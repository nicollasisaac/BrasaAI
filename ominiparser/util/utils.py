# from ultralytics import YOLO
import os, io, base64, time, json, sys, cv2, numpy as np, torch, re
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Union, Dict
from torchvision.ops import box_convert
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
import supervision as sv
from util.box_annotator import BoxAnnotator

# ---- OCR backends ----
import easyocr
reader = easyocr.Reader(['en'])

# PaddleOCR é opcional (lazy)
HAS_PADDLE = False
paddle_ocr = None
def _lazy_load_paddle():
    global HAS_PADDLE, paddle_ocr
    if paddle_ocr is not None:
        return True
    try:
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(
            lang='en', use_angle_cls=False, use_gpu=False,
            show_log=False, max_batch_size=1024, use_dilation=True,
            det_db_score_mode='slow', rec_batch_num=1024
        )
        HAS_PADDLE = True
        return True
    except Exception:
        HAS_PADDLE = False
        paddle_ocr = None
        return False

def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    """
    Retorna dict {'model': ..., 'processor': ...}
    - Para Florence-2: desabilita SDPA, força atenção eager, define generation_config
      com do_sample=False, num_beams=1, use_cache=False pra evitar o bug do past_key_values.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration, GenerationConfig
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        dtype = torch.float16 if device != 'cpu' else torch.float32
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=dtype
        )
        if device != 'cpu':
            model = model.to(device)
        # Configuração de geração segura/consistente
        gen_cfg = GenerationConfig()
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.use_cache = False
        gen_cfg.max_new_tokens = 32
        model.config.use_cache = False
        model.generation_config = gen_cfg

    elif model_name == "florence2":
        # Evita SDPA (que estava quebrando com _supports_sdpa ausente)
        os.environ.setdefault("PYTORCH_SDP_DISABLE_BACKEND", "1")
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, GenerationConfig

        # Pode usar "microsoft/Florence-2-base" (peso off-the-shelf) ou um caminho local/ft
        base_id = "microsoft/Florence-2-base"
        load_id = model_name_or_path if model_name_or_path else base_id

        processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
        config = AutoConfig.from_pretrained(load_id, trust_remote_code=True)

        # Força atenção em modo eager (evita kernels incompatíveis no CPU)
        try:
            setattr(config, "attn_implementation", "eager")
            setattr(config, "_attn_implementation", "eager")
        except Exception:
            pass

        dtype = torch.float32 if device == 'cpu' else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            load_id, config=config, dtype=dtype, trust_remote_code=True
        )

        if device != 'cpu':
            model = model.to(device)

        model.eval()
        # Geração segura para evitar o bug do past_key_values
        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.do_sample = False         # greedy
        gen_cfg.num_beams = 1
        gen_cfg.use_cache = False
        gen_cfg.max_new_tokens = 32

        model.config.use_cache = False
        model.generation_config = gen_cfg

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return {'model': model.to(device), 'processor': processor}

def get_yolo_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=None):
    """
    Gera descrições dos crops com BLIP2/Florence2.
    - Força geração greedy e use_cache=False (no model.generation_config)
    - Remove sampling/early_stopping
    """
    to_pil = ToPILImage()
    non_ocr_boxes = filtered_boxes[starting_idx:] if starting_idx else filtered_boxes
    croped_pil_image = []
    for coord in non_ocr_boxes:
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except Exception:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        # Florence-2 usa tags especiais; para caption genérica "<CAPTION>"
        is_florence = hasattr(model, "config") and isinstance(getattr(model.config, "name_or_path", ""), str) and "Florence" in model.config.name_or_path
        prompt = "<CAPTION>" if is_florence else "The image shows"

    generated_texts = []
    device = model.device
    if batch_size is None:
        batch_size = 8 if device.type == 'cpu' else 64

    # Geração "à prova de bug" (greedy + sem cache)
    # Se o modelo não tiver generation_config, criamos uma aqui.
    try:
        from transformers import GenerationConfig
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is None:
            gen_cfg = GenerationConfig()
        gen_cfg = gen_cfg.clone() if hasattr(gen_cfg, "clone") else gen_cfg
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.use_cache = False
        # Limite curto porque os crops são pequenos
        if not hasattr(gen_cfg, "max_new_tokens") or gen_cfg.max_new_tokens is None:
            gen_cfg.max_new_tokens = 20
    except Exception:
        gen_cfg = None  # fallback: passaremos kwargs diretamente

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if not batch:
            continue

        if device.type == 'cuda':
            inputs = processor(
                images=batch,
                text=[prompt]*len(batch),
                return_tensors="pt",
                do_resize=False
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch,
                text=[prompt]*len(batch),
                return_tensors="pt",
                do_resize=False
            ).to(device=device)

        # Monta kwargs de maneira defensiva (nem todo processor retorna os mesmos campos)
        gen_kwargs = {}
        if "input_ids" in inputs:        gen_kwargs["input_ids"] = inputs["input_ids"]
        if "attention_mask" in inputs:   gen_kwargs["attention_mask"] = inputs["attention_mask"]
        if "pixel_values" in inputs:     gen_kwargs["pixel_values"] = inputs["pixel_values"]

        try:
            if gen_cfg is not None:
                generated_ids = model.generate(
                    **gen_kwargs,
                    generation_config=gen_cfg
                )
            else:
                generated_ids = model.generate(
                    **gen_kwargs,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                    max_new_tokens=20
                )
        except AttributeError:
            # Fallback extra (algumas builds do Florence-2 no CPU ainda reclamam)
            generated_ids = model.generate(
                **gen_kwargs,
                do_sample=False,
                num_beams=1,
                use_cache=False,
                max_new_tokens=20
            )

        # Decodifica
        try:
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception:
            # Alguns processors usam método diferente
            generated_text = [processor.decode(gid, skip_special_tokens=True) for gid in generated_ids]

        generated_texts.extend([str(gen).strip() for gen in generated_text])

    return generated_texts

def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    # usar 'list' em vez de typing.List em isinstance
    assert (ocr_bbox is None) or isinstance(ocr_bbox, list)

    def box_area(b): return (b[2] - b[0]) * (b[3] - b[1])
    def inter_area(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)
    def IoU(b1, b2):
        inter = inter_area(b1,b2)
        union = box_area(b1)+box_area(b2)-inter + 1e-6
        r1 = inter/(box_area(b1) or 1); r2 = inter/(box_area(b2) or 1)
        return max(inter/union, r1, r2)
    def is_inside(b1,b2):
        inter = inter_area(b1,b2)
        return (inter/(box_area(b1) or 1)) > 0.80

    filtered = []
    if ocr_bbox: filtered.extend(ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid = True
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem['bbox']
            if i!=j and IoU(box1, box2) > iou_threshold and (box2[2]-box2[0])*(box2[3]-box2[1]) < (box1[2]-box1[0])*(box1[3]-box1[1]):
                is_valid = False; break
        if is_valid:
            if ocr_bbox:
                box_added = False; ocr_labels = ''
                for box3_elem in list(ocr_bbox):
                    box3 = box3_elem['bbox']
                    if is_inside(box3, box1):
                        ocr_labels += (box3_elem['content'] + ' ')
                        try: filtered.remove(box3_elem)
                        except Exception: pass
                    elif is_inside(box1, box3):
                        box_added = True; break
                if not box_added:
                    filtered.append({'type':'icon','bbox':box1_elem['bbox'],'interactivity':True,'content':(ocr_labels or None)})
            else:
                filtered.append(box1)
    return filtered

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)
    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding, text_thickness=text_thickness, thickness=thickness)
    annotated = image_source.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections, labels=labels, image_size=(w,h))
    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated, label_coordinates

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    if scale_img:
        result = model.predict(source=image, conf=box_threshold, imgsz=imgsz, iou=iou_threshold)
    else:
        result = model.predict(source=image, conf=box_threshold, iou=iou_threshold)
    boxes = result[0].boxes.xyxy
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]
    return boxes, conf, phrases

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    return (int_box[2]-int_box[0])*(int_box[3]-int_box[1])

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9, prompt=None, scale_img=False, imgsz=None, batch_size=64):
    if isinstance(image_source, str):
        image_source = Image.open(image_source).convert("RGB")
    w, h = image_source.size
    if not imgsz: imgsz = (h, w)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_np = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        ocr_bbox = None

    ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt} for box, txt in zip(ocr_bbox or [], ocr_text) if int_box_area(box, w, h) > 0]
    xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x.get('content') is None)
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box.get('content') is None), -1)
    filtered_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_elem]) if filtered_boxes_elem else torch.zeros((0,4))

    if use_local_semantics and caption_model_processor:
        model_dev = caption_model_processor['model'].device if 'model' in caption_model_processor else torch.device('cpu')
        parsed_content_icon = get_parsed_content_icon(filtered_tensor, starting_idx, image_np, caption_model_processor, prompt=prompt, batch_size=(8 if model_dev.type=='cpu' else 64))
        ocr_text_fmt = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text_fmt)
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None and parsed_content_icon:
                box['content'] = parsed_content_icon.pop(0)
        parsed_icon_ls = [f"Icon Box ID {str(i+icon_start)}: {txt}" for i, txt in enumerate(parsed_content_icon)]
        parsed_content_merged = ocr_text_fmt + parsed_icon_ls
    else:
        parsed_content_merged = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]

    if filtered_tensor.numel() > 0:
        filtered_cxcywh = box_convert(boxes=filtered_tensor, in_fmt="xyxy", out_fmt="cxcywh")
    else:
        filtered_cxcywh = torch.zeros((0,4))
    phrases_idx = [i for i in range(len(filtered_cxcywh))]
    annotated_frame, label_coordinates = annotate(image_source=image_np, boxes=filtered_cxcywh, logits=None, phrases=phrases_idx, text_scale=text_scale, text_padding=text_padding)
    pil_img = Image.fromarray(annotated_frame)
    buf = io.BytesIO(); pil_img.save(buf, format="PNG"); encoded_image = base64.b64encode(buf.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
    return encoded_image, label_coordinates, filtered_boxes_elem

def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    return int(x), int(y), int(w), int(h)

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    return int(x), int(y), int(xp), int(yp)

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    return int(x), int(y), int(w), int(h)

def check_ocr_box(image_source: Union[str, Image.Image], display_img=True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    if use_paddleocr and _lazy_load_paddle():
        text_threshold = (easyocr_args or {}).get('text_threshold', 0.5)
        result = paddle_ocr.ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text  = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:
        result = reader.readtext(image_np, **(easyocr_args or {}))
        coord = [item[0] for item in result]
        text  = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item); bb.append((x,y,a,b))
            cv2.rectangle(opencv_img, (x,y), (x+a,y+b), (0,255,0), 2)
        # Visualização opcional (omitida em headless)
    else:
        bb = [get_xywh(item) if output_bb_format=='xywh' else get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering
