# app.py
import os
import io
import json
import time
import base64
from typing import Dict, Any, Tuple

# "spaces" é opcional (só existe em HF Spaces). Se não tiver, vira no-op.
try:
    import spaces  # type: ignore
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(fn):
        return fn

import numpy as np  # <-- IMPORTANTE p/ sanitização
import torch
import gradio as gr
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# -------------------------------------------------------------------
# Pesos do OmniParser (baixar 1x)
# -------------------------------------------------------------------
REPO_ID = "microsoft/OmniParser-v2.0"
WEIGHTS_DIR = "weights"
ICON_DETECT_PT = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")
ICON_CAPTION_DIR = os.path.join(WEIGHTS_DIR, "icon_caption")


def ensure_weights():
    need_download = False
    if not os.path.isdir(WEIGHTS_DIR):
        need_download = True
    else:
        if not os.path.isfile(ICON_DETECT_PT):
            need_download = True
        if not os.path.isdir(ICON_CAPTION_DIR):
            need_download = True

    if need_download:
        print("Baixando repositório de pesos do Hugging Face...")
        snapshot_download(repo_id=REPO_ID, local_dir=WEIGHTS_DIR)
        print(f"Repository downloaded to: {WEIGHTS_DIR}")
    else:
        print(f"Pesos já presentes em: {WEIGHTS_DIR}")


ensure_weights()

# -------------------------------------------------------------------
# Modelos (carregados 1x)
# -------------------------------------------------------------------
yolo_model = get_yolo_model(model_path=ICON_DETECT_PT)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=ICON_CAPTION_DIR
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _sanitize_json(obj):
    """
    Converte numpy/torch/PIL/sets em tipos Python puros para JSON.
    """
    if isinstance(obj, dict):
        return {str(_sanitize_json(k)): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json(x) for x in obj]
    # numpy escalares e arrays
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    return obj


def _xyxy_ratio_to_abs(xyxy_ratio, w, h):
    x1 = int(round(float(xyxy_ratio[0]) * w))
    y1 = int(round(float(xyxy_ratio[1]) * h))
    x2 = int(round(float(xyxy_ratio[2]) * w))
    y2 = int(round(float(xyxy_ratio[3]) * h))
    return [x1, y1, x2, y2]


def _annotated_pil_from_b64(png_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(png_b64)))


def _build_result_json(
    *,
    image_input: Image.Image,
    dino_labeled_img_b64: str,
    label_coordinates_ratio: Dict[str, Any],
    parsed_boxes: list,
    ocr_text_list: list,
    ocr_bbox_xyxy_abs: list,
    box_threshold: float,
    iou_threshold: float,
    imgsz: int,
    use_paddleocr: bool,
    total_ms: float,
    stage_times: Dict[str, int],
    describe_icons: bool
) -> Dict[str, Any]:
    W, H = image_input.size

    # elements “bonitinhos” (xyxy ratio + absolutos)
    elements = []
    for idx, b in enumerate(parsed_boxes):
        b_type = b.get("type")
        b_inter = bool(b.get("interactivity", False))
        b_content = b.get("content", None)
        bbox_ratio = list(map(float, b.get("bbox", [0, 0, 0, 0])))
        bbox_abs = _xyxy_ratio_to_abs(bbox_ratio, W, H)
        elements.append({
            "id": idx,
            "type": b_type,
            "interactivity": b_inter,
            "content": b_content,
            "bbox": {
                "format": "xyxy",
                "ratio": bbox_ratio,
                "absolute": bbox_abs
            }
        })

    # OCR “bonitinho”
    ocr_items = []
    for i, (txt, bb_abs) in enumerate(zip(ocr_text_list, ocr_bbox_xyxy_abs)):
        ocr_items.append({
            "id": i,
            "text": txt,
            "bbox": {"format": "xyxy", "absolute": list(map(int, bb_abs))}
        })

    # label_coordinates (ratio -> absoluto)
    label_coords = {}
    for k, v in (label_coordinates_ratio or {}).items():
        # v pode ser np.ndarray
        v_list = v.tolist() if isinstance(v, np.ndarray) else v
        cx_r, cy_r, w_r, h_r = [float(x) for x in v_list]
        x_abs = int(round(cx_r * W))
        y_abs = int(round(cy_r * H))
        w_abs = int(round(w_r * W))
        h_abs = int(round(h_r * H))
        label_coords[str(k)] = {
            "xywh": {"ratio": [cx_r, cy_r, w_r, h_r], "absolute": [x_abs, y_abs, w_abs, h_abs]}
        }

    # Versões/ambiente
    try:
        import gradio as _gr
        gradio_ver = getattr(_gr, "__version__", None)
    except Exception:
        gradio_ver = None
    torch_ver = torch.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result_json = {
        "processing": {
            "status": "ok",
            "time_ms": int(round(total_ms)),
            "stages_ms": stage_times,
            "describe_icons": describe_icons
        },
        "input": {"image_size": {"width": W, "height": H}},
        "parameters": {
            "box_threshold": float(box_threshold),
            "iou_threshold": float(iou_threshold),
            "imgsz": int(imgsz),
            "use_paddleocr": bool(use_paddleocr)
        },
        "environment": {
            "device": device,
            "torch_version": torch_ver,
            "gradio_version": gradio_ver
        },
        "outputs": {
            "elements": elements,
            "ocr_raw": {
                "texts": ocr_text_list,
                "bboxes_xyxy_absolute": [list(map(int, bb)) for bb in ocr_bbox_xyxy_abs]
            },
            "label_coordinates": label_coords,
            "model_raw": {
                "som": {
                    "annotated_image_b64_png": dino_labeled_img_b64,
                    "label_coordinates_ratio_raw": label_coordinates_ratio,  # pode ter numpy -> sanitiza depois
                    "parsed_boxes_raw": parsed_boxes                         # idem
                },
                "ocr": {
                    "texts_raw": ocr_text_list,
                    "bboxes_xyxy_abs_raw": [list(map(int, bb)) for bb in ocr_bbox_xyxy_abs]
                }
            }
        }
    }

    # GARANTE que tudo é serializável:
    return _sanitize_json(result_json)


# -------------------------------------------------------------------
# Núcleo de pipeline (reutilizado pelo Gradio e pela API)
# -------------------------------------------------------------------
@torch.inference_mode()
def run_pipeline(
    image_input: Image.Image,
    *,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
    describe_icons: bool
) -> Tuple[Dict[str, Any], Image.Image]:
    """
    Executa OCR + detecção + (opcional) legendas + empacota JSON.
    Retorna (result_json, annotated_image)
    """
    t0 = time.perf_counter()
    stages: Dict[str, int] = {}

    # 1) OCR
    t_ocr0 = time.perf_counter()
    ocr_bbox_rslt, _ = check_ocr_box(
        image_input,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )
    ocr_text_list, ocr_bbox_xyxy_abs = ocr_bbox_rslt
    stages["ocr_ms"] = int(round((time.perf_counter() - t_ocr0) * 1000))

    # 2) YOLO + (opcional) caption
    t_det0 = time.perf_counter()
    cap_proc = caption_model_processor if describe_icons else None
    dino_labeled_img_b64, label_coordinates_ratio, parsed_boxes = get_som_labeled_img(
        image_input,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox_xyxy_abs,
        draw_bbox_config=None,
        caption_model_processor=cap_proc,   # <- desliga se describe_icons=False
        ocr_text=ocr_text_list,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    stages["detect_and_caption_ms"] = int(round((time.perf_counter() - t_det0) * 1000))

    annotated_image = _annotated_pil_from_b64(dino_labeled_img_b64)
    total_ms = (time.perf_counter() - t0) * 1000.0

    result_json = _build_result_json(
        image_input=image_input,
        dino_labeled_img_b64=dino_labeled_img_b64,
        label_coordinates_ratio=label_coordinates_ratio,
        parsed_boxes=parsed_boxes,
        ocr_text_list=ocr_text_list,
        ocr_bbox_xyxy_abs=ocr_bbox_xyxy_abs,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        use_paddleocr=use_paddleocr,
        total_ms=total_ms,
        stage_times=stages,
        describe_icons=describe_icons
    )
    return result_json, annotated_image


# -------------------------------------------------------------------
# Gradio UI (streaming + tempos)
# -------------------------------------------------------------------
MARKDOWN = """
# OmniParser V2 for Pure Vision Based General GUI Agent 🔥

OmniParser converte telas de GUI em elementos estruturados. Abaixo você verá **todo** o JSON que o pipeline gera.
"""


@GPU_DECORATOR
@torch.inference_mode()
def gradio_process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz, describe_icons):
    t0 = time.perf_counter()
    yield None, ""  # limpa UI

    if image_input is None:
        yield None, json.dumps({"error": "Envie uma imagem primeiro."}, ensure_ascii=False, indent=2)
        return

    try:
        # feedback inicial
        yield None, json.dumps({"status": "executando pipeline..."}, ensure_ascii=False, indent=2)

        # executa pipeline
        result_json, annotated_image = run_pipeline(
            image_input,
            box_threshold=float(box_threshold),
            iou_threshold=float(iou_threshold),
            use_paddleocr=bool(use_paddleocr),
            imgsz=int(imgsz),
            describe_icons=bool(describe_icons),
        )

        # tempo “ao vivo”
        result_json["processing"]["time_ms_stream"] = int(round((time.perf_counter() - t0) * 1000))
        # Já está sanitizado dentro do run_pipeline/_build_result_json
        yield annotated_image, json.dumps(result_json, ensure_ascii=False, indent=2)

    except Exception as e:
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        yield None, json.dumps(
            {"processing": {"status": "error", "time_ms": elapsed_ms, "message": str(e)}},
            ensure_ascii=False,
            indent=2,
        )


def build_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(type="pil", label="Upload image")
                box_threshold_component = gr.Slider(label="Box Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.05)
                iou_threshold_component = gr.Slider(label="IOU Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                use_paddleocr_component = gr.Checkbox(label="Use PaddleOCR", value=False)
                imgsz_component = gr.Slider(label="Icon Detect Image Size", minimum=640, maximum=1920, step=32, value=640)
                describe_icons_component = gr.Checkbox(label="Descrever ícones (Florence-2)", value=False)
                submit_button_component = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                image_output_component = gr.Image(type="pil", label="Image Output")
                text_output_component = gr.Textbox(
                    label="Parsed screen elements (JSON) — completo",
                    placeholder="Resultados em JSON aparecem aqui",
                    lines=28,
                )

        submit_button_component.click(
            fn=gradio_process,
            inputs=[
                image_input_component,
                box_threshold_component,
                iou_threshold_component,
                use_paddleocr_component,
                imgsz_component,
                describe_icons_component,
            ],
            outputs=[image_output_component, text_output_component],
        )
    return demo


# -------------------------------------------------------------------
# FastAPI (API + monta a UI do Gradio)
# -------------------------------------------------------------------
app = FastAPI(title="OmniParser API", version="1.1.1")

# CORS aberto (ajuste se quiser restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/parse")
async def api_parse(
    file: UploadFile = File(..., description="Imagem (png/jpg/jpeg)"),
    box_threshold: float = Form(0.05),
    iou_threshold: float = Form(0.1),
    use_paddleocr: bool = Form(False),
    imgsz: int = Form(640),
    describe_icons: bool = Form(False),
    return_image: bool = Form(False),
):
    """
    Envie uma imagem via multipart/form-data para obter **todo** o JSON do pipeline.
    - describe_icons=True ativa a legenda dos ícones (lento em CPU)
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result_json, annotated_image = run_pipeline(
            image_input=image,
            box_threshold=float(box_threshold),
            iou_threshold=float(iou_threshold),
            use_paddleocr=bool(use_paddleocr),
            imgsz=int(imgsz),
            describe_icons=bool(describe_icons),
        )

        if return_image:
            buf = io.BytesIO()
            annotated_image.save(buf, format="PNG")
            b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
            result_json["outputs"]["annotated_image_b64_png"] = b64_png

        # Já vem sanitizado de run_pipeline/_build_result_json, mas reforço:
        return JSONResponse(_sanitize_json(result_json))
    except Exception as e:
        return JSONResponse({"processing": {"status": "error", "message": str(e)}}, status_code=500)


# Monta a UI do Gradio dentro do FastAPI em "/"
demo = build_gradio_ui()
# queue() sem args — compatível com sua versão
gr.mount_gradio_app(app, demo.queue(), path="/")


if __name__ == "__main__":
    # workers=1 evita múltiplos loads de modelo na mesma porta
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
