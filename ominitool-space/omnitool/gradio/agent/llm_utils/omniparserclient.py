import os
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any
import requests
from PIL import Image
from tools.screen_capture import get_screenshot  # must return (PIL.Image, Path | None)
from io import BytesIO
from uuid import uuid4

def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _encode_image_pil(im: Image.Image) -> str:
    """Encode PIL image as base64 PNG string (no file I/O)."""
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class OmniParserClient:
    def __init__(self, url: str | None = None) -> None:
        """
        Aceita base URL (ex: http://host:7860) ou endpoint completo (…/api/parse) e normaliza.
        Todos os parâmetros são enviados em JSON, com a imagem em base64.
        """
        base = (url or os.environ.get("OMNIPARSER_URL") or "http://127.0.0.1:7860").rstrip("/")
        self.url = base if base.endswith("/api/parse") else (base + "/api/parse")

        # Parâmetros padrão (podem ser sobrescritos via env)
        self.box_threshold = float(os.environ.get("OMNI_BOX_THRESHOLD", "0.05"))
        self.iou_threshold = float(os.environ.get("OMNI_IOU_THRESHOLD", "0.10"))
        self.use_paddleocr = os.environ.get("OMNI_USE_PADDLEOCR", "1") == "1"
        self.use_florence = os.environ.get("OMNI_USE_FLORENCE", "1") == "1"
        self.describe_icons = os.environ.get("OMNI_DESCRIBE_ICONS", "0") == "1"
        self.imgsz = int(os.environ.get("OMNI_IMGSZ", "1280"))

    def __call__(self) -> Dict[str, Any]:
        """
        1) Captura a tela local (monitor físico), sem salvar arquivo.
        2) Envia JSON: { base64_image, params... } para /api/parse do OmniParser.
        3) Normaliza resposta: screen_info, parsed_content_list, etc.
        4) Não grava SOM/screenshot no disco.
        """
        # 1) captura da tela (sua máquina, não VM)
        screenshot, _maybe_path = get_screenshot()
        width, height = screenshot.size
        screenshot_b64 = _encode_image_pil(screenshot)
        screenshot_uuid = uuid4().hex  # não dependemos mais de nome de arquivo

        # 2) POST JSON para OmniParser
        payload = {
            "base64_image": screenshot_b64,
            "box_threshold": self.box_threshold,
            "iou_threshold": self.iou_threshold,
            "use_paddleocr": self.use_paddleocr,
            "use_florence": self.use_florence,
            "describe_icons": self.describe_icons,
            "imgsz": self.imgsz,
            "return_image": True,
        }
        r = requests.post(self.url, json=payload, timeout=180)
        r.raise_for_status()
        resp = r.json()

        # 3) extrair SOM (base64) e elementos
        som_b64 = (
            resp.get("som_image_base64")
            or resp.get("image_base64")
            or resp.get("preview_base64")
        )

        elements = (resp.get("outputs") or {}).get("elements", []) or []
        input_info = resp.get("input") or {}
        in_w = int(input_info.get("width", width))
        in_h = int(input_info.get("height", height))

        # 4) montar screen_info e lista normalizada
        screen_info_lines = []
        parsed_content_list = []
        for idx, el in enumerate(elements):
            txt  = el.get("text") or el.get("name") or el.get("label") or ""
            role = el.get("role") or el.get("type") or ""
            bbox = el.get("bbox") or el.get("normalized_bbox")

            # normalizar bbox (x1,y1,x2,y2) para [0..1]
            if bbox and max(bbox) > 1.001:  # se vier em pixels
                x1, y1, x2, y2 = bbox
                bbox = [x1 / in_w, y1 / in_h, x2 / in_w, y2 / in_h]

            parsed_content_list.append({
                "bbox": bbox or [0, 0, 0, 0],
                "role": role,
                "text": txt,
            })
            if txt or role:
                screen_info_lines.append(f"Box ID {idx}: [{role}] {txt}")

        screen_info = "\n".join(screen_info_lines) if screen_info_lines else "(no textual elements found)"

        # 5) retorno normalizado (compatível com o resto do app)
        return {
            "latency": float(resp.get("latency", 0.0)),
            "som_image_base64": som_b64,
            "original_screenshot_base64": screenshot_b64,
            "width": width,
            "height": height,
            "screenshot_uuid": screenshot_uuid,
            "parsed_content_list": parsed_content_list,
            "screen_info": screen_info,
        }
