# agent/omnitool/gradio/agent/llm_utils/omniparserclient.py
import os
from typing import Dict, Any, Optional
import requests
from PIL import Image
from io import BytesIO

def _png_bytes(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

class OmniParserClient:
    """
    Cliente para FastAPI /api/parse (multipart/form-data).
    Envia:
      - files={"file": ("screenshot.png", <bytes>, "image/png")}
      - data={
          "box_threshold": float,
          "iou_threshold": float,
          "use_paddleocr": bool,
          "imgsz": int,
          "describe_icons": bool,
          "return_image": bool,
        }
    Retorno: JSON do pipeline (screen_info, parsed_content_list, som_image_base64, etc.)
    """

    def __init__(self, url: Optional[str] = None) -> None:
        base = (url or os.environ.get("OMNIPARSER_URL") or "http://127.0.0.1:7860").rstrip("/")
        # o usuário pode passar já com /api/parse; vamos normalizar:
        self.url = base if base.endswith("/api/parse") else (base + "/api/parse")

        # Tunables via env (mesmo nome que você usa na VM)
        self.box_threshold = float(os.environ.get("OMNI_BOX_THRESHOLD", "0.05"))
        self.iou_threshold = float(os.environ.get("OMNI_IOU_THRESHOLD", "0.10"))
        self.use_paddleocr = os.environ.get("OMNI_USE_PADDLEOCR", "1") == "1"
        self.imgsz = int(os.environ.get("OMNI_IMG_SIZE", "1280"))
        self.describe_icons = os.environ.get("OMNI_DESCRIBE_ICONS", "0") == "1"
        self.return_image = os.environ.get("OMNI_RETURN_IMAGE", "0") == "1"
        self.timeout = int(os.environ.get("OMNI_TIMEOUT", "120"))

    def parse_pil(self, screenshot: Image.Image) -> Dict[str, Any]:
        png_bytes = _png_bytes(screenshot)

        files = {
            "file": ("screenshot.png", png_bytes, "image/png")
        }
        data = {
            "box_threshold": str(self.box_threshold),
            "iou_threshold": str(self.iou_threshold),
            "use_paddleocr": "true" if self.use_paddleocr else "false",
            "imgsz": str(self.imgsz),
            "describe_icons": "true" if self.describe_icons else "false",
            "return_image": "true" if self.return_image else "false",
        }

        # OBS: NÃO enviar headers Content-Type manualmente; requests monta boundary correto.
        resp = requests.post(self.url, files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
