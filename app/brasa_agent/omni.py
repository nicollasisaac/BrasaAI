# brasa_agent/omni.py
# Cliente simples para o endpoint FastAPI /api/parse do OmniParser.
# Envia screenshot (bytes) via multipart/form-data e retorna o JSON.

from typing import Dict, Any, Optional, Union
import requests
import os

__all__ = ["call_omni", "call_omni_from_path"]


def _coerce_form_value(v: Any) -> str:
    """
    Converte valores para strings aceitas em multipart/form-data.
    - bool -> "true"/"false"
    - num/str -> str(v)
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def call_omni(
    base_url: str,
    image_bytes: bytes,
    *,
    box_threshold: Optional[Union[int, float]] = None,
    iou_threshold: Optional[Union[int, float]] = None,
    # ATIVOS POR PADRÃO:
    use_paddleocr: Optional[bool] = True,
    imgsz: Optional[int] = None,
    describe_icons: Optional[bool] = True,
    return_image: Optional[bool] = None,
    timeout: int = 180,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Chama POST {base_url}/api/parse com multipart/form-data.

    Parâmetros opcionais seguem a sua API FastAPI:
      - box_threshold: number
      - iou_threshold: number
      - use_paddleocr: boolean (DEFAULT: True)
      - imgsz: integer
      - describe_icons: boolean (DEFAULT: True)
      - return_image: boolean

    Retorna:
      - dict JSON do servidor (se content-type application/json)
      - caso contrário, {"raw": r.text, "status_code": ..., "content_type": ...}
    """
    url = base_url.rstrip("/") + "/api/parse"

    files = {
        "file": ("screen.png", image_bytes, "image/png"),
    }

    data: Dict[str, str] = {}
    params_map = {
        "box_threshold": box_threshold,
        "iou_threshold": iou_threshold,
        "use_paddleocr": use_paddleocr,
        "imgsz": imgsz,
        "describe_icons": describe_icons,
        "return_image": return_image,
    }
    for k, v in params_map.items():
        if v is not None:
            data[k] = _coerce_form_value(v)

    headers = dict(extra_headers or {})

    resp = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        return resp.json()

    # Fallback quando o servidor não retorna JSON (ex.: string, HTML, etc.)
    return {
        "raw": resp.text,
        "status_code": resp.status_code,
        "content_type": ctype,
    }


def call_omni_from_path(
    base_url: str,
    image_path: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Atalho para ler uma imagem do disco e chamar call_omni.
    """
    with open(image_path, "rb") as f:
        img = f.read()
    return call_omni(base_url, img, **kwargs)


if __name__ == "__main__":
    # Execução rápida de teste manual:
    # export OMNI_API_URL=http://127.0.0.1:7867
    # python -m brasa_agent.omni /caminho/para/screenshot.png
    import sys
    import json as _json

    api = os.getenv("OMNI_API_URL", "http://127.0.0.1:7867")
    if len(sys.argv) < 2:
        print("Uso: python -m brasa_agent.omni /caminho/para/imagem.png")
        sys.exit(1)

    img_path = sys.argv[1]
    out = call_omni_from_path(
        api,
        img_path,
        # por padrão já estão True, mas deixo explícito no exemplo:
        describe_icons=True,
        use_paddleocr=True,
    )
    print(_json.dumps(out, ensure_ascii=False, indent=2))
