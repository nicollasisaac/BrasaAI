# tools/screenshot.py  (ou onde estiver esse utilitário)
from pathlib import Path
from uuid import uuid4
from io import BytesIO
import os

from PIL import Image
from mss import mss

# Se você usa essas bases em outro lugar, mantenha o import de ToolError
from .base import ToolError  # BaseAnthropicTool não é necessário aqui

OUTPUT_DIR = "./tmp/outputs"

def _ensure_output_dir() -> Path:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out

def _resize_image_keep_aspect(im: Image.Image, target_width: int | None = None, target_height: int | None = None) -> Image.Image:
    """
    Redimensiona mantendo proporção.
    - Se só width for passado, calcula height proporcional.
    - Se só height for passado, calcula width proporcional.
    - Se ambos forem passados, encaixa (letterbox não aplicado; apenas 'fit' dentro da caixa).
    """
    if not target_width and not target_height:
        return im

    w, h = im.size
    if target_width and not target_height:
        scale = target_width / float(w)
        new_size = (int(w * scale), int(h * scale))
    elif target_height and not target_width:
        scale = target_height / float(h)
        new_size = (int(w * scale), int(h * scale))
    else:
        # fit dentro da caixa (target_width x target_height)
        scale = min(target_width / float(w), target_height / float(h))
        new_size = (int(w * scale), int(h * scale))

    if new_size == im.size:
        return im
    return im.resize(new_size, Image.LANCZOS)

def get_screenshot(
    *,
    monitor_index: int = 1,
    resize: bool = False,
    target_width: int | None = 1920,
    target_height: int | None = None
):
    """
    Captura um screenshot da **tela física** (monitor) usando mss.

    Parâmetros:
    - monitor_index: índice do monitor (mss usa 1 = primeiro monitor; 0 = TODOS os monitores).
    - resize: se True, faz resize mantendo proporção.
    - target_width / target_height: dimensões desejadas (uma ou ambas). Mantém proporção (fit).

    Retorna:
    - (PIL.Image, caminho_arquivo: Path)

    Exceções:
    - ToolError se falhar.
    """
    output_dir = _ensure_output_dir()
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    try:
        with mss() as sct:
            monitors = sct.monitors  # monitor[0] = virtual screen (tudo); 1..N = monitores físicos
            if monitor_index < 0 or monitor_index >= len(monitors):
                raise ToolError(
                    f"Monitor index {monitor_index} inválido. Disponíveis: 0..{len(monitors)-1} "
                    f"(0=todas as telas, 1=primeiro monitor)."
                )

            region = monitors[monitor_index]
            raw = sct.grab(region)
            # Converte BGRA -> RGBA -> RGB
            im = Image.frombytes("RGB", raw.size, raw.rgb)

            if resize:
                im = _resize_image_keep_aspect(im, target_width=target_width, target_height=target_height)

            im.save(path)
            return im, path

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to capture screenshot from monitor {monitor_index}: {str(e)}")
