# tools/screen_capture.py
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image
from mss import mss

def get_screenshot(
    *,
    monitor_index: int = 1,
    resize: bool = False,
    target_width: Optional[int] = 1920,
    target_height: Optional[int] = None,
) -> Tuple[Image.Image, None]:
    """
    Capture a screenshot from the physical screen using mss.
    Returns (PIL.Image, None). Nothing is written to disk.
    """
    with mss() as sct:
        mons = sct.monitors
        if monitor_index >= len(mons):
            monitor_index = 1 if len(mons) > 1 else 0
        mon = mons[monitor_index]
        raw = sct.grab(mon)
        im = Image.frombytes("RGB", raw.size, raw.rgb)

    if resize and (target_width or target_height):
        w, h = im.size
        if target_width and not target_height:
            scale = target_width / float(w)
            new_size = (int(w * scale), int(h * scale))
        elif target_height and not target_width:
            scale = target_height / float(h)
            new_size = (int(w * scale), int(h * scale))
        else:
            scale = min(target_width / float(w), target_height / float(h))
            new_size = (int(w * scale), int(h * scale))
        if new_size != im.size:
            im = im.resize(new_size, Image.LANCZOS)

    return im, None