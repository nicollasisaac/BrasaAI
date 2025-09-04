import os
import base64
import time
import re
from enum import StrEnum
from typing import Literal, TypedDict, List, Tuple

try:
    import pyautogui
    PYAUTOGUI_OK = True
    pyautogui.FAILSAFE = False  # evita abortar quando o mouse vai ao canto
except Exception:
    PYAUTOGUI_OK = False

import requests
from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .screen_capture import get_screenshot

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "hover",
    "wait",
    "scroll_up",
    "scroll_down",
    "left_press",
]

class Resolution(TypedDict):
    width: int
    height: int

MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),    # 4:3
    "WXGA": Resolution(width=1280, height=800),   # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"

class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

def chunks(s: str, chunk_size: int) -> List[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    """
    Controla o computador local (mouse/teclado/scroll) via pyautogui.
    Fallback: se WINDOWS_HOST_URL estiver definido OU COMPUTER_BACKEND=http,
    usa backend HTTP compatível (mesma API anterior).
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    def __init__(self, is_scaling: bool = False, windows_host_url: str | None = None):
        super().__init__()

        # Detecta backend
        env_url = os.environ.get("WINDOWS_HOST_URL", "").strip()
        self.base_url = (windows_host_url or env_url or "").strip()
        backend_env = os.environ.get("COMPUTER_BACKEND", "").lower()
        if self.base_url or backend_env == "http":
            self.backend = "http"
            self.base_url = (self.base_url or "http://localhost:5000").rstrip("/")
        else:
            self.backend = "local"

        if self.backend == "local" and not PYAUTOGUI_OK:
            raise ToolError("Backend LOCAL selecionado, mas pyautogui não está disponível. Rode: pip install pyautogui pypiwin32")

        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.width, self.height = self.get_screen_size()
        print(f"[ComputerTool] backend={self.backend} | screen size: {self.width}x{self.height}")

        self.key_conversion = {
            "Page_Down": "pagedown",
            "Page_Up": "pageup",
            "Super_L": "win",
            "Escape": "esc",
        }

        self.target_dimension: Resolution | None = None

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(ScalingSource.COMPUTER, self.width, self.height)
        return {"display_width_px": width, "display_height_px": height, "display_number": self.display_num}

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    # ---------------- Core call ----------------
    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(f"[ComputerTool] action={action} text={text} coord={coordinate} scaling={self.is_scaling}")

        # --- ações com coordenadas ---
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of ints")

            if self.is_scaling:
                x, y = self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])
            else:
                x, y = coordinate

            print(f"[ComputerTool] mouse target -> {x},{y}")

            if action == "mouse_move":
                self._mouse_move(x, y)
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            else:  # left_click_drag
                cx, cy = self._cursor_position()
                self._drag_to(x, y, duration=0.5)
                return ToolResult(output=f"Dragged mouse from ({cx}, {cy}) to ({x}, {y})")

        # --- teclado / digitação ---
        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                self._press_combo(text)
                return ToolResult(output=f"Pressed keys: {text}")
            else:  # "type"
                # clique opcional antes de digitar (ajuda a focar)
                self._left_click()
                self._typewrite(text, delay_ms=TYPING_DELAY_MS)
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        # --- ações sem coords ---
        if action in ("left_click", "right_click", "double_click", "middle_click", "screenshot", "cursor_position", "left_press"):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = self._cursor_position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            elif action == "left_click":
                self._left_click()
            elif action == "right_click":
                self._right_click()
            elif action == "middle_click":
                self._middle_click()
            elif action == "double_click":
                self._double_click()
            elif action == "left_press":
                self._mouse_down_up(hold_s=1)
            return ToolResult(output=f"Performed {action}")

        if action in ("scroll_up", "scroll_down"):
            if action == "scroll_up":
                self._scroll(+100)
            else:
                self._scroll(-100)
            return ToolResult(output=f"Performed {action}")

        if action == "hover":
            # no-op sem mudar cursor; já cobrimos com mouse_move
            return ToolResult(output=f"Performed {action}")

        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")

        raise ToolError(f"Invalid action: {action}")

    # ---------------- Backend abstraction ----------------
    def _post_http(self, payload: dict):
        try:
            r = requests.post(f"{self.base_url}/execute", json=payload, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            raise ToolError(f"HTTP backend error: {e}")

    # mouse / teclado / cursor
    def _mouse_move(self, x: int, y: int):
        if self.backend == "http":
            self._post_http({"action": "move", "x": x, "y": y})
        else:
            pyautogui.moveTo(x, y, duration=0.05)

    def _drag_to(self, x: int, y: int, duration: float = 0.5):
        if self.backend == "http":
            self._post_http({"action": "drag_to", "x": x, "y": y, "duration": duration})
        else:
            pyautogui.dragTo(x, y, duration=duration)

    def _left_click(self):
        if self.backend == "http":
            self._post_http({"action": "left_click"})
        else:
            pyautogui.click()

    def _right_click(self):
        if self.backend == "http":
            self._post_http({"action": "right_click"})
        else:
            pyautogui.click(button="right")

    def _middle_click(self):
        if self.backend == "http":
            self._post_http({"action": "middle_click"})
        else:
            pyautogui.middleClick()

    def _double_click(self):
        if self.backend == "http":
            self._post_http({"action": "double_click"})
        else:
            pyautogui.doubleClick()

    def _mouse_down_up(self, hold_s: float = 1.0):
        if self.backend == "http":
            self._post_http({"action": "left_press", "hold": hold_s})
        else:
            pyautogui.mouseDown()
            time.sleep(max(0.0, hold_s))
            pyautogui.mouseUp()

    def _typewrite(self, text: str, delay_ms: int = 12):
        if self.backend == "http":
            self._post_http({"action": "type", "text": text})
        else:
            # para evitar problemas com textos muito longos, dividido em blocos
            for part in chunks(text, TYPING_GROUP_SIZE):
                pyautogui.typewrite(part, interval=delay_ms / 1000.0)

    def _press_combo(self, combo: str):
        # exemplo: "ctrl+shift+esc"
        keys = [self.key_conversion.get(k.strip(), k.strip()).lower() for k in combo.split("+")]
        if self.backend == "http":
            self._post_http({"action": "key_combo", "keys": keys})
        else:
            for k in keys:
                pyautogui.keyDown(k)
            for k in reversed(keys):
                pyautogui.keyUp(k)

    def _cursor_position(self) -> Tuple[int, int]:
        if self.backend == "http":
            resp = self._post_http({"action": "cursor"})
            out = resp.json()
            return int(out["x"]), int(out["y"])
        else:
            p = pyautogui.position()
            return int(p.x), int(p.y)

    def _scroll(self, dy: int):
        if self.backend == "http":
            self._post_http({"action": "scroll", "dy": dy})
        else:
            # pyautogui.scroll: sinal positivo = sobe; negativo = desce
            pyautogui.scroll(dy)

    # ---------------- Screenshot ----------------
    async def screenshot(self):
        # define o alvo de escala uma única vez
        if not hasattr(self, "target_dimension") or self.target_dimension is None:
            self.target_dimension = MAX_SCALING_TARGETS.get("WXGA")

        width, height = self.target_dimension["width"], self.target_dimension["height"]
        screenshot, path = get_screenshot(resize=True, target_width=width, target_height=height)
        time.sleep(0.4)  # dá tempo para a ação anterior refletir
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    # ---------------- Scaling ----------------
    def padding_image(self, screenshot: Image.Image) -> Image.Image:
        """Pad para 16:10 quando a proporção não for 16:10."""
        _, height = screenshot.size
        new_width = height * 16 // 10
        padded = Image.new("RGB", (new_width, height), (255, 255, 255))
        padded.paste(screenshot, (0, 0))
        return padded

    def scale_coordinates(self, source: ScalingSource, x: int, y: int) -> Tuple[int, int]:
        """Converte coords entre a resolução real e a alvo (para modelos)."""
        if not self._scaling_enabled:
            return x, y

        ratio = self.width / self.height
        target_dimension: Resolution | None = None

        for _, dimension in MAX_SCALING_TARGETS.items():
            # tolerância na razão de aspecto
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                break

        if target_dimension is None:
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = target_dimension

        x_sf = target_dimension["width"] / self.width
        y_sf = target_dimension["height"] / self.height

        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # escala "para cima" (coords normalizadas -> coords reais da tela)
            return round(x / x_sf), round(y / y_sf)

        # source == COMPUTER: escala para baixo (coords reais -> alvo)
        return round(x * x_sf), round(y * y_sf)

    def get_screen_size(self) -> Tuple[int, int]:
        """Retorna (width, height) da tela."""
        if self.backend == "http":
            try:
                resp = self._post_http({"action": "get_screen_size"})
                data = resp.json()
                return int(data["width"]), int(data["height"])
            except Exception as e:
                raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")
        else:
            if not PYAUTOGUI_OK:
                raise ToolError("pyautogui indisponível no backend local.")
            w, h = pyautogui.size()
            return int(w), int(h)
