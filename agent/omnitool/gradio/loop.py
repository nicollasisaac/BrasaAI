# agent/omnitool/gradio/loop.py
from enum import Enum
from typing import Callable, Dict, Any, Iterable, List, Optional
import json
import time
import os
import requests
import re
import base64
import io
from PIL import Image

# --- OmniParser client ---
from agent.omnitool.gradio.agent.llm_utils.omniparserclient import OmniParserClient
from agent.omnitool.gradio.agent.llm_utils.oaiclient import run_oci_interleaved

# --- Tools (usa originais se tiver, senão fallback) ---
HAS_TOOLS = True
try:
    from agent.omnitool.gradio.tools.collection import ToolCollection  # type: ignore
    from agent.omnitool.gradio.tools.computer import ComputerTool      # type: ignore
except Exception:
    HAS_TOOLS = False

    class ComputerTool:
        """Mínimo para falar com windows_host.py"""
        def __init__(self, base_url: Optional[str] = None):
            self.base_url = base_url or os.environ.get("WINDOWS_HOST_URL", "http://127.0.0.1:8006")
            self.timeout = float(os.environ.get("COMPUTER_TOOL_TIMEOUT", "15"))

        def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            r = requests.post(f"{self.base_url}/execute", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()

        def _move_if_coord(self, coordinate: Optional[List[int]]):
            if coordinate and len(coordinate) == 2:
                x, y = int(coordinate[0]), int(coordinate[1])
                self._post({"action": "move", "x": x, "y": y})

        def __call__(self, *, action: str, coordinate: Optional[List[int]] = None, text: Optional[str] = None) -> Dict[str, Any]:
            a = (action or "").lower()

            # normaliza 'win' -> 'winleft'
            if a == "key" and text:
                norm = []
                for part in str(text).lower().split("+"):
                    part = part.strip()
                    if part == "win":
                        part = "winleft"
                    norm.append(part)
                text = "+".join(norm)

            if a in ("mouse_move", "move", "hover"):
                self._move_if_coord(coordinate)
                return {"ok": True}
            if a in ("left_click", "click"):
                self._move_if_coord(coordinate)
                return self._post({"action": "left_click"})
            if a in ("double_click", "dblclick"):
                self._move_if_coord(coordinate)
                return self._post({"action": "double_click"})
            if a in ("right_click", "context_click"):
                self._move_if_coord(coordinate)
                return self._post({"action": "right_click"})
            if a == "type":
                return self._post({"action": "type", "text": text or ""})
            if a == "key":
                return self._post({"action": "key", "text": text or ""})
            if a in ("scroll_up",):
                return self._post({"action": "scroll", "dy": 300})
            if a in ("scroll_down",):
                return self._post({"action": "scroll", "dy": -300})
            if a == "drag_to":
                if coordinate and len(coordinate) == 2:
                    x, y = int(coordinate[0]), int(coordinate[1])
                    return self._post({"action": "drag_to", "x": x, "y": y, "duration": 0.5})
                return {"error": "drag_to requires coordinate [x,y]"}
            if a == "wait":
                time.sleep(float(text) if text and text.replace(".", "", 1).isdigit() else 0.5)
                return {"ok": True}
            return {"error": f"Unsupported action: {action}"}

    class ToolCollection:
        def __init__(self, computer: ComputerTool):
            self._computer = computer
        def __call__(self, name: str, **kwargs) -> Dict[str, Any]:
            if name == "computer":
                return self._computer(**kwargs)
            raise ValueError(f"Unknown tool '{name}'")

# =====================

class APIProvider(Enum):
    OCI = "oci"
    GEMINI = "gemini"

OCI_DEFAULT_SYSTEM = """
You are a precise, step-by-step computer control agent.
You have FULL control over a Windows PC.
The user can ask for anything: open apps, browse sites, type text, scroll, etc.

You will be given:
- [screen_info]: structured summary of the screen
- [parsed_content]: parsed UI elements with bounding boxes

Policy:
- ALWAYS propose one actionable step at a time.
- If elements are missing or you are unsure, use exploratory actions (open Start menu with the Windows key, type queries, scroll, hover).
- Prefer keyboard search workflows (Win -> type -> Enter).
- Never output anything outside the JSON.
- Keep trying unless explicitly impossible.

Return STRICT JSON ONLY:
{
  "analysis": "<short reasoning>",
  "next": {
    "action": "mouse_move|left_click|double_click|right_click|type|key|scroll_up|scroll_down|wait|hover",
    "coordinate": [x, y] | null,
    "text": "<if typing or key>",
    "done": false
  }
}
"""

def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", [])
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            return str(first.get("text", ""))
        if isinstance(first, str):
            return first
    return str(last)

def _guess_target_from_user_text(user_text: str) -> Optional[str]:
    if not user_text:
        return None
    t = user_text.strip().lower()
    triggers = ["abrir ", "abra ", "open ", "executar ", "iniciar ", "search ", "buscar "]
    for trig in triggers:
        if trig in t:
            cand = t.split(trig, 1)[1].strip()
            for sep in [";", ",", " e ", " então ", " depois ", " depois,", " depois "]:
                idx = cand.find(sep)
                if idx != -1:
                    cand = cand[:idx]
            cand = cand.strip(" '\"")
            if cand:
                return cand
    words = re.findall(r"[a-z0-9\.\-_/]+", t)
    if words:
        return words[-1]
    return None

def _robust_json_parse(text: str) -> Optional[dict]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None

def _pil_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _keyboard_open(tools, query: str, output_callback):
    output_callback(f"[macro] Opening via keyboard search: {query}")
    tools("computer", action="key", text="win")
    time.sleep(0.3)
    tools("computer", action="type", text=query)
    time.sleep(0.25)
    tools("computer", action="key", text="enter")
    time.sleep(0.9)

def _pick_overlay_b64(parsed: Dict[str, Any]) -> Optional[str]:
    # Tenta várias chaves comuns de overlay do OmniParser
    for k in ("som_image_base64", "image_base64", "overlay_image_base64", "vis_image_base64"):
        b = parsed.get(k)
        if isinstance(b, str) and len(b) > 50:
            return b
    return None

def sampling_loop_sync(
    *,
    model: str,
    provider: str,
    messages: List[Dict[str, Any]],
    output_callback: Callable[[Any], None],
    tool_output_callback: Callable[[Dict[str, Any], str], None],
    api_response_callback: Callable[[Any], None],
    api_key: Optional[str] = None,
    only_n_most_recent_images: Optional[int] = 2,
    max_tokens: int = 1024,
    omniparser_url: str = "http://127.0.0.1:7860",
    save_folder: str = "./tmp/outputs",
) -> Iterable[Optional[List[Any]]]:

    prov = provider.lower().strip()
    if prov not in ("oci", "gemini"):
        output_callback("Unsupported provider. Use 'gemini' or 'oci'.")
        yield None
        return

    computer = ComputerTool(os.environ.get("WINDOWS_HOST_URL"))
    tools = ToolCollection(computer)

    last_actions = []
    def _push_action(sig: str):
        last_actions.append(sig)
        if len(last_actions) > 4:
            last_actions.pop(0)

    user_text = _extract_user_text(messages)
    guessed = _guess_target_from_user_text(user_text)

    for step_i in range(12):
        # 1) Screenshot
        output_callback(f"Step {step_i+1}: Capturing the screen…")
        try:
            from agent.omnitool.gradio.tools.screen_capture import get_screenshot
            pil_image, _ = get_screenshot(resize=False)
        except Exception:
            host = os.environ.get("WINDOWS_HOST_URL", "http://127.0.0.1:8006")
            r = requests.get(f"{host}/screenshot", timeout=10)
            r.raise_for_status()
            b64 = r.json().get("image_base64", "")
            pil_image = Image.open(io.BytesIO(base64.b64decode(b64))) if b64 else None

        # 2) OmniParser
        output_callback("Parsing with OmniParser (VM)…")
        omni = OmniParserClient(url=omniparser_url)
        parsed = omni.parse_pil(pil_image) if pil_image else {}
        screen_info = parsed.get("screen_info", "")
        parsed_content_list = parsed.get("parsed_content_list", "")

        # 2a) escolhe imagem pra exibir (overlay ou screenshot)
        parsed_b64 = _pick_overlay_b64(parsed)
        if not parsed_b64 and pil_image is not None:
            parsed_b64 = _pil_to_b64(pil_image)

        # 3) Prompt LLM
        mm_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Task from user:"},
                {"type": "text", "text": user_text},
                {"type": "text", "text": f"[screen_info]\n{screen_info}"},
                {"type": "text", "text": f"[parsed_content]\n{parsed_content_list}"},
            ]
        }]

        oci_text, _ = run_oci_interleaved(
            mm_messages,
            system=OCI_DEFAULT_SYSTEM,
            max_tokens=max_tokens,
            temperature=float(os.environ.get("LLM_TEMP", "0.2")),
        )
        output_callback(f"Model analysis:\n{oci_text}")

        # 4) Parse JSON
        parsed_json = _robust_json_parse(oci_text)
        if not parsed_json or "next" not in parsed_json:
            if guessed:
                _keyboard_open(tools, guessed, output_callback)
                _push_action(f"macro_open:{guessed}")
                # YIELD imagem + análise (vazia) deste passo
                yield {"step": step_i + 1, "analysis": None, "image_b64": parsed_b64}
                continue
            else:
                tools("computer", action="key", text="win")
                _push_action("key:win")
                time.sleep(0.3)
                yield {"step": step_i + 1, "analysis": None, "image_b64": parsed_b64}
                continue

        step = parsed_json["next"]
        analysis_text = parsed_json.get("analysis", "")

        action = (step.get("action") or "").lower()
        coordinate = step.get("coordinate")
        text = step.get("text")

        sig = f"{action}:{coordinate}:{text}"
        if last_actions.count(sig) >= 2:
            output_callback("[loop-guard] Repeated same action → switching to keyboard search")
            if guessed:
                _keyboard_open(tools, guessed, output_callback)
                _push_action(f"macro_open:{guessed}")
            else:
                tools("computer", action="key", text="win")
                _push_action("key:win")
                time.sleep(0.3)
            # YIELD imagem + análise do passo
            yield {"step": step_i + 1, "analysis": analysis_text, "image_b64": parsed_b64}
            continue

        # 6) Executa ação planejada
        if action and action != "none":
            try:
                result = tools("computer", action=action, coordinate=coordinate, text=text)
                tool_output_callback(result, "computer")
                output_callback(result)
                _push_action(sig)
            except Exception as e:
                output_callback(f"Tool error: {e}")
        else:
            if guessed:
                _keyboard_open(tools, guessed, output_callback)
                _push_action(f"macro_open:{guessed}")
            else:
                tools("computer", action="key", text="win")
                _push_action("key:win")
            # YIELD imagem + análise do passo
            yield {"step": step_i + 1, "analysis": analysis_text, "image_b64": parsed_b64}
            continue

        # 7) YIELD imagem + análise deste passo para a Gallery/log
        yield {"step": step_i + 1, "analysis": analysis_text, "image_b64": parsed_b64}

        if bool(step.get("done")):
            output_callback("Task marked as done.")
            yield None
            return

    # terminou tentativas
    yield None
