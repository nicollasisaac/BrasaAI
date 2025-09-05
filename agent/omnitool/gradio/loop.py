# agent/omnitool/gradio/loop.py
from enum import Enum
from typing import Callable, Dict, Any, Iterable, List, Optional
import json
import time
import os
import requests

# --- OmniParser client (nosso cliente já enviado) ---
from agent.omnitool.gradio.agent.llm_utils.omniparserclient import OmniParserClient
from agent.omnitool.gradio.agent.llm_utils.oaiclient import run_oci_interleaved

# --- Tente usar seus tools originais; se não existir, usamos fallback ---
HAS_TOOLS = True
try:
    # Se o seu projeto tiver esses módulos, ótimo — usaremos eles
    from agent.omnitool.gradio.tools.collection import ToolCollection  # type: ignore
    from agent.omnitool.gradio.tools.computer import ComputerTool      # type: ignore
except Exception:
    HAS_TOOLS = False

    # ===== Fallback mínimo, compatível com o seu windows_host.py =====
    class ComputerTool:
        """
        Implementação mínima: envia ações para o bridge HTTP (windows_host.py).
        Usa WINDOWS_HOST_URL (ex.: http://127.0.0.1:8006).
        """
        def __init__(self, base_url: Optional[str] = None):
            self.base_url = base_url or os.environ.get("WINDOWS_HOST_URL", "http://127.0.0.1:8006")
            self.timeout = float(os.environ.get("COMPUTER_TOOL_TIMEOUT", "15"))

        def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            r = requests.post(f"{self.base_url}/execute", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()

        def __call__(self, *, action: str, coordinate: Optional[List[int]] = None, text: Optional[str] = None) -> Dict[str, Any]:
            # normaliza algumas ações comuns do seu agente
            a = action.lower()
            if a in ("mouse_move", "move", "hover"):
                x, y = (coordinate or [0, 0])
                return self._post({"action": "move", "x": int(x), "y": int(y)})

            if a in ("left_click", "click"):
                return self._post({"action": "left_click"})

            if a in ("double_click", "dblclick"):
                return self._post({"action": "double_click"})

            if a in ("right_click", "context_click"):
                return self._post({"action": "right_click"})

            if a == "type":
                return self._post({"action": "type", "text": text or ""})

            if a == "key":
                # aceita "ctrl+l", "alt+tab", etc.
                return self._post({"action": "key", "text": text or ""})

            if a in ("scroll_up",):
                return self._post({"action": "scroll", "dy": 300})

            if a in ("scroll_down",):
                return self._post({"action": "scroll", "dy": -300})

            if a == "drag_to":
                x, y = (coordinate or [0, 0])
                return self._post({"action": "drag_to", "x": int(x), "y": int(y), "duration": 0.5})

            if a == "wait":
                time.sleep( float(text) if text and text.strip().replace(".","",1).isdigit() else 0.5 )
                return {"ok": True}

            # fallback
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
    OCI = "oci"  # focamos no caminho OCI conforme seu pedido

OCI_DEFAULT_SYSTEM = """You are an automation planner controlling a Windows laptop.
You will be given:
- A structured summary of the current screen (screen_info)
- A list of parsed elements with bounding boxes
Your job is to think step-by-step and propose the SINGLE next action in strict JSON:
{
  "analysis": "<short reasoning>",
  "next": {"action": "mouse_move|left_click|double_click|right_click|type|key|scroll_up|scroll_down|wait|hover", "coordinate": [x, y] | null, "text": "<if type or key>", "done": false}
}
Use (x,y) in pixel coordinates relative to the current screen resolution.
If the task is complete or blocked, set done=true and next={"action":"None"}.
Avoid keyboard shortcuts unless explicitly requested.
"""

def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """
    Extrai texto do último bloco de mensagens, tolerante a formatos (gradio/anthropic-like).
    """
    if not messages:
        return ""
    last = messages[-1]
    # formatos comuns: {"role":"user","content":[{"type":"text","text":"..."}]}
    content = last.get("content", [])
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            return str(first.get("text", ""))
        # se vier só string
        if isinstance(first, str):
            return first
    # fallback
    return str(last)

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

    if provider.lower() != "oci" or not model.lower().startswith("omniparser + gpt-4.1-oci"):
        output_callback("Switch to 'omniparser + gpt-4.1-oci' to use the OCI flow.")
        yield None
        return

    # Tools para controlar o notebook local (via bridge HTTP)
    computer = ComputerTool(os.environ.get("WINDOWS_HOST_URL"))
    tools = ToolCollection(computer)

    # 1) Captura screenshot local (em memória)
    output_callback("Capturing the screen…")
    try:
        from agent.omnitool.gradio.tools.screen_capture import get_screenshot
    except Exception:
        # fallback simples via bridge (sem salvar), caso seu módulo não esteja disponível
        host = os.environ.get("WINDOWS_HOST_URL", "http://127.0.0.1:8006")
        r = requests.get(f"{host}/screenshot", timeout=10)
        r.raise_for_status()
        b64 = r.json().get("image_base64", "")
        if not b64:
            output_callback("Failed to capture screenshot from host bridge.")
            yield None
            return
        # manda a imagem direto pro OmniParser mais à frente
        pil_image = None
    else:
        pil_image, _ = get_screenshot(resize=False)

    # 2) Parse via OmniParser na VM
    output_callback("Parsing with OmniParser (VM)…")
    omni = OmniParserClient(url=omniparser_url)
    if pil_image is not None:
        parsed = omni.parse_pil(pil_image)
    else:
        # Se viemos pelo fallback (base64 do bridge), re-capturamos aqui via bridge e trocamos no client
        host = os.environ.get("WINDOWS_HOST_URL", "http://127.0.0.1:8006")
        r = requests.get(f"{host}/screenshot", timeout=10)
        r.raise_for_status()
        b64 = r.json().get("image_base64", "")
        if not b64:
            output_callback("Failed to capture screenshot from host bridge (fallback).")
            yield None
            return
        # Hack: usar uma versão interna do client que aceite b64 direto
        # (ou poderíamos decodificar para PIL e chamar parse_pil)
        from PIL import Image
        import base64, io
        im = Image.open(io.BytesIO(base64.b64decode(b64)))
        parsed = omni.parse_pil(im)

    screen_info = parsed.get("screen_info", "")
    parsed_content_list = parsed.get("parsed_content_list", "")

    # 3) Monta prompt e chama OCI GPT-4.1
    user_text = _extract_user_text(messages)
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

    # 4) Extrai JSON do passo
    next_action = None
    try:
        start = oci_text.rfind("{")
        end = oci_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            next_action = json.loads(oci_text[start:end+1])
    except Exception as e:
        output_callback(f"Failed to parse JSON from model: {e}")

    if not next_action or "next" not in next_action:
        output_callback("No actionable step returned by the model.")
        yield None
        return

    step = next_action["next"]
    if isinstance(step, dict) and step.get("action") and step.get("action") != "None":
        action = step["action"]
        coordinate = step.get("coordinate")
        text = step.get("text")
        try:
            result = tools("computer", action=action, coordinate=coordinate, text=text)
            tool_output_callback(result, "computer")
            output_callback(result)
        except Exception as e:
            output_callback(f"Tool error: {e}")
    else:
        output_callback("Done or blocked by the model.")
        yield None
        return

    # (1 passo por request)
    yield None
