# app.py
"""
Rodar:
    python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000

Notas:
- O parâmetro --omniparser_server_url é mantido apenas por compatibilidade visual,
  mas o código usa OMNIPARSER_URL (env) definido abaixo.
- Este app NÃO depende de loop.py nem de Anthropic. Tudo aqui é self-contained.
"""

import os, io, json, sys, time, re, pathlib, hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast
import argparse

import base64
import requests
from requests.exceptions import RequestException

from PIL import Image
import pyautogui
import gradio as gr

# ===================== CONFIG BASE =====================
OMNIPARSER_URL    = (os.environ.get("OMNIPARSER_URL") or "http://127.0.0.1:7860").rstrip("/")
OMNI_FORCE_WIDTH  = int(os.environ.get("OMNI_FORCE_WIDTH", "1920"))  # 0 = sem resize forçado
OMNI_FAST_FIRST   = os.environ.get("OMNI_FAST_FIRST", "1") == "1"    # 1 = começa rápido (imgsz<=1280, sem icons)
OMNI_ADAPTIVE     = os.environ.get("OMNI_ADAPTIVE", "1") == "1"      # 1 = tentativa alternativa se lento
LOG_JSON          = os.environ.get("LOG_JSON", "0") == "1"
DRY_RUN           = os.environ.get("DRY_RUN", "0") == "1"
LLM_TEMP          = float(os.environ.get("LLM_TEMP", "0.2"))
MAX_ROUNDS        = int(os.environ.get("MAX_ROUNDS", "5"))
MAX_ACTIONS_PER_ROUND = int(os.environ.get("MAX_ACTIONS_PER_ROUND", "3"))
AUTO_CONTINUE     = os.environ.get("AUTO_CONTINUE", "1") == "1"

LOG_DIR = pathlib.Path("./logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===================== OCI / LLM =====================
HAS_OCI = False
try:
    import oci  # type: ignore
    HAS_OCI = True
except Exception:
    HAS_OCI = False

GPT41_CFG = {
    "name": "GPT 4.1",
    "config_path": "latinoamericaai/.oci/config",
    "profile": "DEFAULT",
    "model_id": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyakhb3pkmf5c6ff7upp3o5sx7kg4bsz6ql6xdeyhlwjpzq",
    "compartment": "ocid1.compartment.oc1..aaaaaaaaev2ipyek53f7sck5ibvtnqrp5w2k54qiuk2cikbfati5bk54yhka",
    "endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    "api_format": "GENERIC",
}
OCI_CONFIG_FILE       = os.path.expanduser(GPT41_CFG["config_path"])
OCI_CONFIG_PROFILE    = GPT41_CFG["profile"]
OCI_MODEL_OCID        = GPT41_CFG["model_id"]
OCI_COMPARTMENT_OCID  = GPT41_CFG["compartment"]
OCI_ENDPOINT          = GPT41_CFG["endpoint"]

def _oci_ready() -> bool:
    return (HAS_OCI
            and os.path.exists(OCI_CONFIG_FILE)
            and isinstance(OCI_MODEL_OCID, str) and OCI_MODEL_OCID.startswith("ocid1.generativeaimodel")
            and isinstance(OCI_COMPARTMENT_OCID, str) and OCI_COMPARTMENT_OCID.startswith("ocid1.compartment"))

def criar_cliente(config_path: str, config_profile: str, endpoint: str):
    cfg = oci.config.from_file(config_path, config_profile)
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=cfg,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )

def _chat_details_generic(prompt: str, max_tokens:int=500, temperature:float=0.2):
    content = oci.generative_ai_inference.models.TextContent(type="TEXT", text=prompt)
    message = oci.generative_ai_inference.models.Message(role="USER", content=[content])
    chat_req = oci.generative_ai_inference.models.GenericChatRequest(
        api_format="GENERIC",
        messages=[message],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        top_k=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return oci.generative_ai_inference.models.ChatDetails(
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=OCI_MODEL_OCID),
        chat_request=chat_req,
        compartment_id=OCI_COMPARTMENT_OCID
    )

def extrair_texto(resp):
    cr = resp.data.chat_response
    if hasattr(cr, "choices"):
        return cr.choices[0].message.content[0].text
    else:
        return cr.text

# ===================== Utils para OmniParser =====================
def ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)

def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def resize_width(im: Image.Image, target_w: int) -> Image.Image:
    if target_w <= 0 or im.width <= target_w:
        return im
    ratio = target_w / im.width
    new_h = max(1, int(im.height * ratio))
    return im.resize((target_w, new_h), Image.LANCZOS)

def quick_hash(im: Image.Image) -> str:
    # hash rápido da imagem para log/controle
    h = hashlib.sha256(_png_bytes(im)).hexdigest()
    return h[:16]

def _omni_post(to_send: Image.Image, describe_icons: bool, imgsz: int) -> Tuple[Dict[str, Any], int]:
    url = OMNIPARSER_URL + "/api/parse"
    params = {
        "box_threshold": 0.05,
        "iou_threshold": 0.10,
        "use_paddleocr": True,
        "use_florence": True,
        "describe_icons": describe_icons,
        "imgsz": imgsz,
        "return_image": False,
    }
    files = {"file": ("screen.png", _png_bytes(to_send), "image/png")}
    data  = {k: (json.dumps(v) if isinstance(v, (dict, list, bool)) else str(v)) for k, v in params.items()}
    t0 = time.perf_counter()
    r = requests.post(url, files=files, data=data, timeout=180)
    r.raise_for_status()
    js = r.json()
    dt = ms_since(t0)
    return js, dt

def send_to_omni(im: Image.Image) -> Tuple[Dict[str, Any], Tuple[int,int], Tuple[int,int], str, int]:
    """
    Retorna (json_omni, (sent_w,sent_h), (screen_w,screen_h), scr_hash, omni_ms_total)
    """
    screen_w, screen_h = im.width, im.height
    to_send = resize_width(im, OMNI_FORCE_WIDTH) if OMNI_FORCE_WIDTH else im
    screen_hash = quick_hash(im)

    # primeira passada (rápida) se solicitado
    if OMNI_FAST_FIRST:
        js, dt = _omni_post(to_send, describe_icons=False, imgsz=min(max(640, to_send.width), 1280))
        sent_w, sent_h = to_send.width, to_send.height
        in_w = int(js.get("input", {}).get("width", sent_w))
        in_h = int(js.get("input", {}).get("height", sent_h))
        elems = len(js.get("outputs", {}).get("elements", []))
        print(f"[Omni] {dt} ms | elements={elems} | sent={sent_w}x{sent_h} | input={in_w}x{in_h} | imgsz=1280 | icons=False")
        total_ms = dt
        # se muito poucos elementos, tenta completo
        if OMNI_ADAPTIVE and (elems < 40 or dt > 25000):
            print("[Omni] Reforçando com describe_icons=True …")
            js2, dt2 = _omni_post(to_send, describe_icons=True, imgsz=min(to_send.width, 1920))
            elems2 = len(js2.get("outputs", {}).get("elements", []))
            print(f"[Omni-2] {dt2} ms | elements={elems2} | imgsz={min(to_send.width,1920)} | icons=True")
            if elems2 >= elems or dt2 < dt:
                js = js2
            total_ms += dt2
    else:
        first_imgsz = max(640, min(to_send.width, 1920))
        js, dt = _omni_post(to_send, describe_icons=True, imgsz=first_imgsz)
        sent_w, sent_h = to_send.width, to_send.height
        in_w = int(js.get("input", {}).get("width", sent_w))
        in_h = int(js.get("input", {}).get("height", sent_h))
        elems = len(js.get("outputs", {}).get("elements", []))
        print(f"[Omni] {dt} ms | elements={elems} | sent={sent_w}x{sent_h} | input={in_w}x{in_h} | imgsz={first_imgsz} | icons=True")
        total_ms = dt
        if OMNI_ADAPTIVE and dt > 25000:
            print("[Omni] Lento. Tentando novamente com icons=False e imgsz<=1280 …")
            alt_imgsz = min(first_imgsz, 1280)
            js2, dt2 = _omni_post(to_send, describe_icons=False, imgsz=alt_imgsz)
            elems2 = len(js2.get("outputs", {}).get("elements", []))
            print(f"[Omni-2] {dt2} ms | elements={elems2} | imgsz={alt_imgsz} | icons=False")
            if elems2 >= elems or dt2 < dt:
                js = js2
            total_ms += dt2

    if LOG_JSON:
        (LOG_DIR / "omni.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")

    # dimensões do frame processado pelo Omni (se reportado)
    in_w = int(js.get("input", {}).get("width", to_send.width))
    in_h = int(js.get("input", {}).get("height", to_send.height))
    return js, (in_w, in_h), (screen_w, screen_h), screen_hash, total_ms

# ===================== CLI Args =====================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    return parser.parse_args()
args = parse_arguments()

# ===================== UI / Estado =====================
CONFIG_DIR = Path("~/.anthropic").expanduser()  # só reaproveitando storage; não usamos Anthropic
API_KEY_FILE = CONFIG_DIR / "api_key"  # não usado, mas mantido para compat

INTRO_TEXT = '''
OmniTool (OCI + OmniParser local). Digite um objetivo e o agente irá:
1) tirar um screenshot da sua tela,
2) parsear com OmniParser (OMNIPARSER_URL), e
3) consultar o GPT-4.1 (OCI) para planejar a próxima ação.
'''

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4.1-oci"
    if "provider" not in state:
        state["provider"] = "oci"
    if "api_key" not in state:
        state["api_key"] = ""  # OCI via arquivo .oci (não precisa aqui)
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False

async def main(state):
    setup_state(state)
    return "Setup completed"

def _api_response_callback(response, response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: Any, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: Any, hide_images=False):
        if isinstance(message, str):
            return message
        # fallback simples
        return str(message)
    message = _render_message(message, hide_images)
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))

def valid_params(user_input, state):
    """Validação leve para não travar UI: só verifica opcionalmente Windows Host."""
    errors = []
    # Checagem do Windows Host (opcional); se não tiver /probe, ignore.
    try:
        url = f'http://{args.windows_host_url}/probe'
        response = requests.get(url, timeout=2)
        if response.status_code != 200:
            errors.append("Windows Host is not responding")
    except Exception:
        # não bloqueia
        pass

    # OCI pronto?
    if not _oci_ready():
        errors.append("OCI não configurado (arquivo .oci/config, profile e OCIDs).")

    if not user_input:
        errors.append("no computer use request provided")

    return errors

# ===================== Prompt & Execução =====================
SYSTEM_INSTRUCTIONS = """You are OmniTool's reasoning agent.
Given the user's goal and the parsed elements of the current screen, plan the next concrete UI action.
Output a short JSON with fields: {"plan": "...", "action": {"type": "...", "target": "...", "notes": "..."}}.
Keep it concise and practical; do not include extra keys.
"""

def build_prompt(user_goal: str, omni_json: Dict[str, Any]) -> str:
    elements = omni_json.get("outputs", {}).get("elements", [])
    # Concatena textos / rótulos que ajudam o agente a decidir:
    lines = []
    for el in elements[:200]:  # limite sanidade
        t = el.get("text") or el.get("name") or el.get("label") or ""
        if t:
            # bbox / role opcionais
            role = el.get("role") or el.get("type") or ""
            lines.append(f"- [{role}] {t}")
    screen_info = "\n".join(lines) if lines else "(no textual elements found)"
    prompt = f"""{SYSTEM_INSTRUCTIONS}

User goal:
{user_goal}

Parsed screen (top elements):
{screen_info}

Now propose the next action in JSON only."""
    return prompt

def process_input(user_input, state):
    # Reset stop
    if state.get("stop"):
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))

    # Log entrada do usuário
    state["messages"].append({"role": "user", "content": user_input})
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']

    # 1) Screenshot
    im = pyautogui.screenshot()  # PIL Image
    # 2) OmniParser
    try:
        omni_js, (in_w, in_h), (screen_w, screen_h), scr_hash, omni_ms = send_to_omni(im)
    except Exception as e:
        chatbot_output_callback(f"Erro no OmniParser: {e}", state['chatbot_messages'])
        yield state['chatbot_messages']
        return

    if LOG_JSON:
        (LOG_DIR / f"omni_{int(time.time())}.json").write_text(json.dumps(omni_js, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Prompt para o LLM (OCI GPT-4.1)
    if not _oci_ready():
        chatbot_output_callback("OCI não está pronto. Verifique seu ~/.oci/config, profile e OCIDs.", state['chatbot_messages'])
        yield state['chatbot_messages']
        return

    try:
        client = criar_cliente(OCI_CONFIG_FILE, OCI_CONFIG_PROFILE, OCI_ENDPOINT)
        prompt = build_prompt(user_input, omni_js)
        chat_details = _chat_details_generic(prompt=prompt, max_tokens=16384, temperature=LLM_TEMP)
        resp = client.chat(chat_details)
        texto = extrair_texto(resp).strip()
    except Exception as e:
        chatbot_output_callback(f"Erro chamando GPT-4.1 (OCI): {e}", state['chatbot_messages'])
        yield state['chatbot_messages']
        return

    # 4) Exibir resposta do agente
    if texto:
        # se vier com linhas extras, mantém. Idealmente ele retorna JSON curto conforme instruções.
        chatbot_output_callback(texto, state['chatbot_messages'])
    else:
        chatbot_output_callback("(sem resposta do modelo)", state['chatbot_messages'])

    yield state['chatbot_messages']

def stop_app(state):
    state["stop"] = True
    return "App stopped"

def get_header_image_base64():
    try:
        script_dir = Path(__file__).parent
        image_path = script_dir.parent / "imgs" / "header_bar_thin.png"  # ajuste se necessário
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        print(f"Failed to load header image: {e}")
        return None

# ===================== Gradio UI =====================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        .markdown-text p {
            font-size: 18px;
        }
        </style>
    """)
    state = gr.State({})
    setup_state(state.value)

    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool Header" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool (OCI + OmniParser)")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")

    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=["omniparser + gpt-4.1-oci"],
                    value="omniparser + gpt-4.1-oci",
                    interactive=False,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=["oci"],
                    value="oci",
                    interactive=False,
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API Key (não usado p/ OCI .oci/config)",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="(ignorado; use ~/.oci/config)",
                    interactive=False,
                )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Digite seu objetivo…", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
        with gr.Column(scale=3):
            iframe = gr.HTML(
                f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                container=False,
                elem_classes="no-padding"
            )

    # Handlers simples
    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value

    def clear_chat(state):
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        return state['chatbot_messages']

    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    submit_button.click(process_input, [chat_input, state], chatbot)
    stop_button.click(stop_app, [state], None)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7888)
