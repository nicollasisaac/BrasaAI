import io
import json
import os
import time
import uuid
import base64
import pathlib
import re
from io import BytesIO
from typing import Callable, Dict, Any, List, Tuple

from PIL import Image, ImageDraw
from mss import mss
import requests

from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import (
    BetaMessage,
    BetaTextBlock,
    BetaToolUseBlock,
    BetaMessageParam,
    BetaUsage,
)

# Helpers from the project (adjust imports to your tree if needed)
from agent.llm_utils.utils import is_image_path

__all__ = ["VLMAgent", "omni_parse_from_image"]

LOG_DIR = pathlib.Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===================== OmniParser CONFIG =====================
OMNIPARSER_URL = (os.environ.get("OMNIPARSER_URL") or "http://127.0.0.1:7860").rstrip("/")
OMNI_FORCE_WIDTH = int(os.environ.get("OMNI_FORCE_WIDTH", "1920"))  # 0 = no resize
OMNI_ADAPTIVE = os.environ.get("OMNI_ADAPTIVE", "1") == "1"         # 1 = second pass if weak/slow
LOG_JSON = os.environ.get("LOG_JSON", "0") == "1"

# ===================== OCI / LLM CONFIG =====================
LLM_TEMP = float(os.environ.get("LLM_TEMP", "0.2"))
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
OCI_CONFIG_FILE = os.path.expanduser(GPT41_CFG["config_path"])
OCI_CONFIG_PROFILE = GPT41_CFG["profile"]
OCI_MODEL_OCID = GPT41_CFG["model_id"]
OCI_COMPARTMENT_OCID = GPT41_CFG["compartment"]
OCI_ENDPOINT = GPT41_CFG["endpoint"]

def _oci_ready() -> bool:
    return (
        HAS_OCI
        and os.path.exists(OCI_CONFIG_FILE)
        and isinstance(OCI_MODEL_OCID, str) and OCI_MODEL_OCID.startswith("ocid1.generativeaimodel")
        and isinstance(OCI_COMPARTMENT_OCID, str) and OCI_COMPARTMENT_OCID.startswith("ocid1.compartment")
    )

def criar_cliente(config_path: str, config_profile: str, endpoint: str):
    cfg = oci.config.from_file(config_path, config_profile)
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=cfg,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240),
    )

def _chat_details_generic(prompt: str, max_tokens: int = 500, temperature: float = 0.2):
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
        presence_penalty=0,
    )
    return oci.generative_ai_inference.models.ChatDetails(
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=OCI_MODEL_OCID),
        chat_request=chat_req,
        compartment_id=OCI_COMPARTMENT_OCID,
    )

def extrair_texto(resp):
    cr = resp.data.chat_response
    if hasattr(cr, "choices"):
        return cr.choices[0].message.content[0].text
    else:
        return cr.text

# ===================== Utils: screenshot + Omni =====================
def _encode_image_pil(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _resize_width_keep_ratio(im: Image.Image, target_w: int) -> Image.Image:
    if target_w <= 0 or im.width <= target_w:
        return im
    ratio = target_w / im.width
    new_h = max(1, int(im.height * ratio))
    return im.resize((target_w, new_h), Image.LANCZOS)

def screenshot_monitor(monitor_index: int = 1) -> Image.Image:
    with mss() as sct:
        mons = sct.monitors
        if monitor_index < 0 or monitor_index >= len(mons):
            raise RuntimeError(f"Monitor index {monitor_index} inválido (0..{len(mons)-1})")
        region = mons[monitor_index]
        raw = sct.grab(region)
        im = Image.frombytes("RGB", raw.size, raw.rgb)
        return im

def _omni_post_multipart(image_pil: Image.Image, describe_icons: bool, imgsz: int, return_image: bool = True):
    """
    Envia a imagem ao OmniParser /api/parse via multipart/form-data (file=...),
    que é o que seu servidor espera.
    """
    url = OMNIPARSER_URL.rstrip("/") + "/api/parse"

    # parâmetros simples (tudo como string)
    data = {
        "box_threshold": "0.05",
        "iou_threshold": "0.10",
        "use_paddleocr": "true",
        "use_florence": "true",
        "describe_icons": "true" if describe_icons else "false",
        "imgsz": str(int(imgsz)),
        "return_image": "true" if return_image else "false",
    }

    # arquivo (PNG) no campo 'file'
    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("screen.png", buf.getvalue(), "image/png")}

    t0 = time.perf_counter()
    r = requests.post(url, data=data, files=files, timeout=180)
    r.raise_for_status()
    js = r.json()
    dt = int((time.perf_counter() - t0) * 1000)
    return js, dt


def omni_parse_from_image(scr: Image.Image):
    to_send = _resize_width_keep_ratio(scr, OMNI_FORCE_WIDTH) if OMNI_FORCE_WIDTH else scr

    js, dt = _omni_post_multipart(to_send, describe_icons=False, imgsz=min(max(640, to_send.width), 1280), return_image=True)
    total_ms = dt
    elems = js.get("outputs", {}).get("elements", []) or []
    if OMNI_ADAPTIVE and (len(elems) < 40 or dt > 25000):
        js2, dt2 = _omni_post_multipart(to_send, describe_icons=True, imgsz=min(to_send.width, 1920), return_image=True)
        elems2 = js2.get("outputs", {}).get("elements", []) or []
        if len(elems2) >= len(elems) or dt2 < dt:
            js = js2
        total_ms += dt2

    input_info = js.get("input") or {}
    in_w = int(input_info.get("width", to_send.width))
    in_h = int(input_info.get("height", to_send.height))
    elements = js.get("outputs", {}).get("elements", []) or []

    lines = []
    parsed_content_list = []
    for idx, el in enumerate(elements):
        txt = el.get("text") or el.get("name") or el.get("label") or ""
        role = el.get("role") or el.get("type") or ""
        bbox = el.get("bbox") or el.get("normalized_bbox") or None
        if bbox and max(bbox) > 1.01:
            x1, y1, x2, y2 = bbox
            bbox = [x1 / in_w, y1 / in_h, x2 / in_w, y2 / in_h]
        if txt or role:
            lines.append(f"Box ID {idx}: [{role}] {txt}")
        parsed_content_list.append({"bbox": bbox or [0, 0, 0, 0], "role": role, "text": txt})

    screen_info = "\n".join(lines) if lines else "(no textual elements found)"

    som_b64 = js.get("som_image_base64") or js.get("image_base64") or js.get("preview_base64") or None
    if not som_b64:
        # fallback: raw screenshot
        buf = BytesIO()
        scr.save(buf, format="PNG")
        som_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    parsed = {
        "original_screenshot_base64": base64.b64encode(_encode_image_pil(scr)).decode("utf-8"),
        "som_image_base64": som_b64,
        "screen_info": screen_info,
        "parsed_content_list": parsed_content_list,
        "width": scr.width,
        "height": scr.height,
        "screenshot_uuid": uuid.uuid4().hex,
        "latency": total_ms / 1000.0,
    }
    if LOG_JSON:
        (LOG_DIR / "omni.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed, js, total_ms

# ===================== Helpers =====================
def extract_data(input_string, data_type):
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches[0][0].strip() if matches else input_string

# ===================== VLMAgent (OCI or existing backends) =====================
class VLMAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        elif model == "omniparser + gpt-4.1-oci":
            self.model = "oci-gpt-4.1"
        else:
            raise ValueError(f"Model {model} not supported")

        self.model_label = model
        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0
        self.system = ""

    def __call__(self, messages: list, parsed_screen: Dict[str, Any] | None):
        self.step_count += 1

        if self.model == "oci-gpt-4.1":
            self.output_callback(f"-- Step {self.step_count}: Capturando tela local e parseando com OmniParser --", sender="bot")
            monitor_idx = int(os.environ.get("SCREEN_MONITOR", "1"))
            im = screenshot_monitor(monitor_index=monitor_idx)
            parsed_screen, _omni_raw, _ms = omni_parse_from_image(im)
        else:
            if not parsed_screen:
                raise ValueError("parsed_screen is required for non-OCI models")

        latency_omniparser = parsed_screen["latency"]
        screen_info = str(parsed_screen["screen_info"])
        screen_width, screen_height = parsed_screen["width"], parsed_screen["height"]

        system = self._get_system_prompt(screen_info)

        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        start = time.time()
        if self.model == "oci-gpt-4.1":
            last_user = ""
            for msg in planner_messages[::-1]:
                if msg.get("role") == "user":
                    cont = msg.get("content", [])
                    if isinstance(cont, list) and cont and hasattr(cont[0], "text"):
                        last_user = cont[0].text
                    elif isinstance(cont, list) and cont and isinstance(cont[0], dict) and "text" in cont[0]:
                        last_user = cont[0]["text"]
                    elif isinstance(cont, str):
                        last_user = cont
                    break

            prompt = self._oci_prompt(user_goal=last_user, screen_info=screen_info)

            if not _oci_ready():
                raise RuntimeError("OCI não configurado (verifique ~/.oci/config, profile e OCIDs).")

            client = criar_cliente(OCI_CONFIG_FILE, OCI_CONFIG_PROFILE, OCI_ENDPOINT)
            chat_details = _chat_details_generic(prompt=prompt, max_tokens=min(16384, self.max_tokens), temperature=LLM_TEMP)
            resp = client.chat(chat_details)
            vlm_response = extrair_texto(resp)
        else:
            from agent.llm_utils.oaiclient import run_oai_interleaved
            from agent.llm_utils.groqclient import run_groq_interleaved

            if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
                vlm_response, token_usage = run_oai_interleaved(
                    messages=planner_messages,
                    system=system,
                    model_name=self.model,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    provider_base_url="https://api.openai.com/v1",
                    temperature=0,
                )
                self.total_token_usage += token_usage
            elif "r1" in self.model:
                vlm_response, token_usage = run_groq_interleaved(
                    messages=planner_messages,
                    system=system,
                    model_name=self.model,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                )
                self.total_token_usage += token_usage
            elif "qwen" in self.model:
                vlm_response, token_usage = run_oai_interleaved(
                    messages=planner_messages,
                    system=system,
                    model_name=self.model,
                    api_key=self.api_key,
                    max_tokens=min(2048, self.max_tokens),
                    provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0,
                )
                self.total_token_usage += token_usage
            else:
                raise ValueError(f"Model {self.model} not supported")

        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        img_to_show_base64 = parsed_screen.get("som_image_base64") or parsed_screen["original_screenshot_base64"]
        if "Box ID" in vlm_response_json and isinstance(vlm_response_json["Box ID"], int):
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                cx = int((bbox[0] + bbox[2]) / 2 * screen_width)
                cy = int((bbox[1] + bbox[3]) / 2 * screen_height)
                vlm_response_json["box_centroid_coordinate"] = [cx, cy]

                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data)).convert("RGB")

                draw = ImageDraw.Draw(img_to_show)
                x, y = cx, cy
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline="red", width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error parsing Box ID to bbox: {e}")

        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
            f"<details><summary>Parsed Screen elements by OmniParser</summary><pre>{screen_info}</pre></details>",
            sender="bot",
        )

        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f"{value}"
            else:
                vlm_plan_str += f"\n{key}: {value}"

        response_content = [BetaTextBlock(text=vlm_plan_str, type="text")]
        if "box_centroid_coordinate" in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(
                id=f"toolu_{uuid.uuid4()}",
                input={"action": "mouse_move", "coordinate": vlm_response_json["box_centroid_coordinate"]},
                name="computer",
                type="tool_use",
            )
            response_content.append(move_cursor_block)

        next_action = vlm_response_json.get("Next Action")
        if next_action == "None":
            pass
        elif next_action == "type":
            response_content.append(
                BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input={"action": next_action, "text": vlm_response_json.get("value", "")},
                    name="computer",
                    type="tool_use",
                )
            )
        elif next_action:
            response_content.append(
                BetaToolUseBlock(
                    id=f"toolu_{uuid.uuid4()}",
                    input={"action": next_action},
                    name="computer",
                    type="tool_use",
                )
            )

        response_message = BetaMessage(
            id=f"toolu_{uuid.uuid4()}",
            content=response_content,
            model="",
            role="assistant",
            type="message",
            stop_reason="tool_use",
            usage=BetaUsage(input_tokens=0, output_tokens=0),
        )
        return response_message, vlm_response_json

    def _oci_prompt(self, user_goal: str, screen_info: str) -> str:
        return f"""{self._get_system_prompt(screen_info)}

User goal:
{user_goal}

Now propose the next action JSON only, following the specified format, no extra commentary."""

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content. 
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{
    "Reasoning": "string",
    "Next Action": "action_type" | "None",
    "Box ID": 0,
    "value": "xxx"
}
```

One Example:
```json
{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": 3
}
```

Another Example:
```json
{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type \"Apple watch\" in the search bar.",
    "Next Action": "type",
    "Box ID": 5,
    "value": "Apple watch"
}
```

Another Example:
```json
{
    "Reasoning": "The current screen does not show 'submit' button, I need to scroll down to see if the button is available.",
    "Next Action": "scroll_down"
}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
2. You should give an analysis to the current screen.
3. Attach the next action prediction in the "Next Action".
4. Avoid keyboard shortcuts.
5. When the task is completed, return "Next Action": "None".
6. Break multi-step tasks into subgoals.
7. Avoid choosing the same action/elements multiple times in a row.
8. If a login/captcha appears or needs user's permission, return "Next Action": "None".
"""
        return main_section

# ===================== history filters =====================
def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg.get("content")
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content
                if not (isinstance(cnt, str) and "som" in cnt and is_image_path(cnt))
            ]

def _maybe_filter_to_n_most_recent_images(
    messages: List[BetaMessageParam],
    images_to_keep: int | None,
    min_removal_threshold: int = 10,
):
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep

    for msg in messages:
        msg_content = msg.get("content")
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                new_content.append(cnt)
            msg["content"] = new_content
