
import os
import json
import base64
import requests
from typing import List, Dict, Any, Tuple

# ======== Helpers from your codebase (import-free light stubs) =========
def is_image_path(p: str) -> bool:
    p = (p or "").lower()
    return any(p.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"])

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ===============================================================
# 1) OpenAI compatible client (unchanged behavior, a bit hardened)
# ===============================================================
def run_oai_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0, provider_base_url: str = "https://api.openai.com/v1"):
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    final_messages = [{"role": "system", "content": system}]

    if isinstance(messages, list):
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item.get("content", []):
                    if isinstance(cnt, str):
                        if is_image_path(cnt) and 'o3-mini' not in model_name:
                            # o3-mini não suporta imagens
                            base64_image = encode_image(cnt)
                            content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        else:
                            content = {"type": "text", "text": cnt}
                    else:
                        # bloco de texto (Anthropic/TextBlock-like)
                        content = {"type": "text", "text": str(cnt)}
                    contents.append(content)
                message = {"role": 'user', "content": contents}
            else:  # string simples
                contents.append({"type": "text", "text": item})
                message = {"role": "user", "content": contents}
            final_messages.append(message)
    elif isinstance(messages, str):
        final_messages = [{"role": "system", "content": system}, {"role": "user", "content": messages}]

    payload = {
        "model": model_name,
        "messages": final_messages,
        "temperature": temperature,
    }
    if 'o1' in model_name or 'o3-mini' in model_name:
        payload['reasoning_effort'] = 'low'
        payload['max_completion_tokens'] = max_tokens
    else:
        payload['max_tokens'] = max_tokens

    resp = requests.post(f"{provider_base_url}/chat/completions", headers=headers, json=payload, timeout=180)

    try:
        j = resp.json()
        text = j['choices'][0]['message']['content']
        token_usage = int(j.get('usage', {}).get('total_tokens', 0))
        return text, token_usage
    except Exception as e:
        try:
            return resp.json(), 0
        except Exception:
            return {"error": f"HTTP {resp.status_code}", "text": resp.text}, 0

# ===============================================================
# 2) OCI GPT‑4.1 Text Chat (Generic) – interleaved wrapper
#    Flattens messages into a single prompt string.
# ===============================================================

HAS_OCI = False
try:
    import oci  # type: ignore
    HAS_OCI = True
except Exception:
    HAS_OCI = False

GPT41_CFG = {
    "name": "GPT 4.1",
    "config_path": os.environ.get("OCI_CONFIG_PATH", "latinoamericaai/.oci/config"),
    "profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
    "model_id": os.environ.get("OCI_MODEL_OCID", "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyakhb3pkmf5c6ff7upp3o5sx7kg4bsz6ql6xdeyhlwjpzq"),
    "compartment": os.environ.get("OCI_COMPARTMENT_OCID", "ocid1.compartment.oc1..aaaaaaaaev2ipyek53f7sck5ibvtnqrp5w2k54qiuk2cikbfati5bk54yhka"),
    "endpoint": os.environ.get("OCI_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"),
}

def _oci_ready() -> bool:
    return (HAS_OCI
            and os.path.exists(os.path.expanduser(GPT41_CFG["config_path"]))
            and str(GPT41_CFG["model_id"]).startswith("ocid1.generativeaimodel")
            and str(GPT41_CFG["compartment"]).startswith("ocid1.compartment"))

def _oci_client():
    cfg = oci.config.from_file(os.path.expanduser(GPT41_CFG["config_path"]), GPT41_CFG["profile"])
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=cfg,
        service_endpoint=GPT41_CFG["endpoint"],
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )

def _oci_chat_details(prompt: str, max_tokens:int=1024, temperature:float=0.2):
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
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=GPT41_CFG["model_id"]),
        chat_request=chat_req,
        compartment_id=GPT41_CFG["compartment"]
    )

def _flatten_to_text(messages: list, system: str) -> str:
    """Flatten mixed text/image messages to a single text prompt for OCI Text Chat."""
    parts = []
    if system:
        parts.append(f"[System]\n{system}\n")
    if isinstance(messages, list):
        for i, item in enumerate(messages):
            role = item.get("role", "user") if isinstance(item, dict) else "user"
            parts.append(f"[{role.title()} #{i+1}]")
            if isinstance(item, dict):
                for cnt in item.get("content", []):
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            parts.append(f"(image omitted: {os.path.basename(cnt)})")
                        else:
                            parts.append(str(cnt))
                    elif isinstance(cnt, dict) and cnt.get("type") == "text":
                        parts.append(str(cnt.get("text", "")))
                    else:
                        parts.append(str(cnt))
            else:
                parts.append(str(item))
            parts.append("")  # blank line
    else:
        parts.append(str(messages))
    return "\n".join(parts).strip()

def run_oci_interleaved(messages: list, system: str, max_tokens=1024, temperature=0.2) -> Tuple[str, int]:
    """
    Convert your interleaved messages to a single text prompt and call OCI GPT‑4.1 Generic Chat.
    Returns (text, token_usage_estimated=0)
    """
    if not _oci_ready():
        raise RuntimeError("OCI não configurado (verifique OCI_* envs, ~/.oci/config, profile e OCIDs).")
    client = _oci_client()
    prompt = _flatten_to_text(messages, system)
    details = _oci_chat_details(prompt, max_tokens=max_tokens, temperature=temperature)
    resp = client.chat(details)
    cr = resp.data.chat_response
    if hasattr(cr, "choices"):
        out = cr.choices[0].message.content[0].text
    else:
        out = cr.text
    return out, 0
