# agent/omnitool/gradio/agent/llm_utils/oaiclient.py
# Requisitos extras: pip install requests
import os, json
from typing import Tuple, List, Dict, Any
import requests  # usado no backend gemini

def _mensagens_para_prompt(messages: List[Dict[str, Any]], system: str) -> str:
    partes = []
    if system:
        partes.append(f"[System]\n{system}\n")
    for i, msg in enumerate(messages or []):
        role = (msg.get("role") or "user").title()
        partes.append(f"[{role} #{i+1}]")
        content = msg.get("content", [])
        if isinstance(content, str):
            partes.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    partes.append(c)
                elif isinstance(c, dict):
                    partes.append(c.get("text", str(c)))
        else:
            partes.append(str(content))
        partes.append("")
    return "\n".join(partes).strip()

# ---------------- GEMINI ----------------
def _gemini_require_key():
    key = os.environ.get("GEMINI_API_KEY", "AIzaSyD7_KEeVR-43uZzciO6epiaFmfkpQp128A").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY não definido")
    return key

def _messages_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contents = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    parts.append({"text": c})
                elif isinstance(c, dict):
                    t = c.get("text")
                    if t is None:
                        t = json.dumps(c, ensure_ascii=False)
                    parts.append({"text": t})
                else:
                    parts.append({"text": str(c)})
        else:
            parts.append({"text": str(content)})
        contents.append({"role": role, "parts": parts})
    return contents

def _gemini_call(messages, system, max_tokens, temperature) -> str:
    api_root = os.environ.get("GEMINI_API_ROOT", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    key = _gemini_require_key()
    url = f"{api_root}/models/{model}:generateContent"

    payload: Dict[str, Any] = {
        "contents": _messages_to_gemini_contents(messages),
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "topP": 1,
            "topK": 1
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        params={"key": key},
        json=payload,
        timeout=float(os.environ.get("GEMINI_TIMEOUT", "60")),
    )
    if r.status_code != 200:
        try:
            j = r.json()
            err = j.get("error") or j
            raise RuntimeError(f"[GEMINI] HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)}")
        except Exception:
            raise RuntimeError(f"[GEMINI] HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        pf = data.get("promptFeedback")
        reason = pf.get("blockReason") if pf else None
        raise RuntimeError(f"[GEMINI] Sem candidates. promptFeedback={reason}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    if parts and isinstance(parts[0], dict) and "text" in parts[0]:
        return parts[0]["text"]
    return cands[0].get("text") or json.dumps(parts, ensure_ascii=False)

# --------------- OCI (lazy import/config) ---------------
def _oci_call(prompt: str, max_tokens: int, temperature: float) -> str:
    import oci  # importa só se realmente for usar OCI

    def _resolve_config_path() -> str:
        env_path = os.environ.get("OCI_CONFIG_FILE")
        candidates = [env_path] if env_path else []
        candidates += [
            r"C:\Users\Inteli\Documents\GitHub\BrasaAI\.oci\config",
            r"C:\Users\Inteli\.oci\config",
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                print(f"[OCI] usando config: {p}")
                return p
        raise FileNotFoundError(f"[OCI] config não encontrado. Tentado: {candidates}")

    cfg_path = _resolve_config_path()
    profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    endpoint = os.environ.get("OCI_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    model_id = os.environ.get("OCI_MODEL_OCID", "")
    compartment_id = os.environ.get("OCI_COMPARTMENT_OCID", "")

    if not model_id or not compartment_id:
        raise RuntimeError("[OCI] OCI_MODEL_OCID e/ou OCI_COMPARTMENT_OCID ausentes")

    client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=oci.config.from_file(cfg_path, profile),
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )

    content = oci.generative_ai_inference.models.TextContent(type="TEXT", text=prompt)
    message = oci.generative_ai_inference.models.Message(role="USER", content=[content])
    chat_req = oci.generative_ai_inference.models.GenericChatRequest(
        api_format="GENERIC",
        messages=[message],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=1, top_k=1,
        frequency_penalty=0, presence_penalty=0
    )
    details = oci.generative_ai_inference.models.ChatDetails(
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id),
        chat_request=chat_req,
        compartment_id=compartment_id
    )
    resp = client.chat(details)
    cr = resp.data.chat_response
    if hasattr(cr, "choices") and cr.choices:
        return cr.choices[0].message.content[0].text
    if hasattr(cr, "text"):
        return cr.text
    return str(cr)

# ---------------- ENTRYPOINT (mantido) ----------------
def run_oci_interleaved(messages: list, system: str, max_tokens=1024, temperature=0.2) -> Tuple[str, int]:
    backend = os.environ.get("LLM_BACKEND", "oci").strip().lower()
    print(f"[LLM] backend={backend}")  # log explícito

    if backend == "gemini":
        # usa o formato messages; GEMINI não precisa do prompt concatenado
        txt = _gemini_call(messages, system, max_tokens, temperature)
        return txt, 0

    # fallback (comportamento original via OCI)
    prompt = _mensagens_para_prompt(messages, system)
    txt = _oci_call(prompt, max_tokens, temperature)
    return txt, 0
