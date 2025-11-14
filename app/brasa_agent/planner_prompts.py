# brasa_agent/planner_prompts.py
import os
from typing import Dict, Any
import requests
import json
import re

BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

PLANNER_SYS = (
    "Você é um planner de 1 passo. Recebe 'goal' e alguns elementos visuais (texto+bbox).\n"
    "Retorne APENAS um JSON com o próximo passo: {type, ...}.\n"
    "Tipos permitidos:\n"
    "- CLICK_TEXT {target_text} → clicar no elemento que contém esse texto\n"
    "- TYPE {text, enter?} → digitar texto no campo focado\n"
    "- HOTKEY {keys:[...]} → hotkey (ex: ['ctrl','k'])\n"
    "- WAIT {ms} → aguardar\n"
    "- DONE {reason} → objetivo atingido\n"
    "Regras: 1 ação por rodada; seja objetivo; se não achar alvo, tente HOTKEY para avançar (ex.: abrir Slack, buscar contato)."
)

JUDGE_SYS = (
    "Você é um juiz. Recebe goal, plan, result e elementos extraídos da tela atual.\n"
    "Responda APENAS JSON: {done: bool, reason: string}. done=true se o objetivo parece atingido."
)

def _chat(messages):
    """
    Chama uma API OpenAI-compatível (BASE) e tenta retornar um JSON.
    """
    url = f"{BASE}/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    txt = data["choices"][0]["message"]["content"].strip()

    # Tenta parsear JSON direto; senão, extrai o último bloco {...}
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # Fallback seguro
    return {"type": "WAIT", "ms": 400}

def local_planner(inp: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PLANNER_SYS},
        {"role": "user", "content": f"Goal: {inp['goal']}\nRound: {inp['round']}\nElements: {inp['elements_sample']}"},
    ]
    return _chat(messages)

def local_judge(inp: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": (
            f"Goal: {inp['goal']}\nRound: {inp['round']}\n"
            f"Plan: {inp['plan']}\nResult: {inp['result']}\n"
            f"Elements: {inp['elements_sample']}"
        )},
    ]
    out = _chat(messages)
    # Garantia de campos esperados
    return {"done": bool(out.get("done")), "reason": str(out.get("reason", ""))}
