# brasa_agent/runtime.py
# Loop do agente com orquestração via N8N em UM endpoint (sem "op").
# O N8N executa: Observer -> (se done) Judge interno -> (senão) Planner e responde o plano.
# Localmente: executamos o plano (PyAutoGUI), recapturamos a tela, e repetimos até o n8n devolver done=true.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, Tuple
import os, io, time, json, datetime
import requests
import pyautogui

from .omni import call_omni
from .actions import execute_action  # CLICK_TEXT / TYPE / HOTKEY / WAIT / DONE

# -------------------------------
# Config via ENV (com defaults)
# -------------------------------
# Mantemos as envs por compatibilidade, mas o loop só para com done=true ou FAILSAFE/erro fatal.
MAX_ROUNDS               = int(os.getenv("MAX_ROUNDS", "0"))  # ignorado para parada; usado só para logging
SLEEP_BETWEEN_ACTIONS    = float(os.getenv("SLEEP_BETWEEN_ACTIONS", "0.25"))
OMNI_API_URL_DEFAULT     = os.getenv("OMNI_API_URL", "http://127.0.0.1:7867")

# -------------------------------
# Utils
# -------------------------------

def _now_iso_tz() -> str:
    # timezone local sem lib externa: usa offset do sistema
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def _screen_size() -> Tuple[int, int]:
    try:
        sz = pyautogui.size()
        return int(sz[0]), int(sz[1])
    except Exception:
        # fallback comum
        return 1920, 1080

def _norm_to_abs(b: List[float], w: int, h: int) -> List[int]:
    # b = [x1,y1,x2,y2] normalizado (0..1) -> pixels inteiros
    x1 = max(0, min(w, int(round(b[0] * w))))
    y1 = max(0, min(h, int(round(b[1] * h))))
    x2 = max(0, min(w, int(round(b[2] * w))))
    y2 = max(0, min(h, int(round(b[3] * h))))
    return [x1, y1, x2, y2]

def _elements_to_sample(elements: List[Dict[str, Any]], w: int, h: int, max_items: int = 64) -> List[Dict[str, Any]]:
    out = []
    for i, el in enumerate(elements[:max_items], start=1):
        t = el.get("type", "")
        text = el.get("content", "")  # Florence/ocr result cai aqui
        bbox = el.get("bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_abs = _norm_to_abs(bbox, w, h)
        else:
            # se vier absoluto por alguma razão, só normaliza o shape
            bbox_abs = [0, 0, 0, 0]
        out.append({
            "id": i,
            "type": t,
            "text": text,
            "bbox": bbox_abs,
        })
    return out

def _post_n8n(n8n_endpoint: str, payload: Dict[str, Any], timeout: int = 90) -> Any:
    r = requests.post(n8n_endpoint, json=payload, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        # Fallback quando não é application/json
        return {"raw": r.text, "status_code": r.status_code}

def _capture_png_bytes() -> bytes:
    shot = pyautogui.screenshot()
    buf = io.BytesIO()
    shot.save(buf, format="PNG")
    return buf.getvalue()

def _unwrap_n8n_response(resp: Any) -> Any:
    """
    Aceita formatos comuns do n8n/LLM:
      - [{ "output": {...}}]
      - { "output": {...} }
      - {...} já objeto final
      - "```json { ... } ```" (string) -> parse
      - [ {...} ] -> pega primeiro
    """
    o = resp
    # Array → primeiro item
    if isinstance(o, list) and len(o) > 0:
        o = o[0]
    # Objeto com 'output'
    if isinstance(o, dict) and "output" in o and isinstance(o["output"], (dict, list, str)):
        o = o["output"]
        # se output ainda for lista, pega primeiro
        if isinstance(o, list) and len(o) > 0:
            o = o[0]
    # String JSON (até cercada por ```json)
    if isinstance(o, str):
        s = o.strip()
        if s.startswith("```"):
            # remove cercas de markdown
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        try:
            o = json.loads(s)
        except Exception:
            # não conseguiu parsear; devolve como string mesmo
            return o
    # Lista de dicionários pura → pega primeiro
    if isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict):
        o = o[0]
    return o

# -------------------------------
# Loop principal
# -------------------------------

def run_loop(
    goal: str,
    omni_initial: Optional[Dict[str, Any]],
    *,
    n8n_endpoint: str,
    on_log: Callable[[str], None],
    omni_base_url: Optional[str] = None,
) -> List[str]:
    """
    Orquestração com N8N em um endpoint. Em cada rodada:
      1) Se não houver omni, captura tela e chama /api/parse;
      2) Monta payload único p/ N8N (goal, round, context, omni, elements_sample, history);
      3) Posta no N8N:
         - Se resposta tiver {"done": true}, finaliza;
         - Senão, espera um PLANO ({"type": "..."}), executa localmente;
      4) Re-captura e repete.
    O loop SÓ para quando o N8N mandar done=true (ou por FAILSAFE/exceção fatal).
    """
    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)
        try:
            on_log(msg)
        except Exception:
            pass

    base_url = (omni_base_url or OMNI_API_URL_DEFAULT).rstrip("/")
    screen_w, screen_h = _screen_size()

    # Estado compartilhado
    state: Dict[str, Any] = {
        "round": 0,
        "goal": goal,
        "last_omni": omni_initial,
        "history": {
            "last_plan": None,
            "last_result": None,
        },
    }

    # Se não veio JSON do Omni, captura agora
    if not state["last_omni"]:
        try:
            img0 = _capture_png_bytes()
            state["last_omni"] = call_omni(base_url, img0)
            log("🖼️ Omni inicial obtido.")
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            raise RuntimeError(f"Falha na captura/parsing inicial: {e}")

    r = 0  # round perpetuado
    while True:
        r += 1
        state["round"] = r

        # extrai elements e OCR (se houver)
        outputs = (state["last_omni"] or {}).get("outputs", {})
        elements = outputs.get("elements", [])
        # ocr não é usado diretamente aqui, mas fica disponível no payload:
        # ocr = outputs.get("ocr", {})

        elements_sample = _elements_to_sample(elements, screen_w, screen_h)

        payload = {
            "goal": goal,
            "round": r,
            "context": {
                "screen": {"width": screen_w, "height": screen_h},
                "ts": _now_iso_tz(),
            },
            "omni": state["last_omni"],           # bruto (inclui elements/ocr/flags)
            "elements_sample": elements_sample,    # lista enxuta p/ LLM
            "history": state["history"],           # último plano/resultado
        }

        # 1 chamada ao N8N: Observer -> (Judge interno se done) ou Planner
        try:
            n8n_resp_raw = _post_n8n(n8n_endpoint, payload)
            log(f"📨 N8N r{r} request: {json.dumps(payload, ensure_ascii=False)[:800]}…")
            log(f"📩 N8N r{r} response: {json.dumps(n8n_resp_raw, ensure_ascii=False)}")
            n8n_resp = _unwrap_n8n_response(n8n_resp_raw)
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            # Erro de rede/HTTP: não para; aplica fallback WAIT e continua
            log(f"🌐 Erro na chamada ao N8N (r{r}): {e} — fallback WAIT 600ms e seguir.")
            n8n_resp = {"type": "WAIT", "ms": 600, "hint": "fallback n8n erro"}

        # DONE do N8N (judge)
        if isinstance(n8n_resp, dict) and n8n_resp.get("done") is True:
            reason = n8n_resp.get("reason", "")
            log(f"✅ Objetivo atingido (judge no N8N). Motivo: {reason}")
            break

        # Caso contrário, esperamos um PLANO de ação
        plan: Dict[str, Any] = {}
        if isinstance(n8n_resp, dict) and "type" in n8n_resp:
            plan = n8n_resp
        else:
            # Fallback: sem 'type' → garantir ação por rodada
            log("⚠️ Resposta sem 'done' e sem 'type' → fallback WAIT 400ms e continuar.")
            plan = {"type": "WAIT", "ms": 400, "hint": "fallback sem plano"}

        # Força round coerente no plano
        plan.setdefault("round", r)

        # Se o Planner devolveu DONE (excepcional), não paramos aqui;
        # apenas registramos e seguimos para próxima iteração (o Judge decide).
        if str(plan.get("type", "")).upper() == "DONE":
            log(f"ℹ️ Planner pediu DONE (r{r}): mantendo loop até Judge confirmar.")
            state["history"] = {
                "last_plan": plan,
                "last_result": {"ok": True, "info": "planner_done_noop"}
            }
        else:
            # Executa 1 ação localmente
            time.sleep(max(0.05, SLEEP_BETWEEN_ACTIONS))
            try:
                result = execute_action(plan, elements)
                log(f"⚙️ Exec r{r}: {json.dumps(result, ensure_ascii=False)}")
            except pyautogui.FailSafeException:
                # Failsafe do PyAutoGUI (mouse no canto superior esquerdo)
                log("⛔ Abortado pelo FAILSAFE do PyAutoGUI.")
                raise
            except Exception as e:
                result = {"ok": False, "error": str(e)}
                log(f"⚙️ Exec r{r} ERRO: {e}")

            # Atualiza histórico (perpetuar round/contexto)
            state["history"] = {
                "last_plan": plan,
                "last_result": result,
            }

        # Re-captura e re-parse SEM PARAR o loop
        try:
            img_bytes = _capture_png_bytes()
            state["last_omni"] = call_omni(base_url, img_bytes)
            log("🔁 Re-parse após ação executada.")
        except pyautogui.FailSafeException:
            log("⛔ Abortado pelo FAILSAFE do PyAutoGUI durante captura/parse.")
            raise
        except Exception as e:
            log(f"🖼️ Re-captura/parse erro r{r}: {e} — mantendo último omni para próxima rodada.")

        time.sleep(max(0.05, SLEEP_BETWEEN_ACTIONS))

    return logs
