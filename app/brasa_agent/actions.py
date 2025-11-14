"""Brasa AI — actions.py
Primitives de automação baseadas em PyAutoGUI.

Schema de ações (planner → executor):
- CLICK_TEXT { target_text: str }
- TYPE { text: str, enter?: bool }
- HOTKEY { keys: list[str] }  # ex.: ["ctrl","k"]
- WAIT { ms: int }
- DONE { reason?: str }

Extras opcionais já suportados pelo executor:
- CLICK_NEAR { near_text: str, dx?: int, dy?: int }   # clica com offset relativo ao texto
- SCROLL { amount: int }                               # positivo = sobe, negativo = desce
- CLICK_AT { x: int, y: int }
- MOVE_TO { x: int, y: int }

Retornos padronizados:
- { ok: bool, reason?: str, done?: bool }
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pyautogui

# Configs de segurança/tempo
pyautogui.PAUSE = float(os.getenv("SLEEP_BETWEEN_ACTIONS", 0.6))
pyautogui.FAILSAFE = True  # mover mouse para canto sup/esq aborta


# -----------------------------
# Helpers de plataforma / hotkeys
# -----------------------------

def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")

def _is_macos() -> bool:
    return sys.platform == "darwin"

def _normalize_keys(keys_in: List[str]) -> List[str]:
    """
    Normaliza nomes de teclas para o esperado pelo pyautogui.
    - Lowercase
    - Remove espaços extras
    - Converte 'win'/'windows'/'cmd'/'meta' → 'winleft' no Windows; → 'command' no macOS
    """
    out: List[str] = []
    for k in keys_in:
        k0 = str(k or "").strip().lower()
        if not k0:
            continue

        # Normaliza sinônimos de tecla "Windows / Command"
        if k0 in ("win", "windows", "cmd", "command", "meta", "super"):
            if _is_windows():
                k0 = "winleft"
            elif _is_macos():
                k0 = "command"
            else:
                # Em Linux, muitas vezes 'win' não é suportado; deixamos 'win' genérico
                # mas pyautogui pode não reconhecer. O planner não deveria usar em Linux.
                k0 = "win"  # melhor esforço
        # Normaliza algumas grafias comuns
        elif k0 in ("control",):
            k0 = "ctrl"
        elif k0 in ("del", "delete"):
            k0 = "delete"
        elif k0 in ("esc", "escape"):
            k0 = "esc"
        elif k0 in ("return", "enter"):
            k0 = "enter"
        elif k0 in ("pgup", "pageup"):
            k0 = "pageup"
        elif k0 in ("pgdn", "pagedown"):
            k0 = "pagedown"
        # demais teclas ficam como estão (ex.: 'r', 'k', 'l', 'tab', ...)

        out.append(k0)
    return out


# -----------------------------
# Helpers de geometria e lookup
# -----------------------------

def _screen_size() -> Tuple[int, int]:
    try:
        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        # fallback comum em headless/VMs sem display válido
        return 1920, 1080


def _clamp_bbox(bbox_abs: List[int]) -> List[int]:
    """Garante bbox dentro do tamanho da tela."""
    x1, y1, x2, y2 = bbox_abs
    sw, sh = _screen_size()
    x1 = max(0, min(sw - 1, int(x1)))
    x2 = max(0, min(sw - 1, int(x2)))
    y1 = max(0, min(sh - 1, int(y1)))
    y2 = max(0, min(sh - 1, int(y2)))
    # garante ordenação
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [x1, y1, x2, y2]


def _center_of(bbox_abs: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = _clamp_bbox(bbox_abs)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def _norm_text(s: str) -> str:
    return (s or "").strip().casefold()


def _bbox_from_el(el: Dict[str, Any]) -> Optional[List[int]]:
    """
    Extrai um bbox absoluto de um elemento do Omni:
    - el['bbox'] pode ser dict com 'absolute' [x1,y1,x2,y2]
    - ou lista [x1,y1,x2,y2] (já absoluto)
    - caso não exista, retorna None
    """
    b = el.get("bbox")
    if isinstance(b, dict):
        absb = b.get("absolute")
        if isinstance(absb, list) and len(absb) == 4:
            return _clamp_bbox(absb)
    elif isinstance(b, (list, tuple)) and len(b) == 4:
        # melhor esforço: assume já absoluto
        return _clamp_bbox(list(b))
    return None


def find_element_bbox(elements: List[Dict[str, Any]], text: str) -> Optional[List[int]]:
    """Procura por um elemento cujo 'content' contenha `text`.
    Prioriza match exato; se não achar, tenta substring.
    Retorna bbox absoluto [x1,y1,x2,y2] ou None.
    """
    query = _norm_text(text)
    best_bbox = None

    # 1) match exato (em qualquer type que tenha content textual)
    for el in elements:
        content = _norm_text(str(el.get("content", "")))
        if content and content == query:
            bbox = _bbox_from_el(el)
            if bbox:
                return bbox

    # 2) substring
    for el in elements:
        content = _norm_text(str(el.get("content", "")))
        if query and content and query in content:
            bbox = _bbox_from_el(el)
            if bbox:
                best_bbox = bbox
                break

    return best_bbox


# -----------------------------
# Primitives de ação
# -----------------------------

def move_to(x: int, y: int, duration: float = 0.2):
    sw, sh = _screen_size()
    x = max(0, min(sw - 1, int(x)))
    y = max(0, min(sh - 1, int(y)))
    pyautogui.moveTo(x, y, duration=duration)


def click_center(bbox_abs: List[int], clicks: int = 1, interval: float = 0.05):
    cx, cy = _center_of(bbox_abs)
    move_to(cx, cy, duration=0.2)
    pyautogui.click(clicks=clicks, interval=interval)


def click_at(x: int, y: int, clicks: int = 1, interval: float = 0.05):
    move_to(x, y, duration=0.2)
    pyautogui.click(clicks=clicks, interval=interval)


def type_text(text: str, press_enter: bool = False):
    pyautogui.typewrite(str(text), interval=0.02)
    if press_enter:
        pyautogui.press("enter")


def press_hotkey(*keys: str):
    if not keys:
        raise ValueError("HOTKEY sem 'keys'")
    norm = _normalize_keys(list(keys))
    pyautogui.hotkey(*norm)


def do_scroll(amount: int):
    # positivo = sobe; negativo = desce (comportamento do pyautogui em muitas plataformas)
    pyautogui.scroll(int(amount))


# -----------------------------
# Executor principal
# -----------------------------

def execute_action(action: Dict[str, Any], elements: List[Dict[str, Any]]):
    """Executa uma ação baseada no schema do planner.

    Tipos suportados:
      - CLICK_TEXT {target_text}
      - TYPE {text, enter?}
      - HOTKEY {keys:[...]}
      - WAIT {ms}
      - DONE {reason?}
      - CLICK_NEAR {near_text, dx?, dy?}
      - SCROLL {amount}
      - CLICK_AT {x, y}
      - MOVE_TO {x, y}
    """
    try:
        atype = (action or {}).get("type", "")
        atype_up = str(atype).strip().upper()

        if atype_up == "CLICK_TEXT":
            target = (action.get("target_text") or "").strip()
            if not target:
                return {"ok": False, "reason": "CLICK_TEXT sem target_text"}
            bbox = find_element_bbox(elements, target)
            if not bbox:
                return {"ok": False, "reason": f"Elemento '{target}' não encontrado"}
            click_center(bbox)
            return {"ok": True}

        elif atype_up == "TYPE":
            txt = action.get("text", "")
            enter = bool(action.get("enter", False))
            type_text(txt, press_enter=enter)
            return {"ok": True}

        elif atype_up == "HOTKEY":
            keys = action.get("keys", [])
            if not keys or not isinstance(keys, list):
                return {"ok": False, "reason": "HOTKEY sem keys (array)"}
            press_hotkey(*[str(k) for k in keys])
            return {"ok": True}

        elif atype_up == "WAIT":
            ms = int(action.get("ms", 400))
            time.sleep(max(0, ms) / 1000)
            return {"ok": True}

        elif atype_up == "DONE":
            # O runtime não para no DONE do Planner; apenas registra o resultado.
            return {"ok": True, "done": True, "reason": action.get("reason", "")}

        elif atype_up == "CLICK_NEAR":
            near = (action.get("near_text") or "").strip()
            dx = int(action.get("dx", 20))
            dy = int(action.get("dy", 0))
            if not near:
                return {"ok": False, "reason": "CLICK_NEAR sem near_text"}
            bbox = find_element_bbox(elements, near)
            if not bbox:
                return {"ok": False, "reason": f"Referência '{near}' não encontrada"}
            cx, cy = _center_of(bbox)
            click_at(cx + dx, cy + dy)
            return {"ok": True}

        elif atype_up == "SCROLL":
            amount = int(action.get("amount", -600))
            do_scroll(amount)
            return {"ok": True}

        elif atype_up == "CLICK_AT":
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
            click_at(x, y)
            return {"ok": True}

        elif atype_up == "MOVE_TO":
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
            move_to(x, y)
            return {"ok": True}

        else:
            return {"ok": False, "reason": f"Tipo desconhecido: {atype}"}

    except pyautogui.FailSafeException:
        return {"ok": False, "reason": "Abortado pelo FAILSAFE (mouse no canto sup/esq)"}
    except Exception as e:
        return {"ok": False, "reason": f"Exceção: {e.__class__.__name__}: {e}"}
