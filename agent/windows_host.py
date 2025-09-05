# windows_host.py
# HTTP bridge to control your local Windows notebook with pyautogui.
# Adds /probe, /execute (action-based), and /screenshot endpoints.
# All screenshots are returned as base64 (not written to disk).

import argparse
import sys
import time
import io
import json
from typing import Optional
from flask import Flask, request, jsonify
from PIL import Image
import base64

# pyautogui needs Pillow and proper permissions
try:
    import pyautogui
    PYAUTOGUI_OK = True
except Exception as e:
    print(f"[windows_host] pyautogui not available: {e}")
    PYAUTOGUI_OK = False

try:
    from mss import mss
    MSS_OK = True
except Exception as e:
    print(f"[windows_host] mss not available: {e}")
    MSS_OK = False

app = Flask(__name__)

def _b64_png_from_image(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def take_screenshot_b64(monitor_index: int = 1) -> str:
    """
    Capture the physical screen without saving, return base64 PNG.
    monitor_index: mss uses 1 as the primary monitor, 0 for all monitors.
    """
    if not MSS_OK:
        raise RuntimeError("mss not available for screenshots")
    with mss() as sct:
        mons = sct.monitors
        if monitor_index >= len(mons):
            monitor_index = 1 if len(mons) > 1 else 0
        mon = mons[monitor_index]
        raw = sct.grab(mon)
        im = Image.frombytes("RGB", raw.size, raw.rgb)
        return _b64_png_from_image(im)

@app.route("/probe", methods=["GET"])
def probe():
    return jsonify({"status": "ok"}), 200

@app.route("/screenshot", methods=["GET"])
def screenshot():
    try:
        b64 = take_screenshot_b64()
        return jsonify({"image_base64": b64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/execute", methods=["POST"])
def execute():
    """
    Action-based API used by ComputerTool HTTP backend.
    Supported actions:
      - move {x,y}
      - drag_to {x,y,duration}
      - left_click | right_click | middle_click | double_click
      - left_press {hold}
      - scroll {dy}
      - type {text}
      - key {text}  (expects combination like 'ctrl+L' or a key name)
      - get_screen_size
      - cursor_position
      - screenshot
    """
    try:
        data = request.get_json(force=True) or {}
        action = data.get("action")
        if not action:
            return jsonify({"error": "Missing 'action' field"}), 400
        if not PYAUTOGUI_OK and action not in ("screenshot", "get_screen_size"):
            return jsonify({"error": "pyautogui not available"}), 500

        if action == "move":
            x, y = int(data["x"]), int(data["y"])
            pyautogui.moveTo(x, y, duration=0.05)
            return jsonify({"ok": True})

        if action == "drag_to":
            x, y = int(data["x"]), int(data["y"])
            duration = float(data.get("duration", 0.5))
            pyautogui.dragTo(x, y, duration=duration)
            return jsonify({"ok": True})

        if action in ("left_click", "right_click", "middle_click", "double_click"):
            if action == "left_click":
                pyautogui.click()
            elif action == "right_click":
                pyautogui.click(button="right")
            elif action == "middle_click":
                pyautogui.middleClick()
            else:
                pyautogui.doubleClick()
            return jsonify({"ok": True})

        if action == "left_press":
            hold = float(data.get("hold", 1.0))
            pyautogui.mouseDown()
            time.sleep(max(0.0, hold))
            pyautogui.mouseUp()
            return jsonify({"ok": True})

        if action == "scroll":
            dy = int(data.get("dy", -100))
            pyautogui.scroll(dy)
            return jsonify({"ok": True})

        if action == "type":
            text = data.get("text", "")
            # A quick click to focus helps on some UIs.
            pyautogui.click()
            # Type with a tiny delay between keys to reduce dropped characters
            pyautogui.typewrite(text, interval=0.012)
            return jsonify({"ok": True})

        if action == "key":
            combo = data.get("text", "")
            # Support combinations like "ctrl+L"
            parts = [p.strip() for p in combo.split("+") if p.strip()]
            if parts:
                pyautogui.hotkey(*parts)
                return jsonify({"ok": True})
            else:
                return jsonify({"error": "invalid key combo"}), 400

        if action == "get_screen_size":
            if not PYAUTOGUI_OK:
                return jsonify({"error": "pyautogui not available"}), 500
            w, h = pyautogui.size()
            return jsonify({"width": int(w), "height": int(h)}), 200

        if action == "cursor_position":
            if not PYAUTOGUI_OK:
                return jsonify({"error": "pyautogui not available"}), 500
            x, y = pyautogui.position()
            return jsonify({"x": int(x), "y": int(y)}), 200

        if action == "screenshot":
            b64 = take_screenshot_b64()
            return jsonify({"image_base64": b64}), 200

        return jsonify({"error": f"Unsupported action: {action}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    print(f"Windows host bridge running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()