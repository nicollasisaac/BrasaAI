# windows_host.py
import argparse
import subprocess
import sys
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/execute", methods=["POST"])
def execute():
    try:
        data = request.get_json(force=True)
        cmd = data.get("command")
        if not isinstance(cmd, list) or not cmd:
            return jsonify({"error": "Invalid command payload"}), 400

        # Executa o comando e captura saída
        # Ex.: ["python", "-c", "import pyautogui; print(pyautogui.size())"]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        return jsonify({
            "returncode": result.returncode,
            "output": result.stdout,
            "error": result.stderr,
        }), 200
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
