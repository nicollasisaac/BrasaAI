# agent/omnitool/gradio/app_new.py
import os
import argparse
import gradio as gr

# ===== PATCH de schema do Gradio (corrige additionalProperties: true/false) =====
try:
    import gradio.blocks as gr_blocks  # type: ignore

    _orig_get_api_info = getattr(gr_blocks.Blocks, "get_api_info", None)

    def _sanitize_schema(obj):
        if isinstance(obj, dict):
            new_d = {}
            for k, v in obj.items():
                if k == "additionalProperties" and isinstance(v, bool):
                    # substitui bool por um schema simples aceito pelo gradio_client
                    new_d[k] = {"type": "string"}
                else:
                    new_d[k] = _sanitize_schema(v)
            return new_d
        elif isinstance(obj, list):
            return [_sanitize_schema(x) for x in obj]
        else:
            return obj

    def _patched_get_api_info(self):
        info = _orig_get_api_info(self) if _orig_get_api_info else {}
        return _sanitize_schema(info)

    if _orig_get_api_info and not getattr(gr_blocks.Blocks, "_brasa_patched_schema", False):
        gr_blocks.Blocks.get_api_info = _patched_get_api_info  # type: ignore[attr-defined]
        setattr(gr_blocks.Blocks, "_brasa_patched_schema", True)
except Exception:
    pass
# ===============================================================================

from agent.omnitool.gradio.loop import sampling_loop_sync

def run_agent(user_input: str, omniparser_url: str):
    """
    Executa o loop do agente e retorna apenas:
      - logs (str)
    """
    msgs = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
    logs = []

    def out_cb(msg):
        logs.append(str(msg))

    def tool_cb(tr, name):
        logs.append(f"[{name}] {tr}")

    def api_cb(resp):
        logs.append(f"[API] {resp}")

    # Executa em modo GEMINI
    for item in sampling_loop_sync(
        model="omniparser + gemini",
        provider="gemini",
        messages=msgs,
        output_callback=out_cb,
        tool_output_callback=tool_cb,
        api_response_callback=api_cb,
        omniparser_url=omniparser_url,
    ):
        # item pode ser None ou dict {"analysis": str, ...}; só texto nos interessa
        if isinstance(item, dict) and item.get("analysis"):
            logs.append(f"[analysis] {item['analysis']}")

    return "\n".join(logs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omniparser_server_url", required=True, help="ex.: 127.0.0.1:7860 ou http://10.0.0.12:7860")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7865)
    args = ap.parse_args()

    omni = args.omniparser_server_url
    if not omni.startswith("http"):
        omni = "http://" + omni

    # interface mínima (sem tema customizado e sem imagem)
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("# BrasaAI — Logs do Agente (Gemini)")
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Comando",
                    placeholder="Ex: Abra o Slack / Abra youtube.com / Buscar 'Ana Maria Braga bolo'",
                    lines=1
                )
                output_box = gr.Textbox(
                    label="Logs do Agente",
                    placeholder="Saída textual do agente…",
                    lines=22
                )
                run_button = gr.Button("▶️ Executar")

        def _runner(text):
            return run_agent(text, omni)

        run_button.click(
            fn=_runner,
            inputs=prompt,
            outputs=output_box
        )

    # manter API ativa (schema já sanitizado); share=True pra evitar problema de localhost
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_TELEMETRY_ENABLED", "False")
    demo.launch(server_name=args.host, server_port=args.port, share=True, show_api=True)

if __name__ == "__main__":
    main()
