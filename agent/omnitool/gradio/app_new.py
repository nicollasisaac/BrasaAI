# agent/omnitool/gradio/app_new.py
import os, argparse, gradio as gr
from agent.omnitool.gradio.loop import sampling_loop_sync  # <-- importa via 'agent.*'

# ===================== Função Principal do Agente =====================
def run_agent(user_input, omniparser_url):
    msgs = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
    outputs = []

    def out_cb(msg): 
        outputs.append(str(msg))
    def tool_cb(tr, name): 
        outputs.append(f"[{name}] {tr}")
    def api_cb(resp): 
        outputs.append(f"[API] {resp}")

    for _ in sampling_loop_sync(
        model="omniparser + gemini",  # Ajuste para Gemini
        provider="gemini",
        messages=msgs,
        output_callback=out_cb,
        tool_output_callback=tool_cb,
        api_response_callback=api_cb,
        omniparser_url=omniparser_url,
    ):
        pass
    return "\n".join(outputs)

# ===================== Função de Inicialização =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omniparser_server_url", required=True, help="ex.: 127.0.0.1:7860 ou http://10.0.0.12:7860")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7865)
    args = ap.parse_args()

    # Normaliza URL do OmniParser
    omni = args.omniparser_server_url
    if not omni.startswith("http"):
        omni = "http://" + omni

    # ===================== Tema Personalizado =====================
    theme = gr.themes.Default(
        primary_hue="green", 
        secondary_hue="green", 
        neutral_hue="gray"
    ).set(
        body_background_fill="#ffffff",
        body_text_color="#1a1a1a",
        button_primary_background_fill="#16a34a",  # Verde vibrante
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#86efac",
        button_secondary_text_color="#064e3b",
        radius="lg"
    )

    css = """
    #title { 
        font-size: 2.5em; 
        font-weight: 800; 
        color: #16a34a; 
        text-align: center; 
        margin-bottom: 0.2em;
    }
    #desc { 
        text-align: center; 
        font-size: 1.2em; 
        color: #1a1a1a; 
        margin-bottom: 1em;
    }
    """

    # ===================== Layout =====================
    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown("# BrasaAI", elem_id="title")
        gr.Markdown("### 🇧🇷 A AI brasileira que controla o seu notebook!", elem_id="desc")
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Textbox(
                    label="Digite seu comando:",
                    placeholder="Ex: Abra o Chrome e pesquise por notícias de tecnologia",
                    lines=1
                )
                output_box = gr.Textbox(
                    label="Respostas do Agente",
                    placeholder="Saída das ações executadas aparecerá aqui...",
                    lines=10
                )
                run_button = gr.Button("▶️ Executar", variant="primary")

        run_button.click(fn=lambda text: run_agent(text, omni), inputs=chatbot, outputs=output_box)

    demo.launch(server_name=args.host, server_port=args.port)

# ===================== EntryPoint =====================
if __name__ == "__main__":
    main()
