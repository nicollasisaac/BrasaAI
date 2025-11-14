# Brasa AI — MVP: Streamlit + OmniParser (/api/parse) + Agent (Observer/Planner/Judge via n8n) + PyAutoGUI
# Loop AUTOMÁTICO: screenshot -> /api/parse -> n8n -> ação -> repete até done.

import os
import json
from typing import List, Callable

import streamlit as st
from dotenv import load_dotenv

# Módulos locais do Brasa AI
from brasa_agent.runtime import run_loop

# -----------------------------------------------------------------------------
# Config & ENV
# -----------------------------------------------------------------------------
load_dotenv()  # carrega .env se existir

OMNI_API_URL_DEFAULT = os.getenv("OMNI_API_URL", "http://127.0.0.1:7867")
N8N_ENDPOINT_URL_DEFAULT = os.getenv("N8N_ENDPOINT_URL", "http://127.0.0.1:5678/webhook/brasa/agent")
MAX_ROUNDS_ENV = os.getenv("MAX_ROUNDS", "8")

st.set_page_config(page_title="Brasa AI — MVP (Loop Automático)", layout="wide")
st.title("🔥 Brasa AI — MVP (Loop Automático)")

st.caption(
    "Loop automático: **captura de tela → OmniParser → n8n (Observer/Planner/Judge) → PyAutoGUI**, "
    "repetindo até o objetivo ser atingido pelo Judge do n8n."
)

# -----------------------------------------------------------------------------
# Sidebar (config em tempo de execução)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Configuração")
    OMNI_API_URL = st.text_input("OMNI_API_URL (FastAPI /api/parse):", value=OMNI_API_URL_DEFAULT)
    N8N_ENDPOINT_URL = st.text_input("N8N_ENDPOINT_URL (Webhook Único do n8n):", value=N8N_ENDPOINT_URL_DEFAULT)
    st.write(f"MAX_ROUNDS (env): **{MAX_ROUNDS_ENV}**")
    st.info("Edite valores aqui ou defina no seu arquivo .env.\n"
            "Para abortar imediatamente, mova o mouse para o canto SUPERIOR ESQUERDO (PyAutoGUI FAILSAFE).")

# -----------------------------------------------------------------------------
# Orquestrador
# -----------------------------------------------------------------------------
st.markdown("### Objetivo do Agente")
user_goal = st.text_input(
    "Objetivo final",
    placeholder="ex.: enviar mensagem 'oi' para ofernando no Slack",
)

# Área de logs “ao vivo”
log_placeholder = st.empty()
logs_list: List[str] = []

def stream_log(msg: str):
    """Callback de log passado ao run_loop: mostra incrementalmente na UI."""
    try:
        logs_list.append(msg)
        # Renderiza como JSON quando possível, senão texto puro
        safe = msg
        try:
            parsed = json.loads(msg)
            safe = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
        log_placeholder.markdown(f"```\n{safe}\n```")
    except Exception:
        pass

st.divider()

# -----------------------------------------------------------------------------
# Execução do loop do agente (100% automático)
# -----------------------------------------------------------------------------
col_run, col_clear = st.columns([2,1])
with col_run:
    if st.button("▶️ Executar Brasa Agent (loop automático)", type="primary", use_container_width=True):
        if not user_goal.strip():
            st.error("Defina um objetivo final.")
        elif not N8N_ENDPOINT_URL.strip():
            st.error("Configure N8N_ENDPOINT_URL (Webhook único do n8n).")
        else:
            st.info("Iniciando loop… (FAILSAFE: mova o mouse para o canto superior esquerdo para ABORTAR)")
            logs_list.clear()
            try:
                # Dica: passe omni_initial=None para o runtime capturar e parsear automaticamente
                logs = run_loop(
                    goal=user_goal,
                    omni_initial=None,              # captura e parse inicial AUTOMÁTICOS
                    n8n_endpoint=N8N_ENDPOINT_URL,  # endpoint ÚNICO do n8n
                    on_log=stream_log,
                    omni_base_url=OMNI_API_URL,     # /api/parse
                )
                st.success("Loop encerrado (done pelo Judge ou limite de rodadas).")
                with st.expander("Ver todos os logs desta execução", expanded=False):
                    for line in logs:
                        st.markdown(f"```\n{line}\n```")
            except Exception as e:
                st.error(f"Erro durante a execução do loop: {e}")

with col_clear:
    if st.button("🧹 Limpar logs", use_container_width=True):
        logs_list.clear()
        log_placeholder.empty()
        st.success("Logs limpos.")

# -----------------------------------------------------------------------------
# Rodapé
# -----------------------------------------------------------------------------
st.caption(
    "Observações:\n"
    "• O runtime captura a tela a cada rodada, envia ao OmniParser e consulta o n8n.\n"
    "• O n8n retorna `{done:true,...}` (Judge) ou 1 ação (Planner). A ação é executada localmente (PyAutoGUI).\n"
    "• `MAX_ROUNDS` pode ser configurado via .env. PaddleOCR/Florence no OmniParser já estão ativados por padrão no cliente."
)
