# BrasaAI

BrasaAI é um agente que **lê a sua tela** (via OmniParser), entende comandos em linguagem natural e executa ações no seu PC em tempo real.  
O LLM usado é o **Google Gemini** via API REST.

---

## 📦 Requisitos

- **Python 3.12+**
- Git
- Ambiente virtual (venv)
- Uma chave de API do **Google Gemini** (Google AI Studio)

---

## 🚀 Instalação

1) **Clone o repositório**
```bash
git clone https://github.com/<seu-user>/BrasaAI.git
cd BrasaAI
````

2. **Crie e ative o venv**

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuração (Gemini)

Crie uma API key no **Google AI Studio**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Windows (PowerShell)

```powershell
$env:LLM_BACKEND="gemini";
$env:GEMINI_API_KEY="AIza...";            # sua chave
$env:GEMINI_MODEL="gemini-2.0-flash";     # opcional
```

---

## 🖥️ Subindo o Windows Host Bridge

O Windows Host é um servidor local que expõe uma API Flask para que o agent possa **executar ações** no Windows.

Inicie o Host Bridge em uma aba separada do terminal:

```powershell
python agent/windows_host.py --host 127.0.0.1 --port 8006
```

Teste se está rodando:

```powershell
curl http://127.0.0.1:8006/probe
```

Saída esperada:

```
Windows host bridge running at http://127.0.0.1:8006
 * Serving Flask app 'windows_host'
 * Debug mode: off
 * Running on http://127.0.0.1:8006
Press CTRL+C to quit
```

---

## 📸 OmniParser

Você pode usar um endpoint remoto (ex.: via Cloudflare Tunnel) ou rodar localmente.
Exemplo de endpoint remoto:

```
https://morris-been-applications-knock.trycloudflare.com
```

---

## ▶️ Rodando o Agent

Com o Windows Host ativo, venv ativado e variáveis configuradas:

```powershell
python -m agent.omnitool.gradio.app_new ^
  --omniparser_server_url https://<SEU_ENDPOINT_OMNIPARSER> ^
  --host 127.0.0.1 ^
  --port 7866
```

Acesse: **[http://127.0.0.1:7866](http://127.0.0.1:7866)**

---

## 🧪 Teste rápido do Gemini

```powershell
$env:LLM_BACKEND="gemini"; $env:GEMINI_API_KEY="AIza..."; $env:GEMINI_MODEL="gemini-2.0-flash"; python - << 'PY'
from agent.omnitool.gradio.agent.llm_utils.oaiclient import run_oci_interleaved
txt,_ = run_oci_interleaved([{"role":"user","content":[{"text":"Say only OK"}]}], system="Be concise.", max_tokens=8)
print("RESPOSTA:", txt)
PY
```

Saída esperada:

```
RESPOSTA: OK
```

---

## 🗂 Estrutura (essencial)

```
BrasaAI/
├─ agent/
│  ├─ windows_host.py             # Host Bridge no Windows (Flask API)
│  └─ omnitool/
│     ├─ gradio/
│     │  ├─ app_new.py            # UI do agente (Gradio)
│     │  ├─ loop.py               # Loop de raciocínio/execução
│     │  └─ agent/llm_utils/
│     │     └─ oaiclient.py       # Cliente LLM (Gemini via REST)
└─ requirements.txt
```

---

## 🔥 Fluxo resumido

1. Ative o ambiente virtual.
2. Configure as variáveis do Gemini.
3. **Suba o Windows Host**:

   ```powershell
   python agent/windows_host.py --host 127.0.0.1 --port 8006
   ```
4. **Suba o Agent**:

   ```powershell
   python -m agent.omnitool.gradio.app_new --omniparser_server_url https://<SEU_ENDPOINT> --host 127.0.0.1 --port 7866
   ```
5. Abra `http://127.0.0.1:7866` e use o agente.

---

## 📝 Licença

MIT. Contribuições são bem-vindas!
