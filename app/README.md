# Brasa AI — MVP

Automação local que converte um **objetivo em texto** (ex.: *“enviar mensagem de 'oi' para ofernando no Slack”*) em **ações na sua tela** usando:

* **Streamlit** (frontend)
* **OmniParser FastAPI** (`POST /api/parse`) para entender a tela via OCR + detecção de elementos
* **Agente** com **Planner** (um passo por rodada) e **Judge** (checa se terminou) — rodando no **n8n** *ou* local via API OpenAI‑compatível
* **PyAutoGUI** para executar cliques, hotkeys e digitação

> **Aviso:** isto é um MVP. Execute em ambiente de testes — automação de tela pode causar cliques indesejados.

---

## 🔧 Requisitos

* Python **3.10+**
* Acesso ao serviço **OmniParser FastAPI** com o endpoint `POST /api/parse` ativo
* (Opcional) **n8n** 1.110+ para orquestrar Planner/Judge
* Permissões do SO para captura de tela e automação de acessibilidade

  * **Windows**: conceder acesso a apps
  * **macOS**: Preferências → Segurança e Privacidade → Acessibilidade + Gravação de Tela

---

## 📁 Estrutura do projeto

```
brasa-ai/
├─ .env.example
├─ requirements.txt
├─ README.md
├─ streamlit_app.py
└─ brasa_agent/
   ├─ __init__.py
   ├─ omni.py              # cliente do FastAPI /api/parse
   ├─ actions.py           # primitives PyAutoGUI
   ├─ runtime.py           # loop: captura → parse → planner → execute → judge
   └─ planner_prompts.py   # prompts + fallback local (OpenAI-compat)
```

---

## 🚀 Instalação

1. Crie o ambiente e instale dependências:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
# Windows (recomendado para estabilidade)
pip install pywin32
```

2. Configure as variáveis copiando `.env.example` → `.env` e ajustando:

```ini
# Endpoint do seu OmniParser FastAPI
OMNI_API_URL=http://127.0.0.1:7867

# Webhooks do n8n (se usar n8n para planner/judge)
N8N_PLANNER_URL=http://127.0.0.1:5678/webhook/brasa/planner
N8N_JUDGE_URL=http://127.0.0.1:5678/webhook/brasa/judge

# Fallback local (se não usar n8n): API OpenAI‑compatível
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Ajustes
MAX_ROUNDS=6
SLEEP_BETWEEN_ACTIONS=0.6
```

3. (Opcional) Suba o **n8n**:

```yaml
# docker-compose.yml
version: "3.8"
services:
  n8n:
    image: n8nio/n8n:latest
    ports: ["5678:5678"]
    restart: unless-stopped
    environment:
      - TZ=America/Sao_Paulo
      - N8N_PROTOCOL=http
      - N8N_PORT=5678
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=troque_essa_senha
      - EXECUTIONS_TIMEOUT=-1
      - EXECUTIONS_TIMEOUT_MAX=0
      - N8N_PAYLOAD_SIZE_MAX=64
    volumes:
      - ./data:/home/node/.n8n
```

```bash
docker compose up -d
```

4. Rode o **frontend**:

```bash
streamlit run streamlit_app.py
```

---

## 🔌 OmniParser — API

**Endpoint:** `POST /api/parse` (multipart/form-data)

**Campos de formulário:**

* `file` *(png/jpg/jpeg, obrigatório)*
* `box_threshold` *(number)*
* `iou_threshold` *(number)*
* `use_paddleocr` *(boolean)*
* `imgsz` *(integer)*
* `describe_icons` *(boolean)* — legenda de ícones (mais lento em CPU)
* `return_image` *(boolean)*

**200 (application/json)**: retorna um objeto com `processing`, `input`, `parameters`, `environment` e `outputs.elements[]` (cada `element` possui `id`, `type`, `content`, `bbox.absolute` e `bbox.ratio`).

Exemplo (reduzido):

```json
{
  "processing": {"status": "ok", "time_ms": 70955},
  "input": {"image_size": {"width": 1918, "height": 1079}},
  "outputs": {
    "elements": [
      {"id": 4, "type": "text", "content": "Visual Studio Code", "bbox": {"absolute": [1037,9,1141,23]}},
      {"id": 28, "type": "text", "content": "TEXTO", "bbox": {"absolute": [585,223,627,237]}}
    ]
  }
}
```

---

## 🧠 Agente (Planner/Judge)

O fluxo do agente segue **rodadas** com **1 passo** por rodada:

1. Captura screenshot → envia para `/api/parse` → recebe elementos
2. **Planner** propõe **um** próximo passo (ex.: `CLICK_TEXT`, `TYPE`, `HOTKEY`, `WAIT`, `DONE`)
3. Executa com **PyAutoGUI**
4. Re-captura e re-parseia
5. **Judge** avalia se o objetivo foi atingido (`done: true`) ou continua

### Tipos de ação suportados

```json
// Planner deve responder um único objeto de ação
{ "type": "CLICK_TEXT", "target_text": "mensagem" }
{ "type": "TYPE", "text": "oi", "enter": true }
{ "type": "HOTKEY", "keys": ["ctrl","k"] }
{ "type": "WAIT", "ms": 400 }
{ "type": "DONE", "reason": "Mensagem enviada" }
```

### Prompts (fallback local)

**Planner (system):**

```
Você é um planner de 1 passo. Recebe 'goal' e alguns elementos visuais (texto+bbox).
Retorne apenas um JSON com o próximo passo: {type, ...}.
Tipos: CLICK_TEXT, TYPE, HOTKEY, WAIT, DONE.
Regras: 1 passo por rodada; seja objetivo; se não achar alvo, tente HOTKEY
(ex.: abrir Slack, buscar contato).
```

**Judge (system):**

```
Você é um juiz. Recebe goal, plan, result e elementos da tela atual.
Responda JSON: {done: bool, reason: string}. done=true se o objetivo foi atingido.
```

---

## 🧩 Configuração no n8n (Planner/Judge via Webhooks)

Crie **dois fluxos**:

### Fluxo 1 — Planner

1. **Webhook (Trigger)**

   * Path: `brasa/planner`
   * Method: `POST`
2. **OpenAI Chat** (ou outro LLM)

   * System: prompt do Planner acima
   * User (Expression):

   ```
   Goal: {{$json["goal"]}}
   Round: {{$json["round"]}}
   Elements: {{$json["elements_sample"]}}
   ```

   * Temperature: 0.2
3. **(Opcional) Code** — normaliza para JSON
4. **Respond to Webhook** — *Last Node*

### Fluxo 2 — Judge

1. **Webhook (Trigger)**

   * Path: `brasa/judge`
   * Method: `POST`
2. **OpenAI Chat**

   * System: prompt do Judge acima
   * User (Expression):

   ```
   Goal: {{$json["goal"]}}
   Round: {{$json["round"]}}
   Plan: {{$json["plan"]}}
   Result: {{$json["result"]}}
   Elements: {{$json["elements_sample"]}}
   ```
3. **Respond to Webhook**

> **Dica**: se o LLM devolver texto, use um **Code node** (JS) para `JSON.parse()`. Em erro, retorne `{type:"WAIT",ms:400}` (Planner) ou `{done:false, reason:"…"}` (Judge).

---

## 🖥️ Fluxo da UI (Streamlit)

1. Informe o **objetivo** (ex.: `enviar mensagem "oi" para ofernando no Slack`).
2. Clique em **“Capturar tela agora”** → **Enviar ao OmniParser** → veja o JSON.
3. Selecione orquestrador: **n8n** ou **Local (OpenAI‑compat)**.
4. Clique **“Executar Brasa Agent (loop)”**.

### Caso de teste (Slack)

Passos esperados (exemplo):

1. `HOTKEY ['ctrl','k']` (abrir busca)
2. `TYPE {text:'ofernando', enter:true}`
3. `TYPE {text:'oi', enter:true}`
4. Judge detecta histórico com “oi” enviado → `done: true`

> Depende do idioma/tema do Slack e da qualidade do OCR; ajuste prompts conforme necessário.

---

## 🧪 Dev Notes (código‑chave)

### Chamar o OmniParser

```python
# brasa_agent/omni.py
r = requests.post(
  f"{OMNI_API_URL}/api/parse",
  files={"file": ("screen.png", image_bytes, "image/png")},
  data={"describe_icons": True}
)
resp = r.json()
```

### Executar ações

```python
# brasa_agent/actions.py
{"type":"CLICK_TEXT","target_text":"mensagem"}
{"type":"TYPE","text":"oi","enter":true}
{"type":"HOTKEY","keys":["ctrl","k"]}
```

### Loop

```python
# brasa_agent/runtime.py
for round in range(MAX_ROUNDS):
  plan = planner(...)
  exec_result = execute_action(plan, elements)
  new_state = parse_screen_again()
  verdict = judge(...)
  if verdict.get("done"): break
```

---

## 🛡️ Segurança e boas práticas

* **PyAutoGUI FAILSAFE** ativo (mova o mouse ao canto superior esquerdo para abortar).
* Use em VM/desktop de testes; feche apps críticos.
* Ajuste `SLEEP_BETWEEN_ACTIONS` se sua máquina for mais lenta.
* Limite de tamanho do payload no n8n: `N8N_PAYLOAD_SIZE_MAX`.

---

## 🐛 Troubleshooting

* **422 no /api/parse**: geralmente `file` ausente ou multipart incorreto. Garanta `files={"file": ("screen.png", bytes, "image/png")}` e **não** envie `Content-Type` manual.
* **n8n devolvendo texto**: normalize com Code node (`JSON.parse`) antes do Respond.
* **Cliques fora de lugar**: tema/zoom alteram posições detectadas. Ajuste prompts e considere novos tipos (e.g., `CLICK_NEAR`).
* **Permissões**: macOS requer Acessibilidade + Gravação de Tela; Windows pode precisar `pywin32`.

---

## 🗺️ Roadmap

* Desenhar bboxes na screenshot (debug visual)
* Novas ações: `CLICK_NEAR({hint_text, dx, dy})`, `FOCUS_INPUT({near_text})`
* Heurística de diff de estado entre rodadas
* Fallback com Slack Web API
* Logs JSONL e gravação de gif/vídeo
* Empacotar como binário (PyInstaller)

---

## 📜 Licença

MVP interno/educacional. Ajuste a licença conforme seu uso (ex.: MIT/Apache‑2.0).
