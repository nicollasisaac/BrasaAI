## Como rodar (local/VM)

### Requisitos

* Python 3.10+ (recomendado 3.12)
* Ubuntu/Debian com `ffmpeg` e build tools (opcional, mas ajuda):

  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg build-essential
  ```
* (Opcional) GPU CUDA instalada para acelerar YOLO/Florence-2

### 1) Clonar e preparar o ambiente

```bash
# ambiente virtual
python -m venv .venv
source .venv/bin/activate

# instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

> CPU apenas? você também pode manter um arquivo `req.cpu.txt` minimizado para ambientes sem GPU e instalar com `pip install -r req.cpu.txt`.

### 2) Executar a aplicação

```bash
python app.py
```

A aplicação sobe:

* **Gradio UI** em `http://0.0.0.0:7860/`
* **FastAPI** (mesmo servidor) com endpoints:

  * `GET /health`
  * `POST /api/parse` (principal)
  * `GET /docs` (Swagger)

> Em VM (AWS/GCP/Azure), libere a **porta 7860** no Security Group / firewall e acesse via `http://SEU_IP_PUBLICO:7860/`.

### 3) Parâmetros importantes

* `box_threshold` (float, default `0.05`) – confiança do YOLO.
* `iou_threshold` (float, default `0.1`) – supressão de sobreposição.
* `imgsz` (int, default `640`) – resolução de inferência do YOLO.
* `use_paddleocr` (bool) – ativa OCR via **PaddleOCR** (se instalado).
* `describe_icons` (bool) – ativa legendas de ícones via **Florence-2** (mais lento em CPU).
* `return_image` (bool) – além do JSON completo, devolve imagem anotada em base64 no topo.

---

## Como usar a API

### `curl` — enviar PNG/JPG

```bash
curl -s -X POST "http://SEU_IP_PUBLICO:7860/api/parse" \
  -F "file=@/caminho/sua_imagem.png" \
  -F "box_threshold=0.05" \
  -F "iou_threshold=0.1" \
  -F "imgsz=640" \
  -F "use_paddleocr=false" \
  -F "describe_icons=false" \
  -F "return_image=false" \
| jq .
```

### `curl` — enviar imagem remota por URL (server baixa)

> Se seu `app.py` estiver com o endpoint `/api/parse_url` habilitado:

```bash
curl -s -X POST "http://SEU_IP_PUBLICO:7860/api/parse_url" \
  -F "url=https://exemplo.com/screenshot.png" \
  -F "box_threshold=0.05" \
  -F "iou_threshold=0.1" \
  -F "imgsz=640" \
  -F "use_paddleocr=false" \
  -F "describe_icons=false" \
  -F "return_image=false" \
| jq .
```

### Python (requests)

```python
import requests

url = "http://SEU_IP_PUBLICO:7860/api/parse"
files = {"file": open("sua_imagem.png", "rb")}
data = {
    "box_threshold": 0.05,
    "iou_threshold": 0.1,
    "imgsz": 640,
    "use_paddleocr": "false",
    "describe_icons": "true",   # ativa Florence-2 (lento em CPU)
    "return_image": "false",
}
resp = requests.post(url, files=files, data=data, timeout=120)
print(resp.status_code)
print(resp.json())
```

---

## Como usar a UI (Gradio)

1. Acesse `http://SEU_IP_PUBLICO:7860/`
2. Faça upload de uma imagem (screenshot de GUI, por exemplo).
3. Ajuste os sliders/checkboxes:

   * **Use PaddleOCR** para melhorar OCR (se tiver a lib instalada).
   * **Descrever ícones (Florence-2)** para legendas (mais lento em CPU).
4. Clique em **Submit**.
   A UI mostra:

   * imagem anotada com bounding boxes
   * **JSON completo** com:

     * `outputs.elements` (ícones e textos unificados, com bbox em ratio e absoluto)
     * `outputs.ocr_raw` (OCR bruto)
     * `outputs.label_coordinates`
     * `outputs.model_raw` (todos os brutos do pipeline, inclusive a imagem anotada em base64)
   * tempos de processamento (`processing.time_ms` e `processing.stages_ms`)

---

## Hugging Face Spaces

O cabeçalho do seu README está configurado para Spaces:

```yaml
---
title: OmniParser V2
emoji: 🏢
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: mit
short_description: OmniParser, turn your LLM into GUI agent
---
```

**Importante:** alinhe `sdk_version` com a versão do Gradio usada no projeto. Se localmente você está em `gradio==5.44.x`, mude para:

```yaml
sdk_version: 5.44.1
```

Ou, se preferir manter `5.16.0`, garanta no `requirements.txt` que a versão do Gradio seja compatível.

> No Spaces, o app sobe automaticamente com `app_file: app.py`. Você não precisa rodar `uvicorn`; o Spaces cuida disso.

---

## Dicas de performance

* **GPU**: se disponível, a detecção e legendagem ficam muito mais rápidas.
* **CPU**: aumente `imgsz` com moderação; `640`–`960` é um bom compromisso.
* **Florence-2**: `describe_icons=true` é útil, porém pode ser pesado em CPU; use quando necessário.

---

## Resolução de problemas

* **“Object of type float32 is not JSON serializable”**
  Já tratamos o JSON para converter floats/ints do NumPy. Se ainda ver algo, limpe o cache e confirme que está rodando a versão mais recente dos arquivos do repositório.

* **Porta não abre na VM**
  Abra a **porta 7860** no Security Group / firewall. Na AWS: Security Group → Inbound rules → Add rule TCP 7860.

* **Conflitos de versão Gradio / gradio-client**
  Sempre instale pares compatíveis (ex.: `gradio==5.44.1` + `gradio-client==1.12.1`). Se usar o Spaces, deixe o `sdk_version` coerente.

