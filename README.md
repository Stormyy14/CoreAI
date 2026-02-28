# CoreAI

Local AI platform built from scratch. Runs entirely on your machine — no cloud, no API keys, no data leaving your device.

![Version](https://img.shields.io/badge/version-1.0.0-white?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10+-white?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-white?style=flat-square)

---

## Features

- **CoreGPT** — custom LLM trained from scratch (27.8M params, RoPE + SwiGLU + RMSNorm)
- **Ollama support** — switch to any Ollama model at runtime (Mistral, Llama, Phi, etc.)
- **Streaming chat** — real-time token streaming via SSE
- **Web search** — automatic DuckDuckGo search when needed, no API key required
- **System monitor** — live CPU, RAM and GPU stats in the sidebar
- **Auto-update** — one-click updates from the UI when a new version is available
- **BPE tokenizer** — byte-level, 4096 vocab, trained on the corpus
- **100% local** — nothing leaves your machine

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (runs on CPU too, but slower)
- ~500 MB disk space (without model weights)

---

## Quick Start

```bash
# Clone
git clone https://github.com/Stormyy14/CoreAI.git
cd CoreAI

# Install dependencies
pip install -r requirements.txt

# GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Start
python server.py
# → open http://localhost:8080
```

---

## Model Weights

Model weights are not included in the repository (too large for git).

**Option A — Train from scratch:**
```bash
python linux_ai.py train-llm
# ~40 min on a modern GPU
```

**Option B — Download pre-trained:**
Check the [Releases](https://github.com/Stormyy14/CoreAI/releases) page for pre-trained `coregpt.pth` and place it in `models/`.

---

## Project Structure

```
CoreAI/
├── linux_ai.py        # Core AI engine (models, training, inference)
├── server.py          # FastAPI web server + SSE + update system
├── requirements.txt
├── version.json       # Version manifest for auto-updates
├── models/            # Trained weights (not tracked in git)
│   ├── coregpt.pth
│   └── tokenizer.json
├── data/
│   └── corpus.txt     # Training corpus
├── web/
│   ├── index.html     # Landing page
│   ├── app.html       # Chat UI
│   └── static/
└── install/
    ├── linux/         # .deb, .rpm, systemd service
    └── windows/       # .bat, .ps1, PyInstaller .exe
```

---

## Architecture

| Component | Details |
|---|---|
| Tokenizer | BPE byte-level, 4096 vocab |
| Embedding | 512 dim |
| Layers | 8 transformer blocks |
| Attention | Multi-head (8 heads) + RoPE |
| FFN | SwiGLU (LLaMA-style) |
| Norm | RMSNorm |
| Context | 512 tokens |
| Parameters | ~27.8M |
| Training | bf16, AdamW, gradient accumulation |

---

## Training

```bash
# Full train (pretrain + finetune) ~40 min on GPU
python linux_ai.py train-llm

# Fine-tune only (requires models/coregpt_pretrain.pth)
python linux_ai.py finetune-only

# Terminal chat
python linux_ai.py chat
```

---

## Linux Install

```bash
chmod +x install/linux/install.sh
./install/linux/install.sh

# Then run from anywhere:
coreai
```

Supports Debian/Ubuntu (`.deb`), Fedora/RHEL (`.rpm`), and Arch.

## Windows Install

Run `install/windows/install.bat` as Administrator, or use the PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass -File install\windows\start_coreai.ps1
```

---

## Auto-Update

CoreAI checks for updates automatically every 5 minutes. When a new version is available, an "Update available" banner appears in the sidebar. Click **Update now** to:

1. Pull latest code from GitHub
2. Download new model weights (if any)
3. Restart the server automatically

---

## License

MIT — do whatever you want with it.
