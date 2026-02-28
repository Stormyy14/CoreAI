"""
CoreAI – FastAPI web server
────────────────────────────────────────────────────────────────────────────────
Serves the CoreAI web UI and exposes REST + SSE endpoints so the browser can
chat with either the built-in CoreGPT or an Ollama model in real time.

Start with:
    python server.py
    (or inside the venv)
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload

Endpoints
---------
GET  /                    → web/index.html
GET  /api/status          → model loaded? (JSON)
GET  /api/stats           → live system stats (JSON)
GET  /api/backends        → list available backends + current selection
POST /api/backends/select → {"backend": "coreai" | "ollama", "model": "mistral"}
POST /api/chat            → full-answer chat (JSON)
GET  /api/stream          → SSE streaming chat (?q=<question>)
POST /api/clear           → clear conversation history
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import AsyncGenerator

import psutil

GITHUB_RAW = "https://raw.githubusercontent.com/Stormyy14/CoreAI/main/version.json"

# ── FastAPI / Starlette ────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, UploadFile, File
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
except ImportError:
    sys.exit("fastapi / uvicorn not installed.\nRun: pip install fastapi uvicorn[standard]")

# ── CoreAI backend ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from coreai import LLMBackend, OllamaBackend, Config  # type: ignore

# ══════════════════════════════════════════════════════════════════════════════
# App setup
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="CoreAI", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB_DIR    = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
WEB_DIR.mkdir(exist_ok=True)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ══════════════════════════════════════════════════════════════════════════════
# Backend manager
# ══════════════════════════════════════════════════════════════════════════════

class BackendManager:
    """
    Manages two backends (CoreAI CoreGPT and Ollama) and lets the user
    switch between them at runtime via the web UI.
    """
    def __init__(self) -> None:
        self._coreai  : LLMBackend   | None = None
        self._ollama  : OllamaBackend| None = None
        self._current : str  = "coreai"   # "coreai" | "ollama"
        self._loading : bool = False
        self._error   : str  = ""

    # ── CoreAI (CoreGPT) ──────────────────────────────────────────────────────

    def load_coreai(self) -> None:
        """Load (or return cached) CoreGPT backend."""
        if self._coreai is not None:
            return
        self._loading = True
        self._error   = ""
        try:
            self._coreai = LLMBackend.load()
        except Exception as exc:
            self._error  = str(exc)
            self._coreai = None
        finally:
            self._loading = False

    # ── Ollama ────────────────────────────────────────────────────────────────

    def load_ollama(self, model: str | None = None) -> None:
        cfg = Config()
        self._ollama = OllamaBackend(model or cfg.ollama_model)

    def ollama_available(self) -> bool:
        return OllamaBackend.is_available()

    def ollama_models(self) -> list[str]:
        return OllamaBackend.list_models()

    # ── Active backend ────────────────────────────────────────────────────────

    def select(self, backend: str, model: str | None = None) -> str:
        """Switch active backend. Returns error string or ''."""
        if backend == "ollama":
            if not self.ollama_available():
                return "Ollama is not running. Install and start it first."
            self.load_ollama(model)
            self._current = "ollama"
            return ""
        elif backend == "coreai":
            self._current = "coreai"
            return ""
        return f"Unknown backend: {backend}"

    def get(self) -> LLMBackend | OllamaBackend | None:
        if self._current == "ollama":
            return self._ollama
        return self._coreai

    @property
    def current(self) -> str:
        return self._current

    @property
    def is_ready(self) -> bool:
        if self._current == "ollama":
            return self._ollama is not None and self.ollama_available()
        return self._coreai is not None

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> str:
        return self._error

    def clear_history(self) -> None:
        if self._coreai:
            self._coreai.history.clear()
        if self._ollama:
            self._ollama.history.clear()


manager = BackendManager()


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load CoreAI (CoreGPT) in a thread at startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, manager.load_coreai)


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    html_path = WEB_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>web/index.html not found</h1>", status_code=500)
    return FileResponse(str(html_path))


@app.get("/app", response_class=HTMLResponse)
async def chat_app() -> FileResponse:
    html_path = WEB_DIR / "app.html"
    if not html_path.exists():
        return HTMLResponse("<h1>web/app.html not found</h1>", status_code=500)
    return FileResponse(str(html_path))


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "ready"   : manager.is_ready,
        "loading" : manager.loading,
        "error"   : manager.error,
        "backend" : manager.current,
        "model"   : (
            "CoreGPT v2 (88M params)"
            if manager.current == "coreai"
            else (manager._ollama.model if manager._ollama else "—")
        ),
    })


# ── Backends list + selection ─────────────────────────────────────────────────

@app.get("/api/backends")
async def list_backends() -> JSONResponse:
    loop = asyncio.get_event_loop()
    # Run blocking urllib calls in a thread so they don't freeze the event loop
    ollama_ok     = await loop.run_in_executor(None, manager.ollama_available)
    ollama_models = await loop.run_in_executor(None, manager.ollama_models) if ollama_ok else []
    return JSONResponse({
        "current"         : manager.current,
        "coreai_ready"    : manager._coreai is not None,
        "coreai_loading"  : manager.loading,
        "ollama_available": ollama_ok,
        "ollama_models"   : ollama_models,
    })


@app.post("/api/backends/select")
async def select_backend(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    backend = body.get("backend", "coreai")
    model   = body.get("model")

    loop = asyncio.get_event_loop()
    err  = await loop.run_in_executor(None, lambda: manager.select(backend, model))
    if err:
        return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"ok": True, "backend": manager.current})


# ── System stats ──────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def system_stats() -> JSONResponse:
    cpu = psutil.cpu_percent(interval=0.2)
    ram = psutil.virtual_memory()
    stats: dict = {
        "cpu_pct"  : cpu,
        "ram_used" : ram.used  // (1024 ** 2),
        "ram_total": ram.total // (1024 ** 2),
        "ram_pct"  : ram.percent,
    }
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        ).strip()
        name, temp, util, vram_used, vram_total = [s.strip() for s in out.split(",")]
        stats["gpu"] = {
            "name"      : name,
            "temp_c"    : int(temp),
            "util_pct"  : int(util),
            "vram_used" : int(vram_used),
            "vram_total": int(vram_total),
        }
    except Exception:
        stats["gpu"] = None
    return JSONResponse(stats)


# ── Full-answer chat ──────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is empty"}, status_code=400)

    backend = manager.get()
    if backend is None:
        return JSONResponse({"error": "No backend ready. Try again in a moment."}, status_code=503)

    web_search  : bool  = body.get("web_search", True)
    temperature : float = float(body.get("temperature", 0.75))
    max_tokens  : int   = int(body.get("max_tokens", 400))

    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()
    answer = await loop.run_in_executor(
        None,
        lambda: backend.chat(question, use_web_search=web_search,
                             temperature=temperature, max_new_tokens=max_tokens),
    )
    elapsed = round(time.perf_counter() - t0, 2)
    return JSONResponse({"answer": answer, "elapsed_s": elapsed, "backend": manager.current})


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────

@app.get("/api/stream")
async def stream_endpoint(
    q          : str   = "",
    web_search : bool  = True,
    temperature: float = 0.75,
    max_tokens : int   = 400,
) -> StreamingResponse:
    question = q.strip()
    if not question:
        return StreamingResponse(_sse_error("question is empty"), media_type="text/event-stream")

    backend = manager.get()
    if backend is None:
        return StreamingResponse(
            _sse_error("No backend ready. Try again in a moment."),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _sse_stream(backend, question, web_search, temperature, max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse_stream(
    backend    : LLMBackend | OllamaBackend,
    question   : str,
    web_search : bool,
    temperature: float,
    max_tokens : int,
) -> AsyncGenerator[str, None]:
    loop  = asyncio.get_event_loop()
    queue : asyncio.Queue[str | None] = asyncio.Queue()

    def _produce() -> None:
        try:
            for token in backend.stream_chat(
                question,
                use_web_search=web_search,
                temperature=temperature,
                max_new_tokens=max_tokens,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _produce)
    yield "event: start\ndata: {}\n\n"

    while True:
        token = await queue.get()
        if token is None:
            break
        yield f"data: {json.dumps({'token': token})}\n\n"

    yield "event: done\ndata: {}\n\n"


async def _sse_error(msg: str) -> AsyncGenerator[str, None]:
    yield f"event: error\ndata: {json.dumps({'error': msg})}\n\n"


# ── Clear history ─────────────────────────────────────────────────────────────

@app.post("/api/clear")
async def clear_history() -> JSONResponse:
    manager.clear_history()
    return JSONResponse({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# Conversation history (SQLite)
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = ROOT / "data" / "history.db"


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _init_db() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    with _db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                backend TEXT DEFAULT 'coreai',
                created_at REAL,
                updated_at REAL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL
            )""")
        c.commit()


_init_db()


@app.get("/api/history")
async def list_conversations() -> JSONResponse:
    with _db() as c:
        rows = c.execute(
            "SELECT id, title, backend, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.post("/api/history")
async def new_conversation(request: Request) -> JSONResponse:
    body   = await request.json()
    cid    = str(uuid.uuid4())
    now    = time.time()
    title  = (body.get("title") or "New Chat")[:60]
    backend = body.get("backend", "coreai")
    with _db() as c:
        c.execute(
            "INSERT INTO conversations (id, title, backend, created_at, updated_at) VALUES (?,?,?,?,?)",
            (cid, title, backend, now, now),
        )
        c.commit()
    return JSONResponse({"id": cid})


@app.get("/api/history/{conv_id}")
async def get_conversation(conv_id: str) -> JSONResponse:
    with _db() as c:
        conv = c.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if not conv:
            return JSONResponse({"error": "not found"}, status_code=404)
        msgs = c.execute(
            "SELECT role, content FROM messages WHERE conv_id=? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
    return JSONResponse({"conv": dict(conv), "messages": [dict(m) for m in msgs]})


@app.delete("/api/history/{conv_id}")
async def delete_conversation(conv_id: str) -> JSONResponse:
    with _db() as c:
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        c.commit()
    return JSONResponse({"ok": True})


@app.post("/api/history/{conv_id}/message")
async def save_message(conv_id: str, request: Request) -> JSONResponse:
    body    = await request.json()
    role    = body.get("role", "user")
    content = body.get("content", "")
    title   = body.get("title")          # optional: update title
    now     = time.time()
    with _db() as c:
        c.execute(
            "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?,?,?,?)",
            (conv_id, role, content, now),
        )
        if title:
            c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                      (title[:60], now, conv_id))
        else:
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))
        c.commit()
    return JSONResponse({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# File upload
# ══════════════════════════════════════════════════════════════════════════════

MAX_FILE_CHARS = 12_000


def _extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            text = "[pypdf not installed — install with: pip install pypdf]"
        except Exception as e:
            text = f"[PDF error: {e}]"
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n...[truncated — showing {MAX_FILE_CHARS} of {len(text)} chars]"
    return text


@app.post("/api/history/restore")
async def restore_message(request: Request) -> JSONResponse:
    """Restore a Q/A pair into the active backend's in-memory history."""
    body = await request.json()
    q, a = body.get("q",""), body.get("a","")
    be = manager.get()
    if be and q and a:
        be.history.append((q, a))
    return JSONResponse({"ok": True})


# ── Ollama model management ────────────────────────────────────────────────────

@app.post("/api/ollama/pull")
async def ollama_pull(request: Request) -> JSONResponse:
    body  = await request.json()
    model = body.get("model","").strip()
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: subprocess.Popen(["ollama", "pull", model],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    )
    return JSONResponse({"ok": True, "msg": f"Pulling {model} in background…"})


@app.post("/api/ollama/remove")
async def ollama_remove(request: Request) -> JSONResponse:
    body  = await request.json()
    model = body.get("model","").strip()
    if not model:
        return JSONResponse({"error": "model required"}, status_code=400)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(["ollama", "rm", model], capture_output=True, text=True),
    )
    if result.returncode != 0:
        return JSONResponse({"error": result.stderr.strip()}, status_code=400)
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    data     = await file.read()
    filename = file.filename or "file"
    text     = _extract_text(filename, data)
    return JSONResponse({
        "filename": filename,
        "content" : text,
        "chars"   : len(text),
        "size_kb" : round(len(data) / 1024, 1),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Update system
# ══════════════════════════════════════════════════════════════════════════════

def _read_local_version() -> dict:
    p = ROOT / "version.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"version": "0.0.0"}


def _fetch_remote_version() -> dict:
    with urllib.request.urlopen(GITHUB_RAW, timeout=8) as r:
        return json.loads(r.read())


def _version_tuple(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.split("."))
    except Exception:
        return (0, 0, 0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.get("/api/version")
async def get_version() -> JSONResponse:
    return JSONResponse(_read_local_version())


@app.get("/api/check-update")
async def check_update() -> JSONResponse:
    try:
        loop   = asyncio.get_event_loop()
        remote = await loop.run_in_executor(None, _fetch_remote_version)
        local  = _read_local_version()
        has_update = _version_tuple(remote["version"]) > _version_tuple(local["version"])
        return JSONResponse({
            "has_update"    : has_update,
            "local_version" : local.get("version", "?"),
            "remote_version": remote.get("version", "?"),
            "changelog"     : remote.get("changelog", ""),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc), "has_update": False}, status_code=200)


@app.get("/api/update")
async def do_update() -> StreamingResponse:
    return StreamingResponse(
        _update_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _update_stream() -> AsyncGenerator[str, None]:
    def sse(msg: str, kind: str = "log") -> str:
        return f"data: {json.dumps({'type': kind, 'msg': msg})}\n\n"

    loop = asyncio.get_event_loop()

    # ── 1. git pull ────────────────────────────────────────────────────────────
    yield sse("Pulling latest code from GitHub…")
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            ),
        )
        if result.returncode != 0:
            yield sse(f"git pull error: {result.stderr.strip()}", "error")
            return
        yield sse(result.stdout.strip() or "Already up to date.")
    except Exception as exc:
        yield sse(f"git pull failed: {exc}", "error")
        return

    # ── 2. Download new model weights if listed in remote version.json ─────────
    try:
        remote = await loop.run_in_executor(None, _fetch_remote_version)
    except Exception as exc:
        yield sse(f"Could not fetch remote version.json: {exc}", "error")
        return

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    for name, info in remote.get("models", {}).items():
        url      = info.get("url", "")
        expected = info.get("sha256", "")
        filename = info.get("filename", f"{name}.pth")
        dest     = models_dir / filename

        if not url:
            continue

        # Skip if already up to date
        if dest.exists() and expected and _sha256(dest) == expected:
            yield sse(f"Model {filename} already up to date.")
            continue

        yield sse(f"Downloading {filename}…")
        try:
            def _download():
                urllib.request.urlretrieve(url, str(dest))
            await loop.run_in_executor(None, _download)

            if expected:
                actual = await loop.run_in_executor(None, lambda: _sha256(dest))
                if actual != expected:
                    dest.unlink(missing_ok=True)
                    yield sse(f"Checksum mismatch for {filename}!", "error")
                    return
            yield sse(f"Downloaded {filename}.")
        except Exception as exc:
            yield sse(f"Download failed: {exc}", "error")
            return

    # ── 3. Write new local version.json ────────────────────────────────────────
    (ROOT / "version.json").write_text(json.dumps(remote, indent=2))

    # ── 4. Restart server ──────────────────────────────────────────────────────
    yield sse("Update complete — restarting…", "done")
    await asyncio.sleep(0.5)

    async def _restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    asyncio.create_task(_restart())


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n  ╔═══════════════════════════════════════════╗")
    print("  ║          CoreAI  Web Server               ║")
    print("  ║   http://localhost:8080                   ║")
    print("  ║                                           ║")
    print("  ║   Backends: CoreAI (CoreGPT) + Ollama    ║")
    print("  ╚═══════════════════════════════════════════╝\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
