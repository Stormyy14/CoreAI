#!/usr/bin/env python3
"""
LinuxAI – Comprehensive AI-Powered Linux Intelligence & Control Platform
=========================================================================
Version : 2.0.0  |  Python 3.9+  |  Linux only

Install dependencies:
    pip install -r requirements.txt
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Run modes:
    python coreai.py              → interactive menu  (default)
    python coreai.py train        → train / retrain ALL models (CNN + LLM + sklearn)
    python coreai.py train-llm    → train only the built-in LLM (downloads corpus automatically)
    python coreai.py chat         → conversational AI shell (powered by the built-in LLM)
    python coreai.py monitor      → live system dashboard (Ctrl-C to exit)
    python coreai.py agent        → natural-language AI agent (system control + LLM)
    python coreai.py predict      → infer a digit from the MNIST test set
    python coreai.py control      → one-shot system-control snapshot

Built-in LLM architecture (GPT-style, trained from scratch):
    Tokenizer  : character-level (no external deps)
    Attention  : multi-head causal self-attention (flash-attention when available)
    Block      : pre-norm TransformerBlock (attn + GELU-FFN, both with residual)
    Model      : CoreGPT  (~3 M parameters by default)
    Training   : Phase-1 pretraining on text corpus, Phase-2 instruction fine-tuning
    Inference  : temperature + top-k autoregressive sampling

Quick usage after training:
    >>> from coreai import load_and_predict, LLMBackend
    >>> digit = load_and_predict("models/mnist_cnn.pth", dataset_index=42)
    >>> llm = LLMBackend.load(); print(llm.chat("What is a neural network?"))
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import io
import json
import logging
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Numeric / ML
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────────────────────────────────
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI
# ─────────────────────────────────────────────────────────────────────────────
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Directory scaffold
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
LOGS_DIR   = Path("logs")
DATA_DIR   = Path("data")

for _d in (MODELS_DIR, LOGS_DIR, DATA_DIR):
    _d.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "coreai.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("coreai")


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Training hyper-parameters ─────────────────────────────────────────────
    batch_size  : int   = 128
    cnn_epochs  : int   = 10
    lr          : float = 1e-3

    # ── Device (auto-select GPU if available) ─────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Saved-model paths ─────────────────────────────────────────────────────
    cnn_path      : Path = MODELS_DIR / "mnist_cnn.pth"
    iso_path      : Path = MODELS_DIR / "anomaly_detector.joblib"
    proc_clf_path : Path = MODELS_DIR / "process_classifier.joblib"
    scaler_path   : Path = MODELS_DIR / "scaler.joblib"

    # ── Built-in LLM paths ────────────────────────────────────────────────────
    llm_path         : Path = MODELS_DIR / "coregpt.pth"
    llm_pretrain_path: Path = MODELS_DIR / "coregpt_pretrain.pth"  # saved after pretrain, before fine-tuning
    llm_ckpt_path    : Path = MODELS_DIR / "coregpt_checkpoint.pth"
    tok_path         : Path = MODELS_DIR / "tokenizer.json"
    corpus_path      : Path = DATA_DIR   / "corpus.txt"
    llm_save_every   : int  = 500   # save checkpoint every N training steps

    # ── CoreGPT hyper-parameters ──────────────────────────────────────────────
    llm_tokenizer  : str   = "bpe"   # "bpe" (recommended) or "char"
    llm_bpe_vocab  : int   = 8192    # BPE vocabulary size
    llm_context    : int   = 512     # context window (tokens)
    llm_d_model    : int   = 768     # embedding dimension  (↑ from 512)
    llm_n_heads    : int   = 12      # attention heads      (↑ from 8)
    llm_n_layers   : int   = 12      # transformer blocks   (↑ from 8)  → ~88 M params
    llm_dropout    : float = 0.10
    llm_lr         : float = 3e-4
    llm_batch      : int   = 4       # micro-batch (smaller to fit 88M model in VRAM)
    llm_grad_accum : int   = 8       # grad accum → effective batch = 4×8 = 32
    llm_pretrain_steps : int = 20_000  # more data → more steps
    llm_finetune_steps : int = 2_000

    # ── Ollama backend settings ───────────────────────────────────────────────
    ollama_url   : str = "http://localhost:11434"
    ollama_model : str = "mistral"   # change to "llama3", "phi3", "gemma2", etc.

    # ── Command safety: any command containing these substrings is blocked ─────
    dangerous_cmds: tuple = (
        "rm -rf /",
        "dd if=",
        "mkfs",
        "> /dev/sd",
        ":(){ :|:& };:",
        "chmod -R 777 /",
        "wget -O- | bash",
        "curl | sh",
        "curl -s | sh",
    )


cfg = Config()


# ══════════════════════════════════════════════════════════════════════════════
# §2  DEEP-LEARNING MODEL  –  Convolutional Neural Network for MNIST
# ══════════════════════════════════════════════════════════════════════════════

class DigitCNN(nn.Module):
    """
    Compact CNN for 28×28 greyscale digit images (MNIST).

    Architecture:
        Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)
        Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
        Flatten → Linear(3136→256) → ReLU → Dropout(0.4) → Linear(256→10)

    Typically reaches ≥99 % test accuracy after 10 epochs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 14×14×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 14×14×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # →  7×7×64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# §3  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """Orchestrates training for every model in the platform."""

    # ── §3a  CNN on MNIST ─────────────────────────────────────────────────────

    @staticmethod
    def train_cnn() -> "DigitCNN":
        """Download MNIST, train the CNN, save the best checkpoint."""
        console.rule("[bold cyan]Training CNN on MNIST[/]")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean / std
        ])

        train_ds = torchvision.datasets.MNIST(
            str(DATA_DIR), train=True,  download=True, transform=transform
        )
        test_ds = torchvision.datasets.MNIST(
            str(DATA_DIR), train=False, download=True, transform=transform
        )

        train_dl = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
        )
        test_dl = DataLoader(
            test_ds, batch_size=512, shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
        )

        model     = DigitCNN().to(cfg.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        # Halve the LR every 5 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as prog:

            for epoch in range(1, cfg.cnn_epochs + 1):
                # ── Training pass ──────────────────────────────────────────────
                model.train()
                total_loss = n_correct = n_total = 0
                task_id = prog.add_task(
                    f"Epoch {epoch:02d}/{cfg.cnn_epochs}", total=len(train_dl)
                )

                for imgs, labels in train_dl:
                    imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * imgs.size(0)
                    n_correct  += (logits.argmax(1) == labels).sum().item()
                    n_total    += imgs.size(0)
                    prog.advance(task_id)

                train_loss = total_loss / n_total
                train_acc  = n_correct  / n_total

                # ── Validation pass ────────────────────────────────────────────
                model.eval()
                val_correct = val_total = 0
                with torch.no_grad():
                    for imgs, labels in test_dl:
                        imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                        val_correct += (model(imgs).argmax(1) == labels).sum().item()
                        val_total   += imgs.size(0)

                val_acc = val_correct / val_total
                console.print(
                    f"  Epoch {epoch:2d}  │  "
                    f"loss [yellow]{train_loss:.4f}[/]  │  "
                    f"train_acc [green]{train_acc:.4f}[/]  │  "
                    f"val_acc [bold green]{val_acc:.4f}[/]"
                )
                log.info(
                    "CNN epoch=%d train_loss=%.4f train_acc=%.4f val_acc=%.4f",
                    epoch, train_loss, train_acc, val_acc,
                )

                # Save best weights
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), cfg.cnn_path)

                scheduler.step()

        console.print(
            f"\n[bold green]Best validation accuracy: {best_val_acc:.4f}[/]\n"
            f"Model saved → [cyan]{cfg.cnn_path}[/]\n"
        )
        return model

    # ── §3b  Anomaly detector (Isolation Forest) ──────────────────────────────

    @staticmethod
    def train_anomaly_detector() -> IsolationForest:
        """
        Collect 30 s of live system snapshots and train an Isolation Forest
        to recognise unusual resource-usage patterns.
        """
        console.rule("[bold cyan]Training Anomaly Detector[/]")
        console.print("[yellow]Sampling live system metrics for 30 s …[/]")

        samples: List[List[float]] = []
        for _ in range(30):
            samples.append(SystemMonitor.snapshot_vector())
            time.sleep(1)

        normal = np.array(samples)

        # Augment with synthetic extreme-load anomalies so the model
        # has a non-trivial decision boundary even on a quiet machine.
        rng   = np.random.default_rng(42)
        spikes = normal.mean(0) + normal.std(0) * rng.standard_normal((15, normal.shape[1])) * 6
        X     = np.vstack([normal, spikes])

        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)

        iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
        iso.fit(X_scaled)

        joblib.dump(iso,    cfg.iso_path)
        joblib.dump(scaler, cfg.scaler_path)
        console.print(
            f"[green]Anomaly detector saved → {cfg.iso_path}[/]\n"
            f"[green]Scaler saved           → {cfg.scaler_path}[/]\n"
        )
        log.info("Anomaly detector trained on %d samples", len(X))
        return iso

    # ── §3c  Process classifier (Random Forest) ───────────────────────────────

    @staticmethod
    def train_process_classifier() -> RandomForestClassifier:
        """
        Build a Random Forest that flags processes as 'normal' vs 'suspicious'
        based on heuristic CPU / memory / thread-count features.
        Labels are intentionally conservative: only truly extreme combinations
        are marked suspicious so that false-positive rates stay low.
        """
        console.rule("[bold cyan]Training Process Classifier[/]")

        rows: List[List[float]] = []
        labels: List[int]       = []

        for proc in psutil.process_iter(
            ["cpu_percent", "memory_percent", "num_threads", "name"]
        ):
            try:
                info = proc.info
                cpu  = float(info.get("cpu_percent")   or 0.0)
                mem  = float(info.get("memory_percent") or 0.0)
                thr  = int(info.get("num_threads")      or 1)
                name_len = len(info.get("name") or "")
                # Heuristic: very high CPU *and* high memory simultaneously → suspicious
                label = 1 if (cpu > 80 and mem > 20) else 0
                rows.append([cpu, mem, thr, name_len])
                labels.append(label)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Ensure we have enough data even on idle machines
        rng = np.random.default_rng(0)
        if len(rows) < 30:
            rows   += [
                [float(rng.uniform(0, 5)),    float(rng.uniform(0, 2)),  1,  8] for _ in range(60)
            ]
            labels += [0] * 60
        rows   += [
            [float(rng.uniform(85, 100)), float(rng.uniform(25, 80)), 50, 6] for _ in range(15)
        ]
        labels += [1] * 15

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )
        clf.fit(X_tr, y_tr)

        acc = accuracy_score(y_te, clf.predict(X_te))
        console.print(f"[green]Process classifier accuracy: {acc:.4f}[/]")
        console.print(
            classification_report(
                y_te, clf.predict(X_te),
                target_names=["normal", "suspicious"],
                zero_division=0,
            )
        )

        joblib.dump(clf, cfg.proc_clf_path)
        console.print(f"[green]Process classifier saved → {cfg.proc_clf_path}[/]\n")
        log.info("Process classifier trained, accuracy=%.4f", acc)
        return clf


# ══════════════════════════════════════════════════════════════════════════════
# §4  SYSTEM MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class SystemMonitor:
    """Real-time system metrics, anomaly scoring, and Rich-formatted reports."""

    # Feature order must remain stable – anomaly detector depends on it.
    FEATURE_NAMES = (
        "cpu_%", "mem_%", "disk_%", "n_pids",
        "net_sent_MB", "net_recv_MB",
    )

    # ── §4a  Raw feature vector ───────────────────────────────────────────────

    @staticmethod
    def snapshot_vector() -> List[float]:
        """Return one fixed-length observation of the current system state."""
        cpu   = psutil.cpu_percent(interval=0.1)
        mem   = psutil.virtual_memory()
        disk  = psutil.disk_usage("/")
        net   = psutil.net_io_counters()
        return [
            cpu,
            mem.percent,
            disk.percent,
            float(len(psutil.pids())),
            net.bytes_sent / 1e6,
            net.bytes_recv / 1e6,
        ]

    # ── §4b  Anomaly detection ────────────────────────────────────────────────

    @staticmethod
    def is_anomalous(vector: Optional[List[float]] = None) -> Tuple[bool, float]:
        """
        Returns (is_anomalous, anomaly_score).
        Falls back to (False, 0.0) when the detector has not been trained yet.
        """
        if not cfg.iso_path.exists() or not cfg.scaler_path.exists():
            return False, 0.0
        iso: IsolationForest = joblib.load(cfg.iso_path)
        scaler: StandardScaler = joblib.load(cfg.scaler_path)
        v = vector or SystemMonitor.snapshot_vector()
        X = scaler.transform([v])
        score = float(iso.decision_function(X)[0])
        anomalous = iso.predict(X)[0] == -1   # -1 = anomaly in sklearn
        return anomalous, score

    # ── §4c  Top processes ────────────────────────────────────────────────────

    @staticmethod
    def top_processes(n: int = 12) -> List[Dict]:
        """Return the top-N processes ordered by CPU usage."""
        procs: List[Dict] = []
        for p in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "status", "username"]
        ):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(procs, key=lambda x: x.get("cpu_percent") or 0, reverse=True)[:n]

    # ── §4d  Rich system summary panel ───────────────────────────────────────

    @staticmethod
    def system_summary() -> Panel:
        cpu    = psutil.cpu_percent(interval=0.1)
        mem    = psutil.virtual_memory()
        disk   = psutil.disk_usage("/")
        net    = psutil.net_io_counters()
        boot   = datetime.fromtimestamp(psutil.boot_time())
        uptime_s = (datetime.now() - boot).total_seconds()
        uptime = f"{int(uptime_s // 3600)}h {int((uptime_s % 3600) // 60)}m"

        try:
            load1, load5, load15 = os.getloadavg()
            load_str = f"{load1:.2f} / {load5:.2f} / {load15:.2f}"
        except AttributeError:
            load_str = "n/a"

        anomalous, score = SystemMonitor.is_anomalous()
        anom_colour = "bold red" if anomalous else "green"
        anom_label  = "ANOMALY DETECTED" if anomalous else "Normal"
        anom_str    = f"[{anom_colour}]{anom_label}[/] (score={score:.3f})"

        lines = [
            f"[bold cyan]Host:[/]       {os.uname().nodename}   Uptime: {uptime}",
            f"[bold cyan]OS:[/]         {os.uname().sysname} {os.uname().release}",
            f"[bold cyan]CPU:[/]        {cpu:5.1f}%   Cores: {psutil.cpu_count()}   Load: {load_str}",
            f"[bold cyan]Memory:[/]     {mem.percent:5.1f}%   "
            f"Used: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB",
            f"[bold cyan]Disk (/):[/]   {disk.percent:5.1f}%   "
            f"Used: {disk.used/1e9:.1f} GB / {disk.total/1e9:.1f} GB",
            f"[bold cyan]Network:[/]    ↑ {net.bytes_sent/1e6:.1f} MB   "
            f"↓ {net.bytes_recv/1e6:.1f} MB",
            f"[bold cyan]Anomaly:[/]    {anom_str}",
        ]
        return Panel(
            "\n".join(lines),
            title=f"[bold]System Status – {datetime.now().strftime('%H:%M:%S')}[/]",
            border_style="cyan",
        )

    # ── §4e  Rich process table ───────────────────────────────────────────────

    @staticmethod
    def process_table(n: int = 14) -> Table:
        """
        Process table annotated with the AI process-classifier output when
        the model is available.
        """
        clf_loaded = cfg.proc_clf_path.exists()
        clf = joblib.load(cfg.proc_clf_path) if clf_loaded else None

        tbl = Table(
            title="Top Processes (by CPU)", box=box.SIMPLE_HEAVY,
            header_style="bold magenta",
        )
        for col in ("PID", "Name", "CPU %", "MEM %", "Threads", "Status", "User", "AI Flag"):
            tbl.add_column(col)

        for p in SystemMonitor.top_processes(n):
            cpu      = float(p.get("cpu_percent")    or 0.0)
            mem      = float(p.get("memory_percent") or 0.0)
            name     = (p.get("name") or "")[:24]
            thr_cnt  = 1

            if clf_loaded:
                feat  = np.array([[cpu, mem, thr_cnt, len(name)]], dtype=np.float32)
                label = clf.predict(feat)[0]
                flag  = "[bold red]SUSPICIOUS[/]" if label == 1 else "[green]normal[/]"
            else:
                flag = "[dim]–[/]"

            tbl.add_row(
                str(p.get("pid",  "")),
                name,
                f"{cpu:5.1f}",
                f"{mem:5.2f}",
                str(thr_cnt),
                p.get("status", ""),
                (p.get("username") or "")[:14],
                flag,
            )
        return tbl

    # ── §4f  Live dashboard ───────────────────────────────────────────────────

    @staticmethod
    def live_dashboard(duration: int = 3600) -> None:
        """Refresh-every-second live TUI dashboard. Press Ctrl-C to quit."""
        console.print("[dim]Press Ctrl-C to exit the dashboard.[/]\n")
        layout = Layout()
        layout.split_column(
            Layout(name="top",    size=11),
            Layout(name="bottom"),
        )
        start = time.time()
        try:
            with Live(layout, console=console, refresh_per_second=1, screen=True):
                while time.time() - start < duration:
                    layout["top"].update(SystemMonitor.system_summary())
                    layout["bottom"].update(SystemMonitor.process_table())
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        console.print("[green]Dashboard closed.[/]")


# ══════════════════════════════════════════════════════════════════════════════
# §5  SYSTEM CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class SystemController:
    """
    Safe, fully logged system-control primitives.

    Destructive or state-changing operations always ask for confirmation
    unless `force=True` is explicitly passed (used internally by the agent
    only for read-only commands).
    """

    # ── §5a  Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_dangerous(cmd: str) -> bool:
        return any(danger in cmd for danger in cfg.dangerous_cmds)

    @staticmethod
    def _run(cmd: str) -> Tuple[int, str, str]:
        """Execute a shell command; return (returncode, stdout, stderr)."""
        log.info("EXEC: %s", cmd)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout or "", result.stderr or ""

    # ── §5b  Shell execution ──────────────────────────────────────────────────

    @staticmethod
    def run_command(cmd: str, force: bool = False) -> str:
        """
        Execute `cmd` with safety checks.
        Set force=True to skip the interactive confirmation prompt
        (only appropriate for read-only commands).
        """
        if SystemController._is_dangerous(cmd):
            msg = f"BLOCKED – matched a dangerous pattern: {cmd}"
            console.print(f"[bold red]{msg}[/]")
            log.warning(msg)
            return "BLOCKED"

        if force:
            rc, out, err = SystemController._run(cmd)
            return out or err or f"(exit {rc})"

        if Confirm.ask(f"[yellow]Execute:[/] {cmd} ?", default=False):
            rc, out, err = SystemController._run(cmd)
            output = out or err or f"(exit {rc})"
            console.print(output)
            return output

        return "Cancelled"

    # ── §5c  Process management ───────────────────────────────────────────────

    @staticmethod
    def kill_process(pid: int) -> None:
        """SIGTERM a process by PID after user confirmation."""
        try:
            proc = psutil.Process(pid)
            console.print(
                f"[yellow]About to SIGTERM PID {pid} ({proc.name()}, "
                f"user={proc.username()})[/]"
            )
            if Confirm.ask("Confirm?", default=False):
                proc.terminate()
                console.print(f"[green]PID {pid} terminated.[/]")
                log.info("Terminated PID %d (%s)", pid, proc.name())
        except psutil.NoSuchProcess:
            console.print(f"[red]PID {pid} does not exist.[/]")
        except psutil.AccessDenied:
            console.print(f"[red]Permission denied for PID {pid}.[/]")

    @staticmethod
    def launch(cmd: str) -> None:
        """Spawn a background process detached from this terminal."""
        log.info("LAUNCH: %s", cmd)
        subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        console.print(f"[green]Launched:[/] {cmd}")

    # ── §5d  File system ──────────────────────────────────────────────────────

    @staticmethod
    def list_dir(path: str = ".") -> Table:
        p = Path(path)
        tbl = Table(title=f"Directory listing: {p.resolve()}", box=box.SIMPLE)
        for col in ("Type", "Name", "Size (bytes)", "Modified"):
            tbl.add_column(col)

        if not p.exists():
            console.print(f"[red]Path not found: {path}[/]")
            return tbl

        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        for entry in entries[:60]:
            try:
                st   = entry.stat()
                size = str(st.st_size) if entry.is_file() else ""
                mod  = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                kind = "[blue]DIR [/]" if entry.is_dir() else "file"
                tbl.add_row(kind, entry.name, size, mod)
            except PermissionError:
                tbl.add_row("?", entry.name, "", "")
        return tbl

    @staticmethod
    def disk_usage_tree(path: str = "/") -> Table:
        """Top-20 directories by size under `path`."""
        _, out, _ = SystemController._run(
            f"du -h --max-depth=1 {path} 2>/dev/null | sort -rh | head -20"
        )
        tbl = Table(title=f"Disk usage: {path}", box=box.SIMPLE)
        tbl.add_column("Size", style="yellow")
        tbl.add_column("Path")
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                tbl.add_row(parts[0], parts[1])
        return tbl

    # ── §5e  Service management (systemd) ─────────────────────────────────────

    @staticmethod
    def list_services() -> Table:
        _, out, _ = SystemController._run(
            "systemctl list-units --type=service --no-pager --no-legend 2>/dev/null | head -30"
        )
        tbl = Table(title="Systemd Services", box=box.SIMPLE)
        for col in ("Unit", "Load", "Active", "Sub", "Description"):
            tbl.add_column(col)
        for line in out.strip().splitlines():
            parts = line.split(None, 4)
            while len(parts) < 5:
                parts.append("")
            tbl.add_row(*parts[:5])
        return tbl

    @staticmethod
    def service_action(name: str, action: str) -> None:
        if action not in ("start", "stop", "restart", "status"):
            console.print("[red]action must be start | stop | restart | status[/]")
            return
        if action == "status":
            SystemController.run_command(f"systemctl {action} {name}", force=True)
        else:
            SystemController.run_command(f"systemctl {action} {name}", force=False)

    # ── §5f  Network helpers ──────────────────────────────────────────────────

    @staticmethod
    def network_info() -> None:
        _, out, _ = SystemController._run("ip addr show")
        console.print(out)
        _, out, _ = SystemController._run("ss -tulnp 2>/dev/null | head -25")
        console.print(out)

    # ── §5g  Package management (read-only queries only) ─────────────────────

    @staticmethod
    def installed_packages(limit: int = 20) -> None:
        """List recently installed packages (dnf / apt / pacman as available)."""
        for cmd in (
            f"dnf list installed 2>/dev/null | tail -n +2 | head -{limit}",
            f"dpkg -l 2>/dev/null | grep '^ii' | head -{limit}",
            f"pacman -Q 2>/dev/null | head -{limit}",
        ):
            rc, out, _ = SystemController._run(cmd)
            if rc == 0 and out.strip():
                console.print(out)
                return
        console.print("[yellow]No supported package manager found.[/]")


# ══════════════════════════════════════════════════════════════════════════════
# §6  AI AGENT SHELL  –  natural-language command router
# ══════════════════════════════════════════════════════════════════════════════

class AIAgent:
    """
    Intent-recognising agent that maps free-text instructions to concrete
    system actions. No external LLM or network connection required.

    Design:
      - Keyword matching against a static intent vocabulary
      - Lightweight entity extraction (PID, path, service name) via regex
      - Each intent maps to one or more SystemController / SystemMonitor calls
      - Unknown input falls back to confirmed shell execution
    """

    # Vocabulary: intent → trigger keywords (any match wins, longest first)
    INTENTS: Dict[str, List[str]] = {
        "monitor"   : ["live dashboard", "monitor", "dashboard", "watch stats", "live view"],
        "top_procs" : ["top processes", "show processes", "running processes",
                       "cpu usage", "process list", "what's running", "htop"],
        "kill"      : ["kill process", "kill pid", "terminate process",
                       "stop process", "end process"],
        "launch"    : ["launch", "open app", "start program", "run program", "execute"],
        "disk"      : ["disk usage", "disk space", "storage", "how much space", "du "],
        "ls"        : ["list files", "list directory", "show files", "show directory",
                       "ls ", "what's in"],
        "services"  : ["list services", "systemd services", "service status",
                       "daemon", "systemctl"],
        "train"     : ["train", "retrain", "fit model", "train ai", "learn"],
        "predict"   : ["predict", "classify image", "recognize digit",
                       "what digit", "identify number"],
        "network"   : ["network", "ip address", "open ports", "connections",
                       "ifconfig", "ip addr", "show ports"],
        "sysinfo"   : ["system info", "uname", "hostname", "uptime", "os version",
                       "cpu info"],
        "packages"  : ["installed packages", "list packages", "what's installed",
                       "show packages"],
        "anomaly"   : ["anomaly", "check anomaly", "is anything suspicious",
                       "security check"],
        "help"      : ["help", "what can you do", "commands", "?"],
        "quit"      : ["quit", "exit", "bye", "goodbye", ":q"],
    }

    def __init__(self) -> None:
        self.cnn    : Optional[DigitCNN]    = None
        self.llm    : Optional[LLMBackend]  = None
        self.history: List[str]             = []
        self._try_load_cnn()
        self._try_load_llm()

    def _try_load_llm(self) -> None:
        """Load the built-in CoreGPT silently; fail gracefully if not trained yet."""
        try:
            self.llm = LLMBackend.load()
        except FileNotFoundError:
            self.llm = None

    def _try_load_cnn(self) -> None:
        if cfg.cnn_path.exists():
            self.cnn = DigitCNN()
            self.cnn.load_state_dict(
                torch.load(cfg.cnn_path, map_location=cfg.device)
            )
            self.cnn.eval()
            self.cnn.to(cfg.device)

    # ── Intent + entity detection ─────────────────────────────────────────────

    def detect_intent(self, text: str) -> Tuple[str, Dict[str, Any]]:
        lower = text.lower()
        # Check longer phrases first to avoid partial matches
        for intent, keywords in self.INTENTS.items():
            for kw in sorted(keywords, key=len, reverse=True):
                if kw in lower:
                    return intent, self._extract_entities(text)
        return "shell", {"cmd": text}

    @staticmethod
    def _extract_entities(text: str) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        # PID: 3-to-7-digit standalone number
        m = re.search(r'(?<!\d)(\d{3,7})(?!\d)', text)
        if m:
            entities["pid"] = int(m.group(1))
        # Absolute path
        m = re.search(r'(/[\w/.\-]+)', text)
        if m:
            entities["path"] = m.group(1)
        # Service name (common services)
        m = re.search(
            r'\b(nginx|apache2|httpd|sshd|ssh|docker|mysql|mariadb|'
            r'postgresql|redis|mongod|cron|NetworkManager|firewalld)\b',
            text, re.I,
        )
        if m:
            entities["service"] = m.group(1).lower()
        # Lifecycle action
        for action in ("start", "stop", "restart", "status", "enable", "disable"):
            if re.search(rf'\b{action}\b', text, re.I):
                entities["action"] = action
                break
        return entities

    # ── Response dispatch ─────────────────────────────────────────────────────

    def respond(self, text: str) -> None:
        self.history.append(text)
        intent, entities = self.detect_intent(text)

        handlers: Dict[str, Any] = {
            "monitor":   self._handle_monitor,
            "top_procs": self._handle_top_procs,
            "kill":      self._handle_kill,
            "launch":    self._handle_launch,
            "disk":      self._handle_disk,
            "ls":        self._handle_ls,
            "services":  self._handle_services,
            "train":     self._handle_train,
            "predict":   self._handle_predict,
            "network":   self._handle_network,
            "sysinfo":   self._handle_sysinfo,
            "packages":  self._handle_packages,
            "anomaly":   self._handle_anomaly,
            "help":      self._print_help,
            "quit":      self._handle_quit,
            "shell":     self._handle_shell,
        }
        handler = handlers.get(intent, self._handle_shell)
        handler(text=text, entities=entities)

    # ── Per-intent handlers ───────────────────────────────────────────────────

    def _handle_monitor(self, **_: Any) -> None:
        SystemMonitor.live_dashboard(duration=300)

    def _handle_top_procs(self, **_: Any) -> None:
        console.print(SystemMonitor.process_table())

    def _handle_kill(self, entities: Dict, text: str, **_: Any) -> None:
        pid = entities.get("pid")
        if pid is None:
            try:
                pid = int(Prompt.ask("PID to terminate"))
            except ValueError:
                console.print("[red]Invalid PID.[/]"); return
        SystemController.kill_process(pid)

    def _handle_launch(self, text: str, **_: Any) -> None:
        cmd = re.sub(r'^(launch|open|start|run|execute)\s+', '', text, flags=re.I).strip()
        if not cmd:
            cmd = Prompt.ask("Command to launch")
        SystemController.launch(cmd)

    def _handle_disk(self, entities: Dict, **_: Any) -> None:
        path = entities.get("path", "/")
        console.print(SystemController.disk_usage_tree(path))

    def _handle_ls(self, entities: Dict, **_: Any) -> None:
        path = entities.get("path", ".")
        console.print(SystemController.list_dir(path))

    def _handle_services(self, entities: Dict, **_: Any) -> None:
        console.print(SystemController.list_services())
        svc    = entities.get("service")
        action = entities.get("action")
        if svc and action:
            SystemController.service_action(svc, action)

    def _handle_train(self, **_: Any) -> None:
        ModelTrainer.train_cnn()
        ModelTrainer.train_anomaly_detector()
        ModelTrainer.train_process_classifier()
        LLMTrainer.train()
        self._try_load_cnn()
        self._try_load_llm()

    def _handle_predict(self, **_: Any) -> None:
        self._predict_interactive()

    def _handle_network(self, **_: Any) -> None:
        SystemController.network_info()

    def _handle_sysinfo(self, **_: Any) -> None:
        _, out, _ = SystemController._run("uname -a && uptime && hostname -I")
        console.print(out)

    def _handle_packages(self, **_: Any) -> None:
        SystemController.installed_packages()

    def _handle_anomaly(self, **_: Any) -> None:
        v = SystemMonitor.snapshot_vector()
        anomalous, score = SystemMonitor.is_anomalous(v)
        colour = "bold red" if anomalous else "green"
        label  = "ANOMALY DETECTED" if anomalous else "All clear"
        console.print(
            Panel(
                f"[{colour}]{label}[/]\n"
                f"Score: {score:.4f}\n"
                f"Features: { {k: f'{v:.2f}' for k, v in zip(SystemMonitor.FEATURE_NAMES, v)} }",
                title="Anomaly Check",
                border_style=colour.replace("bold ", ""),
            )
        )

    def _handle_shell(self, text: str, **_: Any) -> None:
        """
        Fallback handler.  If the built-in LLM is loaded, route the input
        through it so the agent can answer any free-form question.
        Raw shell commands are still forwarded to the system (with confirmation)
        when the text starts with a known shell prefix or the LLM is not ready.
        """
        SHELL_PREFIXES = (
            "sudo ", "apt ", "dnf ", "pacman ", "pip ", "git ",
            "cat ", "echo ", "cd ", "cp ", "mv ", "mkdir ", "touch ",
            "chmod ", "chown ", "grep ", "find ", "awk ", "sed ", "curl ",
            "wget ", "tar ", "zip ", "unzip ", "/",
        )
        looks_like_shell = any(text.lstrip().startswith(p) for p in SHELL_PREFIXES)

        if self.llm is not None and not looks_like_shell:
            # Let the LLM answer
            console.print("[dim]Thinking…[/]")
            try:
                answer = self.llm.chat(text)
                console.print(
                    Panel(answer, title="[bold cyan]LinuxAI[/]", border_style="cyan")
                )
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/]")
        else:
            SystemController.run_command(text)

    @staticmethod
    def _handle_quit(**_: Any) -> None:
        console.print("[green]Goodbye![/]")
        sys.exit(0)

    # ── MNIST prediction ──────────────────────────────────────────────────────

    def _predict_interactive(self) -> None:
        if self.cnn is None:
            console.print("[red]CNN not found. Run the agent command 'train' first.[/]")
            return
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_ds = torchvision.datasets.MNIST(
            str(DATA_DIR), train=False, download=True, transform=transform
        )
        try:
            idx = int(Prompt.ask("MNIST test-set index (0 – 9999)", default="42"))
            idx = max(0, min(idx, len(test_ds) - 1))
        except ValueError:
            idx = 42

        img, true_label = test_ds[idx]
        with torch.no_grad():
            pred = self.cnn(img.unsqueeze(0).to(cfg.device)).argmax(1).item()

        colour = "green" if pred == true_label else "red"
        console.print(
            Panel(
                f"True label: [bold]{true_label}[/]\n"
                f"Predicted:  [bold {colour}]{pred}[/]\n"
                f"Result:     [bold {'green' if pred == true_label else 'red'}]"
                f"{'CORRECT ✓' if pred == true_label else 'WRONG ✗'}[/]",
                title=f"Digit Prediction — index {idx}",
                border_style=colour,
            )
        )

    # ── Help table ────────────────────────────────────────────────────────────

    @staticmethod
    def _print_help(**_: Any) -> None:
        tbl = Table(
            title="AI Agent — Available Commands",
            box=box.ROUNDED, header_style="bold cyan",
        )
        tbl.add_column("Intent", style="cyan")
        tbl.add_column("Example phrase(s)")
        rows = [
            ("monitor",   "monitor / live dashboard / watch stats"),
            ("top_procs", "show processes / what's using CPU"),
            ("kill",      "kill process 1234 / terminate pid 5678"),
            ("launch",    "launch firefox / start htop"),
            ("disk",      "disk usage / how much space on /home"),
            ("ls",        "list /etc / show files in /tmp"),
            ("services",  "list services / restart nginx / status sshd"),
            ("train",     "train / retrain AI models"),
            ("predict",   "predict / classify image 42"),
            ("network",   "show network / ip address / open ports"),
            ("sysinfo",   "system info / uptime / hostname"),
            ("packages",  "installed packages / list packages"),
            ("anomaly",   "check anomaly / is anything suspicious"),
            ("help",      "help / ?"),
            ("quit",      "quit / exit / bye"),
            ("<other>",   "Any text → confirmed shell execution"),
        ]
        for intent, ex in rows:
            tbl.add_row(intent, ex)
        console.print(tbl)

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        llm_status = (
            "[green]Built-in LLM ready[/]" if self.llm is not None
            else "[yellow]LLM not trained – run 'train' or 'python coreai.py train-llm'[/]"
        )
        console.print(
            Panel(
                "[bold cyan]LinuxAI Agent Shell[/]\n"
                f"{llm_status}\n"
                "Type [green]help[/] for a command overview, "
                "[red]quit[/] to exit.\n"
                "[dim]Any free-form question → answered by the built-in LLM.[/]\n"
                "[dim]System commands (ls, cat …) → forwarded to the shell.[/]",
                border_style="cyan",
            )
        )
        while True:
            try:
                text = Prompt.ask("[bold green]ai >[/]").strip()
                if text:
                    self.respond(text)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[green]Goodbye![/]")
                break


# ══════════════════════════════════════════════════════════════════════════════
# §7  BUILT-IN LLM  –  GPT-style transformer trained entirely from scratch
# ══════════════════════════════════════════════════════════════════════════════
#
#  Capabilities: language modelling, instruction following, Q&A, conversation.
#  Architecture: CharTokenizer → CoreGPT (causal transformer) → autoregressive
#  Training:     Phase-1 pretraining on a downloaded text corpus
#                Phase-2 instruction fine-tuning on Q&A pairs
#  Notes:        With default hyper-params (~3 M params, 256-char context) the
#                model trains on CPU in ≈ 15-30 min.  Quality grows with time –
#                raise llm_pretrain_steps / llm_finetune_steps for richer output.
# ══════════════════════════════════════════════════════════════════════════════


# ── §7a  Character-level tokenizer ────────────────────────────────────────────

class CharTokenizer:
    """
    Maps every unique character in the training corpus to an integer id.
    Special tokens:
        <PAD>  id=0   padding / unknown character
        <BOS>  id=1   beginning of sequence
        <EOS>  id=2   end of sequence
    """

    PAD, BOS, EOS = 0, 1, 2
    SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]

    def __init__(self) -> None:
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ── build from raw text ───────────────────────────────────────────────────

    def build(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        # special tokens occupy ids 0-2; normal chars start at 3
        vocab = self.SPECIAL + chars
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        return self

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.PAD) for c in text]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        parts = []
        for i in ids:
            ch = self.itos.get(i, "")
            if skip_special and ch in self.SPECIAL:
                continue
            parts.append(ch)
        return "".join(parts)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        tok = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ── §7a'  Byte-level BPE tokenizer ────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-level Byte Pair Encoding (BPE) tokenizer built entirely from scratch.

    - Base vocabulary: 256 raw bytes + 3 special tokens (PAD/BOS/EOS)
    - Learns merge rules from a text corpus subsample (300 K chars for speed)
    - Encodes any UTF-8 text without unknown-token issues
    - ~4× fewer tokens per character than char-level → much longer effective context
    - Fast encoding via greedy longest-match using a reverse-vocab lookup table

    Default vocab size: 4096  (256 bytes + 3 special + 3837 learned merges)
    """

    PAD, BOS, EOS = 0, 1, 2
    SPECIAL       = ["<PAD>", "<BOS>", "<EOS>"]
    NUM_SPECIAL   = 3
    _BASE         = 256 + NUM_SPECIAL   # first byte id after special tokens

    def __init__(self, vocab_size: int = 4096) -> None:
        self.target_vocab_size : int                          = vocab_size
        self.merges            : List[Tuple[int, int]]        = []
        self.merge_map         : Dict[Tuple[int, int], int]   = {}
        self.vocab             : Dict[int, bytes]             = {}
        self.vocab_size        : int                          = 0
        self._rev              : Dict[bytes, int]             = {}   # bytes → id cache

    # ── Build from corpus ─────────────────────────────────────────────────────

    def build(self, text: str) -> "BPETokenizer":
        """Learn BPE merge rules from a text corpus (uses first 300 K chars)."""
        ns   = self.NUM_SPECIAL
        base = self._BASE
        num_merges = max(0, self.target_vocab_size - base)

        # Initialise base vocabulary
        for i, sp in enumerate(self.SPECIAL):
            self.vocab[i] = sp.encode("utf-8")
        for b in range(256):
            self.vocab[ns + b] = bytes([b])

        # Subsample for speed  (merge rules generalise from a 300 K sample)
        sample = text[:300_000]
        ids: List[int] = [ns + b for b in sample.encode("utf-8")]

        console.print(
            f"[cyan]BPE:[/] learning {num_merges} merges "
            f"on {len(ids):,} tokens …"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]BPE merges[/] {task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("", total=num_merges)

            for merge_i in range(num_merges):
                # Count adjacent pair frequencies
                counts: Dict[Tuple[int, int], int] = {}
                for a, b in zip(ids, ids[1:]):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
                if not counts:
                    break

                best   = max(counts, key=counts.get)
                new_id = base + merge_i
                self.merges.append(best)
                self.merge_map[best] = new_id
                self.vocab[new_id]   = self.vocab[best[0]] + self.vocab[best[1]]

                # Apply merge to working ids
                merged: List[int] = []
                i = 0
                while i < len(ids):
                    if (i + 1 < len(ids)
                            and ids[i] == best[0]
                            and ids[i + 1] == best[1]):
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                ids = merged

                prog.advance(task, 1)
                if (merge_i + 1) % 500 == 0:
                    try:
                        tok_str = self.vocab[new_id].decode("utf-8")
                    except Exception:
                        tok_str = repr(self.vocab[new_id])
                    prog.update(
                        task,
                        description=(
                            f"merge {merge_i+1}/{num_merges}  "
                            f"'{tok_str}' (freq={counts[best]})"
                        ),
                    )

        self.vocab_size = len(self.vocab)
        self._build_rev()
        return self

    def _build_rev(self) -> None:
        """Build bytes→id lookup table for fast greedy encoding."""
        # Longer tokens take priority (greedy longest-match)
        self._rev = {
            v: k
            for k, v in sorted(self.vocab.items(), key=lambda x: len(x[1]))
            if k >= self.NUM_SPECIAL
        }

    # ── Encode / decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode UTF-8 text → list of token ids (greedy longest-match)."""
        if not self._rev:
            self._build_rev()
        data = text.encode("utf-8")
        n    = len(data)
        ids  : List[int] = []
        i    = 0
        # Pre-compute max token byte length for the search window
        max_len = max((len(v) for v in self._rev), default=1)
        while i < n:
            end = min(i + max_len, n)
            matched = False
            while end > i:
                chunk = bytes(data[i:end])
                if chunk in self._rev:
                    ids.append(self._rev[chunk])
                    i += len(chunk)
                    matched = True
                    break
                end -= 1
            if not matched:
                ids.append(self.NUM_SPECIAL + data[i])
                i += 1
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token ids → UTF-8 string."""
        pieces: List[bytes] = []
        for tok_id in ids:
            if tok_id < self.NUM_SPECIAL:
                if not skip_special:
                    pieces.append(self.vocab[tok_id])
            elif tok_id in self.vocab:
                pieces.append(self.vocab[tok_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {
            "type"             : "bpe",
            "target_vocab_size": self.target_vocab_size,
            "vocab_size"       : self.vocab_size,
            "merges"           : list(self.merges),
            "vocab"            : {str(k): list(v) for k, v in self.vocab.items()},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok                  = cls(vocab_size=data["target_vocab_size"])
        tok.vocab_size       = data["vocab_size"]
        tok.merges           = [tuple(m) for m in data["merges"]]  # type: ignore[assignment]
        base                 = cls._BASE
        tok.merge_map        = {
            tuple(m): base + i                                      # type: ignore[misc]
            for i, m in enumerate(tok.merges)
        }
        tok.vocab            = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok._build_rev()
        return tok


# ── §7b  Normalisation: RMSNorm (faster than LayerNorm, used by LLaMA) ────────

class RMSNorm(nn.Module):
    """
    Root-Mean-Square layer normalisation (Zhang & Sennrich, 2019).
    Omits the mean-centering step of LayerNorm, which is faster and equally
    effective for transformer language models.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ── §7c  Multi-head causal self-attention with RoPE ───────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

    RoPE (Su et al., 2021) encodes position by rotating the Q/K vectors rather
    than adding a fixed position embedding.  Benefits:
      - Relative-position awareness: the dot product naturally captures distance
      - Length generalisation: works for sequences longer than those seen in training
      - No extra parameters vs learned position embeddings

    Uses F.scaled_dot_product_attention → FlashAttention when available.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.drop_p   = dropout

        # Fused Q/K/V projection (no bias, as in LLaMA)
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,     bias=False)
        self.drop = nn.Dropout(dropout)

    # ── RoPE helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embeddings to Q and K tensors.

        q, k: (B, H, T, head_dim)
        """
        T        = q.shape[2]
        head_dim = q.shape[-1]
        half     = head_dim // 2
        device   = q.device

        # Frequency bands  θ_i = 1 / 10000^(2i/d)
        theta  = 1.0 / (10_000.0 ** (
            torch.arange(0, half, device=device).float() / half
        ))
        # Position × frequency grid  →  (T, half)
        freqs  = torch.outer(torch.arange(T, device=device).float(), theta)
        # Duplicate for the two halves  →  (T, head_dim)
        freqs  = torch.cat([freqs, freqs], dim=-1)

        cos_f  = freqs.cos()[None, None, :, :]   # (1, 1, T, head_dim)
        sin_f  = freqs.sin()[None, None, :, :]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = q * cos_f + rotate_half(q) * sin_f
        k_rot = k * cos_f + rotate_half(k) * sin_f
        return q_rot, k_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self._apply_rope(q, k)

        attn_drop = self.drop_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_drop)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


# ── §7d  Position-wise feed-forward: SwiGLU (LLaMA-style gated FFN) ──────────

class SwiGLUFFN(nn.Module):
    """
    Gated FFN with SiLU (Swish) activation — used by LLaMA, Mistral, and Gemma.

    SwiGLU(x) = (W_gate · x  ⊙  silu(W_up · x)) · W_down
                ↑ gate branch   ↑ value branch

    The gating mechanism gives the model finer control over information flow.
    Uses an 8/3 × d_model hidden dimension (rounded to the nearest 64) to keep
    parameter count comparable to a 4× GELU FFN.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        # 8/3 expansion (from LLaMA paper), rounded for hardware alignment
        hidden = int(d_model * 8 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ── §7e  Transformer block ────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block using modern components (LLaMA-style):
        x → RMSNorm → CausalSelfAttention (RoPE) → residual
          → RMSNorm → SwiGLUFFN                  → residual

    RMSNorm replaces LayerNorm (faster, same quality).
    SwiGLU replaces GELU-FFN (better empirical performance on language).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── §7f  CoreGPT – the full language model (LLaMA-style) ─────────────────────

class CoreGPT(nn.Module):
    """
    GPT-style causal language model with modern LLaMA-inspired architecture.

    Architecture improvements over the original GPT-2 design:
      - RoPE positional encoding (no learned pos_emb → saves params, better generalisation)
      - RMSNorm instead of LayerNorm (faster, same quality)
      - SwiGLU FFN instead of GELU-FFN (empirically better on language)
      - Weight tying between token embedding and output head

    Default (~25 M parameters):
        vocab_size=4096 (BPE), context_len=512, d_model=512, n_heads=8, n_layers=8
    """

    def __init__(
        self,
        vocab_size : int,
        context_len: int   = 512,
        d_model    : int   = 512,
        n_heads    : int   = 8,
        n_layers   : int   = 8,
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()
        self.context_len = context_len

        # ── Embedding (no position embedding – RoPE handles it in attention) ──
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # ── Transformer body ──────────────────────────────────────────────────
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.norm_f  = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: output head shares weights with the token embedding
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        idx    : torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx     : (B, T) integer token ids
        targets : (B, T) shifted ids for next-token prediction loss

        Returns (logits, loss).  loss is None when targets is None.
        """
        _, T = idx.shape
        assert T <= self.context_len, (
            f"Input length {T} exceeds context_len {self.context_len}"
        )

        x      = self.emb_drop(self.tok_emb(idx))   # (B, T, d_model)
        x      = self.blocks(x)
        x      = self.norm_f(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,   # PAD token id = 0 for both BPE and CharTokenizer
            )

        return logits, loss

    # ── Autoregressive generation ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx               : torch.Tensor,
        max_new_tokens    : int   = 300,
        temperature       : float = 0.8,
        top_k             : int   = 50,
        top_p             : float = 0.9,
        repetition_penalty: float = 1.2,
        stop_token        : Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample tokens one at a time.

        Parameters
        ----------
        idx                : (1, T) seed token ids
        max_new_tokens     : maximum new tokens to generate
        temperature        : >1 more random, <1 more focused  (0.7–0.9 recommended)
        top_k              : keep only top-k candidates (0 = disabled)
        top_p              : nucleus sampling — keep smallest set with cumprob ≥ top_p
        repetition_penalty : >1 penalises repeating tokens (1.2 is a good default)
        stop_token         : stop early when this token id is sampled

        Returns the full sequence (seed + generated).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond          = idx[:, -self.context_len:]
            logits, _         = self(idx_cond)
            logits            = logits[:, -1, :]          # (1, vocab_size)

            # ── Repetition penalty ────────────────────────────────────────────
            if repetition_penalty != 1.0:
                seen = set(idx[0].tolist())
                for tok_id in seen:
                    if logits[0, tok_id] > 0:
                        logits[0, tok_id] /= repetition_penalty
                    else:
                        logits[0, tok_id] *= repetition_penalty

            logits = logits / temperature

            # ── Top-k filtering ───────────────────────────────────────────────
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            # ── Top-p (nucleus) filtering ─────────────────────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens beyond the nucleus
                to_remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

            if stop_token is not None and next_tok.item() == stop_token:
                break

        return idx

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── §7f  LLMTrainer – download corpus, pretrain, fine-tune ───────────────────

class LLMTrainer:
    """
    Two-phase training for CoreGPT:
      Phase 1 – character-level language-model pretraining on a mixed text corpus
                 (downloads public-domain books from Project Gutenberg automatically)
      Phase 2 – instruction fine-tuning on a curated Q&A dataset formatted as:
                 ### Human: <question>\\n### AI: <answer>\\n\\n
    """

    # ── Corpus sources (Project Gutenberg, all public domain) ─────────────────
    GUTENBERG = [
        # Fiction / literature
        ("Sherlock Holmes",          "https://www.gutenberg.org/files/1661/1661-0.txt"),
        ("Pride and Prejudice",      "https://www.gutenberg.org/files/1342/1342-0.txt"),
        ("Moby Dick",                "https://www.gutenberg.org/files/2701/2701-0.txt"),
        ("Frankenstein",             "https://www.gutenberg.org/files/84/84-0.txt"),
        ("Alice in Wonderland",      "https://www.gutenberg.org/files/11/11-0.txt"),
        ("Adventures of Tom Sawyer", "https://www.gutenberg.org/files/74/74-0.txt"),
        ("Great Expectations",       "https://www.gutenberg.org/files/1400/1400-0.txt"),
        ("A Tale of Two Cities",     "https://www.gutenberg.org/files/98/98-0.txt"),
        ("Crime and Punishment",     "https://www.gutenberg.org/files/2554/2554-0.txt"),
        ("War of the Worlds",        "https://www.gutenberg.org/files/36/36-0.txt"),
        ("Dracula",                  "https://www.gutenberg.org/files/345/345-0.txt"),
        ("The Count of Monte Cristo","https://www.gutenberg.org/files/1184/1184-0.txt"),
        ("Don Quixote",              "https://www.gutenberg.org/files/996/996-0.txt"),
        ("Les Misérables",           "https://www.gutenberg.org/files/135/135-0.txt"),
        ("Anna Karenina",            "https://www.gutenberg.org/files/1399/1399-0.txt"),
        ("Ulysses",                  "https://www.gutenberg.org/files/4300/4300-0.txt"),
        ("The Brothers Karamazov",   "https://www.gutenberg.org/files/28054/28054-0.txt"),
        ("Jane Eyre",                "https://www.gutenberg.org/files/1260/1260-0.txt"),
        ("Wuthering Heights",        "https://www.gutenberg.org/files/768/768-0.txt"),
        ("The Picture of Dorian Gray","https://www.gutenberg.org/files/174/174-0.txt"),
        # Non-fiction / science / philosophy
        ("The Art of War",           "https://www.gutenberg.org/files/132/132-0.txt"),
        ("On the Origin of Species", "https://www.gutenberg.org/files/1228/1228-0.txt"),
        ("The Republic (Plato)",     "https://www.gutenberg.org/files/1497/1497-0.txt"),
        ("Meditations (Marcus Aurelius)",
                                     "https://www.gutenberg.org/files/2680/2680-0.txt"),
        ("Principia Mathematica (Newton)",
                                     "https://www.gutenberg.org/files/28233/28233-0.txt"),
        ("The Wealth of Nations",    "https://www.gutenberg.org/files/3300/3300-0.txt"),
        ("The Prince (Machiavelli)", "https://www.gutenberg.org/files/1232/1232-0.txt"),
        ("Beyond Good and Evil",     "https://www.gutenberg.org/files/4363/4363-0.txt"),
        ("Critique of Pure Reason",  "https://www.gutenberg.org/files/4280/4280-0.txt"),
        ("Relativity (Einstein)",    "https://www.gutenberg.org/files/5001/5001-0.txt"),
    ]

    # ── WikiText-103 (Wikipedia articles, ~180 MB, much richer knowledge) ──────
    WIKITEXT_URLS = [
        # Primary: HuggingFace mirror (reliable, no redirects)
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/wikitext-103-raw-v1.zip",
        # Fallback: original S3 (may redirect)
        "https://s3.amazonaws.com/research.metamind.io/wikitext/"
        "wikitext-103-raw-v1.zip",
    ]

    # ── Instruction fine-tuning Q&A pairs ─────────────────────────────────────
    # General knowledge + Linux + Programming + Science
    QA_PAIRS: List[Tuple[str, str]] = [
        # Linux / system
        ("What is Linux?",
         "Linux is a free, open-source operating system kernel created by Linus Torvalds in 1991. "
         "It powers everything from personal computers to servers, supercomputers, smartphones (Android), "
         "and embedded devices. Hundreds of distributions (Ubuntu, Fedora, Arch, Debian) build on it."),
        ("What does the 'ls' command do?",
         "The 'ls' command lists the contents of a directory. "
         "Common flags: -l (long format with permissions, size, date), -a (show hidden files), "
         "-h (human-readable sizes), -R (recursive). Example: ls -lah /home"),
        ("How do I check memory usage in Linux?",
         "Use 'free -h' for a quick overview of RAM and swap. "
         "For a detailed live view use 'htop' or 'top'. "
         "The file /proc/meminfo contains raw kernel memory statistics."),
        ("What is a process in Linux?",
         "A process is a running instance of a program. Each has a unique PID (Process ID). "
         "Processes can be listed with 'ps aux' or 'top'. "
         "Kill a process with 'kill <PID>' (SIGTERM) or 'kill -9 <PID>' (SIGKILL, forceful)."),
        ("What is SSH?",
         "SSH (Secure Shell) is a cryptographic network protocol for securely accessing remote machines. "
         "Use 'ssh user@hostname' to connect. Key-based authentication (ssh-keygen) is more secure "
         "than passwords. The default port is 22."),
        ("What is a file permission in Linux?",
         "Linux file permissions control who can read (r), write (w), or execute (x) a file. "
         "They are shown as a 9-character string like rwxr-xr-- for owner/group/others. "
         "Use 'chmod' to change them and 'chown' to change ownership."),
        ("What is grep?",
         "grep searches for patterns in text files using regular expressions. "
         "Example: 'grep -r error /var/log' searches all files in /var/log for the word 'error'. "
         "grep -i is case-insensitive, -n shows line numbers, -v inverts the match."),
        ("What is a shell script?",
         "A shell script is a text file containing a sequence of shell commands. "
         "It starts with a shebang line like #!/bin/bash. "
         "Scripts automate repetitive tasks, system administration, and deployments."),
        ("What is systemd?",
         "systemd is the init system and service manager used by most modern Linux distributions. "
         "It starts services at boot, manages dependencies between them, and provides logging via journald. "
         "Commands: systemctl start/stop/restart/status <service>, journalctl -u <service>."),
        ("How do I install software on Linux?",
         "It depends on the distribution. On Debian/Ubuntu use 'apt install <package>'. "
         "On Fedora/RHEL use 'dnf install <package>'. On Arch use 'pacman -S <package>'. "
         "You can also compile from source or use universal formats like Flatpak, Snap, or AppImage."),
        # Programming
        ("What is Python?",
         "Python is a high-level, interpreted, general-purpose programming language known for its "
         "clean syntax and readability. It is widely used in data science, web development, automation, "
         "AI/ML, and scripting. Created by Guido van Rossum in 1991."),
        ("What is a neural network?",
         "A neural network is a machine learning model loosely inspired by the brain. "
         "It consists of layers of interconnected nodes (neurons). Each connection has a weight. "
         "During training, weights are adjusted via backpropagation and gradient descent "
         "to minimise a loss function. Deep networks (many layers) can learn complex representations."),
        ("What is a transformer in deep learning?",
         "A transformer is a neural network architecture introduced in the 2017 paper "
         "'Attention Is All You Need'. It relies entirely on self-attention mechanisms rather than "
         "recurrence or convolution. Transformers are the foundation of large language models "
         "like GPT, BERT, T5, and LLaMA."),
        ("What is gradient descent?",
         "Gradient descent is an iterative optimisation algorithm that minimises a loss function "
         "by updating parameters in the direction opposite to the gradient. "
         "Stochastic Gradient Descent (SGD) uses random mini-batches for efficiency. "
         "Adaptive variants like Adam adjust the learning rate per parameter."),
        ("What is overfitting?",
         "Overfitting happens when a model learns the training data too well, including its noise, "
         "and fails to generalise to new data. Signs: high train accuracy but low validation accuracy. "
         "Remedies: more data, dropout, weight decay (L2 regularisation), early stopping, "
         "data augmentation, or a simpler model."),
        ("What is the difference between RAM and storage?",
         "RAM (Random Access Memory) is fast, temporary memory used by running programs. "
         "Storage (HDD/SSD) is slow, persistent memory that retains data when powered off. "
         "When a program runs, its code and data are loaded from storage into RAM for fast access."),
        # Science / general
        ("What is artificial intelligence?",
         "Artificial intelligence (AI) is the simulation of human-like intelligence in machines. "
         "It encompasses machine learning (systems that learn from data), deep learning (neural networks), "
         "natural language processing, computer vision, robotics, and more. "
         "Modern AI is largely driven by large neural networks trained on massive datasets."),
        ("What is the speed of light?",
         "The speed of light in a vacuum is approximately 299,792,458 metres per second (≈ 300,000 km/s). "
         "It is denoted by 'c' and is a fundamental constant of nature. "
         "Nothing with mass can reach or exceed c according to Einstein's special relativity."),
        ("What is DNA?",
         "DNA (deoxyribonucleic acid) is the molecule that carries the genetic instructions for "
         "the development, functioning, growth, and reproduction of all known organisms. "
         "It is a double helix of two strands made of nucleotides (adenine, thymine, guanine, cytosine)."),
        ("What is quantum computing?",
         "Quantum computing uses quantum-mechanical phenomena (superposition and entanglement) "
         "to perform computations. Quantum bits (qubits) can represent 0 and 1 simultaneously, "
         "enabling certain algorithms (Shor's, Grover's) to run exponentially faster than classical computers."),
        ("How does the internet work?",
         "The internet is a global network of computers connected via physical cables, fibre, and wireless links. "
         "Data is broken into packets and routed through intermediate nodes using the TCP/IP protocol. "
         "The World Wide Web (websites) runs on top of the internet using HTTP/HTTPS."),
        ("What is machine learning?",
         "Machine learning is a branch of AI where systems learn from data rather than being explicitly programmed. "
         "Types: supervised (labelled data, e.g. classification), unsupervised (unlabelled, e.g. clustering), "
         "reinforcement (agent learns by reward/penalty). ML powers recommendation systems, spam filters, "
         "image recognition, and language models."),
        ("What is a CPU?",
         "A CPU (Central Processing Unit) is the primary processor of a computer. "
         "It executes instructions from programs sequentially (or in parallel via multiple cores). "
         "Key specs: clock speed (GHz), number of cores, cache size. Modern CPUs have 4–64 cores. "
         "GPUs (Graphics Processing Units) are better for highly parallel workloads like ML training."),
        ("What is open source?",
         "Open-source software has its source code publicly available for anyone to read, modify, and distribute. "
         "Governed by licences like MIT, GPL, and Apache. "
         "Famous examples: Linux, Python, Firefox, Git, TensorFlow, PyTorch. "
         "Open source fosters collaboration, transparency, and community-driven development."),
        ("Tell me a fun fact.",
         "Here is one: a group of flamingos is called a 'flamboyance'. "
         "Another: honey never spoils – archaeologists found 3000-year-old honey in Egyptian tombs that was still edible. "
         "And: the shortest war in history lasted only 38 to 45 minutes (Anglo-Zanzibar War, 1896)."),
        # ── Extended Linux / DevOps ───────────────────────────────────────────
        ("How do I find files in Linux?",
         "Use the 'find' command. Examples: "
         "'find /home -name '*.log'' finds all .log files under /home. "
         "'find . -type f -mtime -7' finds files modified in the last 7 days. "
         "For fast name-based search, 'locate filename' uses a pre-built index (run 'updatedb' first). "
         "For content search inside files, use 'grep -r pattern /path'."),
        ("What is a cron job?",
         "A cron job is a scheduled task managed by the cron daemon in Linux. "
         "Cron reads a crontab file where each line defines a schedule and command. "
         "Format: 'min hour day month weekday command'. "
         "Example: '0 2 * * * /backup.sh' runs /backup.sh at 2:00 AM every day. "
         "Edit your crontab with 'crontab -e'. List jobs with 'crontab -l'."),
        ("What is a firewall?",
         "A firewall controls network traffic by allowing or blocking connections based on rules. "
         "On Linux, 'iptables' and 'nftables' are the kernel-level packet filters. "
         "'ufw' (Uncomplicated Firewall) provides a simpler front-end: "
         "'ufw allow 22' permits SSH, 'ufw deny 80' blocks HTTP, 'ufw enable' activates the firewall."),
        ("What is Docker?",
         "Docker is a containerisation platform that packages applications and their dependencies "
         "into isolated, portable containers. Containers share the host OS kernel but have separate "
         "filesystems and process spaces. Key commands: 'docker run', 'docker build', 'docker ps', "
         "'docker stop'. Docker Compose lets you define multi-container apps in a YAML file."),
        ("What is Git and how do I use it?",
         "Git is a distributed version control system created by Linus Torvalds. "
         "Key commands: 'git init' – create a new repo. 'git clone URL' – copy a remote repo. "
         "'git add file' – stage changes. 'git commit -m msg' – save a snapshot. "
         "'git push' – upload to remote. 'git pull' – fetch + merge from remote. "
         "'git branch name' – create a branch. 'git merge branch' – merge it back."),
        ("How does a CPU work?",
         "A CPU (Central Processing Unit) is the brain of the computer. "
         "It repeatedly executes the fetch-decode-execute cycle: "
         "1. Fetch an instruction from RAM. "
         "2. Decode it (determine the operation). "
         "3. Execute it (ALU does arithmetic, registers store temporary values). "
         "Modern CPUs use pipelining, branch prediction, and out-of-order execution for speed. "
         "Multiple cores allow true parallel execution of independent tasks."),
        ("What is the difference between TCP and UDP?",
         "TCP (Transmission Control Protocol) guarantees ordered, reliable delivery: "
         "it establishes a connection (3-way handshake), retransmits lost packets, and ensures order. "
         "Used for HTTP, SSH, FTP, email. "
         "UDP (User Datagram Protocol) is connectionless and faster but unreliable: "
         "no guarantee of delivery or order. Used for video streaming, gaming, DNS, and VoIP."),
        ("What is machine learning?",
         "Machine learning (ML) is a branch of AI where systems learn patterns from data "
         "rather than following explicit rules. "
         "Main types: supervised learning (learns from labelled examples), "
         "unsupervised learning (finds structure in unlabelled data), "
         "reinforcement learning (learns by trial and error with rewards). "
         "Common algorithms: linear regression, decision trees, neural networks, SVMs."),
        ("What is CUDA?",
         "CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform "
         "that lets developers run code on GPUs. "
         "GPUs have thousands of small cores ideal for massively parallel tasks like matrix multiplication. "
         "PyTorch and TensorFlow use CUDA automatically when a compatible GPU is available. "
         "Check GPU availability in PyTorch with 'torch.cuda.is_available()'."),
        ("How does the internet work?",
         "The internet is a global network of computers connected via the TCP/IP protocol stack. "
         "When you visit a website: 1. DNS resolves the hostname to an IP address. "
         "2. Your browser opens a TCP connection to the server (port 80 HTTP, 443 HTTPS). "
         "3. It sends an HTTP request. 4. The server responds with HTML/CSS/JS. "
         "5. Your browser renders the page. Data travels as IP packets through routers worldwide."),
        ("What is recursion in programming?",
         "Recursion is when a function calls itself to solve a smaller version of the same problem. "
         "Every recursive function needs a base case (stop condition) to prevent infinite loops. "
         "Example: factorial(n) = n * factorial(n-1), base case factorial(0) = 1. "
         "Recursion is elegant but can cause stack overflow for very deep call chains. "
         "Iterative solutions with explicit stacks are sometimes preferred for performance."),
        ("What are environment variables in Linux?",
         "Environment variables are named values available to processes in their environment. "
         "View all: 'env' or 'printenv'. Print one: 'echo $HOME'. "
         "Set temporarily: 'MYVAR=value command'. Set persistently: add 'export MYVAR=value' "
         "to ~/.bashrc or ~/.profile. "
         "Important variables: PATH (executable search paths), HOME, USER, SHELL, LANG."),
        # ── AI / self-awareness Q&A ───────────────────────────────────────────
        ("Who are you?",
         "I am LinuxAI, an artificial intelligence assistant running natively on Linux. "
         "I was built from scratch using Python and PyTorch — a GPT-style transformer language model "
         "trained on public-domain texts and a curated Q&A dataset. "
         "I can answer questions about Linux, programming, science, and many other topics. "
         "I also have access to your live system information and can help manage your computer."),
        ("What can you do?",
         "I can help you with many tasks: "
         "Answer questions about Linux, programming, AI, science, and general knowledge. "
         "Monitor your system (CPU, RAM, disk, processes, network). "
         "Run shell commands and explain their output. "
         "Help debug code and explain concepts. "
         "Browse your filesystem and list running services. "
         "I was trained entirely from scratch — no external AI API is used."),
        ("How were you trained?",
         "I was trained in two phases using PyTorch on a text corpus. "
         "Phase 1 (pretraining): I read millions of characters from public-domain books "
         "(Sherlock Holmes, Moby Dick, philosophical texts, science books, etc.) "
         "to learn grammar, facts, and reasoning patterns. "
         "Phase 2 (instruction fine-tuning): I was shown hundreds of question-answer pairs "
         "to learn how to follow instructions and give helpful responses. "
         "The architecture is a causal transformer with RoPE attention and SwiGLU feed-forward layers."),
        ("What is a large language model?",
         "A large language model (LLM) is a neural network trained on vast amounts of text "
         "to predict the next token in a sequence. "
         "During training it implicitly learns grammar, facts, reasoning, and world knowledge. "
         "At inference time, it generates text autoregressively — one token at a time — "
         "conditioned on all previous tokens. "
         "Famous LLMs: GPT-4 (OpenAI), Claude (Anthropic), LLaMA (Meta), Gemini (Google). "
         "They have billions of parameters; I am a compact version trained from scratch."),
        ("What is attention in transformers?",
         "Self-attention is the core mechanism of transformer models. "
         "For each token, it computes a weighted sum over all other tokens in the sequence, "
         "where the weights reflect how relevant each other token is. "
         "This allows the model to relate words that are far apart in the text. "
         "Multi-head attention runs several attention heads in parallel, each learning "
         "different types of relationships (e.g., syntax, coreference, semantics). "
         "The formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V"),

        # ── Mathematics ───────────────────────────────────────────────────────
        ("What is the Pythagorean theorem?",
         "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse "
         "(the side opposite the right angle) equals the sum of squares of the other two sides. "
         "Formula: a² + b² = c², where c is the hypotenuse. "
         "For example, a triangle with sides 3, 4, and 5: 9 + 16 = 25. ✓ "
         "It is one of the most fundamental results in geometry, with hundreds of known proofs."),
        ("What is a derivative in calculus?",
         "A derivative measures how a function changes as its input changes — it is the instantaneous rate of change. "
         "Geometrically, it is the slope of the tangent line to the curve at a given point. "
         "Notation: f'(x) or df/dx. Basic rules: "
         "d/dx[x^n] = nx^(n-1) (power rule), "
         "d/dx[sin x] = cos x, d/dx[e^x] = e^x, "
         "chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x). "
         "Derivatives are used in physics, economics, and optimisation."),
        ("What is a prime number?",
         "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. "
         "Examples: 2, 3, 5, 7, 11, 13, 17, 19, 23 … "
         "2 is the only even prime. There are infinitely many primes (Euclid's proof). "
         "The Sieve of Eratosthenes is a classic algorithm to find all primes up to a limit. "
         "Prime numbers are fundamental in cryptography (RSA encryption relies on factoring large primes)."),
        ("What is the difference between mean, median, and mode?",
         "Mean: the arithmetic average — sum all values, divide by count. Sensitive to outliers. "
         "Median: the middle value when sorted. Robust to outliers. For even counts, average the two middle values. "
         "Mode: the most frequently occurring value. A dataset can have multiple modes or none. "
         "Example: [1, 2, 2, 3, 100] → Mean=21.6, Median=2, Mode=2. "
         "The median best represents 'typical' income in a skewed distribution."),
        ("What is Big-O notation?",
         "Big-O notation describes the upper bound on an algorithm's time or space complexity as input size n grows. "
         "O(1) — constant time (e.g., array lookup). "
         "O(log n) — logarithmic (binary search). "
         "O(n) — linear (scanning a list). "
         "O(n log n) — efficient sorting (merge sort, heapsort). "
         "O(n²) — quadratic (bubble sort, nested loops). "
         "O(2^n) — exponential (brute-force combinatorics). "
         "Choosing the right algorithm can make the difference between milliseconds and centuries."),

        # ── Physics & Science ─────────────────────────────────────────────────
        ("What is Newton's second law?",
         "Newton's second law of motion states: Force = mass × acceleration (F = ma). "
         "A larger force produces a greater acceleration; a larger mass requires more force to accelerate. "
         "Units: force in Newtons (N), mass in kilograms (kg), acceleration in m/s². "
         "This law is the foundation of classical mechanics and explains everything from falling objects "
         "to rocket propulsion."),
        ("What is energy and what are its forms?",
         "Energy is the capacity to do work. It is measured in Joules (J). "
         "Main forms: Kinetic (energy of motion, ½mv²), Potential (stored energy, e.g. gravitational mgh), "
         "Thermal (heat), Chemical (stored in bonds, e.g. food, fuel), "
         "Electrical (moving charges), Nuclear (from atomic nuclei), Radiant (electromagnetic radiation). "
         "The law of conservation of energy: energy cannot be created or destroyed, only transformed."),
        ("What is the periodic table?",
         "The periodic table organises all 118 known chemical elements by atomic number (number of protons). "
         "Elements in the same column (group) share similar chemical properties. "
         "Rows are called periods; as you go right, atomic number increases. "
         "Created by Dmitri Mendeleev in 1869. Key groups: alkali metals (Group 1), halogens (Group 17), "
         "noble gases (Group 18). Metals make up ~75% of all elements."),
        ("What is photosynthesis?",
         "Photosynthesis is the process by which plants, algae, and some bacteria convert "
         "sunlight, water (H₂O), and carbon dioxide (CO₂) into glucose (C₆H₁₂O₆) and oxygen (O₂). "
         "Equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. "
         "It occurs in the chloroplasts. It is the basis of almost all food chains on Earth "
         "and produces the oxygen we breathe."),
        ("What is gravity?",
         "Gravity is a fundamental force that attracts any two objects with mass toward each other. "
         "Newton's law of gravitation: F = G·m₁·m₂/r², where G is the gravitational constant, "
         "m₁ and m₂ are masses, and r is the distance between them. "
         "Einstein's general relativity describes gravity as the curvature of spacetime caused by mass. "
         "On Earth's surface, g ≈ 9.81 m/s². Gravity holds planets in orbit and shapes the large-scale "
         "structure of the universe."),

        # ── History & Geography ───────────────────────────────────────────────
        ("When did World War II happen and who were the main parties?",
         "World War II lasted from 1939 to 1945. "
         "The Allies (USA, UK, Soviet Union, France, China, and others) fought the Axis powers "
         "(Nazi Germany, Fascist Italy, Imperial Japan). "
         "It started when Germany invaded Poland on September 1, 1939. "
         "Key events: Battle of Britain (1940), German invasion of USSR (1941), "
         "US entry after Pearl Harbor (Dec 1941), D-Day Normandy landings (June 6, 1944), "
         "atomic bombs on Hiroshima and Nagasaki (Aug 1945). "
         "It ended in September 1945 and killed an estimated 70–85 million people."),
        ("What is the capital of Portugal and what language is spoken there?",
         "The capital of Portugal is Lisbon (Lisboa). It is the westernmost capital city in continental Europe. "
         "The official language is Portuguese (Português), spoken by about 260 million people worldwide — "
         "making it the 6th most spoken language. "
         "Portugal is located on the Iberian Peninsula in southwestern Europe. "
         "Its currency is the Euro (€). Portugal joined the EU in 1986."),
        ("What was the Renaissance?",
         "The Renaissance was a cultural and intellectual movement that began in Italy in the 14th century "
         "and spread across Europe through the 17th century. "
         "It marked the transition from the Medieval period to modernity, characterised by renewed interest "
         "in classical Greek and Roman culture, humanism, scientific inquiry, and artistic innovation. "
         "Key figures: Leonardo da Vinci, Michelangelo, Raphael (art), Copernicus, Galileo (science), "
         "Dante, Petrarch (literature). The printing press (Gutenberg, ~1440) accelerated the spread of ideas."),
        ("What is the European Union?",
         "The European Union (EU) is a political and economic union of 27 member states, primarily in Europe. "
         "Founded by the Maastricht Treaty in 1993, built on earlier communities dating to the 1950s. "
         "Key institutions: European Parliament, European Commission, European Council, Court of Justice. "
         "19 member states share the euro (€) currency (Eurozone). "
         "The EU guarantees free movement of people, goods, services, and capital across member states. "
         "Combined GDP makes it one of the world's largest economies."),

        # ── Programming ───────────────────────────────────────────────────────
        ("What is object-oriented programming?",
         "Object-oriented programming (OOP) organises code around objects — instances of classes that "
         "bundle data (attributes) and behaviour (methods). "
         "Core principles: Encapsulation (hiding internal details), Inheritance (classes share behaviour "
         "from parent classes), Polymorphism (different objects respond to the same interface differently), "
         "Abstraction (hiding complexity). "
         "OOP languages: Python, Java, C++, C#, Ruby. "
         "It models real-world entities naturally and promotes code reuse."),
        ("What is the difference between a list and a tuple in Python?",
         "Both list and tuple store ordered sequences of elements in Python. Key differences: "
         "Lists are mutable (can be changed after creation): my_list.append(4), my_list[0] = 99. "
         "Tuples are immutable (cannot be changed): coordinates = (10.5, 48.3). "
         "Tuples are faster and use less memory. Use tuples for fixed data (coordinates, RGB colours, "
         "database rows) and lists for data that changes (shopping cart, log entries)."),
        ("What is a REST API?",
         "REST (Representational State Transfer) is an architectural style for web APIs. "
         "It uses standard HTTP methods: GET (read), POST (create), PUT/PATCH (update), DELETE (remove). "
         "Resources are identified by URLs (e.g., /api/users/42). "
         "Responses are usually JSON. REST APIs are stateless — each request contains all needed information. "
         "Example: GET /api/products returns all products; POST /api/products with JSON body creates one."),
        ("What is a database and what is SQL?",
         "A database is an organised collection of structured data, managed by a Database Management System (DBMS). "
         "SQL (Structured Query Language) is the standard language for relational databases. "
         "Key statements: SELECT col FROM table WHERE condition — retrieve data. "
         "INSERT INTO table VALUES (...) — add a row. "
         "UPDATE table SET col=val WHERE condition — modify rows. "
         "DELETE FROM table WHERE condition — remove rows. "
         "JOIN combines rows from multiple tables. Popular DBs: PostgreSQL, MySQL, SQLite, MariaDB."),
        ("What is a regular expression?",
         "A regular expression (regex) is a sequence of characters that defines a search pattern. "
         "Used for finding, matching, and replacing text. "
         "Common syntax: . (any char), * (0 or more), + (1 or more), ? (0 or 1), "
         "^ (start of line), $ (end), [abc] (any of a,b,c), \\d (digit), \\w (word char). "
         "Example: \\d{3}-\\d{4} matches phone numbers like 123-4567. "
         "In Python: import re; re.findall(r'\\d+', 'I have 3 cats and 12 dogs') → ['3', '12']."),
        ("What is asynchronous programming?",
         "Asynchronous programming lets a program handle multiple tasks concurrently without blocking. "
         "Instead of waiting for a slow operation (network call, disk I/O) to finish, "
         "the program can do other work meanwhile. "
         "In Python, 'async/await' syntax and the 'asyncio' library enable this. "
         "Example: async def fetch_data(): data = await http_get(url) — doesn't block while waiting. "
         "Other models: threads (true parallelism, CPU-bound), processes (separate memory space). "
         "Async is ideal for I/O-bound tasks like web servers handling many simultaneous requests."),
        ("What is the difference between compiled and interpreted languages?",
         "Compiled languages (C, C++, Rust, Go) translate source code to machine code before running. "
         "Result: fast executables, but require a compilation step per platform. "
         "Interpreted languages (Python, JavaScript, Ruby) are executed line-by-line by an interpreter at runtime. "
         "Result: slower execution but faster development, cross-platform by default. "
         "Some languages are both: Java compiles to bytecode run by the JVM; "
         "Python can be compiled to .pyc bytecode for faster loading. "
         "Modern JIT (Just-In-Time) compilers blur this distinction."),

        # ── Health & Biology ──────────────────────────────────────────────────
        ("How does the human heart work?",
         "The human heart is a muscular organ that pumps blood through the circulatory system. "
         "It has four chambers: right atrium, right ventricle, left atrium, left ventricle. "
         "The right side pumps deoxygenated blood to the lungs (pulmonary circulation). "
         "The left side pumps oxygenated blood to the body (systemic circulation). "
         "The heart beats ~60–100 times per minute at rest, driven by electrical signals "
         "from the sinoatrial (SA) node. The aorta is the body's largest artery."),
        ("What are vitamins and why are they important?",
         "Vitamins are essential organic compounds needed in small amounts for normal body function. "
         "They cannot be synthesised in sufficient quantities by the body, so must come from food. "
         "Fat-soluble: A (vision, immunity), D (calcium absorption, bones), E (antioxidant), K (blood clotting). "
         "Water-soluble: C (immunity, collagen), B1 (energy), B12 (nerve function, red blood cells), "
         "Folate (cell division, critical in pregnancy). "
         "Deficiencies cause diseases: lack of C → scurvy, lack of D → rickets, lack of B12 → anaemia."),
        ("What is the immune system?",
         "The immune system defends the body against pathogens (bacteria, viruses, fungi, parasites). "
         "Innate immunity — fast, non-specific first response: skin, mucus, fever, macrophages, neutrophils. "
         "Adaptive immunity — slower, specific, has memory: B cells produce antibodies, "
         "T cells kill infected cells. Memory cells remember past infections for faster future responses. "
         "Vaccines work by presenting harmless antigen fragments to trigger adaptive immunity without disease. "
         "Autoimmune diseases occur when the immune system attacks the body's own tissues."),

        # ── Economics & Society ───────────────────────────────────────────────
        ("What is inflation?",
         "Inflation is the rate at which the general level of prices for goods and services rises, "
         "reducing purchasing power. Measured by the Consumer Price Index (CPI). "
         "Causes: too much money in circulation (demand-pull), rising production costs (cost-push), "
         "supply shocks (e.g. oil crisis). "
         "Central banks (e.g. European Central Bank, US Federal Reserve) manage inflation by raising "
         "or lowering interest rates. Moderate inflation (~2%) is considered healthy for an economy; "
         "hyperinflation (thousands of % per year) is catastrophic."),
        ("What is cryptocurrency?",
         "Cryptocurrency is a digital or virtual currency secured by cryptography, operating on "
         "decentralised blockchain networks. "
         "Bitcoin (BTC), created in 2009, was the first. Others include Ethereum (ETH), which supports "
         "smart contracts and decentralised apps. "
         "Transactions are recorded on a distributed ledger (blockchain) maintained by many nodes. "
         "Mining uses computational power to validate transactions and earn new coins. "
         "Cryptocurrencies are highly volatile and unregulated in many jurisdictions."),

        # ── Cultura Portuguesa / Portuguese ──────────────────────────────────
        ("O que é o fado?",
         "O fado é um género musical português reconhecido pela UNESCO como Património Cultural Imaterial da Humanidade. "
         "Caracteriza-se por melodias melancólicas que expressam 'saudade' — um sentimento único de nostalgia. "
         "Instrumentado tradicionalmente com guitarra portuguesa, viola baixo e voz. "
         "Surgiu em Lisboa no século XIX. Grandes nomes: Amália Rodrigues, Carlos do Carmo, Mariza. "
         "Há também o fado de Coimbra, mais académico e cantado exclusivamente por homens."),
        ("Qual é a capital de Portugal e o que é famoso lá?",
         "Lisboa (Lisbon) é a capital de Portugal, situada na foz do rio Tejo. "
         "É famosa pelo Castelo de São Jorge, a Torre de Belém (Património da UNESCO), "
         "o Mosteiro dos Jerónimos, o elétrico 28, o Alfama (bairro histórico), "
         "os Pastéis de Belém, e o Museu do Azulejo. "
         "Tem uma das mais belas baías da Europa. O porto de Lisboa foi fundamental "
         "nas Descobertas Portuguesas (séculos XV-XVI)."),
        ("O que foram os Descobrimentos Portugueses?",
         "Os Descobrimentos Portugueses (séculos XV e XVI) foram uma série de expedições marítimas "
         "que permitiram a Portugal cartografar e estabelecer rotas comerciais para a África, Ásia e Brasil. "
         "Figuras-chave: Infante D. Henrique (impulsionou a exploração), Bartolomeu Dias (Cabo da Boa Esperança, 1488), "
         "Vasco da Gama (rota para a Índia, 1498), Pedro Álvares Cabral (Brasil, 1500). "
         "Portugal foi a primeira potência colonial europeia e a sua influência moldou culturas em 5 continentes."),

        # ── CoreAI self-awareness (updated) ──────────────────────────────────
        ("Who are you?",
         "I am CoreAI, an artificial intelligence assistant built entirely from scratch using Python and PyTorch. "
         "I am a GPT-style transformer language model with 88 million parameters, trained on a diverse corpus "
         "including literature, science, philosophy, and encyclopaedic knowledge. "
         "I can answer questions on many topics, help with Linux and programming, monitor your system, "
         "and search the web for up-to-date information. "
         "I run 100% locally on your machine — no data is ever sent to external AI services."),
        ("What can you do?",
         "I can help you with many tasks: "
         "Answer questions about science, mathematics, history, programming, Linux, and general knowledge. "
         "Search the web (DuckDuckGo) and fetch current weather for real-time information. "
         "Monitor your system: CPU, RAM, GPU, disk usage, running processes. "
         "Run and explain shell commands. Debug code and explain concepts. "
         "Have a conversation in English or Portuguese. "
         "I was trained entirely from scratch — no external AI API is used."),
        ("How were you trained?",
         "I was trained in two phases using PyTorch on an RTX GPU. "
         "Phase 1 (pretraining, ~20,000 steps): I read millions of tokens from a large corpus including "
         "classic literature, philosophical texts, scientific works, and Wikipedia articles (WikiText-103). "
         "This taught me grammar, facts, reasoning patterns, and general world knowledge. "
         "Phase 2 (instruction fine-tuning, ~2,000 steps): I was shown 150+ question-answer pairs "
         "covering Linux, programming, science, mathematics, history, and general knowledge, "
         "to learn how to follow instructions and give helpful, concise answers. "
         "Architecture: causal transformer with RoPE positional encoding, SwiGLU FFN, and RMSNorm."),
        ("What is your architecture?",
         "I use a decoder-only transformer architecture similar to GPT, with modern improvements: "
         "RoPE (Rotary Position Embeddings) — encodes token positions in the attention scores, "
         "allowing better generalisation to longer sequences than learned embeddings. "
         "SwiGLU FFN — LLaMA-style feed-forward with gated linear units (8/3 expansion ratio), "
         "more expressive than standard GELU feed-forward. "
         "RMSNorm — simpler and faster than LayerNorm, used in LLaMA and Mistral. "
         "BPE tokenizer (8192 vocab) — byte-level byte-pair encoding. "
         "Size: 88M parameters, 12 layers, 12 attention heads, d_model=768, context=512 tokens."),
    ]

    # ── Instruction format ────────────────────────────────────────────────────
    HUMAN_TAG = "### Human: "
    AI_TAG    = "### AI: "
    SEP       = "\n\n"

    @classmethod
    def format_qa(cls, question: str, answer: str = "") -> str:
        """Return an instruction-formatted string ready for tokenisation."""
        s = cls.HUMAN_TAG + question + "\n" + cls.AI_TAG + answer
        if answer:
            s += cls.SEP
        return s

    # ── Corpus download ───────────────────────────────────────────────────────

    @staticmethod
    def _download_text(url: str, name: str) -> str:
        """Download text from `url`, returning raw content."""
        console.print(f"  [dim]↓ Downloading {name} …[/]", end="")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LinuxAI/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            # Strip Project Gutenberg header/footer boilerplate
            for marker in ("*** START OF", "***START OF"):
                pos = raw.find(marker)
                if pos != -1:
                    raw = raw[raw.index("\n", pos) + 1:]
                    break
            for marker in ("*** END OF", "***END OF"):
                pos = raw.find(marker)
                if pos != -1:
                    raw = raw[:pos]
                    break
            console.print(f" [green]{len(raw):,} chars[/]")
            return raw.strip()
        except Exception as exc:
            console.print(f" [red]FAILED ({exc})[/]")
            return ""

    @classmethod
    def _download_wikitext(cls) -> str:
        """
        Download WikiText-103-raw (zip ~180 MB) and extract the training split.
        Returns the text content or '' on failure.
        """
        import zipfile, io
        cache_path = DATA_DIR / "wikitext103_train.txt"
        if cache_path.exists():
            console.print(f"  [green]WikiText-103 cache found ({cache_path.stat().st_size // 1_000_000} MB).[/]")
            return cache_path.read_text(encoding="utf-8", errors="ignore")

        console.print("  [cyan]↓ Downloading WikiText-103 (~180 MB) … this is a one-time download[/]")
        for url in cls.WIKITEXT_URLS:
            try:
                console.print(f"  [dim]Trying: {url[:60]}…[/]")
                # Use a redirect-following opener
                opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
                req = urllib.request.Request(url, headers={"User-Agent": "CoreAI/1.0"})
                with opener.open(req, timeout=120) as resp:
                    data = resp.read()
                console.print(f"  [green]Downloaded {len(data) // 1_000_000} MB. Extracting …[/]")
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    train_file = next(
                        (n for n in zf.namelist() if "train" in n and n.endswith(".raw")),
                        None,
                    )
                    if train_file is None:
                        train_file = next((n for n in zf.namelist() if "train" in n), None)
                    if train_file is None:
                        console.print("  [red]Could not find training file in zip.[/]")
                        continue
                    text = zf.read(train_file).decode("utf-8", errors="ignore")
                cache_path.write_text(text, encoding="utf-8")
                console.print(f"  [green]WikiText-103 saved: {len(text):,} chars.[/]")
                return text
            except Exception as exc:
                console.print(f"  [yellow]Failed ({exc}), trying next URL…[/]")
        console.print("  [red]WikiText-103 unavailable. Continuing with Gutenberg books only.[/]")
        return ""

    @classmethod
    def build_corpus(cls) -> str:
        """
        Build training corpus:
          1. WikiText-103 (Wikipedia articles, ~180 MB) — primary source
          2. Project Gutenberg books (classic literature + non-fiction)
          3. Q&A pairs (for pretraining signal, repeated a few times)
        """
        if cfg.corpus_path.exists():
            console.print(f"[green]Corpus cache found ({cfg.corpus_path}).[/]")
            return cfg.corpus_path.read_text(encoding="utf-8")

        console.print("[cyan]Building training corpus …[/]")
        parts: List[str] = []

        # 1. WikiText-103 (biggest source of knowledge)
        console.print("[bold]WikiText-103 (Wikipedia articles):[/]")
        wiki_text = cls._download_wikitext()
        if wiki_text:
            # Use up to 50M chars (~1/3 of full dataset) to keep training manageable
            parts.append(wiki_text[:50_000_000])

        # 2. Gutenberg books
        console.print("[bold]Project Gutenberg books:[/]")
        for name, url in cls.GUTENBERG:
            text = cls._download_text(url, name)
            if text:
                parts.append(text)

        # 3. Q&A pairs (repeated 3× to give fine-tuning signal in pretrain too)
        qa_block = "\n\n".join(
            cls.format_qa(q, a) for q, a in cls.QA_PAIRS
        ) * 3
        parts.append(qa_block)

        corpus = "\n\n".join(parts)
        cfg.corpus_path.write_text(corpus, encoding="utf-8")
        console.print(f"[green]Corpus saved: {len(corpus):,} characters.[/]")
        return corpus

    # ── Mini-batch sampler ────────────────────────────────────────────────────

    @staticmethod
    def _random_batch(
        data   : torch.Tensor,
        context: int,
        batch  : int,
        device : str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample `batch` random (x, y) pairs from a flat token tensor."""
        ix = torch.randint(len(data) - context, (batch,))
        x  = torch.stack([data[i     : i + context    ] for i in ix])
        y  = torch.stack([data[i + 1 : i + context + 1] for i in ix])
        return x.to(device), y.to(device)

    # ── Training loop ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_inference_ckpt(model: "CoreGPT", tok: Any, path: Path) -> None:
        """Save a self-contained inference checkpoint (no optimizer state)."""
        torch.save(
            {
                "arch"           : "core-gpt-v2",
                "tokenizer_type" : cfg.llm_tokenizer,
                "model_state"    : model.state_dict(),
                "vocab_size"     : tok.vocab_size,
                "context_len"    : cfg.llm_context,
                "d_model"        : cfg.llm_d_model,
                "n_heads"        : cfg.llm_n_heads,
                "n_layers"       : cfg.llm_n_layers,
            },
            path,
        )

    @classmethod
    def train(cls, skip_pretrain: bool = False) -> "CoreGPT":
        """
        Full two-phase training pipeline.
        skip_pretrain=True loads weights from coregpt_pretrain.pth and only
        runs fine-tuning — useful for tweaking fine-tuning without retraining.
        """
        console.rule("[bold magenta]Phase 1 – Building corpus[/]")
        corpus = cls.build_corpus()

        # ── Build / load tokenizer ────────────────────────────────────────────
        if cfg.tok_path.exists():
            console.rule("[bold magenta]Phase 2 – Loading tokenizer (cached)[/]")
            tok_data = json.loads(cfg.tok_path.read_text(encoding="utf-8"))
            tok_type = tok_data.get("type", "char")
            if tok_type == "bpe":
                tok: Any = BPETokenizer.load(cfg.tok_path)
            else:
                tok = CharTokenizer.load(cfg.tok_path)
            console.print(f"[green]Tokenizer loaded:[/] {tok.vocab_size} tokens ({tok_type})")
        else:
            console.rule("[bold magenta]Phase 2 – Building tokenizer[/]")
            if cfg.llm_tokenizer == "bpe":
                tok = BPETokenizer(vocab_size=cfg.llm_bpe_vocab).build(corpus)
                console.print(
                    f"[green]BPE vocabulary: {tok.vocab_size} tokens "
                    f"(target {cfg.llm_bpe_vocab})[/]"
                )
            else:
                tok = CharTokenizer().build(corpus)
                console.print(f"[green]Char vocabulary: {tok.vocab_size} characters[/]")
            tok.save(cfg.tok_path)

        # ── Encode corpus ─────────────────────────────────────────────────────
        console.print("[cyan]Encoding corpus …[/]")
        data_ids  = torch.tensor(tok.encode(corpus), dtype=torch.long)
        console.print(
            f"[green]Corpus:[/] {len(corpus):,} chars → "
            f"[green]{len(data_ids):,} tokens[/] "
            f"(compression {len(corpus)/max(len(data_ids),1):.2f}×)"
        )
        split     = int(0.9 * len(data_ids))
        train_ids = data_ids[:split]
        val_ids   = data_ids[split:]

        # ── Check for resumable checkpoint ────────────────────────────────────
        resume_ckpt   = None
        resume_phase  = None
        resume_step   = 1
        if cfg.llm_ckpt_path.exists():
            resume_ckpt  = torch.load(cfg.llm_ckpt_path, map_location=cfg.device, weights_only=False)
            resume_phase = resume_ckpt.get("phase")
            resume_step  = resume_ckpt.get("step", 1) + 1
            console.print(
                f"[yellow]Resumable checkpoint found:[/] "
                f"phase=[cyan]{resume_phase}[/]  "
                f"step=[cyan]{resume_ckpt['step']}/{resume_ckpt['total_steps']}[/]"
            )

        # ── Build model ───────────────────────────────────────────────────────
        model = CoreGPT(
            vocab_size  = tok.vocab_size,
            context_len = cfg.llm_context,
            d_model     = cfg.llm_d_model,
            n_heads     = cfg.llm_n_heads,
            n_layers    = cfg.llm_n_layers,
            dropout     = cfg.llm_dropout,
        ).to(cfg.device)

        if skip_pretrain and cfg.llm_pretrain_path.exists():
            pretrain_ckpt = torch.load(cfg.llm_pretrain_path, map_location=cfg.device, weights_only=False)
            model.load_state_dict(pretrain_ckpt["model_state"])
            console.print(
                f"[green]Pretrained weights loaded[/] from {cfg.llm_pretrain_path}  "
                f"[dim](skipping pretraining)[/]"
            )
        elif resume_ckpt is not None:
            model.load_state_dict(resume_ckpt["model_state"])
            console.print("[green]Model weights restored from checkpoint.[/]")
        else:
            console.rule("[bold magenta]Phase 3 – Pretraining CoreGPT[/]")

        console.print(
            f"[cyan]Parameters: {model.num_parameters:,}[/]  |  "
            f"[cyan]Device: {cfg.device}[/]  |  "
            f"[cyan]Context: {cfg.llm_context} tokens[/]  |  "
            f"[cyan]Architecture: RoPE + SwiGLU + RMSNorm[/]"
        )

        # ── Pretraining phase ─────────────────────────────────────────────────
        if resume_phase not in ("Fine-tuning",) and not skip_pretrain:
            if resume_phase != "Pretraining":
                console.rule("[bold magenta]Phase 3 – Pretraining CoreGPT[/]")
            pre_start = resume_step if resume_phase == "Pretraining" else 1
            optimizer  = optim.AdamW(model.parameters(), lr=cfg.llm_lr, weight_decay=0.01)
            scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.llm_pretrain_steps, eta_min=cfg.llm_lr / 10
            )
            if resume_ckpt is not None and resume_phase == "Pretraining":
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            cls._run_loop(
                model, optimizer, scheduler,
                train_ids, val_ids,
                steps      = cfg.llm_pretrain_steps,
                phase      = "Pretraining",
                start_step = pre_start,
            )
            # Save pretrained weights so fine-tuning can be redone without retraining
            cls._save_inference_ckpt(model, tok, path=cfg.llm_pretrain_path)
            console.print(f"[dim]  Pretrain weights saved → {cfg.llm_pretrain_path}[/]")

        # ── Fine-tuning phase ─────────────────────────────────────────────────
        console.rule("[bold magenta]Phase 4 – Instruction fine-tuning[/]")
        qa_text = "\n\n".join(
            cls.format_qa(q, a) for q, a in cls.QA_PAIRS
        ) * 5   # 5× repetition — enough to learn patterns without memorising
        qa_ids       = torch.tensor(tok.encode(qa_text), dtype=torch.long)
        ft_start     = resume_step if resume_phase == "Fine-tuning" else 1
        ft_optimizer = optim.AdamW(
            model.parameters(), lr=cfg.llm_lr / 5, weight_decay=0.01
        )
        ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            ft_optimizer, T_max=cfg.llm_finetune_steps, eta_min=cfg.llm_lr / 50
        )
        if resume_ckpt is not None and resume_phase == "Fine-tuning":
            ft_optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            ft_scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        cls._run_loop(
            model, ft_optimizer, ft_scheduler,
            qa_ids, qa_ids,
            steps      = cfg.llm_finetune_steps,
            phase      = "Fine-tuning",
            start_step = ft_start,
        )

        # ── Save final model + clean up mid-train checkpoint ──────────────────
        cls._save_inference_ckpt(model, tok, path=cfg.llm_path)
        if cfg.llm_ckpt_path.exists():
            cfg.llm_ckpt_path.unlink()
        console.print(f"\n[bold green]CoreGPT saved → {cfg.llm_path}[/]")
        log.info("CoreGPT trained and saved to %s", cfg.llm_path)
        return model

    @classmethod
    def _run_loop(
        cls,
        model    : CoreGPT,
        optimizer: optim.Optimizer,
        scheduler: Any,
        train_ids: torch.Tensor,
        val_ids  : torch.Tensor,
        steps    : int,
        phase    : str,
        start_step: int = 1,
    ) -> None:
        """
        Generic training loop with:
          - Automatic mixed precision (bf16) → ~50% less VRAM, ~2× faster on RTX
          - Gradient accumulation (effective batch = micro_batch × grad_accum)
          - Checkpoint saving every cfg.llm_save_every steps → safe to Ctrl+C and resume
        """
        model.train()
        log_every  = max(steps // 20, 50)
        eval_every = max(steps // 10, 100)
        best_val   = float("inf")

        use_amp   = (cfg.device == "cuda") and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        if use_amp:
            console.print(
                "[dim]  Mixed precision: [green]bf16[/] "
                "(~2× faster, ~50% less VRAM)[/]"
            )
        if start_step > 1:
            console.print(
                f"[yellow]  Resuming {phase} from step {start_step}/{steps}[/]"
            )

        grad_accum = cfg.llm_grad_accum

        def _save_checkpoint(step: int) -> None:
            torch.save(
                {
                    "phase"          : phase,
                    "step"           : step,
                    "total_steps"    : steps,
                    "model_state"    : model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "vocab_size"     : model.tok_emb.weight.shape[0],
                    "context_len"    : model.context_len,
                    "d_model"        : model.norm_f.weight.shape[0],
                    "n_heads"        : cfg.llm_n_heads,
                    "n_layers"       : cfg.llm_n_layers,
                    "tokenizer_type" : cfg.llm_tokenizer,
                    "arch"           : "core-gpt-v2",
                },
                cfg.llm_ckpt_path,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{phase}[/] [cyan]{{task.description}}[/]"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("", total=steps, completed=start_step - 1)
            running_loss = 0.0
            optimizer.zero_grad()

            for step in range(start_step, steps + 1):
                accum_loss = 0.0
                for _ in range(grad_accum):
                    x, y = cls._random_batch(
                        train_ids, cfg.llm_context, cfg.llm_batch, cfg.device
                    )
                    with torch.autocast(device_type=cfg.device, dtype=amp_dtype, enabled=use_amp):
                        _, loss = model(x, y)
                    (loss / grad_accum).backward()
                    accum_loss += loss.item() / grad_accum

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += accum_loss
                prog.advance(task, 1)
                prog.update(task, description=f"step {step}/{steps} loss={accum_loss:.3f}")

                # ── Periodic checkpoint (safe to Ctrl+C after this) ───────────
                if step % cfg.llm_save_every == 0:
                    _save_checkpoint(step)
                    log.info("Checkpoint saved at %s step=%d", phase, step)

                if step % log_every == 0:
                    avg = running_loss / log_every
                    running_loss = 0.0
                    log.info("%s step=%d avg_loss=%.4f", phase, step, avg)

                if step % eval_every == 0 and len(val_ids) > cfg.llm_context:
                    val_loss = cls._eval_loss(model, val_ids, use_amp=use_amp, amp_dtype=amp_dtype)
                    colour   = "green" if val_loss < best_val else "yellow"
                    console.print(
                        f"  step {step:>5}  train_loss {accum_loss:.4f}"
                        f"  val_loss [{colour}]{val_loss:.4f}[/{colour}]"
                        f"  lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    if val_loss < best_val:
                        best_val = val_loss

        # Final checkpoint at end of phase
        _save_checkpoint(steps)

    @staticmethod
    @torch.no_grad()
    def _eval_loss(
        model     : CoreGPT,
        val_ids   : torch.Tensor,
        n_batches : int   = 8,
        use_amp   : bool  = False,
        amp_dtype : torch.dtype = torch.float32,
    ) -> float:
        model.eval()
        losses = []
        for _ in range(n_batches):
            x, y = LLMTrainer._random_batch(
                val_ids, cfg.llm_context, cfg.llm_batch, cfg.device
            )
            with torch.autocast(device_type=cfg.device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))


# ── §7g  WebSearch – real-time web search and weather ────────────────────────

class WebSearch:
    """
    Provides real-time information by querying the web.

    - General search: DuckDuckGo (free, no API key, via duckduckgo_search package)
    - Weather: wttr.in (free, no API key, HTTP request)
    - Gracefully returns empty string if anything fails
    """

    # Questions that likely need current / real-world information
    SEARCH_TRIGGERS = (
        "today", "current", "agora", "hoje", "now", "latest", "recent",
        "news", "noticias", "notícias", "weather", "temperatura", "temperature",
        "tempo", "clima", "price", "preço", "stock", "who is", "quem é",
        "when did", "what happened", "tonight", "this week", "2025", "2026",
        "forecast", "previsão",
    )

    WEATHER_TRIGGERS = (
        "weather", "temperatura", "temperature", "tempo", "clima",
        "chuva", "rain", "sun", "sol", "frio", "cold", "hot", "quente",
        "forecast", "previsão", "graus", "degrees", "celsius",
    )

    @staticmethod
    def needs_search(question: str) -> bool:
        q = question.lower()
        return any(t in q for t in WebSearch.SEARCH_TRIGGERS)

    @staticmethod
    def is_weather(question: str) -> bool:
        q = question.lower()
        return any(t in q for t in WebSearch.WEATHER_TRIGGERS)

    @staticmethod
    def get_weather(location: str = "") -> str:
        """Query wttr.in for current weather (no API key needed)."""
        try:
            loc = location.strip().replace(" ", "+") if location else ""
            url = f"https://wttr.in/{loc}?format=3&lang=pt"
            req = urllib.request.Request(url, headers={"User-Agent": "CoreAI/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                return resp.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    @staticmethod
    def search(query: str, max_results: int = 3) -> str:
        """Search DuckDuckGo and return a short summary of results."""
        try:
            from duckduckgo_search import DDGS   # type: ignore
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""
            lines = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")[:250]
                lines.append(f"• {title}: {body}")
            return "\n".join(lines)
        except ImportError:
            return ""   # duckduckgo_search not installed
        except Exception:
            return ""

    @classmethod
    def enrich_prompt(cls, question: str) -> str:
        """
        Return additional context to inject into the prompt, or empty string.
        Tries weather first (fast), then general search.
        """
        extra = ""

        if cls.is_weather(question):
            # Try to extract a location from the question
            weather = cls.get_weather()   # use IP-based location as default
            if weather:
                extra = f"[Current weather: {weather}]\n\n"

        if not extra and cls.needs_search(question):
            results = cls.search(question)
            if results:
                extra = f"[Web search results:\n{results}]\n\n"

        return extra


# ── §7h  OllamaBackend – wraps a locally running Ollama server ───────────────

class OllamaBackend:
    """
    Talks to a locally running Ollama server (https://ollama.com).
    Exposes the same chat() / stream_chat() interface as LLMBackend.

    Install Ollama:
        Linux:   curl -fsSL https://ollama.com/install.sh | sh
        Windows: https://ollama.com/download
    Pull a model:
        ollama pull mistral       (7B, ~4 GB, recommended)
        ollama pull llama3        (8B, ~5 GB)
        ollama pull phi3          (3.8B, ~2.5 GB, fastest)
        ollama pull gemma2        (9B, ~6 GB)
    """

    def __init__(self, model: str | None = None) -> None:
        self.model    = model or cfg.ollama_model
        self.base_url = cfg.ollama_url.rstrip("/")
        self.history  : List[Tuple[str, str]] = []

    # ── Availability check ────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """Return True if Ollama is reachable at the configured URL."""
        try:
            url = cfg.ollama_url.rstrip("/") + "/api/tags"
            req = urllib.request.Request(url, headers={"User-Agent": "CoreAI/1.0"})
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    @classmethod
    def list_models(cls) -> List[str]:
        """Return list of locally available Ollama model names."""
        try:
            url = cfg.ollama_url.rstrip("/") + "/api/tags"
            req = urllib.request.Request(url, headers={"User-Agent": "CoreAI/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Prompt builder (same style as LLMBackend) ────────────────────────────

    def _build_prompt(self, question: str, web_context: str = "") -> str:
        system = (
            "You are CoreAI, a highly capable AI assistant running locally on the user's machine. "
            "You are helpful, precise, and honest. Answer in the same language as the question. "
            "Be concise but complete."
        )
        if web_context:
            system += f"\n\nReal-time information retrieved from the web:\n{web_context}"

        # Build chat messages for Ollama's /api/chat endpoint
        messages = [{"role": "system", "content": system}]
        for human, ai in self.history[-6:]:
            messages.append({"role": "user",      "content": human})
            messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": question})
        return json.dumps({
            "model"   : self.model,
            "messages": messages,
            "stream"  : False,
            "options" : {"temperature": 0.7, "num_predict": 400},
        })

    # ── Full response ─────────────────────────────────────────────────────────

    def chat(
        self,
        question        : str,
        use_web_search  : bool  = True,
        **_kw,
    ) -> str:
        web_context = WebSearch.enrich_prompt(question) if use_web_search else ""
        payload = self._build_prompt(question, web_context).encode()
        try:
            url = self.base_url + "/api/chat"
            req = urllib.request.Request(
                url,
                data   = payload,
                headers= {"Content-Type": "application/json", "User-Agent": "CoreAI/1.0"},
                method = "POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data   = json.loads(resp.read())
            answer = data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            answer = f"[Ollama error: {exc}]"
        self.history.append((question, answer))
        return answer

    # ── Streaming response ────────────────────────────────────────────────────

    def stream_chat(
        self,
        question       : str,
        use_web_search : bool = True,
        **_kw,
    ) -> Generator[str, None, None]:
        web_context = WebSearch.enrich_prompt(question) if use_web_search else ""
        # Build streaming payload
        data = json.loads(self._build_prompt(question, web_context))
        data["stream"] = True
        payload = json.dumps(data).encode()
        full = ""
        try:
            url = self.base_url + "/api/chat"
            req = urllib.request.Request(
                url,
                data   = payload,
                headers= {"Content-Type": "application/json", "User-Agent": "CoreAI/1.0"},
                method = "POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full += token
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            yield f"[Ollama error: {exc}]"
            full = f"[Ollama error: {exc}]"
        self.history.append((question, full))


# ── §7i  LLMBackend – inference wrapper + conversation manager ────────────────

class LLMBackend:
    """
    High-level inference interface for CoreGPT.

    Manages:
      - Loading the saved checkpoint and tokenizer
      - Conversation history (rolling window)
      - Prompt construction (system message + history + new turn)
      - Autoregressive generation with temperature / top-k controls
      - Enriching system-related questions with live system data
    """

    SYSTEM_MSG = (
        "You are CoreAI, a highly capable AI assistant. "
        "You answer any question clearly, helpfully, and concisely – "
        "technical, scientific, creative, or casual. "
        "When web search results are provided, use them to give accurate, up-to-date answers. "
        "When live system data is provided, use it to answer system-related questions. "
        "Always respond in the same language the user writes in.\n\n"
    )
    MAX_HISTORY = 6   # keep last N exchanges to avoid exceeding context window

    def __init__(self, model: CoreGPT, tok: Any) -> None:
        self.model   = model
        self.tok     = tok          # BPETokenizer or CharTokenizer
        self.history : List[Tuple[str, str]] = []

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "LLMBackend":
        """Load a trained CoreGPT checkpoint from disk."""
        if not cfg.llm_path.exists() or not cfg.tok_path.exists():
            raise FileNotFoundError(
                "LLM not found. Run:  python coreai.py train-llm"
            )
        ckpt  = torch.load(cfg.llm_path, map_location=cfg.device, weights_only=False)
        model = CoreGPT(
            vocab_size  = ckpt["vocab_size"],
            context_len = ckpt["context_len"],
            d_model     = ckpt["d_model"],
            n_heads     = ckpt["n_heads"],
            n_layers    = ckpt["n_layers"],
            dropout     = 0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(cfg.device).eval()

        # Load the correct tokenizer class based on what was used during training
        tok_type = ckpt.get("tokenizer_type", "char")
        if tok_type == "bpe":
            tok: Any = BPETokenizer.load(cfg.tok_path)
        else:
            tok = CharTokenizer.load(cfg.tok_path)

        arch = ckpt.get("arch", "core-gpt-v1")
        console.print(
            f"[green]CoreGPT loaded[/] – "
            f"{model.num_parameters:,} params  |  "
            f"vocab {tok.vocab_size}  |  "
            f"tokenizer [cyan]{tok_type}[/]  |  "
            f"arch [cyan]{arch}[/]"
        )
        return cls(model, tok)

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(self, question: str, web_context: str = "") -> str:
        """
        Assemble the full prompt:
          system message + web/weather context + live system data + history + question
        """
        parts = [self.SYSTEM_MSG]

        # 1. Web search / weather context (fetched by caller)
        if web_context:
            parts.append(web_context)

        # 2. Live system data for system-related questions
        syswords = ("cpu", "memory", "ram", "disk", "process", "load", "network",
                    "uptime", "usage", "service", "temperatura", "temperatura cpu")
        if any(w in question.lower() for w in syswords):
            v      = SystemMonitor.snapshot_vector()
            labels = SystemMonitor.FEATURE_NAMES
            sysinfo = "  ".join(f"{l}={v[i]:.1f}" for i, l in enumerate(labels))
            parts.append(f"[Live system state: {sysinfo}]\n\n")

        # 3. Rolling conversation history
        for q, a in self.history[-self.MAX_HISTORY:]:
            parts.append(LLMTrainer.format_qa(q, a))

        # 4. New turn
        parts.append(LLMTrainer.HUMAN_TAG + question + "\n" + LLMTrainer.AI_TAG)
        return "".join(parts)

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _generate_answer(
        self,
        question          : str,
        temperature       : float,
        top_k             : int,
        top_p             : float,
        repetition_penalty: float,
        max_new_tokens    : int,
        web_context       : str = "",
    ) -> str:
        """Core generation — builds prompt, runs model, extracts answer."""
        prompt = self._build_prompt(question, web_context=web_context)
        enc    = self.tok.encode(prompt)
        max_prompt = cfg.llm_context - max_new_tokens
        if len(enc) > max_prompt:
            enc = enc[-max_prompt:]

        idx     = torch.tensor([enc], dtype=torch.long, device=cfg.device)
        out_ids = self.model.generate(
            idx,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k,
            top_p              = top_p,
            repetition_penalty = repetition_penalty,
        )
        full_text = self.tok.decode(out_ids[0].tolist(), skip_special=True)

        ai_marker = LLMTrainer.AI_TAG
        pos = full_text.rfind(ai_marker)
        if pos != -1:
            answer = full_text[pos + len(ai_marker):]
        else:
            prompt_text = self.tok.decode(enc, skip_special=True)
            answer = full_text[len(prompt_text):]

        stop = answer.find(LLMTrainer.HUMAN_TAG)
        if stop != -1:
            answer = answer[:stop]
        return answer.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        question          : str,
        temperature       : float = 0.75,
        top_k             : int   = 40,
        top_p             : float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens    : int   = 300,
        use_web_search    : bool  = True,
    ) -> str:
        """Generate a response, optionally enriched with web search / weather."""
        web_context = WebSearch.enrich_prompt(question) if use_web_search else ""
        answer = self._generate_answer(
            question, temperature, top_k, top_p,
            repetition_penalty, max_new_tokens, web_context,
        )
        self.history.append((question, answer))
        return answer

    def stream_chat(
        self,
        question          : str,
        temperature       : float = 0.75,
        top_k             : int   = 40,
        top_p             : float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens    : int   = 300,
        use_web_search    : bool  = True,
    ) -> Generator[str, None, None]:
        """
        Generate a response and yield it word-by-word for streaming UIs.
        The full answer is also appended to conversation history.
        """
        web_context = WebSearch.enrich_prompt(question) if use_web_search else ""
        answer = self._generate_answer(
            question, temperature, top_k, top_p,
            repetition_penalty, max_new_tokens, web_context,
        )
        self.history.append((question, answer))

        # Stream word by word with punctuation attached
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


# ══════════════════════════════════════════════════════════════════════════════
# §8  STANDALONE PREDICTION UTILITY  (usage example / import API)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_predict(model_path: str | Path, dataset_index: int = 0) -> int:
    """
    Load a saved CNN checkpoint and predict the digit for one MNIST test image.

    Usage example
    -------------
    >>> from coreai import load_and_predict
    >>> predicted = load_and_predict("models/mnist_cnn.pth", dataset_index=42)
    >>> print(f"Predicted digit: {predicted}")

    Parameters
    ----------
    model_path    : str | Path – path to the .pth weights file
    dataset_index : int        – index in the MNIST test set (0 – 9 999)

    Returns
    -------
    int – predicted class label (0 – 9)
    """
    model = DigitCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = torchvision.datasets.MNIST(
        str(DATA_DIR), train=False, download=True, transform=transform
    )
    img, true_label = test_ds[dataset_index]

    with torch.no_grad():
        pred = model(img.unsqueeze(0)).argmax(1).item()

    console.print(
        f"Index [cyan]{dataset_index}[/] — "
        f"True: [bold]{true_label}[/]  "
        f"Predicted: [bold {'green' if pred == true_label else 'red'}]{pred}[/]"
    )
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# §8  INTERACTIVE MAIN MENU
# ══════════════════════════════════════════════════════════════════════════════

_BANNER = r"""
[bold cyan]
  ██╗     ██╗███╗   ██╗██╗   ██╗██╗  ██╗ █████╗ ██╗
  ██║     ██║████╗  ██║██║   ██║╚██╗██╔╝██╔══██╗██║
  ██║     ██║██╔██╗ ██║██║   ██║ ╚███╔╝ ███████║██║
  ██║     ██║██║╚██╗██║██║   ██║ ██╔██╗ ██╔══██║██║
  ███████╗██║██║ ╚████║╚██████╔╝██╔╝ ██╗██║  ██║██║
  ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝[/]
  [dim]AI-Powered Linux Intelligence & Control Platform  v1.0.0[/]
"""


def _run_chat_session(agent: AIAgent) -> None:
    """
    Dedicated conversational REPL powered entirely by the built-in CoreGPT LLM.
    Every message goes directly to LLMBackend.chat(); no intent routing.
    """
    if agent.llm is None:
        console.print(
            "[red]Built-in LLM is not trained yet.[/]\n"
            "Run option [bold]1[/] (Train ALL) or [bold]2[/] (Train LLM only) first."
        )
        return

    console.print(
        Panel(
            "[bold cyan]LinuxAI Chat[/]  –  powered by CoreGPT (built from scratch)\n"
            "[dim]Type your question in any language. "
            "Type [bold]clear[/] to reset history, [bold]quit[/] to exit.[/]",
            border_style="magenta",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]you >[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[green]Chat closed.[/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", ":q"):
            console.print("[green]Chat closed.[/]")
            break
        if user_input.lower() == "clear":
            agent.llm.history.clear()
            console.print("[dim]History cleared.[/]")
            continue

        console.print("[dim]Generating…[/]")
        try:
            answer = agent.llm.chat(user_input)
            console.print(
                Panel(answer, title="[bold cyan]LinuxAI[/]", border_style="cyan")
            )
        except Exception as exc:
            console.print(f"[red]LLM error: {exc}[/]")


def interactive_menu() -> None:
    console.print(_BANNER)
    agent = AIAgent()   # pre-load CNN once for the session

    MENU = [
        ("1", "Train ALL models  (CNN + LLM + sklearn)"),
        ("2", "Train built-in LLM only  (CoreGPT from scratch)"),
        ("3", "Chat with built-in LLM  (answer any question)"),
        ("4", "AI agent shell  (system control + LLM)"),
        ("5", "Live system dashboard"),
        ("6", "Process table (AI-annotated)"),
        ("7", "File browser"),
        ("8", "Systemd services"),
        ("9", "Predict an MNIST digit"),
        ("0", "Run a raw shell command"),
        ("q", "Quit"),
    ]

    while True:
        tbl = Table(show_header=False, box=box.ROUNDED, border_style="cyan", width=52)
        tbl.add_column("Key", style="bold yellow", width=4)
        tbl.add_column("Action")
        for k, v in MENU:
            tbl.add_row(k, v)
        console.print(tbl)

        choice = Prompt.ask("[bold yellow]Select[/]").strip().lower()

        if choice == "1":
            ModelTrainer.train_cnn()
            ModelTrainer.train_anomaly_detector()
            ModelTrainer.train_process_classifier()
            LLMTrainer.train()
            agent._try_load_cnn()
            agent._try_load_llm()

        elif choice == "2":
            LLMTrainer.train()
            agent._try_load_llm()

        elif choice == "3":
            _run_chat_session(agent)

        elif choice == "4":
            agent.run()

        elif choice == "5":
            SystemMonitor.live_dashboard(duration=3600)

        elif choice == "6":
            console.print(SystemMonitor.process_table())

        elif choice == "7":
            path = Prompt.ask("Path to list", default=".")
            console.print(SystemController.list_dir(path))

        elif choice == "8":
            console.print(SystemController.list_services())

        elif choice == "9":
            if not cfg.cnn_path.exists():
                console.print("[red]No trained CNN found – run option 1 first.[/]")
            else:
                try:
                    idx = int(Prompt.ask("MNIST test-set index", default="0"))
                except ValueError:
                    idx = 0
                load_and_predict(cfg.cnn_path, idx)

        elif choice == "0":
            cmd = Prompt.ask("Shell command")
            SystemController.run_command(cmd)

        elif choice in ("q", "quit", "exit"):
            console.print("[green]Goodbye![/]")
            break


# ══════════════════════════════════════════════════════════════════════════════
# §9  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="coreai",
        description="LinuxAI – AI-Powered Linux Intelligence & Control Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python coreai.py                   # interactive menu\n"
            "  python coreai.py train             # train all models\n"
            "  python coreai.py monitor           # live dashboard\n"
            "  python coreai.py agent             # AI shell\n"
            "  python coreai.py predict --index 7 # classify digit\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train",        help="Train / retrain ALL models (CNN + LLM + sklearn)")
    sub.add_parser("train-llm",    help="Train only the built-in CoreGPT LLM from scratch")
    sub.add_parser("finetune-only", help="Redo only the fine-tuning phase (uses saved pretrain weights)")
    sub.add_parser("chat",         help="Chat with the built-in LLM (answer any question)")
    sub.add_parser("monitor",   help="Launch the live system dashboard")
    sub.add_parser("agent",     help="Open the natural-language AI agent shell")
    sub.add_parser("control",   help="One-shot system snapshot + control panel")

    p_pred = sub.add_parser("predict", help="Predict an MNIST digit")
    p_pred.add_argument(
        "--index", "-i", type=int, default=0,
        help="Index in the MNIST test set (0–9999, default: 0)",
    )

    args = parser.parse_args()

    if args.cmd == "train":
        ModelTrainer.train_cnn()
        ModelTrainer.train_anomaly_detector()
        ModelTrainer.train_process_classifier()
        LLMTrainer.train()

    elif args.cmd == "train-llm":
        LLMTrainer.train()

    elif args.cmd == "finetune-only":
        if not cfg.llm_pretrain_path.exists():
            console.print(
                "[red]No pretrained weights found.[/] "
                "Run [cyan]python coreai.py train-llm[/] first "
                "(pretrain weights are saved automatically after pretraining)."
            )
            sys.exit(1)
        LLMTrainer.train(skip_pretrain=True)

    elif args.cmd == "chat":
        agent = AIAgent()
        _run_chat_session(agent)

    elif args.cmd == "monitor":
        SystemMonitor.live_dashboard(duration=3600)

    elif args.cmd == "agent":
        AIAgent().run()

    elif args.cmd == "control":
        console.print(SystemMonitor.system_summary())
        console.print(SystemMonitor.process_table())
        console.print(SystemController.list_dir("."))

    elif args.cmd == "predict":
        if not cfg.cnn_path.exists():
            console.print(
                "[red]No trained CNN found.[/] "
                "Run [cyan]python coreai.py train[/] first."
            )
            sys.exit(1)
        load_and_predict(cfg.cnn_path, args.index)

    else:
        # Default: launch the interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
