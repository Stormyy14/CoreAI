#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CoreAI — Linux Installer (Fedora / Debian / Ubuntu / Arch)
# Usage:  sudo bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP="coreai"
APP_NAME="CoreAI"
APP_VERSION="1.0.0"
INSTALL_DIR="/opt/coreai"
BIN_DIR="/usr/local/bin"
ICON_DIR="/usr/share/icons/hicolor/256x256/apps"
DESKTOP_DIR="/usr/share/applications"
VENV_DIR="$INSTALL_DIR/venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Check root ────────────────────────────────────────────────────────────────
[[ "$EUID" -ne 0 ]] && die "Please run as root: sudo bash install.sh"

# ── Find source directory ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(realpath "$SCRIPT_DIR/../..")"   # CoreAI root

[[ -f "$SRC_DIR/coreai.py" ]] || die "coreai.py not found in $SRC_DIR"
[[ -f "$SRC_DIR/server.py"   ]] || die "server.py not found in $SRC_DIR"

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   CoreAI Linux Installer v${APP_VERSION}          ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# ── Detect OS & package manager ───────────────────────────────────────────────
if   command -v dnf  &>/dev/null; then PKG="dnf";  DISTRO="fedora"
elif command -v apt  &>/dev/null; then PKG="apt";  DISTRO="debian"
elif command -v pacman &>/dev/null; then PKG="pacman"; DISTRO="arch"
else warn "Unknown package manager — skipping system packages"; PKG="none"; DISTRO="unknown"
fi
info "Detected: $DISTRO (package manager: $PKG)"

# ── Install Python 3 if missing ───────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  info "Installing Python 3…"
  case "$PKG" in
    dnf)    dnf install -y python3 python3-pip ;;
    apt)    apt update -qq && apt install -y python3 python3-pip python3-venv ;;
    pacman) pacman -Sy --noconfirm python python-pip ;;
    *)      die "Python 3 not found. Please install it manually." ;;
  esac
fi
PYTHON=$(command -v python3)
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
ok "Python $PY_VER found at $PYTHON"

# ── Copy application files ────────────────────────────────────────────────────
info "Installing to $INSTALL_DIR…"
mkdir -p "$INSTALL_DIR"
rsync -a --delete \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='install' \
  --exclude='venv' \
  "$SRC_DIR/" "$INSTALL_DIR/"
ok "Application files copied"

# ── Create Python virtualenv ──────────────────────────────────────────────────
info "Creating virtual environment…"
$PYTHON -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

info "Installing Python dependencies (this may take a few minutes)…"
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt" --quiet \
  || warn "Some dependencies failed — try running pip manually inside $VENV_DIR"
ok "Dependencies installed"

# ── Launcher script ───────────────────────────────────────────────────────────
cat > "$BIN_DIR/coreai" <<EOF
#!/usr/bin/env bash
# CoreAI launcher
cd "$INSTALL_DIR"
source "$VENV_DIR/bin/activate"
if [[ "\$1" == "train" ]]; then
    python coreai.py train-llm
elif [[ "\$1" == "chat" ]]; then
    python coreai.py chat
else
    echo "Starting CoreAI web server at http://localhost:8080"
    python server.py
fi
EOF
chmod +x "$BIN_DIR/coreai"
ok "Launcher created at $BIN_DIR/coreai"

# ── Desktop icon (SVG) ────────────────────────────────────────────────────────
mkdir -p "$ICON_DIR"
cat > "$ICON_DIR/coreai.svg" <<'SVGEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#6d8ef7"/>
      <stop offset="100%" style="stop-color:#a78bfa"/>
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="48" fill="url(#g)"/>
  <text x="128" y="176" font-family="sans-serif" font-size="140" font-weight="900"
        fill="white" text-anchor="middle">C</text>
</svg>
SVGEOF
ok "Icon created"

# ── .desktop entry ────────────────────────────────────────────────────────────
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/coreai.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CoreAI
GenericName=Local AI Assistant
Comment=Chat with CoreAI or Ollama — 100% local, no cloud
Exec=bash -c "coreai & sleep 2 && xdg-open http://localhost:8080"
Icon=coreai
Terminal=false
Categories=Science;Education;Utility;
Keywords=AI;LLM;chat;assistant;ollama;
StartupWMClass=coreai
EOF
if command -v update-desktop-database &>/dev/null; then
  update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi
ok ".desktop entry created"

# ── systemd user service (optional) ──────────────────────────────────────────
SERVICE_DIR="/etc/systemd/system"
cat > "$SERVICE_DIR/coreai.service" <<EOF
[Unit]
Description=CoreAI Local AI Platform
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
if command -v systemctl &>/dev/null; then
  systemctl daemon-reload
  info "systemd service installed. Enable with:  sudo systemctl enable --now coreai"
fi
ok "systemd service created at $SERVICE_DIR/coreai.service"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   CoreAI installed successfully!          ║"
echo "  ╠═══════════════════════════════════════════╣"
echo "  ║  Start:   coreai                          ║"
echo "  ║  Train:   coreai train                    ║"
echo "  ║  Chat:    coreai chat                     ║"
echo "  ║  Web UI:  http://localhost:8080           ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""
