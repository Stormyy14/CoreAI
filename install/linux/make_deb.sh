#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CoreAI — Build Debian/Ubuntu .deb package
#
# Requirements:  dpkg-deb (part of dpkg, pre-installed on Debian/Ubuntu)
#                OR: fpm  (gem install fpm / pip install fpm)
#
# Usage:  bash make_deb.sh
# Output: coreai_1.0.0_amd64.deb  (in the CoreAI root)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(realpath "$SCRIPT_DIR/../..")"

APP="coreai"
VERSION="1.0.0"
ARCH="amd64"
PKG_NAME="${APP}_${VERSION}_${ARCH}"
BUILD_DIR="$SRC_DIR/build_deb/$PKG_NAME"

echo "Building $PKG_NAME.deb …"

# ── Prepare package tree ──────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p \
  "$BUILD_DIR/DEBIAN" \
  "$BUILD_DIR/opt/coreai" \
  "$BUILD_DIR/usr/local/bin" \
  "$BUILD_DIR/usr/share/applications" \
  "$BUILD_DIR/usr/share/icons/hicolor/256x256/apps" \
  "$BUILD_DIR/lib/systemd/system"

# ── Copy source ───────────────────────────────────────────────────────────────
rsync -a --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
  --exclude='build_deb' --exclude='venv' --exclude='install' \
  "$SRC_DIR/" "$BUILD_DIR/opt/coreai/"

# ── Launcher ──────────────────────────────────────────────────────────────────
cat > "$BUILD_DIR/usr/local/bin/coreai" <<'EOF'
#!/usr/bin/env bash
cd /opt/coreai
if [[ ! -d venv ]]; then
    echo "Creating virtual environment (first run)…"
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt --quiet
fi
source venv/bin/activate
case "${1:-}" in
    train) python coreai.py train-llm ;;
    chat)  python coreai.py chat ;;
    *)
        echo "Starting CoreAI at http://localhost:8080"
        python server.py ;;
esac
EOF
chmod 0755 "$BUILD_DIR/usr/local/bin/coreai"

# ── Desktop icon ──────────────────────────────────────────────────────────────
cat > "$BUILD_DIR/usr/share/icons/hicolor/256x256/apps/coreai.svg" <<'SVGEOF'
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

# ── .desktop entry ────────────────────────────────────────────────────────────
cat > "$BUILD_DIR/usr/share/applications/coreai.desktop" <<EOF
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
Keywords=AI;LLM;chat;assistant;
EOF

# ── systemd service ───────────────────────────────────────────────────────────
cat > "$BUILD_DIR/lib/systemd/system/coreai.service" <<EOF
[Unit]
Description=CoreAI Local AI Platform
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/coreai
ExecStartPre=/bin/bash -c 'test -d /opt/coreai/venv || (python3 -m venv /opt/coreai/venv && /opt/coreai/venv/bin/pip install -r /opt/coreai/requirements.txt -q)'
ExecStart=/opt/coreai/venv/bin/python /opt/coreai/server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ── DEBIAN/control ────────────────────────────────────────────────────────────
cat > "$BUILD_DIR/DEBIAN/control" <<EOF
Package: coreai
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.10), python3-venv, python3-pip
Maintainer: CoreAI Project <coreai@localhost>
Description: CoreAI Local AI Platform
 Chat with a from-scratch MiniGPT or Ollama models.
 Runs 100% locally — no cloud, no API keys required.
 Includes a FastAPI web server with SSE streaming and
 real-time system stats.
EOF

# ── DEBIAN/postinst ───────────────────────────────────────────────────────────
cat > "$BUILD_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database /usr/share/applications
fi
if command -v systemctl &>/dev/null; then
    systemctl daemon-reload
fi
echo "CoreAI installed. Run 'coreai' to start."
exit 0
EOF
chmod 0755 "$BUILD_DIR/DEBIAN/postinst"

# ── DEBIAN/prerm ──────────────────────────────────────────────────────────────
cat > "$BUILD_DIR/DEBIAN/prerm" <<'EOF'
#!/bin/bash
systemctl stop    coreai 2>/dev/null || true
systemctl disable coreai 2>/dev/null || true
exit 0
EOF
chmod 0755 "$BUILD_DIR/DEBIAN/prerm"

# ── Build .deb ────────────────────────────────────────────────────────────────
OUT="$SRC_DIR/${PKG_NAME}.deb"
dpkg-deb --build --root-owner-group "$BUILD_DIR" "$OUT"
rm -rf "$SRC_DIR/build_deb"

echo ""
echo "  Package built: $OUT"
echo "  Install with:  sudo dpkg -i $OUT"
echo "                 sudo apt-get install -f   # fix deps if needed"
echo ""
