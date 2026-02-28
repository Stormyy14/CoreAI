#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CoreAI — Build Fedora/RHEL/openSUSE .rpm package
#
# Requirements:  rpm-build   (dnf install rpm-build)
#
# Usage:  bash make_rpm.sh
# Output: ~/rpmbuild/RPMS/x86_64/coreai-1.0.0-1.x86_64.rpm
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(realpath "$SCRIPT_DIR/../..")"

APP="coreai"
VERSION="1.0.0"
RELEASE="1"

# ── Check rpm-build ───────────────────────────────────────────────────────────
if ! command -v rpmbuild &>/dev/null; then
  echo "rpm-build not found. Installing…"
  if command -v dnf &>/dev/null; then
    sudo dnf install -y rpm-build
  else
    echo "Please install rpm-build manually." && exit 1
  fi
fi

# ── Prepare rpmbuild tree ─────────────────────────────────────────────────────
RPMBUILD="$HOME/rpmbuild"
mkdir -p "$RPMBUILD"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# ── Create source tarball ─────────────────────────────────────────────────────
TARNAME="${APP}-${VERSION}"
TARBALL="$RPMBUILD/SOURCES/${TARNAME}.tar.gz"

echo "Creating source tarball…"
tar -czf "$TARBALL" \
  --transform "s|^|${TARNAME}/|" \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='build_deb' \
  --exclude='venv' \
  --exclude='install' \
  -C "$SRC_DIR" .

# ── Write spec file ───────────────────────────────────────────────────────────
cat > "$RPMBUILD/SPECS/coreai.spec" <<SPEC
Name:           coreai
Version:        ${VERSION}
Release:        ${RELEASE}%{?dist}
Summary:        Local AI Platform — CoreAI MiniGPT + Ollama
License:        MIT
URL:            https://github.com/yourname/coreai
Source0:        %{name}-%{version}.tar.gz

Requires:       python3 >= 3.10
Requires:       python3-pip

BuildArch:      noarch

%description
CoreAI is a local AI platform that lets you chat with a from-scratch
MiniGPT model or any Ollama-compatible model. Runs 100% locally with
no cloud, no API keys, and a sleek web UI.

%prep
%setup -q

%install
rm -rf %{buildroot}

# Application files
install -d %{buildroot}/opt/coreai
cp -r . %{buildroot}/opt/coreai/

# Launcher
install -d %{buildroot}/usr/local/bin
cat > %{buildroot}/usr/local/bin/coreai <<'EOF'
#!/usr/bin/env bash
cd /opt/coreai
if [[ ! -d venv ]]; then
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt --quiet
fi
source venv/bin/activate
case "\${1:-}" in
    train) python linux_ai.py train-llm ;;
    chat)  python linux_ai.py chat ;;
    *)     echo "Starting CoreAI at http://localhost:8080"
           python server.py ;;
esac
EOF
chmod 0755 %{buildroot}/usr/local/bin/coreai

# Desktop icon
install -d %{buildroot}/usr/share/icons/hicolor/256x256/apps
cat > %{buildroot}/usr/share/icons/hicolor/256x256/apps/coreai.svg <<'SVGEOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="g" x1="0%%" y1="0%%" x2="100%%" y2="100%%">
      <stop offset="0%%" style="stop-color:#6d8ef7"/>
      <stop offset="100%%" style="stop-color:#a78bfa"/>
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="48" fill="url(#g)"/>
  <text x="128" y="176" font-family="sans-serif" font-size="140" font-weight="900"
        fill="white" text-anchor="middle">C</text>
</svg>
SVGEOF

# .desktop entry
install -d %{buildroot}/usr/share/applications
cat > %{buildroot}/usr/share/applications/coreai.desktop <<'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=CoreAI
GenericName=Local AI Assistant
Comment=Chat with CoreAI or Ollama — 100%% local, no cloud
Exec=bash -c "coreai & sleep 2 && xdg-open http://localhost:8080"
Icon=coreai
Terminal=false
Categories=Science;Education;Utility;
Keywords=AI;LLM;chat;assistant;
EOF

# systemd service
install -d %{buildroot}/usr/lib/systemd/system
cat > %{buildroot}/usr/lib/systemd/system/coreai.service <<'EOF'
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

%files
/opt/coreai/
/usr/local/bin/coreai
/usr/share/applications/coreai.desktop
/usr/share/icons/hicolor/256x256/apps/coreai.svg
/usr/lib/systemd/system/coreai.service

%post
update-desktop-database /usr/share/applications 2>/dev/null || true
systemctl daemon-reload 2>/dev/null || true
echo "CoreAI installed. Run 'coreai' to start."

%preun
systemctl stop    coreai 2>/dev/null || true
systemctl disable coreai 2>/dev/null || true

%postun
systemctl daemon-reload 2>/dev/null || true

%changelog
* $(date '+%a %b %d %Y') CoreAI Project <coreai@localhost> - ${VERSION}-${RELEASE}
- Initial package
SPEC

# ── Build RPM ─────────────────────────────────────────────────────────────────
echo "Building RPM…"
rpmbuild -bb "$RPMBUILD/SPECS/coreai.spec"

RPM_FILE=$(find "$RPMBUILD/RPMS" -name "coreai-*.rpm" | head -1)
echo ""
echo "  Package built: $RPM_FILE"
echo "  Install with:  sudo dnf install '$RPM_FILE'"
echo "                 # or: sudo rpm -ivh '$RPM_FILE'"
echo ""
