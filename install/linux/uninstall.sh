#!/usr/bin/env bash
# CoreAI — Uninstaller
set -euo pipefail

[[ "$EUID" -ne 0 ]] && echo "Please run as root: sudo bash uninstall.sh" && exit 1

echo "Removing CoreAI…"

systemctl stop    coreai 2>/dev/null || true
systemctl disable coreai 2>/dev/null || true

rm -f /usr/local/bin/coreai
rm -f /usr/share/applications/coreai.desktop
rm -f /usr/share/icons/hicolor/256x256/apps/coreai.svg
rm -f /etc/systemd/system/coreai.service
systemctl daemon-reload 2>/dev/null || true

read -rp "Remove application data in /opt/coreai? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    rm -rf /opt/coreai
    echo "Application data removed."
else
    echo "Application data kept at /opt/coreai"
fi

echo "CoreAI uninstalled."
