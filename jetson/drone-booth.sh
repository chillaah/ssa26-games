#!/usr/bin/env bash
# Desktop launcher for the drone booth (Caught by Scouts) on the Jetson + SIYI A8 mini.
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/drone-booth" || exit 1

if pgrep -f -- "-m dronebooth" >/dev/null; then
  zenity --question --title "Drone Booth" --text "The drone booth is already running. Stop it and start again?" || exit 0
  pkill -f -- "-m dronebooth"
  sleep 2
fi

echo "Starting the drone booth with configs/jetson.yaml (log: /tmp/dronebooth.log)"
echo "Press q or Esc in the booth window to quit, f to toggle fullscreen."
.venv/bin/python -m dronebooth --config configs/jetson.yaml --fullscreen 2>&1 | tee /tmp/dronebooth.log
echo
read -r -p "Drone booth stopped. Press Enter to close this window."
