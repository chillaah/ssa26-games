#!/usr/bin/env bash
# Desktop launcher for the drone booth (Caught by Scouts) on the Jetson + SIYI A8 mini.
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO/drone-booth" || exit 1

if pgrep -f -- "-m dronebooth" >/dev/null; then
  zenity --question --title "Drone Booth" --text "The drone booth is already running. Stop it and start again?" || exit 0
  pkill -f -- "-m dronebooth"  # SIGTERM: it shuts down cleanly (camera, share server, window)
  for _ in $(seq 20); do pgrep -f -- "-m dronebooth" >/dev/null || break; sleep 0.5; done
fi

echo "Starting the drone booth with configs/jetson.yaml (log: /tmp/dronebooth.log)"
echo "Close: the X button at the top right (tap twice), or q / Esc. f toggles fullscreen."
# tee ignores hang-ups so the cleanup log still reaches the file if this terminal is closed.
.venv/bin/python -m dronebooth --config configs/jetson.yaml --fullscreen 2>&1 | (trap '' HUP INT TERM; exec tee /tmp/dronebooth.log)
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo
  read -r -p "The drone booth stopped with an error (exit code $rc, log in /tmp/dronebooth.log). Press Enter to close."
fi
