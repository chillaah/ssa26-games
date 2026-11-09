#!/usr/bin/env bash
# Desktop launcher for the Scouts Drone Booth on the Jetson + SIYI A8 mini.
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
# Event time zone for photo timestamps, collage dates and logs, whatever the system is set to.
export TZ=Asia/Singapore
cd "$REPO/drone-booth" || exit 1

if pgrep -f -- "-m dronebooth" >/dev/null; then
  zenity --question --title "Drone Booth" --text "The drone booth is already running. Stop it and start again?" || exit 0
  pkill -f -- "-m dronebooth"  # SIGTERM: it shuts down cleanly (camera, share server, window)
  for _ in $(seq 20); do pgrep -f -- "-m dronebooth" >/dev/null || break; sleep 0.5; done
fi

LOGDIR="$HOME/ssa26-logs"  # kept across reboots (unlike /tmp)
mkdir -p "$LOGDIR"
LOG="$LOGDIR/dronebooth.log"
# Keep at most ~2 x 50 MB of log.
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt 52428800 ]; then
  mv -f "$LOG" "$LOG.1"
fi
ln -sf "$LOG" /tmp/dronebooth.log  # old location still works

# The camera may forget its stream setting after a power cycle: make sure it is 1080p.
python3 "$REPO/jetson/camera_stream.py" 1920 1080 2000 2>&1 | tee -a "$LOG"

echo "Starting the Drone Booth with configs/jetson.yaml (log: $LOG)"
echo "Close: the X button at the top right (tap twice), or q / Esc. f toggles fullscreen."
# Restart after a crash or a camera stall (non-zero exit); stop after a deliberate quit (exit 0).
# tee ignores hang-ups so the cleanup log still reaches the file if this terminal is closed.
restarts=0
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start" >> "$LOG"
  .venv/bin/python -m dronebooth --config configs/jetson.yaml --fullscreen 2>&1 | (trap '' HUP INT TERM; exec tee -a "$LOG")
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && break
  restarts=$((restarts + 1))
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') exited with code $rc; restart $restarts in 3 s (Ctrl+C to stop)" | tee -a "$LOG"
  sleep 3 || break
done
