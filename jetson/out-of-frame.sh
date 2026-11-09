#!/usr/bin/env bash
# Desktop launcher for Out of Frame on the Jetson: pick a Jetson + SIYI config, then run the game.
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
# Event time zone for photo timestamps, collage dates and logs, whatever the system is set to.
export TZ=Asia/Singapore
cd "$REPO/out-of-frame" || exit 1
DEFAULT=jetson_steady

describe() {
  case "$1" in
    jetson_steady)  echo "Gimbal leans toward the box, picture held still" ;;
    jetson_follow)  echo "Gimbal leans toward the box, picture moves slightly" ;;
    jetson_virtual) echo "Gimbal holds still, the box glides over the view" ;;
    jetson)         echo "Gimbal swings to random angles, the whole image is the photo" ;;
    jetson_table)   echo "Like jetson, aimed for a camera upright on a desk" ;;
    *)              sed -n '1s/^# *//p' "configs/$1.yaml" ;;
  esac
}

# Default first and pre-selected, then the rest of configs/jetson*.yaml.
rows=()
if [ -f "configs/$DEFAULT.yaml" ]; then
  rows+=(TRUE "$DEFAULT" "$(describe "$DEFAULT")")
fi
for f in configs/jetson*.yaml; do
  n="$(basename "$f" .yaml)"
  [ "$n" = "$DEFAULT" ] && continue
  rows+=(FALSE "$n" "$(describe "$n")")
done

cfg="$(zenity --list --radiolist --title "Out of Frame" --text "Pick a setup (Jetson + SIYI A8 mini):" \
  --column "" --column "Config" --column "What it does" --print-column 2 \
  --width 760 --height 340 "${rows[@]}")" || exit 0
[ -n "$cfg" ] || exit 0

if pgrep -f -- "-m outofframe" >/dev/null; then
  zenity --question --title "Out of Frame" --text "Out of Frame is already running. Stop it and start again?" || exit 0
  pkill -f -- "-m outofframe"  # SIGTERM: it shuts down cleanly (camera, gimbal, window)
  for _ in $(seq 20); do pgrep -f -- "-m outofframe" >/dev/null || break; sleep 0.5; done
fi

LOGDIR="$HOME/ssa26-logs"  # kept across reboots (unlike /tmp)
mkdir -p "$LOGDIR"
LOG="$LOGDIR/outofframe.log"
# Keep at most ~2 x 50 MB of log.
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt 52428800 ]; then
  mv -f "$LOG" "$LOG.1"
fi
ln -sf "$LOG" /tmp/outofframe.log  # old location still works

# The camera may forget its stream setting after a power cycle: make sure it is 1080p.
python3 "$REPO/jetson/camera_stream.py" 1920 1080 2000 2>&1 | tee -a "$LOG"

echo "Starting Out of Frame with configs/$cfg.yaml (log: $LOG)"
echo "Close: the X button at the top right (tap twice), or q / Esc. f toggles fullscreen."
# Restart after a crash or a camera stall (non-zero exit); stop after a deliberate quit (exit 0).
# tee ignores hang-ups so the cleanup log still reaches the file if this terminal is closed.
restarts=0
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start" >> "$LOG"
  .venv/bin/python -m outofframe --config "configs/$cfg.yaml" --fullscreen 2>&1 | (trap '' HUP INT TERM; exec tee -a "$LOG")
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && break
  restarts=$((restarts + 1))
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') exited with code $rc; restart $restarts in 3 s (Ctrl+C to stop)" | tee -a "$LOG"
  sleep 3 || break
done
