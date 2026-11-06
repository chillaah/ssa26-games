#!/usr/bin/env bash
# Desktop launcher for Out of Frame on the Jetson: pick a Jetson + SIYI config, then run the game.
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
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
  pkill -f -- "-m outofframe"
  sleep 2
fi

echo "Starting Out of Frame with configs/$cfg.yaml (log: /tmp/outofframe.log)"
echo "Press q or Esc in the game window to quit, f to toggle fullscreen."
.venv/bin/python -m outofframe --config "configs/$cfg.yaml" --fullscreen 2>&1 | tee /tmp/outofframe.log
echo
read -r -p "Out of Frame stopped. Press Enter to close this window."
