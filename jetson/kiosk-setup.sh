#!/usr/bin/env bash
# Keep the Jetson's screen on and free of pop-ups for days of unattended running:
# no screen blanking, lock screen, dimming, notification banners or software-update pop-ups.
# User settings only (no sudo). Saves the current values first; undo with:  ./jetson/kiosk-setup.sh --undo
set -u
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
BACKUP="$HOME/ssa26-logs/kiosk-before.txt"
AUTOSTART="$HOME/.config/autostart"
mkdir -p "$(dirname "$BACKUP")" "$AUTOSTART"

SETTINGS=(
  "org.gnome.desktop.session idle-delay uint32 0"
  "org.gnome.desktop.screensaver lock-enabled false"
  "org.gnome.desktop.screensaver idle-activation-enabled false"
  "org.gnome.desktop.screensaver ubuntu-lock-on-suspend false"
  "org.gnome.settings-daemon.plugins.power idle-dim false"
  "org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'"
  "org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'"
  "org.gnome.desktop.notifications show-banners false"
  "org.gnome.software download-updates false"
  "org.gnome.software allow-updates false"
)
POPUPS=(update-notifier.desktop)  # system autostart entries hidden for this user

if [ "${1:-}" = "--undo" ]; then
  [ -f "$BACKUP" ] || { echo "No backup at $BACKUP"; exit 1; }
  while read -r schema key value; do
    gsettings set "$schema" "$key" "$value" && echo "restored $schema $key = $value"
  done < "$BACKUP"
  for f in "${POPUPS[@]}"; do rm -f "$AUTOSTART/$f"; done
  exit 0
fi

if [ ! -f "$BACKUP" ]; then  # keep the very first backup
  for s in "${SETTINGS[@]}"; do
    read -r schema key _ <<< "$s"
    v=$(gsettings get "$schema" "$key" 2>/dev/null) && echo "$schema $key $v" >> "$BACKUP"
  done
  echo "Saved current settings to $BACKUP"
fi
for s in "${SETTINGS[@]}"; do
  read -r schema key value <<< "$s"
  value=${value#uint32 }
  if gsettings writable "$schema" "$key" >/dev/null 2>&1; then
    gsettings set "$schema" "$key" "$value" && echo "set $schema $key = $value"
  fi
done
for f in "${POPUPS[@]}"; do
  if [ -f "/etc/xdg/autostart/$f" ]; then
    printf '[Desktop Entry]\nHidden=true\n' > "$AUTOSTART/$f" && echo "hid autostart $f"
  fi
done
echo "Done. Undo with: $0 --undo"
