#!/usr/bin/env bash
# Put "Out of Frame" and "Drone Booth" icons on the Jetson desktop (and in the app menu).
# Run once on the Jetson, from anywhere:  ./jetson/install-shortcuts.sh
set -eu
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
APPS="$HOME/.local/share/applications"
mkdir -p "$DESK" "$APPS"
chmod +x "$REPO/jetson/out-of-frame.sh" "$REPO/jetson/drone-booth.sh"

write() {  # file name comment script icon
  cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Name=$2
Comment=$3
Exec=$4
Icon=$5
Terminal=true
Categories=Game;
EOF
  chmod +x "$1"
}

for dir in "$DESK" "$APPS"; do
  write "$dir/out-of-frame.desktop" "Out of Frame" "Pick a setup and start the Out of Frame game" \
    "$REPO/jetson/out-of-frame.sh" "$REPO/out-of-frame/assets/brand/logos/logomark-dutch.png"
  write "$dir/drone-booth.desktop" "Drone Booth" "Start the Caught by Scouts drone booth" \
    "$REPO/jetson/drone-booth.sh" "$REPO/drone-booth/assets/brand/logos/logomark-powder.png"
done

# GNOME only launches desktop icons marked as trusted ("Allow Launching").
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
for f in "$DESK/out-of-frame.desktop" "$DESK/drone-booth.desktop"; do
  gio set "$f" metadata::trusted true 2>/dev/null \
    || echo "Could not mark $f as trusted; right-click it on the desktop and pick Allow Launching."
done
echo "Shortcuts installed in $DESK"
