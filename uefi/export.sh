#!/bin/bash
# The reverse of uefi/apply.sh: what was changed in Mu-Silicium/ goes back here. The changes to Silicium's own files go to
# hakios.patch, HakiOS's new files under Silicon/ to files/, and each device's platform to devices/<name>/uefi/files/
# (devices/<name>/uefi/export.sh). The banner pictures are made by branding/banner.py and are not saved: change that instead.
set -e
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(dirname "$HERE")
MU=${MU:-$ROOT/Mu-Silicium}
[ "$(git -C "$MU" rev-parse HEAD)" = "$(cat "$HERE/BASE")" ] || { echo "uefi/export.sh: $MU is not at BASE"; exit 1; }
git -C "$MU" diff -- BootShim Silicon ':!Silicon/Silicium/SiliciumPkg/Include/Resources' > "$HERE/hakios.patch"
rm -rf "$HERE/files"
mkdir -p "$HERE/files/Silicon/Silicium/SiliciumPkg/Library/BootManagerLib"
cp "$MU"/Silicon/Silicium/SiliciumPkg/Library/BootManagerLib/XnuLoader.[ch] "$HERE/files/Silicon/Silicium/SiliciumPkg/Library/BootManagerLib/"
for d in "$ROOT"/devices/*/uefi/export.sh; do [ -x "$d" ] && "$d"; done
echo "uefi/export.sh: patch $(wc -l < "$HERE/hakios.patch") lines, $(find "$HERE/files" -type f | wc -l) files"
