#!/bin/bash
# HakiOS's UEFI: Project Silicium's Mu-Silicium at the commit in BASE, with HakiOS's changes on top, and the platform of each
# device that has one (devices/<name>/uefi/files/). The result is Mu-Silicium/ at the top of the repository, which mkxnuimg.py
# and tools/update-prebuilt.sh take the image from.
#
#   uefi/apply.sh [DEVICE...]      default: every device with a devices/<name>/uefi/files/
#
# What HakiOS changes in Mu-Silicium (hakios.patch, files/, branding/):
#   BootShim.S       copies the "XNUP" container behind the UEFI image to 0xF0000000, where the loader finds the kernel
#   BootManagerLib   starts the XNU loader from that container (XnuLoader.c) instead of showing a boot menu
#   the banner       BootLogo.bmp, NoBootOS.bmp and MassStorage.bmp are HakiOS's, made by branding/banner.py, in place of
#                    Silicium's: the logo the phone shows while it starts is ours, always
#   the vendor       the firmware says "HakiOS" (PcdFirmwareVendor, the SMBIOS tables) and its first log line is HakiOS's
set -e
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(dirname "$HERE")
MU=${MU:-$ROOT/Mu-Silicium}
[ -e "$MU" ] && { echo "uefi/apply.sh: $MU already exists (uefi/export.sh saves what is in it; move it away for a fresh one)"; exit 1; }
git clone https://github.com/Project-Silicium/Mu-Silicium.git "$MU"
git -C "$MU" checkout -q "$(cat "$HERE/BASE")"
git -C "$MU" submodule update --init --recursive
git -C "$MU" apply --whitespace=nowarn "$HERE/hakios.patch"
cp -r "$HERE/files/." "$MU/"
python3 "$HERE/branding/banner.py" "$MU/Silicon/Silicium/SiliciumPkg/Include/Resources"
devices=("$@")
if [ ${#devices[@]} -eq 0 ]; then
	for d in "$ROOT"/devices/*/uefi/files; do [ -d "$d" ] && devices+=("$(basename "$(dirname "$(dirname "$d")")")"); done
fi
for d in "${devices[@]}"; do
	[ -d "$ROOT/devices/$d/uefi/files" ] || { echo "uefi/apply.sh: devices/$d has no uefi/files/" >&2; exit 1; }
	cp -r "$ROOT/devices/$d/uefi/files/." "$MU/"
	echo "uefi/apply.sh: the platform of $d"
done
echo "uefi/apply.sh: done; build with: cd Mu-Silicium && python3 build_uefi.py -d <device> -r RELEASE (uefi/README.md)"
