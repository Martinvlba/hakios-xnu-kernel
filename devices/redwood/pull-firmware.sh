#!/bin/bash
# Pull the phone's own firmware out of its Android into the fw-*/ directories at the top of the
# repository (never committed: it is the vendor's). The phone is on adb, in Android.
set -e
cd "$(dirname "$0")/../.."
pull() { # DIR SOURCE...  (missing files are told, not fatal)
	local dir=$1; shift
	mkdir -p "$dir"
	for f in "$@"; do adb pull "$f" "$dir/" >/dev/null 2>&1 || echo "pull-firmware.sh: $f: not on the phone"; done
	echo "pull-firmware.sh: $dir: $(ls -1 "$dir" | wc -l | tr -d ' ') files"
}
adb get-state >/dev/null
pull fw-gpu /vendor/firmware/a660_gmu.bin /vendor/firmware/a660_sqe.fw /vendor/firmware/a660_zap.mdt /vendor/firmware/a660_zap.b00 /vendor/firmware/a660_zap.b01 /vendor/firmware/a660_zap.b02
pull fw-wifi $(for i in mdt b02 b03 b04 b05 b06 b07; do echo /vendor/firmware_mnt/image/wpss.$i; done)
pull fw-audio /vendor/firmware_mnt/image/adsp.mdt $(for i in $(seq -w 2 36); do echo /vendor/firmware_mnt/image/adsp.b$i; done)
pull fw-haptic /vendor/firmware/aw8697_haptic.bin
pull fw-modem /vendor/firmware_mnt/image/modem.mdt $(for i in $(seq -w 0 28); do echo /vendor/firmware_mnt/image/modem.b$i; done)
mkdir -p fw-modem/image && adb pull /vendor/firmware_mnt/image fw-modem/ >/dev/null 2>&1 && echo "pull-firmware.sh: fw-modem/image: the modem's own files (TFTP)"
pull fw-camera $(adb shell ls /vendor/lib64/camera/com.qti.sensormodule.hakios_*.bin /vendor/lib64/camera/com.qti.tuned.hakios_*.bin 2>/dev/null | tr -d '\r')
echo "pull-firmware.sh: done; Wi-Fi also wants fw-wifi/mac (the radio's address) and fw-wifi/network (the network's name and passphrase)"
