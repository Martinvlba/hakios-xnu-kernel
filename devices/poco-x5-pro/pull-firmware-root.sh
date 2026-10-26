#!/bin/bash
# pull-firmware.sh, through root. On some ROMs of this phone (seen on a redwood unit running a
# spoofed-brand KernelSU build) /vendor/firmware_mnt is a vfat mount owned by system with
# fmask=0337, and /vendor/firmware is not readable by the adb shell user either: the plain
# `adb pull` of pull-firmware.sh gets nothing ("not on the phone" for every file) although every
# file is there. This does the same pull by way of su: the firmware is copied, under root, to a
# world-readable staging directory in /data/local/tmp, pulled from there, and the staging directory
# is removed again. Nothing on the phone is changed. Needs the phone in Android with adb and a root
# that answers `su -c` (Magisk, KernelSU).
#
# The layout it leaves is pull-firmware.sh's: fw-gpu/, fw-wifi/, fw-audio/, fw-haptic/, fw-modem/
# (modem.* and image/, the modem's own files for TFTP). Camera firmware is not pulled: its files
# carry the first phone's renamed names and are not needed for the modem or the network.
set -e
cd "$(dirname "$0")/../.."
S=/data/local/tmp/fw
adb get-state >/dev/null
adb shell "su -c 'sh -c \"
S=$S; rm -rf \$S; mkdir -p \$S/gpu \$S/wifi \$S/audio \$S/haptic \$S/modem
cp /vendor/firmware/a660_gmu.bin /vendor/firmware/a660_sqe.fw /vendor/firmware/a660_zap.* \$S/gpu/ 2>/dev/null
cp /vendor/firmware_mnt/image/wpss.* \$S/wifi/
cp /vendor/firmware_mnt/image/adsp.* \$S/audio/
cp /vendor/firmware/aw8697_haptic.bin \$S/haptic/
cp /vendor/firmware_mnt/image/modem.* \$S/modem/
cp -r /vendor/firmware_mnt/image \$S/modem/
chmod -R a+rX \$S\"'"
rm -rf fw-stage
adb pull "$S" fw-stage >/dev/null
rm -rf fw-gpu fw-wifi fw-audio fw-haptic fw-modem
mv fw-stage/gpu fw-gpu; mv fw-stage/wifi fw-wifi; mv fw-stage/audio fw-audio
mv fw-stage/haptic fw-haptic; mv fw-stage/modem fw-modem; rmdir fw-stage
adb shell "su -c 'rm -rf $S'"
for d in fw-gpu fw-wifi fw-audio fw-haptic fw-modem fw-modem/image; do
	echo "pull-firmware-root.sh: $d: $(ls -1 "$d" | wc -l | tr -d ' ') files"
done
echo "pull-firmware-root.sh: done; Wi-Fi also wants fw-wifi/mac (the radio's address) and fw-wifi/network (the network's name and passphrase)"
