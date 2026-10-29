#!/bin/bash
# Boot the XNU loader + kernel payload under QEMU the way the phone does with HakiOS's U-Boot: U-Boot (not edk2) is
# the firmware, the loader and its payload are in memory before it starts (an HKUB container at 0x50000000, where a
# phone's ABL would have put its ramdisk), and the hakios command puts the payload at 0xF0000000 and starts the loader.
# usage: run-u-boot.sh [seconds] [extra qemu args...]
# env: QEMU_U_BOOT   U-Boot for QEMU (default ../u-boot/build-qemu/u-boot.bin: uefi/u-boot/apply.sh qemu &&
#                    uefi/u-boot/build.sh qemu)
#      QEMU_SMP, QEMU_MEM, QEMU_TB_MB as for run.sh
cd "$(dirname "$0")"
T=${1:-20}; shift
UB=${QEMU_U_BOOT:-../u-boot/build-qemu/u-boot.bin}
[ -f "$UB" ] || { echo "run-u-boot.sh: no $UB (uefi/u-boot/apply.sh qemu && uefi/u-boot/build.sh qemu)"; exit 1; }
[ -f payload.bin ] || { echo "run-u-boot.sh: no payload.bin (make qemu-open makes one)"; exit 1; }
L=../xnuloader/loader.efi; [ -f $L ] || L=../prebuilt/loader.efi
python3 - payload.bin $L u-boot.hkub <<'PY' || exit 1
import os, sys
sys.path.insert(0, os.path.join('..', 'tools'))
from mkubootimg import container
payload, loader, out = sys.argv[1:4]
open(out, 'wb').write(container([('payload', open(payload, 'rb').read()), ('loader', open(loader, 'rb').read())]))
PY
timeout "$T" qemu-system-aarch64 -M virt,gic-version=3 -cpu cortex-a76 -smp ${QEMU_SMP:-1} -m ${QEMU_MEM:-6G} \
  -accel tcg,tb-size=${QEMU_TB_MB:-384} \
  -bios "$UB" \
  -device loader,file=u-boot.hkub,addr=0x50000000,force-raw=on \
  -display none -serial stdio -monitor none -no-reboot "$@" 2>&1
