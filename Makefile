# HakiOS: one Makefile at the top, the way an Android tree has one. Every target is a script
# below it; run the script itself for its options.
#
#   make                 help
#   make fetch           the OS files from the iOS 16.5 IPSW into sysimage/ (sysimage/fetch.sh)
#   make images          userland/root.apfs and userland/system.apfs (userland/build.sh)
#   make kernel          the kernel from source (xnu/apply.sh, xnu-deps/setup.sh, xnu-build.sh)
#   make loader          the boot loader, with identity.h made from your IPSW (xnuloader/build.sh)
#   make firmware        the phone's firmware off the phone over adb (devices/$(DEVICE)/pull-firmware.sh)
#   make qemu            a 20-minute session in QEMU (needs the iOS images)
#   make qemu-open       boot the open kernel in QEMU with NO Apple software
#                        (kernel + the hello RAM disk + a placeholder loader)
#   make phone           a session on the phone, then back to Android (tools/modem-cycle.sh)
#   make prebuilt        refresh prebuilt/ from what was built (tools/update-prebuilt.sh)
#   make clean           build output of the kernel, loader and images
#   make doctor          check this machine and a phone on adb for what HakiOS needs (tools/doctor.sh)
#   make esp             an EFI system partition tree in build/esp: the loader and a payload it reads itself,
#                        to copy to a FAT32 volume (UEFI-native boot, no fastboot; docs/uefi-boot.md)
#
# Configuration, the way Linux's make does it (configs/, tools/hakios-config.py; docs/configs.md):
#   make redwood_defconfig   .config from configs/defconfig + redwood_defconfig, which is common_defconfig + sc7280.config + redwood.config
#                            (any NAME_defconfig; make common_defconfig is the common kernel alone; docs/configs.md)
#   make virt.config         lay a fragment (configs/virt.config) over the .config there is
#   make config              show the .config: its symbols, the kernel's arguments, the firmware it wants
#   make savedefconfig       the .config as a minimal defconfig
#   make phone               with a .config, its arguments are the kernel's; without, the phone line below
#   FRAGMENTS="open trace"   fragments for a NAME_defconfig: make redwood_defconfig FRAGMENTS=open
#
#   DEVICE=redwood   the device directory under devices/ (default redwood)
#   ARGS="..."           kernel arguments for make phone (default: the .config's, else everything the phone has)
#   SECS=600             the session's length for make phone / make qemu
#   JOBS=8               parallel jobs for the kernel build

DEVICE ?= redwood
JOBS ?= 8
SECS ?= 600
# qemu-open runs the open configuration: configs/defconfig (nothing of a phone's own, one core)
ARGS_OPEN ?= $(shell tools/hakios-config.py args --defconfig defconfig)
ifneq ($(wildcard .config),)
# a .config is the kernel's arguments; the runner then does not add its own four (cores, clock, storage): they are in it
ARGS ?= $(shell tools/hakios-config.py args)
RUNNER_BASE = BASE_ARGS=
else
ARGS ?= touchscreen=1 gpu=1 adsp=1 sound=4 wifi=1 haptic=5 modem=1 modem_again=1 modem_buffer=0xa8000000
RUNNER_BASE =
endif
IMAGE_OPTS ?= REST=1
ESP_DIR ?= build/esp
PAYLOAD ?= qemu/payload.bin

.PHONY: help fetch images kernel loader loader-open hello qemu qemu-open phone prebuilt clean all doctor defconfig config savedefconfig esp

help:
	@awk '{ if (!/^#/) exit; sub(/^# ?/, ""); print }' Makefile

doctor:
	tools/doctor.sh

all: kernel loader images

fetch:
	sysimage/fetch.sh

images:
	$(IMAGE_OPTS) userland/build.sh

kernel:
	[ -d xnu-src/xnu-8796.121.2 ] || xnu/apply.sh
	[ -e xnu-deps/host/bin/ld64 ] || xnu-deps/setup.sh
	tools/make-kernel.sh -j$(JOBS)

loader:
	xnuloader/build.sh

firmware:
	devices/$(DEVICE)/pull-firmware.sh

qemu:
	mkdir -p logs
	qemu/run-ios.sh $(SECS) -display gtk | tee logs/qemu.log

# The Apple-free path: no IPSW, no iOS user space. The loader is built with a placeholder identity
# (xnuloader/mkidentity.sh --placeholder), and the kernel boots the raw-syscall `hello` RAM disk.
# This is the configuration an outside contributor can build and run with nothing of Apple's.
loader-open:
	PLACEHOLDER=1 xnuloader/mkidentity.sh
	xnuloader/build.sh

# `hello` on its own: a static, raw-syscall program (no iOS SDK, no Apple libraries).
hello:
	. ./xnu-env.sh && xnu-clang -arch arm64 -Os -static -nostdlib -ffreestanding \
		-fno-stack-protector -fno-builtin -Wl,-e,_start -Wl,-pagezero_size,0x100000000 \
		-o userland/hello userland/src/hello.c

qemu-open: kernel loader-open hello
	mkdir -p logs
	xnuloader/mkpayload.py -o qemu/payload.bin \
		kernel=xnu-src/xnu-8796.121.2/BUILD/obj/DEVELOPMENT_ARM64_BCM2837/mach.development.bcm2837 \
		ramdisk=userland/hello --cmdline "$(ARGS_OPEN)"
	qemu/run.sh $(SECS) | tee logs/qemu-open.log

phone:
	[ ! -f .config ] || tools/hakios-config.py check
	XL_SB_DELAY=8 $(RUNNER_BASE) tools/modem-cycle.sh "$(ARGS)" $(SECS)

# The loader and a payload (make qemu-open makes qemu/payload.bin; xnuloader/mkpayload.py makes any other) as the files an
# EFI system partition holds, which the loader reads itself: UEFI-native boot. The default tree is made again each time.
esp:
	[ "$(ESP_DIR)" != build/esp ] || rm -rf build/esp
	tools/mkesp.py $(ESP_DIR) $(PAYLOAD)

# ---- configuration (configs/, tools/hakios-config.py) ---------------------------------------------------------
defconfig:
	tools/hakios-config.py defconfig defconfig $(FRAGMENTS)

%_defconfig:
	tools/hakios-config.py defconfig $* $(FRAGMENTS)

%.config:
	tools/hakios-config.py merge $*

config:
	tools/hakios-config.py show

savedefconfig:
	tools/hakios-config.py savedefconfig

prebuilt:
	tools/update-prebuilt.sh

clean:
	rm -rf xnu-src/xnu-8796.121.2/BUILD xnuloader/loader.obj xnuloader/loader.efi userland/root.apfs userland/system.apfs userland/obj
