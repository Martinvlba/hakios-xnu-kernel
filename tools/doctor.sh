#!/bin/bash
# tools/doctor.sh: check this machine, and a phone on adb if there is one, for what HakiOS needs, and say
# what to do about every problem. Read-only: it changes nothing on the host or on the phone.
#
#   tools/doctor.sh          (or: make doctor)
#
# Exit status: 1 if something that is needed is missing, 0 otherwise (warnings are not failures).
# What it knows comes from bringing the project up on a Windows machine with a phone behind a spoofed
# identity: see docs/windows.md.
cd "$(dirname "$0")/.." || exit 1
export MSYS_NO_PATHCONV=1                       # Git Bash must not rewrite /proc/... into C:/... for adb

npass=0; nwarn=0; nfail=0
if [ -t 1 ]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'; else G=; Y=; R=; B=; N=; fi
ok()   { printf '  %sok%s    %s\n' "$G" "$N" "$1"; npass=$((npass + 1)); }
warn() { printf '  %swarn%s  %s\n        -> %s\n' "$Y" "$N" "$1" "$2"; nwarn=$((nwarn + 1)); }
bad()  { printf '  %sFAIL%s  %s\n        -> %s\n' "$R" "$N" "$1" "$2"; nfail=$((nfail + 1)); }
note() { printf '        %s\n' "$1"; }
head1() { printf '\n%s%s%s\n' "$B" "$1" "$N"; }
have() { command -v "$1" >/dev/null 2>&1; }
# a build tool: a failure where the images are built, only a warning in Git Bash (they are built in WSL there)
need() { if [ "$host" = gitbash ]; then warn "$1" "$2 -- in WSL, where the images are built (docs/windows.md)"; else bad "$1" "$2"; fi; }

case "$(uname -s)" in
Linux) if grep -qi microsoft /proc/version 2>/dev/null; then host=wsl; else host=linux; fi ;;
Darwin) host=macos ;;
MINGW* | MSYS* | CYGWIN*) host=gitbash ;;
*) host=other ;;
esac

head1 "This machine ($host)"

# python3 (the build scripts need 3.8+); on Windows `python3` can be a Store stub that does not run
if have python3 && v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) && [ -n "$v" ]; then
	if [ "${v%%.*}" -ge 3 ] && [ "${v##*.}" -ge 8 ]; then ok "python3 $v"; else bad "python3 $v is too old" "install Python 3.8 or newer"; fi
else
	bad "python3 does not run" "install Python 3 (on Windows: python.org, not the Microsoft Store stub)"
fi

# 7-Zip 23 or newer: sysimage/fetch.sh unpacks APFS with it
z=""; for t in 7zz 7z 7za; do have $t && { z=$t; break; }; done
if [ -n "$z" ]; then
	zv=$($z 2>&1 | head -3 | grep -oE '[0-9]+\.[0-9]+' | head -1)
	if [ -n "$zv" ] && [ "${zv%%.*}" -ge 23 ]; then ok "7-Zip $zv ($z)"; else need "7-Zip ${zv:-?} is older than 23 (cannot read APFS)" "install 7-Zip 23 or newer (apt: 7zip, brew: sevenzip)"; fi
else
	need "7-Zip not found" "install 7-Zip 23 or newer (apt: 7zip, brew: sevenzip)"
fi

if have cc || have clang; then ok "C compiler ($(have cc && echo cc || echo clang)): builds unzbm for sysimage/fetch.sh"; else need "no C compiler" "install clang or gcc"; fi
have git && ok "git" || bad "git not found" "install git"
if have lld-link; then ok "lld-link: the boot loader can be built (xnuloader/build.sh)"; else warn "lld-link not found" "needed to build the boot loader with the iPhone identity from your own IPSW (make loader), which the iOS run needs (apt: lld; brew: lld); the open kernel (make qemu-open) runs with prebuilt/loader.efi, which has no Apple data"; fi
have lz4 && ok "lz4" || warn "lz4 not found" "needed by some image steps (apt: lz4)"

if have flock; then
	ok "flock (xl needs it)"
elif [ "$host" = gitbash ]; then
	warn "no flock in Git Bash" "export PATH=\"\$PWD/tools/win-bin:\$PATH\"  (tools/win-bin/flock is a shim for it)"
else
	bad "flock not found" "install util-linux (macOS: see tools/macos-bin/README.md)"
fi

# room for the IPSW parts and the images: about 8 GB while fetching, 5.3 GB afterwards, 2.5 GB system image
kb=$(df -Pk . 2>/dev/null | awk 'NR == 2 { print $4 }')
if [ -n "$kb" ] && [ "$kb" -ge 12582912 ]; then ok "disk: $((kb / 1048576)) GB free here"; else warn "disk: ${kb:+$((kb / 1048576)) GB }free here, 12 GB wanted" "free some space, or work in another directory"; fi

# the trap: fetch.sh makes files named NAME:com.apple.decmpfs, and ':' is not a legal NTFS name character
case "$host:$PWD" in
wsl:/mnt/*) warn "this checkout is on a Windows drive (/mnt/...)" "build in WSL's own filesystem: git clone it to ~/hakios and copy fw-* there (docs/windows.md)" ;;
gitbash:*) warn "Git Bash cannot run sysimage/fetch.sh (NTFS forbids ':' in file names)" "build the images in WSL, copy them back; run xl from here (docs/windows.md)" ;;
esac

# rebuilding the kernel or userland needs theos's SDK and its Linux toolchain's cctools linker (xnu-deps/setup.sh)
theos=${THEOS:-$HOME/theos}
if [ -d "$theos/sdks/iPhoneOS16.5.sdk" ] && [ -x "$theos/toolchain/linux/iphone/bin/ld" ]; then
	ok "theos with the iPhoneOS 16.5 SDK and the cctools linker: the kernel and userland can be rebuilt"
elif [ -d "$theos/sdks/iPhoneOS16.5.sdk" ]; then
	warn "theos has the SDK but not its Linux toolchain (no cctools linker: $theos/toolchain/linux/iphone/bin/ld)" "optional, only to rebuild the kernel or userland: install theos's Linux toolchain, then xnu-deps/setup.sh; PREBUILT=1 and prebuilt/ work without"
else
	warn "no theos SDK ($theos/sdks/iPhoneOS16.5.sdk)" "optional, only to rebuild the kernel or userland (theos + xnu-deps/setup.sh); PREBUILT=1 and prebuilt/ work without"
fi
if have qemu-system-aarch64; then
	ok "qemu-system-aarch64: make qemu-open works"
	# what the iOS run (qemu/run-ios.sh) needs on top: a display device, the UEFI firmware, memory. Without the display
	# SpringBoard does not start, and the window stays black, which is what people reported first.
	if qemu-system-aarch64 -device help 2>/dev/null | grep -q '"ramfb"'; then ok "qemu has the ramfb display device (the screen of qemu/run-ios.sh)"
	else bad "this qemu-system-aarch64 has no ramfb display device" "the screen stays black without it: install a QEMU of 7.0 or newer (the full package, e.g. qemu-system-arm on Debian/Ubuntu)"; fi
	fw=""; for f in "${QEMU_EFI:-}" /usr/share/edk2/aarch64/QEMU_EFI.fd /usr/share/AAVMF/AAVMF_CODE.fd; do [ -n "$f" ] && [ -f "$f" ] && { fw=$f; break; }; done
	if [ -n "$fw" ]; then ok "UEFI firmware for the guest: $fw"
	else bad "no UEFI firmware for aarch64 guests" "install edk2-aarch64 (Arch) or qemu-efi-aarch64 (Debian/Ubuntu), or set QEMU_EFI and QEMU_VARS"; fi
	# the guest has 7 GB of RAM by default (QEMU_RAM_GB, never below 6: the system image sits in it, 3 GB up)
	mem_mb=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo 2>/dev/null)
	if [ -n "$mem_mb" ]; then
		if [ "$mem_mb" -ge 7200 ]; then ok "memory: ${mem_mb} MB available (the iOS run wants about 7 GB)"
		else warn "memory: only ${mem_mb} MB available; the iOS run asks QEMU for 7 GB" "close other programs, or on a small machine give WSL/the VM more memory; below that the host swaps and the window stays black for a long time"; fi
	fi
	note "this is an emulated CPU, so the run is slow: the window is black for the first minute or two, then the Apple logo,"
	note "then the lock screen after about 3 to 10 minutes. 'tail -f logs/qemu.log' shows it is alive; qemu/screendump.sh saves the screen."
else
	warn "no qemu-system-aarch64" "optional: only for the QEMU targets"
fi

head1 "What is already built here"
if [ -f fw-modem/modem.mdt ]; then ok "phone firmware pulled (fw-*/)"; else warn "no fw-modem/ yet" "with the phone on adb: devices/redwood/pull-firmware.sh, or pull-firmware-root.sh if that gets nothing"; fi
if [ -f userland/system.apfs ] && [ -f userland/root.apfs ]; then ok "userland/system.apfs and root.apfs"; else warn "no iOS images yet" "make fetch, then PREBUILT=1 REST=1 MODEM=1 SYSTEM=1 userland/build.sh (needs iOS 16.5 from Apple's servers, ~8 GB)"; fi

head1 "Phone"
adb=""
for c in adb adb.exe /mnt/c/platform-tools/adb.exe /c/platform-tools/adb.exe tools/macos-bin/platform-tools/adb; do
	if have "$c" || [ -x "$c" ]; then adb=$c; break; fi
done
if [ -z "$adb" ]; then
	warn "adb not found" "install Android platform-tools (docs/windows.md); skipping the phone checks"
else
	fb=""; for c in fastboot fastboot.exe /mnt/c/platform-tools/fastboot.exe /c/platform-tools/fastboot.exe; do { have "$c" || [ -x "$c" ]; } && { fb=$c; break; }; done
	devs=$("$adb" devices 2>/dev/null | tr -d '\r' | awk 'NR > 1 && $2 == "device" { print $1 }')
	if [ -z "$devs" ]; then
		if "$adb" devices 2>/dev/null | grep -q unauthorized; then
			warn "a phone is attached but has not authorised this computer" "unlock it and accept the USB debugging prompt"
		elif [ -n "$fb" ] && "$fb" devices 2>/dev/null | grep -q fastboot; then
			ok "a phone is in fastboot (./xl state tells loader from fastboot)"
		else
			warn "no phone on adb or fastboot" "plug it in with USB debugging on; if it was left in the HakiOS loader and is gone, hold power ~15 s to restart it into Android"
		fi
	else
		A() { "$adb" shell "$1" 2>/dev/null | tr -d '\r'; }
		model=$(A 'getprop ro.product.model'); board=$(A 'getprop ro.product.board'); soc=$(A 'getprop ro.soc.model')
		dtm=$(A 'cat /proc/device-tree/model 2>/dev/null'); dtc=$(A 'cat /proc/device-tree/compatible 2>/dev/null | tr "\\0" " "')
		note "reports: model='$model' board='$board' soc='$soc'"
		note "device tree: model='$dtm' compatible='$dtc'"
		if echo "$dtm $board" | grep -qi redwood && [ "$soc" = SM7325 ]; then
			ok "redwood / SM7325: the reference platform (devices/redwood)"
			if ! echo "$model" | grep -qiE 'poco|redwood|2210132'; then
				note "the reported model '$model' is not a POCO X5 Pro name: it is probably spoofed; the device tree is what counts"
			fi
		elif [ -n "$dtm$soc" ]; then
			warn "not the reference platform ($soc)" "a port is a new devices/<name>/: devices/redwood/README.md; adb pull /sys/firmware/fdt and run tools/soc-args.py on it"
		else
			warn "could not read the device tree" "an adb shell may be restricted; try with root"
		fi
		if A 'su -c id' | grep -q 'uid=0'; then
			ok "root answers (su)"; root=1
		else
			warn "no root" "some ROMs keep the modem, GPU and DSP firmware unreadable to the adb shell"; root=0
		fi
		plain=$(A 'ls /vendor/firmware_mnt/image/modem.mdt 2>&1')
		if echo "$plain" | grep -q '^/vendor/firmware_mnt/image/modem.mdt$'; then
			ok "modem firmware is readable without root: devices/redwood/pull-firmware.sh works"
		elif [ "$root" = 1 ] && A 'su -c "ls /vendor/firmware_mnt/image/modem.mdt"' | grep -q modem.mdt; then
			warn "the firmware is there but only root can read it" "use devices/redwood/pull-firmware-root.sh instead of pull-firmware.sh"
		else
			warn "modem firmware not found under /vendor/firmware_mnt/image" "check the ROM's firmware layout (docs/windows.md)"
		fi
		lk=$(A 'getprop ro.boot.flash.locked'); vb=$(A 'getprop ro.boot.verifiedbootstate')
		note "bootloader as reported: locked=$lk verifiedboot=$vb (root-hiding modules can fake this; the real test is ./xl enter)"
		bat=$(A 'cat /sys/class/power_supply/battery/capacity')
		if [ -n "$bat" ] && [ "$bat" -lt 30 ] 2>/dev/null; then warn "battery $bat%" "charge it: HakiOS does not charge the phone while it runs"; else ok "battery ${bat:-?}%"; fi

		# What a persistent install would stand on (docs/install.md). Only looked at: nothing here writes to the phone.
		slot=$(A 'getprop ro.boot.slot_suffix')
		if [ -n "$slot" ]; then
			ok "A/B slots: Android runs from slot ${slot#_}; a persistent install would use the other slot and leave Android on this one"
		else
			warn "no A/B slot suffix reported" "a persistent install keeps Android on one slot and HakiOS on the other: without two slots there is no safe place for it (docs/install.md)"
		fi
		if [ "$root" = 1 ]; then
			if A 'su -c "ls /dev/block/by-name"' | grep -qx darwin; then
				ok "the partition darwin exists (the port's own: persistence and the iOS system image)"
			else
				note "no partition darwin yet: only persistence and a flash install need it (tools/userdata-carve.py plans it, and says what it does to Android's data)"
			fi
		fi
	fi
fi

printf '\n%s%d ok, %d warnings, %d failures%s\n' "$B" "$npass" "$nwarn" "$nfail" "$N"
[ "$nfail" -eq 0 ]
