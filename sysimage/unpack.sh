#!/bin/bash
# Unpack the files of iOS 16.5 (20F66, iPhone X) that the system image is made of, out of an IPSW
# you have on your own disk, and lay them out the way ../userland/build.sh expects them:
#
#   rootfs/                  the chosen part of the root filesystem (see select-system.py)
#   system.files/.manifest   what was chosen, with owners, modes and symbolic links
#   var.skel                 the skeleton of /private/var (see mkvarskel.py)
#   cryptex/                 the OS cryptex without the dyld shared cache
#   cryptex-app.dmg          the App cryptex as it is in the IPSW (10 MB, an APFS container: Safari's
#                            daemons, its settings and the bundles around it, which the root volume's
#                            launchd jobs and links reach through /System/Cryptexes/App)
#   cryptex-app/             ... and its files
#   20F66__iPhone10,3_6/     the dyld shared cache (arm64)
#   DeviceTree.d22ap.im4p    the device tree, for ../xnuloader/mkidentity.sh
#
# The project ships no file of Apple's and downloads none. The IPSW is yours: get
# iPhone10,3,iPhone10,6_16.5_20F66_Restore.ipsw yourself (6 165 698 198 bytes) and give its path:
#
#   sysimage/unpack.sh ~/Downloads/iPhone10,3,iPhone10,6_16.5_20F66_Restore.ipsw
#   IPSW=... sysimage/unpack.sh              the same, from the environment (make unpack IPSW=...)
#
# The disk images are copied out of the IPSW (ipsw-member.py) and unpacked with 7z. Needs: python3,
# 7z (7-Zip 23 or newer: it reads APFS), cc. About 8 GB of disk while it runs, 5.3 GB afterwards.
#
#   KEEP=1   keep the copied disk images (rootfs.dmg is needed again only after changing
#            select-system.py; with it in place a second run does not read the IPSW for it)
set -e
IPSW=${1:-${IPSW:-}}
if [ -z "$IPSW" ]; then
	sed -n '16,20p' "$0" | sed 's/^# \{0,1\}//' >&2
	echo "unpack.sh: give the path of your iOS 16.5 IPSW" >&2
	exit 1
fi
[ -f "$IPSW" ] || { echo "unpack.sh: $IPSW: no such file" >&2; exit 1; }
IPSW=$(cd "$(dirname "$IPSW")" && pwd)/$(basename "$IPSW")
size=$(wc -c < "$IPSW" | tr -d ' ')
# select-system.py and the checks below are made for this one build; another IPSW would unpack, then fail somewhere later
if [ "$size" != 6165698198 ] && [ -z "${ANY_IPSW:-}" ]; then
	echo "unpack.sh: $IPSW is $size bytes, not the 6165698198 of iOS 16.5 20F66 for iPhone10,3/10,6 (ANY_IPSW=1 tries it anyway)" >&2
	exit 1
fi
cd "$(dirname "$0")"
ROOTFS_DMG=096-04088-077.dmg                    # the root filesystem, APFS
CRYPTEX_DMG=096-04666-078.dmg                   # the OS cryptex: dyld shared cache and a few libraries
APP_CRYPTEX_DMG=096-03536-083.dmg               # the App cryptex (BuildManifest: Cryptex1,AppOS)
CACHE=20F66__iPhone10,3_6
UNZBM=../xnu-deps/host/bin/unzbm

for tool in python3 7z cc; do
	command -v $tool >/dev/null || { echo "unpack.sh: $tool is not installed"; exit 1; }
done
# 7z leaves files compressed with LZBITMAP as an empty file and two attribute files; unzbm finishes them
if [ ! -x $UNZBM ]; then
	mkdir -p "$(dirname $UNZBM)"
	cc -O2 -w -o $UNZBM ../xnu-deps/build/lzbitmap/unzbm.c ../xnu-deps/build/lzbitmap/libzbitmap.c
fi
unzbm_all() { # unzbm_all DIR: expand every LZBITMAP file below DIR
	find "$1" -type f -name '*:com.apple.decmpfs' | sed 's/:com\.apple\.decmpfs$//' | tr '\n' '\0' | xargs -0 -r -n 200 $UNZBM
	local left
	left=$(find "$1" -name '*:com.apple.*' | wc -l)
	[ "$left" -eq 0 ] || { echo "unpack.sh: $left files in $1 are still compressed"; exit 1; }
}

# --- the root filesystem --------------------------------------------------------------------------
if [ ! -f .done-rootfs ] || [ select-system.py -nt .done-rootfs ]; then
	if [ ! -f rootfs.dmg ]; then
		echo "unpack.sh: copying the root filesystem image out of the IPSW (4.7 GB)"
		python3 ipsw-member.py "$IPSW" $ROOTFS_DMG rootfs.dmg.part
		mv rootfs.dmg.part rootfs.dmg
	fi
	echo "unpack.sh: listing the image"
	7z l -slt rootfs.dmg > rootfs.slt
	python3 select-system.py rootfs.slt system.files system.manifest
	python3 mkvarskel.py rootfs.slt > var.skel
	echo "unpack.sh: extracting $(wc -l < system.files) files"
	rm -rf rootfs
	7z x -y -orootfs rootfs.dmg @system.files > extract-system.log
	# the two fonts too big for the image as they are, made smaller in place (see select-system.py)
	python3 ../tools/font_trim.py emoji rootfs/System/Library/Fonts/CoreAddition/AppleColorEmoji-160px.ttc \
		rootfs/System/Library/Fonts/CoreAddition/AppleColorEmoji-160px.ttc 40,64,96
	python3 ../tools/font_trim.py faces rootfs/System/Library/Fonts/LanguageSupport/PingFang.ttc \
		rootfs/System/Library/Fonts/LanguageSupport/PingFang.ttc '^\.?PingFang(SC|TC|HK)-(Regular|Semibold)$'
	touch .done-rootfs
	[ -n "$KEEP" ] || rm -f rootfs.dmg
fi

# --- the OS cryptex -------------------------------------------------------------------------------
if [ ! -f .done-cryptex ]; then
	if [ ! -f cryptex-os.dmg ]; then
		echo "unpack.sh: copying the OS cryptex image out of the IPSW (2.7 GB)"
		python3 ipsw-member.py "$IPSW" $CRYPTEX_DMG cryptex-os.dmg.part
		mv cryptex-os.dmg.part cryptex-os.dmg
	fi
	echo "unpack.sh: extracting the cryptex"
	rm -rf cryptex cryptex.cache $CACHE
	# 7z reports every LZBITMAP file as an error ("Unsupported Method"): expected, unzbm_all checks the result
	7z x -y -ocryptex cryptex-os.dmg '-xr!com.apple.dyld' '-x!System/DriverKit' > extract-cryptex.log 2>&1 || true
	unzbm_all cryptex
	# the dyld shared cache; its .symbols file (0.6 GB) is never mapped and is left out
	7z x -y -ocryptex.cache cryptex-os.dmg System/Library/Caches/com.apple.dyld '-xr!*.symbols*' > extract-dyld.log 2>&1 || true
	unzbm_all cryptex.cache
	mv cryptex.cache/System/Library/Caches/com.apple.dyld $CACHE
	rm -rf cryptex.cache
	[ "$(ls $CACHE | wc -l)" -eq 43 ] || { echo "unpack.sh: the dyld shared cache is incomplete"; exit 1; }
	touch .done-cryptex
	[ -n "$KEEP" ] || rm -f cryptex-os.dmg
fi

# --- the App cryptex ------------------------------------------------------------------------------
# kept as it is (the boot off the flash puts it in memory unchanged, ../userland/build.sh FLASHBOOT=1)
# and unpacked (the system image has its files at /System/Cryptexes/App)
if [ ! -f .done-cryptex-app ]; then
	if [ ! -f cryptex-app.dmg ]; then
		echo "unpack.sh: copying the App cryptex image out of the IPSW (10 MB)"
		python3 ipsw-member.py "$IPSW" $APP_CRYPTEX_DMG cryptex-app.dmg.part
		mv cryptex-app.dmg.part cryptex-app.dmg
	fi
	echo "unpack.sh: extracting the App cryptex"
	rm -rf cryptex-app
	7z x -y -ocryptex-app cryptex-app.dmg > extract-cryptex-app.log 2>&1 || true
	unzbm_all cryptex-app
	[ -f cryptex-app/usr/libexec/webbookmarksd ] || { echo "unpack.sh: the App cryptex is incomplete"; exit 1; }
	touch .done-cryptex-app
fi
# --- the device tree, for xnuloader/mkidentity.sh -----------------------------------------------
[ -f DeviceTree.d22ap.im4p ] || python3 ipsw-member.py "$IPSW" Firmware/all_flash/DeviceTree.d22ap.im4p DeviceTree.d22ap.im4p
echo "unpack.sh: done: $(du -sh rootfs | cut -f1) rootfs, $(du -sh cryptex | cut -f1) cryptex, $(du -sh cryptex-app | cut -f1) App cryptex, $(du -sh $CACHE | cut -f1) dyld shared cache"
