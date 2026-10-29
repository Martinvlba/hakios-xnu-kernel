#!/bin/bash
# Refresh prebuilt/ from what is built in this tree: the kernel, the loader, our user-space programs
# and the pieces of the phone's UEFI. With them somebody without the cross toolchain can still make the
# images and run the system (QEMU or the phone); see README.md.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
# KERNEL=... to take the kernel from a tree built elsewhere (xnu/apply.sh makes one anywhere)
K=${KERNEL:-hakios-kernel}
MU=Mu-Silicium
mkdir -p prebuilt/userland devices/redwood/prebuilt
cp $K prebuilt/hakios-kernel
# The loader in prebuilt/ is built with the PLACEHOLDER identity (xnuloader/mkidentity.sh --placeholder: no Apple data), in a
# temporary copy so that the real xnuloader/identity.h of this tree, made from your own IPSW, is left alone and never copied
# here. It boots the open kernel (make qemu-open); to start Darwin, build the loader yourself (make loader).
if command -v lld-link >/dev/null && command -v clang >/dev/null; then
	T=$(mktemp -d)
	mkdir "$T/xnuloader"
	cp xnuloader/loader.c xnuloader/efi.h xnuloader/build.sh xnuloader/mkidentity.sh xnuloader/mkidentity.py xnuloader/im4p.py "$T/xnuloader/"
	( cd "$T/xnuloader" && PLACEHOLDER=1 ./mkidentity.sh >/dev/null && ./build.sh >/dev/null ) && cp "$T/xnuloader/loader.efi" prebuilt/loader.efi ||
		echo "update-prebuilt.sh: the placeholder loader did not build; prebuilt/loader.efi kept"
	rm -rf "$T"
else
	echo "update-prebuilt.sh: no lld-link or clang here; prebuilt/loader.efi kept"
fi
cp userland/hello prebuilt/userland/
for f in init child stage0 hello_dyn svc_test surf_test rwtouch mg_test cpu_test sound_test libhakios.dylib libhakios.sb.dylib \
	libhakios.media.dylib libhakios.app.dylib libhakios.tweaks.dylib HakiosAudio net_test; do
	cp userland/obj/$f prebuilt/userland/
done
if [ -f $MU/Build/redwoodPkg/RELEASE_CLANGPDB/FV/SILICIUM_UEFI.fd ]; then
	cp $MU/Build/redwoodPkg/RELEASE_CLANGPDB/FV/SILICIUM_UEFI.fd $MU/BootShim/BootShim.bin $MU/Resources/DTBs/redwood.dtb \
		$MU/Resources/ramdisk $MU/Resources/Scripts/mkbootimg.py devices/redwood/prebuilt/
fi
{
	echo "Built $(date '+%Y-%m-%d %H:%M') from this tree."
	echo
	sha256sum prebuilt/hakios-kernel prebuilt/loader.efi prebuilt/userland/* devices/redwood/prebuilt/* 2>/dev/null
} > prebuilt/SHA256SUMS
ls -la prebuilt prebuilt/userland devices/redwood/prebuilt
