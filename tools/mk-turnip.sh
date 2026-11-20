#!/bin/bash
# Build Turnip, Mesa's Vulkan driver for the Adreno, for the Darwin userland: the first piece of a Metal that runs on
# Vulkan (UniXNU's AdrenoMetal: metal2vulkan over Turnip). The tree is tools/mk-mesa.sh's (the pinned release with
# tools/mesa/*.patch on, turnip-darwin.patch among them); run that first. Turnip's msm backend talks to the kernel's
# GPU driver through tools/mesa/hakios-drm, the stand-in for libdrm, whose syncobjs (Turnip's fences and semaphores)
# are kept in the process over the kernel's fences.
#
# What comes out: xnu-deps/build/mesa/out/libvulkan_freedreno.dylib (install name /redwood/lib/libvulkan_freedreno.dylib),
# loaded by its vk_icdGetInstanceProcAddr like any Vulkan driver.
#
# Needs on this machine what mk-mesa.sh needs, and glslangValidator (apt: glslang-tools) for Turnip's own shaders.
set -e
# ARCH=arm64e (the default since 11-20: the iPhone 13 mini's userland, d16ap, and the newer iOS releases'; an arm64
# library does not load into an arm64e process) or arm64 (the iPhone X's, d22ap). arm64 builds where it always did; another architecture beside it, its name on the
# cross file and on the build and output directories, so that neither build takes the other's objects.
ARCH=${ARCH:-arm64e}
case $ARCH in arm64) SFX= ;; arm64e) SFX=-arm64e ;; *) echo "$0: ARCH=$ARCH: arm64 or arm64e" >&2; exit 1 ;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERSION=25.0.7
B=$ROOT/xnu-deps/build/mesa
BUILD=build$SFX-turnip
OUT=out$SFX
CROSS=cross$SFX.ini
cd "$B"
[ -d mesa-$VERSION ] && [ -f $CROSS ] || { echo "mk-turnip.sh: no Mesa tree in $B (or no $CROSS): ARCH=$ARCH tools/mk-mesa.sh first" >&2; exit 1; }
grep -q with_freedreno_vk mesa-$VERSION/meson.build ||
	{ echo "mk-turnip.sh: the tree has not got turnip-darwin.patch: tools/mk-mesa.sh again" >&2; exit 1; }
command -v glslangValidator >/dev/null || { echo "mk-turnip.sh: no glslangValidator (apt: glslang-tools)" >&2; exit 1; }
export PATH=$B/venv/bin:$PATH
[ -f $BUILD/build.ninja ] || meson setup $BUILD mesa-$VERSION --cross-file $CROSS --wrap-mode=nofallback \
	--prefix=/redwood --libdir=lib --buildtype=release -Db_ndebug=true \
	-Dplatforms= -Dgallium-drivers= -Dvulkan-drivers=freedreno -Dfreedreno-kmds=msm -Dopengl=false -Dgles1=disabled \
	-Dgles2=disabled -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dllvm=disabled -Dzstd=disabled -Dxmlconfig=disabled \
	-Dexpat=disabled -Dshader-cache=disabled -Dvalgrind=disabled -Dlibunwind=disabled -Dglvnd=disabled -Dvideo-codecs= \
	-Dtools= -Dbuild-tests=false > turnip-setup$SFX.log
ninja -C $BUILD src/freedreno/vulkan/libvulkan_freedreno.dylib > turnip-ninja$SFX.log
mkdir -p $OUT
cp $BUILD/src/freedreno/vulkan/libvulkan_freedreno.dylib $OUT/
install_name_tool -id /redwood/lib/libvulkan_freedreno.dylib $OUT/libvulkan_freedreno.dylib 2>/dev/null ||
	"$ROOT/xnu-deps/host/bin/install_name_tool" -id /redwood/lib/libvulkan_freedreno.dylib $OUT/libvulkan_freedreno.dylib
echo "mk-turnip.sh: Turnip $VERSION: $B/$OUT/libvulkan_freedreno.dylib ($(stat -c %s $OUT/libvulkan_freedreno.dylib) bytes)"
