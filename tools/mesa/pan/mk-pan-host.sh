#!/bin/bash
# Mesa 26.2.5's panfrost compiler built on this machine (the host), and tools/mesa/pan/pan_compile_test with it: the
# shaders the Pixel 8a's Mali-G715 (arch 11) runs, compiled by Mesa's own compiler -- the one the Metal driver's Mali
# back end will use (devices/akita/PLAN.md). Mesa is fetched (a pinned release); nothing of it is in the repository.
#
#   tools/mesa/pan/mk-pan-host.sh            builds, and writes userland/src/mali_store_index.h (a shader built in NIR)
#                                            and userland/src/mali_air_fill_id.h (hakios-metal's fill_id kernel, AIR,
#                                            through our front end tools/mesa/air: air_pan)
#
# Mesa's NIR library is a stub unless some driver is configured, so softpipe and OpenGL are (nothing of them is used:
# LLVM-free, so no OpenCL precompiler, which panvk would want). Needs: meson, ninja, python3-mako, libdrm-dev, zlib.
set -e
VERSION=26.2.5
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
B=$ROOT/xnu-deps/build/mesa26
mkdir -p "$B"
cd "$B"
[ -f mesa-$VERSION.tar.xz ] || curl -sL --fail -o mesa-$VERSION.tar.xz https://archive.mesa3d.org/mesa-$VERSION.tar.xz
[ -d mesa-$VERSION ] || tar xJf mesa-$VERSION.tar.xz
# our AIR front end (tools/mesa/air) and the one line of meson that reaches it, as tools/mk-mesa.sh does for 25.0.7
rm -rf mesa-$VERSION/src/compiler/air
# tools/mesa/air-mesa26: the front end the Mali build links (Mesa 26 names); tools/mesa/air, merged from upstream, still has
# Mesa 25 names and does not build here. AIR_SRC= for another.
cp -r "${AIR_SRC:-$ROOT/tools/mesa/air-mesa26}" mesa-$VERSION/src/compiler/air
grep -q "subdir('air')" mesa-$VERSION/src/compiler/meson.build || patch -s -d mesa-$VERSION -p1 < "$ROOT/tools/mesa/air-frontend.patch"
[ -f build-host/build.ninja ] || meson setup build-host mesa-$VERSION --buildtype=debugoptimized -Dplatforms= \
	-Dgallium-drivers=softpipe -Dvulkan-drivers= -Dtools=panfrost -Dglx=disabled -Degl=disabled -Dgbm=disabled \
	-Dllvm=disabled -Dopengl=true -Dgles1=disabled -Dgles2=enabled -Dzstd=disabled -Dexpat=disabled -Dvalgrind=disabled \
	-Dlibunwind=disabled -Dglvnd=disabled -Dvideo-codecs= -Dlmsensors=disabled -Dbuild-tests=false \
	-Dshader-cache=disabled > meson-host.log
LIBS="src/panfrost/compiler/libpanfrost_compiler.a src/panfrost/compiler/bifrost/libpanfrost_bifrost.a
src/panfrost/compiler/bifrost/libpanfrost_bifrost_disasm.a src/panfrost/compiler/bifrost/valhall/libpanfrost_valhall_disasm.a
src/panfrost/compiler/midgard/libpanfrost_midgard.a src/panfrost/compiler/midgard/libpanfrost_midgard_disasm.a
src/panfrost/model/libpanfrost_model_lib.a src/compiler/nir/libnir.a src/compiler/libcompiler.a src/util/libmesa_util.a
src/util/libmesa_util_simd.a src/util/libmesa_util_clflush.a src/util/libmesa_util_clflushopt.a src/util/blake3/libblake3.a
src/c11/impl/libmesa_util_c11.a"
ninja -C build-host $LIBS src/compiler/air/libair.a src/panfrost/bifrost_compiler > ninja-host.log
cd build-host
# the flags Mesa's own build gives its panfrost tool (the configuration's defines and include paths)
FLAGS=$(python3 - <<'EOF'
import json, shlex
for e in json.load(open('compile_commands.json')):
    if e['file'].endswith('bifrost/cmdline.c'):
        a, out, skip = shlex.split(e['command']), [], False
        for x in a[1:]:
            if skip: skip = False; continue
            if x in ('-o', '-MF', '-MQ'): skip = True; continue
            if x == '-c' or x.endswith('.c') or x.startswith('-M'): continue
            out.append(x)
        print(' '.join(shlex.quote(x) for x in out))
EOF
)
eval cc $FLAGS -c -o pan_compile_test.o "$ROOT/tools/mesa/pan/pan_compile_test.c"
c++ -o pan_compile_test pan_compile_test.o -Wl,--start-group $LIBS -Wl,--end-group -lm -lpthread -lz -ldl
./pan_compile_test "$ROOT/userland/src/mali_store_index.h"
# AIR: hakios-metal's fill_id kernel (beside this repository: ../hakios-metal or HAKIOS_METAL=) through air_pan
eval cc $FLAGS -DAIR_MESA_NEW_NIR -c -o air_pan.o "$ROOT/tools/mesa/pan/air_pan.c"
c++ -o air_pan air_pan.o -Wl,--start-group src/compiler/air/libair.a $LIBS -Wl,--end-group -lm -lpthread -lz -ldl
HM=${HAKIOS_METAL:-$(cd "$ROOT/.." && pwd)/hakios-metal}
[ -f "$HM/tools/kernels/fill_id.ll" ] || HM=$HOME/akita/ref/hakios-metal
llvm-as "$HM/tools/kernels/fill_id.ll" -o fill_id.bc
./air_pan fill_id.bc "" "$ROOT/userland/src/mali_air_fill_id.h"
# hakios-metal's Mali blits (beside this repository): copy and fill shaders
eval cc $FLAGS -c -o pan_blit_shaders.o "$ROOT/tools/mesa/pan/pan_blit_shaders.c"
c++ -o pan_blit_shaders pan_blit_shaders.o -Wl,--start-group $LIBS -Wl,--end-group -lm -lpthread -lz -ldl
if [ -d "$HM/src" ] && [ "$HM" != "$HOME/akita/ref/hakios-metal" ]; then
	mkdir -p "$HM/src/mali"
	./pan_blit_shaders "$HM/src/mali/blit_shaders.h"
fi
echo "mk-pan-host.sh: $B/build-host/pan_compile_test; userland/src/mali_store_index.h written"
