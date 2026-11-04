#!/bin/bash
# Mesa 26.2.5's panfrost compiler built on this machine (the host), and tools/mesa/pan/pan_compile_test with it: the
# shaders the Pixel 8a's Mali-G715 (arch 11) runs, compiled by Mesa's own compiler -- the one the Metal driver's Mali
# back end will use (devices/akita/PLAN.md). Mesa is fetched (a pinned release); nothing of it is in the repository.
#
#   tools/mesa/pan/mk-pan-host.sh            builds, and writes userland/src/mali_store_index.h
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
ninja -C build-host $LIBS src/panfrost/bifrost_compiler > ninja-host.log
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
echo "mk-pan-host.sh: $B/build-host/pan_compile_test; userland/src/mali_store_index.h written"
