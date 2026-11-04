#!/bin/bash
# Mesa 26.2.5's panfrost compiler cross-built for the phone (arm64 iOS 16.5): the libraries the Metal driver's Mali
# back end links, to compile shaders on the Pixel 8a itself for its Mali-G715 -- what tools/mk-mesa.sh does with 25.0.7
# for redwood's Adreno, with the same cross compiler (xnu-deps/host/bin/xnu-clang) and SDK. Only the compiler's
# archives are built (NIR, the compiler's glue, panfrost's compiler, util); the configuration is mk-pan-host.sh's.
#
#   tools/mesa/pan/mk-pan-ios.sh       -> xnu-deps/build/mesa26/build-ios/...  (archives listed in out-ios/archives)
set -e
VERSION=26.2.5
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
source "$ROOT/xnu-env.sh"
B=$ROOT/xnu-deps/build/mesa26
SDK=$THEOS/sdks/iPhoneOS16.5.sdk
CC=$ROOT/xnu-deps/host/bin/xnu-clang
mkdir -p "$B/out-ios"
cd "$B"
[ -f mesa-$VERSION.tar.xz ] || curl -sL --fail -o mesa-$VERSION.tar.xz https://archive.mesa3d.org/mesa-$VERSION.tar.xz
[ -d mesa-$VERSION ] || tar xJf mesa-$VERSION.tar.xz
# panfrost's OpenCL precompiler (src/panfrost/clc) runs a host program against panfrost's own libraries, which a cross
# build cannot have: it is for the drivers (libpan's kernels), not the compiler, so only the drivers take it in
python3 - mesa-$VERSION/src/panfrost/meson.build <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("if with_gallium_panfrost or with_panfrost_vk or with_tools.contains('panfrost')\n  subdir('clc')",
              "if with_gallium_panfrost or with_panfrost_vk\n  subdir('clc')")
open(p, 'w').write(s)
PY
# our AIR front end (tools/mesa/air), as mk-pan-host.sh puts it there
rm -rf mesa-$VERSION/src/compiler/air
cp -r "$ROOT/tools/mesa/air" mesa-$VERSION/src/compiler/air
grep -q "subdir('air')" mesa-$VERSION/src/compiler/meson.build || patch -s -d mesa-$VERSION -p1 < "$ROOT/tools/mesa/air-frontend.patch"
cat > cross-ios.ini <<INI
[binaries]
c = ['$CC', '-arch', 'arm64', '-isysroot', '$SDK', '-miphoneos-version-min=16.0']
cpp = ['$CC++', '-arch', 'arm64', '-isysroot', '$SDK', '-miphoneos-version-min=16.0', '-fuse-cxa-atexit']
objc = ['$CC', '-arch', 'arm64', '-isysroot', '$SDK']
ar = 'llvm-ar'
ranlib = 'llvm-ranlib'
strip = 'llvm-strip'
pkg-config = 'false'

[host_machine]
system = 'darwin'
subsystem = 'ios'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'

[properties]
needs_exe_wrapper = true
INI
[ -f build-ios/build.ninja ] || meson setup build-ios mesa-$VERSION --cross-file cross-ios.ini --wrap-mode=nofallback \
	--buildtype=release -Db_ndebug=true -Dplatforms= -Dgallium-drivers=softpipe -Dvulkan-drivers= -Dglx=disabled \
	-Degl=disabled -Dgbm=disabled -Dllvm=disabled -Dopengl=true -Dgles1=disabled -Dgles2=enabled -Dzstd=disabled \
	-Dexpat=disabled -Dvalgrind=disabled -Dlibunwind=disabled -Dglvnd=disabled -Dvideo-codecs= -Dlmsensors=disabled \
	-Dbuild-tests=false -Dshader-cache=disabled -Dxmlconfig=disabled -Dtools=panfrost > meson-ios.log
LIBS="src/compiler/air/libair.a src/panfrost/compiler/libpanfrost_compiler.a src/panfrost/compiler/bifrost/libpanfrost_bifrost.a
src/panfrost/compiler/bifrost/libpanfrost_bifrost_disasm.a src/panfrost/compiler/bifrost/valhall/libpanfrost_valhall_disasm.a
src/panfrost/compiler/midgard/libpanfrost_midgard.a src/panfrost/compiler/midgard/libpanfrost_midgard_disasm.a
src/panfrost/model/libpanfrost_model_lib.a src/compiler/nir/libnir.a src/compiler/libcompiler.a src/util/libmesa_util.a
src/util/blake3/libblake3.a src/c11/impl/libmesa_util_c11.a"
ninja -C build-ios $LIBS > ninja-ios.log
for l in $LIBS; do echo "$B/build-ios/$l"; done > out-ios/archives
echo "mk-pan-ios.sh: $(wc -l < out-ios/archives) archives for arm64 iOS in $B/build-ios"
# the flags Mesa's iOS build compiles its compiler with (defines, include paths), for a C file that calls it from
# outside the build -- hakios-metal's src/mali/compile_mali.c (its Makefile reads out-ios/cflags and out-ios/archives)
(cd build-ios && python3 - <<'PY'
import json, shlex
for e in json.load(open('compile_commands.json')):
    if e['file'].endswith('compiler/pan_compiler.c'):
        a, out, skip = shlex.split(e['command']), [], False
        for x in a[1:]:
            if skip: skip = False; continue
            if x in ('-o', '-MF', '-MQ'): skip = True; continue
            if x == '-c' or x.endswith('.c') or x.startswith('-M'): continue
            if x.startswith('-I') and not x.startswith('-I/'): x = '-I' + __import__('os').path.abspath(x[2:])
            out.append(x)
        print(' '.join(shlex.quote(x) for x in out)); break
PY
) > out-ios/cflags
echo "mk-pan-ios.sh: flags in $B/out-ios/cflags"
