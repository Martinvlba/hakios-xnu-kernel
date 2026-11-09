#!/usr/bin/env python3
"""ca-metallib.py OUTDIR [SYSIMAGE]: QuartzCore's default.metallib for CoreAnimation drawing through Metal.

This image is the iPhone X's, and its QuartzCore carries no default.metallib: only default.pipelinelib, whose
"functions" file holds the four uber shaders (fixed_vert_lph_gen/_spc, fixed_frag_lph_cpf/_cph) as 451 variants
already specialised for the pipelines pipelines.desc names. The window server's Metal renderer wants the library
itself, with the rest of CoreAnimation's functions (blurs, clears, paths, copies): those are taken from the air64
slice of THIS MAC's QuartzCore.framework/default.metallib (a later CoreAnimation, read where it lies and never put
in the repository), with one variant of each of the four uber shaders put in their place -- the plug-in compiles the
variant a pipeline's label names (userland/src/HakiosMetalCA.m, hm_bitcode_for()). From an iPhone's own library of
the same CoreAnimation (below) the uber shaders are kept generic and specialised on the phone with each pipeline's
function constants.

OUTDIR/QuartzCore-default.metallib is written; build the image with EXTRA_SHARE=OUTDIR and run with
ca_metallib=/redwood/share/<OUTDIR's name>/QuartzCore-default.metallib (userland/src/inject.c).

Off a Mac (Windows, WSL, Linux): CA_QC_METALLIB=FILE names the later CoreAnimation's library to take the air64 slice
from -- a copy of a Mac's QuartzCore.framework/Versions/A/Resources/default.metallib, or iPadOS/iOS 17's
QuartzCore.framework/default.metallib out of its IPSW (7z x the root filesystem's dmg), both universal with an air64
slice -- or an iOS 16 iPhone's own (the iPhone 11's 20F66 has one: a plain MTLB of the same CoreAnimation as this
image, taken as it is). Without it, the Mac's own path above.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_metallib  # noqa: E402

MAC_QC = os.environ.get('CA_QC_METALLIB') or '/System/Library/Frameworks/QuartzCore.framework/Versions/A/Resources/default.metallib'
AIR64 = 0x01000017


def air64_slice(path):
    with open(path, 'rb') as f:
        if f.read(4) == b'MTLB':
            f.seek(0)
            return f.read()             # a library of one platform already (an iPhone's: iOS 16's own, e.g. iPhone 11's)
        f.seek(0)
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 0xcafebabe, '%s: not a universal file' % path
        for _ in range(n):
            cputype, _sub, off, size, _align = struct.unpack('>iiIII', f.read(20))
            if cputype == AIR64:
                f.seek(off)
                return f.read(size)
    raise SystemExit('%s: no air64 slice' % path)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    image = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'sysimage', 'rootfs')
    functions = open(os.path.join(image, 'System/Library/Frameworks/QuartzCore.framework/default.pipelinelib/functions'),
                     'rb').read()
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, '.part.metallib')
    variants, at = {}, 0
    while (at := functions.find(b'MTLB', at)) >= 0:
        total = struct.unpack_from('<Q', functions, at + 0x10)[0]
        open(tmp, 'wb').write(functions[at:at + total])
        lib = merge_metallib.read(tmp)
        variants.setdefault(dict(lib['funcs'][0]['tags'])[b'NAME'], (lib['funcs'][0], lib['head']))
        at += total
    if not os.path.isfile(MAC_QC):
        raise SystemExit('%s: not found. Off a Mac, name a later CoreAnimation\'s library: CA_QC_METALLIB=FILE '
                         '(a Mac\'s QuartzCore default.metallib copied over, or iOS 17\'s out of its IPSW)' % MAC_QC)
    mac = os.path.join(out, '.air64.metallib')
    air = air64_slice(MAC_QC)
    open(mac, 'wb').write(air)
    lib = merge_metallib.read(mac)
    # An iPhone's own library (a plain MTLB: the iPhone 11's 20F66, the same CoreAnimation as this image) keeps its
    # uber shaders as they are: generic, with the function constants CoreAnimation sets per pipeline (fc_color_opcodes0
    # 58, fc_color_inputs0 56, ...), which the plug-in's compiler specialises on the phone for every pipeline exactly.
    # Swapped for one of this image's variants each (a later CoreAnimation's, whose constants are not this one's, has
    # to be), the pipelines the variants do not name were drawn by the nearest variant: whole frames blue or cyan at an
    # app's launch (11-09). CA_UBER=variants swaps them anyway.
    keep = air[:4] == b'MTLB' and os.environ.get('CA_UBER') != 'variants'
    if not keep:
        lib['funcs'] = [variants[n][0] if (n := dict(f['tags'])[b'NAME']) in variants else f for f in lib['funcs']]
        lib['head'] = next(iter(variants.values()))[1]
    merge_metallib.write([lib], os.path.join(out, 'QuartzCore-default.metallib'))
    os.remove(tmp)
    os.remove(mac)
    print('%s: %d functions, %s' % (os.path.join(out, 'QuartzCore-default.metallib'), len(lib['funcs']),
          'the uber shaders its own (generic)' if keep else '%d of them this image\'s uber shader variants' % len(variants)))


if __name__ == '__main__':
    main()
