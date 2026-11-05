#!/usr/bin/env python3
"""A .metallib of compute kernels from their AIR bitcode, as a C header (for metal_test's compute step).

    mk-metallib.py OUT.h NAME FILE.bc [FILE.bc ...]     each function named after its file (fill_id.bc -> fill_id)

The container is the one hakios-metal's plugin/hakios_metallib.c reads (and its tests/metallib_test.cpp builds): the
header, the function list (NAME, TYPE 2 = kernel, VERS, OFFT, MDSZ per function), the modules. Nothing of Apple's is in
it but the layout.
"""
import os
import struct
import sys


def tag(name, value):
    return name.encode() + struct.pack('<H', len(value)) + value


def entry(name, offset, size):
    e = tag('NAME', name.encode() + b'\0') + tag('TYPE', bytes([2])) + tag('VERS', struct.pack('<4H', 2, 8, 4, 0))
    e += tag('OFFT', struct.pack('<3Q', 0, 0, offset)) + tag('MDSZ', struct.pack('<Q', size)) + b'ENDT'
    return struct.pack('<I', len(e) + 4) + e


def main():
    out, var, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    modules, entries = b'', b''
    for f in files:
        data = open(f, 'rb').read()
        entries += entry(os.path.splitext(os.path.basename(f))[0], len(modules), len(data))
        modules += data
    lst = struct.pack('<I', len(files)) + entries
    head = bytearray(0x58)
    head[0:4] = b'MTLB'
    struct.pack_into('<HH', head, 4, 32769, 2)
    list_at = len(head)
    modules_at = list_at + len(lst)
    total = modules_at + len(modules)
    struct.pack_into('<QQQ', head, 0x10, total, list_at, len(lst) - 4)
    struct.pack_into('<QQ', head, 0x48, modules_at, len(modules))
    lib = bytes(head) + lst + modules
    with open(out, 'w') as h:
        h.write('/* made by tools/mesa/pan/mk-metallib.py from %s: a .metallib of compute kernels (AIR) */\n'
                % ' '.join(os.path.basename(f) for f in files))
        h.write('static const unsigned char %s[%u] = {' % (var, len(lib)))
        for i, b in enumerate(lib):
            h.write(('\n\t' if i % 16 == 0 else ' ') + '0x%02x,' % b)
        h.write('\n};\n')


main()
