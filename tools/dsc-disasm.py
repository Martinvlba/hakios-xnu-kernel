#!/usr/bin/env python3
"""Disassemble code of a dyld shared cache at the addresses the cache was built for (as tools/dsc-symbols.py says
them): for working out a protocol from the user-space side -- which IOKit selector a library's function calls, with
what scalars and structures.

usage: dsc-disasm.py CACHE_DIR ADDRESS [BYTES]      (CACHE_DIR holds dyld_shared_cache_arm64 and its .NN files;
                                                     BYTES default 400; -s CACHE.symbols names branch targets)
       dsc-disasm.py CACHE_DIR --bytes ADDRESS N    N bytes as hex and text (a string, a table)

Needs python3-capstone. Nothing of Apple's is kept: it reads the cache you extracted from your own IPSW.
"""
import bisect
import glob
import mmap
import os
import struct
import sys


class Cache:
    def __init__(self, directory):
        self.maps = []
        for path in sorted(glob.glob(os.path.join(directory, 'dyld_shared_cache_arm64*'))):
            if path.endswith('.symbols'):
                continue
            f = open(path, 'rb')
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if m[:7] != b'dyld_v1':
                continue
            offset, count = struct.unpack_from('<II', m, 0x10)
            for i in range(count):
                address, size, file_offset = struct.unpack_from('<QQQ', m, offset + 32 * i)
                self.maps.append((address, size, file_offset, m))

    def read(self, address, n):
        for start, size, file_offset, m in self.maps:
            if start <= address < start + size:
                at = file_offset + address - start
                return m[at:at + min(n, start + size - address)]
        sys.exit('dsc-disasm: %#x is in none of the cache\'s mappings' % address)


def symbols(path):
    with open(path, 'rb') as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    local_off, _ = struct.unpack_from('<QQ', m, 0x48)
    nlist_off, nlist_count, str_off, _, _, _ = struct.unpack_from('<IIIIII', m, local_off)
    out = []
    for i in range(nlist_count):
        strx, n_type, _, _, value = struct.unpack_from('<IBBHQ', m, local_off + nlist_off + 16 * i)
        if value and (n_type & 0x0e) == 0x0e:
            end = m.find(b'\0', local_off + str_off + strx)
            out.append((value, m[local_off + str_off + strx:end].decode('utf-8', 'replace')))
    out.sort()
    return out


def main():
    args = sys.argv[1:]
    names = []
    if '-s' in args:
        i = args.index('-s')
        names = symbols(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    cache = Cache(args[0])
    if args[1] == '--bytes':
        data = cache.read(int(args[2], 0), int(args[3], 0))
        for i in range(0, len(data), 16):
            row = data[i:i + 16]
            print('%#x  %-48s %s' % (int(args[2], 0) + i, row.hex(' '), ''.join(chr(b) if 32 <= b < 127 else '.' for b in row)))
        return
    import capstone
    address = int(args[1], 0)
    data = cache.read(address, int(args[2], 0) if len(args) > 2 else 400)
    keys = [v for v, _ in names]

    def name(a):
        i = bisect.bisect_right(keys, a) - 1
        if i >= 0 and a - keys[i] < 0x4000:
            return ' <%s%s>' % (names[i][1], '+%#x' % (a - keys[i]) if a != keys[i] else '')
        return ''

    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    for ins in md.disasm(bytes(data), address):
        target = ''
        if ins.mnemonic in ('bl', 'b', 'adr', 'adrp') or ins.mnemonic.startswith('b.') or ins.mnemonic in ('cbz', 'cbnz', 'tbz', 'tbnz'):
            try:
                target = name(int(ins.op_str.split('#')[-1], 0))
            except ValueError:
                pass
        print('%#x  %-8s %s%s' % (ins.address, ins.mnemonic, ins.op_str, target))


if __name__ == '__main__':
    main()
