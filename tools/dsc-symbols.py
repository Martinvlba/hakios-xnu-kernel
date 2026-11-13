#!/usr/bin/env python3
"""Local symbols of a dyld shared cache: the names the cache's .symbols file keeps for what the libraries do not export
(C++ vtables, static functions) -- for the addresses userland/src/inject*.c is built around, when the iOS build changes.

usage: dsc-symbols.py CACHE.symbols REGEX...     every local symbol whose name matches one, with its unslid address

The .symbols file is a subcache whose header (dyld_cache_header) gives localSymbolsOffset/Size; there
dyld_cache_local_symbols_info {nlistOffset, nlistCount, stringsOffset, stringsSize, entriesOffset, entriesCount}, each
nlist_64 (16 bytes: string index, type, section, desc, value) with the address as the library was laid out in the cache.
Nothing of Apple's is kept: it reads the file you extracted from your own IPSW.
"""
import mmap
import re
import struct
import sys


def main():
    path, patterns = sys.argv[1], [re.compile(p) for p in sys.argv[2:]]
    with open(path, 'rb') as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if m[:7] != b'dyld_v1':
        sys.exit('dsc-symbols: %s is not a dyld cache file' % path)
    local_off, local_size = struct.unpack_from('<QQ', m, 0x48)
    nlist_off, nlist_count, str_off, str_size, entries_off, entries_count = struct.unpack_from('<IIIIII', m, local_off)
    strings = local_off + str_off
    base = local_off + nlist_off
    for i in range(nlist_count):
        strx, n_type, n_sect, n_desc, value = struct.unpack_from('<IBBHQ', m, base + 16 * i)
        end = m.find(b'\0', strings + strx)
        name = m[strings + strx:end].decode('utf-8', 'replace')
        if any(p.search(name) for p in patterns):
            print('%#x %s' % (value, name))


if __name__ == '__main__':
    main()
