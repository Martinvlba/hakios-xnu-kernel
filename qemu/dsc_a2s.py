#!/usr/bin/env python3
"""Names for addresses in the Darwin dyld shared cache, without ipsw: the library an address is in and the nearest
exported symbol at or before it.

usage: dsc_a2s.py ADDRESS...           unslid addresses (0x1...), as qemu/symstack.py passes them
       from qemu.dsc_a2s import Cache; Cache(path).name(address)

Reads the split cache of sysimage/20F66__iPhone10,3_6 as Apple lays it out (iOS 16): the main file's header names its
images (path, load address) and its subcaches; every file has its own mappings from addresses to file offsets; each
image's Mach-O header gives its __TEXT range, its __LINKEDIT segment and its LC_SYMTAB, whose offsets are __LINKEDIT's
(nlist_64 entries and their strings, in the .dyldlinkedit subcache). This cache has no .symbols file, so a library's
local (not exported) functions have no names here: an address in one is named by the exported symbol before it, with
the distance -- the library is right, the function may not be. The table is built once and kept beside the cache
(a2s-table.pickle; a few seconds to build).
"""
import bisect
import glob
import os
import pickle
import struct
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, 'sysimage/20F66__iPhone10,3_6/dyld_shared_cache_arm64')

LC_SEGMENT_64, LC_SYMTAB = 0x19, 0x2
N_STAB, N_TYPE, N_SECT = 0xe0, 0x0e, 0x0e


class Cache:
    def __init__(self, path=CACHE):
        table = os.path.join(os.path.dirname(path), 'a2s-table.pickle')
        if os.path.exists(table) and os.path.getmtime(table) >= os.path.getmtime(path):
            self.images, self.symbols = pickle.load(open(table, 'rb'))
        else:
            self.images, self.symbols = self._build(path)
            pickle.dump((self.images, self.symbols), open(table, 'wb'))
        self.image_starts = [i[0] for i in self.images]
        self.symbol_addrs = [s[0] for s in self.symbols]

    @staticmethod
    def _build(path):
        files = [path] + sorted(f for f in glob.glob(path + '.*') if not f.endswith('.pickle'))
        maps = []                               # (address, size, file, offset)
        blobs = {}
        for f in files:
            data = open(f, 'rb').read()
            blobs[f] = data
            if data[:6] != b'dyld_v':
                continue
            mo, mc = struct.unpack_from('<II', data, 0x10)
            for i in range(mc):
                addr, size, off = struct.unpack_from('<QQQ', data, mo + 32 * i)
                maps.append((addr, size, f, off))

        def at(addr, n):
            for a, s, f, o in maps:
                if a <= addr < a + s:
                    return blobs[f][o + addr - a:o + addr - a + n]
            return None

        main = blobs[path]
        io, ic = struct.unpack_from('<II', main, 0x1c0)
        images, symbols = [], {}
        for i in range(ic):
            addr, _mt, _ino, pathoff = struct.unpack_from('<QQQI', main, io + 32 * i)
            name = main[pathoff:main.index(b'\0', pathoff)].decode()
            hdr = at(addr, 32)
            if hdr is None or struct.unpack_from('<I', hdr)[0] != 0xfeedfacf:
                continue
            ncmds, sizeofcmds = struct.unpack_from('<II', hdr, 16)
            cmds = at(addr + 32, sizeofcmds)
            text = linkedit = symtab = None
            p = 0
            for _ in range(ncmds):
                cmd, size = struct.unpack_from('<II', cmds, p)
                if cmd == LC_SEGMENT_64:
                    seg = cmds[p + 8:p + 24].rstrip(b'\0').decode()
                    vmaddr, vmsize, fileoff = struct.unpack_from('<QQQ', cmds, p + 24)
                    if seg == '__TEXT':
                        text = (vmaddr, vmsize)
                    elif seg == '__LINKEDIT':
                        linkedit = (vmaddr, fileoff)
                elif cmd == LC_SYMTAB:
                    symtab = struct.unpack_from('<IIII', cmds, p + 8)
                p += size
            if text:
                images.append((text[0], text[0] + text[1], os.path.basename(name)))
            if not (symtab and linkedit):
                continue
            symoff, nsyms, stroff, strsize = symtab
            nl = at(linkedit[0] + symoff - linkedit[1], 16 * nsyms)
            strs = at(linkedit[0] + stroff - linkedit[1], strsize)
            if nl is None or strs is None:
                continue
            for k in range(nsyms):
                strx, ntype, _sect, _desc, value = struct.unpack_from('<IBBHQ', nl, 16 * k)
                if ntype & N_STAB or (ntype & N_TYPE) != N_SECT or value == 0:
                    continue
                end = strs.find(b'\0', strx)
                symbols.setdefault(value, strs[strx:end].decode(errors='replace'))
        images.sort()
        return images, sorted(symbols.items())

    def name(self, addr):
        i = bisect.bisect_right(self.image_starts, addr) - 1
        if i < 0 or addr >= self.images[i][1]:
            return '?'
        lib = self.images[i][2]
        j = bisect.bisect_right(self.symbol_addrs, addr) - 1
        if j >= 0 and self.symbol_addrs[j] >= self.images[i][0]:
            return f'{lib}: {self.symbols[j][1]} + {addr - self.symbol_addrs[j]:#x}'
        return f'{lib} + {addr - self.images[i][0]:#x}'


if __name__ == '__main__':
    c = Cache()
    for a in sys.argv[1:]:
        print(a, c.name(int(a, 16)))
