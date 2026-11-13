#!/usr/bin/env python3
"""Names for a "userabort:" line of the kernel's log (shared cache slide SLIDE; called from: ADDR ...), from the
shared cache's local symbols (the .symbols file; tools/dsc-symbols.py says what that is).

usage: dsc-backtrace.py CACHE.symbols < LINES      each line's frames, the nearest symbol below and the distance
"""
import bisect
import mmap
import re
import struct
import sys

f = open(sys.argv[1], 'rb')
m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
lo, _ = struct.unpack_from('<QQ', m, 0x48)
no, nc, so, ss, eo, ec = struct.unpack_from('<IIIIII', m, lo)
syms = []
for i in range(nc):
    strx, t, s, d, v = struct.unpack_from('<IBBHQ', m, lo + no + 16 * i)
    if v and s == 1:
        syms.append((v, strx))
syms.sort()
keys = [v for v, _ in syms]
for line in sys.stdin:
    slide = re.search(r'slide (0x[0-9a-f]+)', line)
    if not slide or 'called from:' not in line:
        continue
    slide = int(slide.group(1), 16)
    print(line.split('called from:')[0].strip()[-90:])
    for a in re.findall(r'0x[0-9a-f]+', line.split('called from:')[1]):
        u = int(a, 16) - slide
        if not 0x180000000 <= u < 0x280000000:
            print('    %s (not in the cache)' % a)
            continue
        v, strx = syms[bisect.bisect_right(keys, u) - 1]
        e = m.find(b'\0', lo + so + strx)
        print('    %#x %s + %d' % (u, m[lo + so + strx:e].decode('utf-8', 'replace')[:120], u - v))
