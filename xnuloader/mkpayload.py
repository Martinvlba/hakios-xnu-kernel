#!/usr/bin/env python3
"""
Build the "XNUP" payload consumed by the boot shim and the XNU loader.

  header : u32 magic "XNUP", u32 size (bytes after the header)
  entries: char name[16], u32 size, u32 reserved, data padded to 16 bytes
  end    : an entry whose name is empty

usage: mkpayload.py -o payload.bin name=file [name=file ...] [--cmdline "..."]
"""
import argparse, struct

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True)
ap.add_argument('--cmdline', default='')
ap.add_argument('--physbase', default='', help='hex: force XNU physical base (testing)')
ap.add_argument('--usb', action='store_true', help='make the loader stay resident and take commands over USB')
ap.add_argument('--rdres', default='', help='hex size of a root image already placed at the resident address (QEMU)')
ap.add_argument('--checkpoint', default='', help='kernel checkpoint to stop at; 999 = loader self test')
ap.add_argument('--kernel-version', default='',
                help='the kernel.ver the loader prints under its banner (default: tools/kernel-version.py)')
ap.add_argument('--no-version', action='store_true', help='no kernel.ver entry')
ap.add_argument('items', nargs='*')
args = ap.parse_args()

body = b''
def add(name, data):
    global body
    body += struct.pack('<16sII', name.encode(), len(data), 0) + data + b'\0' * (-len(data) % 16)

for item in args.items:
    name, path = item.split('=', 1)
    add(name, open(path, 'rb').read())
if args.cmdline:
    add('cmdline', args.cmdline.encode() + b'\0')
if args.physbase:
    add('physbase', args.physbase.lower().replace('0x', '').encode() + b'\0')
if args.rdres:
    add('rdres', args.rdres.lower().replace('0x', '').encode() + b'\0')
if args.usb:
    add('usb', b'1\0')
if args.checkpoint:
    add('checkpoint', args.checkpoint.encode() + b'\0')
# The version of the kernel in this payload, which the loader prints as "kernel.ver=" under its banner and
# hands to the kernel as the boot-arg hakios_kver. A loader that does not know the entry skips it.
if not args.no_version:
    version = args.kernel_version
    if not version:
        import os, subprocess, sys
        tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'kernel-version.py')
        try:
            version = subprocess.run([sys.executable, tool], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            print('mkpayload.py: no kernel.ver (tools/kernel-version.py failed: %s)' % e, file=sys.stderr)
    if version:
        add('kernel.ver', version.encode() + b'\0')
body += b'\0' * 24
open(args.output, 'wb').write(struct.pack('<II', 0x50554E58, len(body)) + body)
print(f'{args.output}: {len(body) + 8} bytes')
