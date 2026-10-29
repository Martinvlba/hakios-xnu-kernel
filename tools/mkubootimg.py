#!/usr/bin/env python3
"""
HakiOS's U-Boot as an Android boot image for `fastboot boot`: started from RAM by the phone's own bootloader (ABL),
nothing flashed. U-Boot takes the HakiOS loader and its payload from the image's ramdisk and starts them from memory
(the hakios command, uefi/u-boot/files/cmd/hakios.c); the boot partition is never read or written.

  kernel  = gzip(u-boot.bin) + the phone's Android device tree   (what ABL expects, as for Silicium's image)
  ramdisk = an "HKUB" container: loader.efi and an XNUP payload  (or nothing: U-Boot then waits for them over fastboot)

The same container on its own (-c) is what the host sends to a U-Boot that is already running:

  fastboot stage x.hkub && fastboot oem run:"hakios load fastboot" && fastboot continue

  HKUB, little-endian: u32 magic "HKUB", u32 version 1, u32 count, u32 header size,
                       count x { char name[16]; u32 offset; u32 size; u32 crc32; u32 reserved }, entries 4 KiB aligned

usage: mkubootimg.py [-d DEVICE] [-u u-boot.bin] [-l loader.efi] [-p payload.bin] [-o out.img] [-c out.hkub] [--env F | --install]
       mkubootimg.py -p payload.bin -c x.hkub        the container alone
"""
import argparse, gzip, os, struct, subprocess, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HKUB_MAGIC, HKUB_VERSION = b'HKUB', 1
PAYLOAD_MAX = 0x04000000 - 0x100000          # the loader's payload area less the panic log (xnuloader/loader.c)


def container(entries):
    """entries: [(name, bytes)] -> an HKUB container"""
    header_size = 16 + 32 * len(entries)
    offset = (header_size + 0xfff) & ~0xfff
    table, body = b'', b''
    for name, data in entries:
        assert len(name) < 16
        table += struct.pack('<16sIIII', name.encode(), offset + len(body), len(data), zlib.crc32(data), 0)
        body += data + b'\0' * (-len(data) % 0x1000)
    head = struct.pack('<4sIII', HKUB_MAGIC, HKUB_VERSION, len(entries), header_size) + table
    return head + b'\0' * (offset - len(head)) + body


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-d', '--device', default='redwood')
    ap.add_argument('-u', '--u-boot', help='default: u-boot/build-DEVICE/u-boot.bin, else devices/DEVICE/prebuilt/u-boot.bin')
    ap.add_argument('-l', '--loader', help='loader.efi (default: xnuloader/loader.efi, else prebuilt/loader.efi)')
    ap.add_argument('-p', '--payload', help='an XNUP payload (xnuloader/mkpayload.py)')
    ap.add_argument('-o', '--output', help='the boot image (default: hakios-u-boot.img)')
    ap.add_argument('-c', '--container', help='write the HKUB container on its own (for fastboot stage); without -o, only it')
    ap.add_argument('--no-ramdisk', action='store_true', help='U-Boot only: it waits for the loader over fastboot')
    ap.add_argument('--env', help='an environment for U-Boot to take with the loader (name=value lines): what the image does')
    ap.add_argument('--install', action='store_true',
                    help='the image for a boot partition: with devices/DEVICE/u-boot/install.env (the menu: Android or HakiOS)')
    args = ap.parse_args()

    dev = os.path.join(ROOT, 'devices', args.device)
    ub = args.u_boot
    if not ub:
        for c in (os.path.join(ROOT, 'u-boot', f'build-{args.device}', 'u-boot.bin'),
                  os.path.join(dev, 'prebuilt', 'u-boot.bin')):
            if os.path.isfile(c):
                ub = c
                break
        else:
            sys.exit(f'mkubootimg.py: no u-boot.bin for {args.device}: uefi/u-boot/apply.sh {args.device} && '
                     f'uefi/u-boot/build.sh {args.device}')
    loader = args.loader
    if not loader:
        loader = os.path.join(ROOT, 'xnuloader', 'loader.efi')
        if not os.path.isfile(loader):
            loader = os.path.join(ROOT, 'prebuilt', 'loader.efi')
    dtb = os.path.join(dev, 'prebuilt', 'device-tree.dtb')
    mkbootimg = os.path.join(dev, 'prebuilt', 'mkbootimg.py')
    for path in (ub, loader, dtb, mkbootimg):
        if not os.path.isfile(path):
            sys.exit(f'mkubootimg.py: {path} not found')

    u_boot = open(ub, 'rb').read()
    # ABL starts it as an arm64 Linux kernel: the header's magic "ARM\x64" at 0x38
    if u_boot[0x38:0x3c] != b'ARM\x64':
        sys.exit(f'mkubootimg.py: {ub} has no Linux image header (CONFIG_LINUX_KERNEL_IMAGE_HEADER)')

    entries = []
    if not args.no_ramdisk:
        efi = open(loader, 'rb').read()
        if efi[:2] != b'MZ':
            sys.exit(f'mkubootimg.py: {loader} is not a UEFI application')
        if args.payload:
            data = open(args.payload, 'rb').read()
            if data[:4] != b'XNUP' or len(data) > PAYLOAD_MAX:
                sys.exit(f'mkubootimg.py: {args.payload}: not an XNUP payload, or over {PAYLOAD_MAX} bytes')
            entries.append(('payload', data))
        # after the payload: a loader.efi the payload carries (xl enter's) gives way to this one
        entries.append(('loader', efi))
        env = args.env or (os.path.join(dev, 'u-boot', 'install.env') if args.install else None)
        if env:
            if not os.path.isfile(env):
                sys.exit(f'mkubootimg.py: {env} not found')
            # last: U-Boot takes it once the loader and the payload are in place
            entries.append(('env', open(env, 'rb').read()))
    hkub = container(entries) if entries else b''
    if args.container:
        open(args.container, 'wb').write(hkub)
        print(f'{args.container}: {len(hkub)} bytes ({", ".join(n for n, _ in entries)})')
        if not args.output:
            return                          # the container alone, for a U-Boot that is already running

    out = args.output or os.path.join(ROOT, 'hakios-u-boot.img')
    kernel, ramdisk = out + '.kernel~', out + '.ramdisk~'
    open(kernel, 'wb').write(gzip.compress(u_boot, 6) + open(dtb, 'rb').read())
    # no container: one zero byte for ABL, which hakios load does not take, so U-Boot waits for the host instead
    open(ramdisk, 'wb').write(hkub or b'\0')
    try:
        subprocess.check_call([sys.executable, mkbootimg, '--kernel', kernel, '--ramdisk', ramdisk,
                               '--header_version', '1', '-o', out])
    finally:
        os.unlink(kernel)
        os.unlink(ramdisk)
    print(f'{out}: {os.path.getsize(out)} bytes (u-boot {len(u_boot)}, ramdisk {len(hkub)}: '
          f'{", ".join(f"{n} {len(d)}" for n, d in entries) or "none"})')


if __name__ == '__main__':
    main()
