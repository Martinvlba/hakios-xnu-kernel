#!/usr/bin/env python3
"""
HakiOS's U-Boot as an Android boot image for `fastboot boot`: started from RAM by the phone's own bootloader (ABL),
nothing flashed. U-Boot takes the HakiOS loader and its payload from the image's ramdisk and starts them from memory
(the hakios command, uefi/u-boot/files/cmd/hakios.c); the boot partition is never read or written.

  kernel  = gzip(u-boot.bin) + the phone's Android device tree   (what ABL expects, as for Silicium's image)
  ramdisk = an "HKUB" container: loader.efi and an XNUP payload  (or nothing: U-Boot then waits for them over fastboot)

A device whose bootloader wants another kind of image says so in devices/DEVICE/u-boot/boot-image (boot_image_style():
the Pixel 8a's is header version 4, u-boot.bin as it is, the device tree vendor_boot's).

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


def avb_hash_footer(image, partition_size, partition_name='boot'):
    """`image` with an unsigned AVB hash footer, `partition_size` bytes long: what avbtool add_hash_footer --algorithm NONE
    makes. A bootloader whose vbmeta chains the boot partition (Google's on zuma) refuses an image with no footer at all,
    unlocked or not; an unlocked one takes an unsigned footer as a verification error, which it allows. Laid out as
    libavb reads it (avb_footer.h, avb_vbmeta_image.h, avb_hash_descriptor.h; all big-endian): the image padded to 4 KiB,
    the vbmeta image (a 256-byte header, no authentication block, an auxiliary block with one hash descriptor), zeros,
    and the 64-byte footer in the partition's last bytes."""
    import hashlib
    block = 4096
    salt = b''
    digest = hashlib.sha256(salt + image).digest()
    name = partition_name.encode()
    # the hash descriptor: tag 2, the bytes after the first 16, then image size, algorithm, lengths, flags, reserved
    body = struct.pack('>Q32sLLLL60s', len(image), b'sha256', len(name), len(salt), len(digest), 0, b'')
    body += name + salt + digest
    body += b'\0' * (-(16 + len(body)) % 8)
    descriptor = struct.pack('>QQ', 2, len(body)) + body
    aux = descriptor + b'\0' * (-len(descriptor) % 64)
    header = struct.pack('>4sLLQQL' 'QQQQQQQQQQ' 'QLL48s80s',
                         b'AVB0', 1, 0,             # required libavb 1.0
                         0, len(aux),               # authentication block (none: unsigned), auxiliary block
                         0,                         # algorithm NONE
                         0, 0, 0, 0,                # hash, signature: none
                         0, 0, 0, 0,                # public key and its metadata: none
                         0, len(descriptor),        # the descriptors, at the start of the auxiliary block
                         0, 0, 0,                   # rollback index, flags, rollback index location
                         b'hakios mkubootimg', b'')
    assert len(header) == 256
    vbmeta = header + aux
    vbmeta_offset = len(image) + (-len(image) % block)
    end = vbmeta_offset + len(vbmeta)
    if end + 64 > partition_size:
        sys.exit(f'mkubootimg.py: the image and its vbmeta ({end} bytes) do not fit a partition of {partition_size}')
    footer = struct.pack('>4sLLQQQ28s', b'AVBf', 1, 0, len(image), vbmeta_offset, len(vbmeta), b'')
    assert len(footer) == 64
    return (image + b'\0' * (vbmeta_offset - len(image)) + vbmeta + b'\0' * (partition_size - 64 - end) + footer)


def with_container(u_boot, hkub):
    """u-boot.bin with the HKUB container behind it, for `container kernel`. Before it relocates itself U-Boot keeps its
    stack 512 KiB past the start of its BSS (INIT_SP_RELATIVE, SYS_INIT_SP_BSS_OFFSET), which is past even the image
    size its Linux header declares: the container starts beyond both, at a 64 KiB boundary (where image_container()
    looks), and the header's image_size is made to cover it, so that the bootloader puts nothing of its own there."""
    image_size = struct.unpack_from('<Q', u_boot, 16)[0]
    at = max(image_size, len(u_boot) + 0x80000)
    at += -at % 0x10000
    out = bytearray(u_boot + b'\0' * (at - len(u_boot)) + hkub)
    struct.pack_into('<Q', out, 16, len(out) + (-len(out) % 0x1000))
    return bytes(out)


def boot_image_style(dev):
    """How the device's bootloader wants the image: devices/DEVICE/u-boot/boot-image, `key value` lines.

      header_version  1 (Qualcomm's ABL, the default) or 3/4 (a bootloader that takes the device tree from
                      vendor_boot and puts vendor_boot's ramdisks in front of this one: Google's on a Pixel)
      kernel          gzip+dtb (the default: gzip(u-boot.bin) with the phone's Android tree appended, as ABL wants),
                      raw (u-boot.bin as it is; the bootloader hands over its own device tree), or lz4 (the same in
                      LZ4's legacy frame, `lz4 -l`: what a Pixel's own boot partition holds)
      os_version, os_patch_level   written into the header when given (mkbootimg's --os_version, --os_patch_level)
      container       ramdisk (the default: the HKUB container is the image's ramdisk) or kernel (behind u-boot.bin,
                      in the kernel: for a bootloader that never loads the boot image's ramdisk -- Google's on the
                      Pixel 8a takes init_boot's -- found there by `hakios load`, cmd/hakios.c image_container())
      avb_footer      the boot partition's size: the image is made that long, with an unsigned AVB hash footer
                      (avb_hash_footer()), for a bootloader that wants one (Google's on zuma)
    """
    style = {'header_version': '1', 'kernel': 'gzip+dtb'}
    path = os.path.join(dev, 'u-boot', 'boot-image')
    if os.path.isfile(path):
        for n, line in enumerate(open(path), 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            key, _, value = line.partition(' ')
            value = value.strip()
            if key not in ('header_version', 'kernel', 'os_version', 'os_patch_level', 'avb_footer', 'container') \
                    or not value:
                sys.exit(f'mkubootimg.py: {path}:{n}: not a known setting: {line}')
            style[key] = value
    if style['header_version'] not in ('1', '2', '3', '4') or style['kernel'] not in ('gzip+dtb', 'raw', 'lz4'):
        sys.exit(f'mkubootimg.py: {path}: header_version 1-4 and kernel gzip+dtb, raw or lz4')
    if style['header_version'] in ('3', '4') and style['kernel'] == 'gzip+dtb':
        sys.exit(f'mkubootimg.py: {path}: a header version {style["header_version"]} image has its device tree in '
                 'vendor_boot, so kernel raw or lz4')
    return style


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

    style = boot_image_style(dev)
    if style.get('container', 'ramdisk') == 'kernel' and hkub:
        u_boot = with_container(u_boot, hkub)
        hkub = b''                          # the ramdisk is then the one zero byte
    out = args.output or os.path.join(ROOT, 'hakios-u-boot.img')
    kernel, ramdisk = out + '.kernel~', out + '.ramdisk~'
    if style['kernel'] == 'gzip+dtb':
        open(kernel, 'wb').write(gzip.compress(u_boot, 6) + open(dtb, 'rb').read())
    elif style['kernel'] == 'lz4':          # LZ4's legacy frame, as a Pixel's own kernel; the device tree is the bootloader's
        subprocess.run(['lz4', '-l', '-12', '-f', '-q', '-', kernel], input=u_boot, check=True)
    else:                                   # 'raw': the device tree is the bootloader's own (vendor_boot)
        open(kernel, 'wb').write(u_boot)
    # no container: one zero byte for ABL, which hakios load does not take, so U-Boot waits for the host instead
    open(ramdisk, 'wb').write(hkub or b'\0')
    cmd = [sys.executable, mkbootimg, '--kernel', kernel, '--ramdisk', ramdisk,
           '--header_version', style['header_version'], '-o', out]
    for key in ('os_version', 'os_patch_level'):
        if style.get(key):
            cmd += [f'--{key}', style[key]]
    try:
        subprocess.check_call(cmd)
        if style.get('avb_footer'):
            img = open(out, 'rb').read()
            open(out, 'wb').write(avb_hash_footer(img, int(style['avb_footer'], 0)))
    finally:
        os.unlink(kernel)
        os.unlink(ramdisk)
    print(f'{out}: {os.path.getsize(out)} bytes (u-boot {len(u_boot)}, ramdisk {len(hkub)}: '
          f'{", ".join(f"{n} {len(d)}" for n, d in entries) or "none"})')


if __name__ == '__main__':
    main()
