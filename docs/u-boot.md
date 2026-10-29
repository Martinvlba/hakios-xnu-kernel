# U-Boot

HakiOS's loader can be started by **Das U-Boot** instead of Project Silicium's UEFI. The phone's own bootloader (ABL) starts
U-Boot from RAM with `fastboot boot`, U-Boot finds the loader and its payload **in memory** and starts the loader as a UEFI
application. Nothing is flashed: not `boot`, not any other partition, and a reboot gives the phone back to Android.

Status in one line: **works on the POCO X5 Pro and in QEMU, the whole chain to the kernel's user space, with `xl` as it was.**
The sections below say exactly what was run.

## Why

| | Silicium (`mkxnuimg.py`) | U-Boot (`tools/mkubootimg.py`) |
|---|---|---|
| What it is made of | Mu-Silicium and the **phone vendor's own UEFI drivers**, which are not in the repositories: you extract them from your own phone first | open source only: U-Boot's Qualcomm support and the phone's device tree |
| Where the loader comes from | the payload the boot shim copies to `0xF0000000` | the boot image's ramdisk, or U-Boot's own fastboot: the hakios command copies the payload to `0xF0000000` |
| A new loader or payload | a new boot image and `fastboot boot` through ABL again | the same, or sent to the running U-Boot over USB (below), without going back through ABL |
| `xl` (run, ramdisk, log, screenshot) | yes | yes: U-Boot provides the same USB protocol the loader uses under Qualcomm's UEFI |

## The chain

```
ABL ──fastboot boot──> U-Boot (RAM) ──hakios load──> payload at 0xF0000000, loader.efi ──bootefi──> loader ──> XNU
                         │                               ▲
                         └── no payload in the image ────┘ fastboot stage x.hkub; fastboot oem run:hakios load fastboot
```

- **The boot image** (`tools/mkubootimg.py`): the kernel is `gzip(u-boot.bin)` with the phone's Android device tree appended,
  exactly what ABL expects for Silicium's image (header version 1); the ramdisk is an `HKUB` container with the loader and a payload,
  or nothing.
- **U-Boot's device tree** (`devices/redwood/u-boot/files/dts/upstream/src/arm64/qcom/sm7325-xiaomi-redwood.dts`): the panel's
  framebuffer at `0xE1000000` as ABL left it (the loader knows the phone by that address), the keys, USB as a device, UFS, and every
  carve-out of the Android device tree and of Silicium's memory map. The size of RAM is not in it: ABL writes it into the device
  tree it hands over and U-Boot reads it from there.
- **`0xF0000000`, 64 MiB** is reserved in that device tree (`hakios-payload`): nothing in U-Boot takes it, the hakios command
  writes the payload there, and the loader's memory window stays clear of it, as under Silicium.
- **The hakios command** (`uefi/u-boot/files/cmd/hakios.c`):

  | | |
  |---|---|
  | `hakios load` | from the ramdisk ABL loaded (the device tree ABL handed over says where) |
  | `hakios load fastboot` | what the host sent last with `fastboot stage` |
  | `hakios load ADDR [SIZE]` | from anywhere in RAM; a container or a payload says its own size |
  | `hakios boot` | `bootefi` the loader |
  | `hakios info` | what is in memory |

  It takes an `HKUB` container, a bare XNUP payload or a bare `loader.efi`. A payload with an entry `loader.efi` (the one
  `xl enter` makes) brings its loader with it. Every entry of a container carries a CRC-32, and a damaged one is never started.
- **Qualcomm's USB device protocol** (`uefi/u-boot/files/lib/efi_loader/efi_qcom_usb.c`): the loader talks fastboot to the host
  through `EFI_USB_DEVICE_PROTOCOL`, the interface of Qualcomm's UEFI (published in its open-source ABL). U-Boot provides it on its
  own USB gadget stack: the control endpoint is answered from the loader's descriptors, the loader sees its bulk transfers and the
  cable's state. So `xl` works unchanged.
- **What U-Boot will not do**: write the phone's storage. Capsule updates are off (they flash a partition) and U-Boot's watchdog is
  not started (the loader has its own, and keeps U-Boot's boot services while the kernel runs).

## Build

```sh
tools/hakios-device.py sync          # devices/redwood and uefi/, as for everything else
make u-boot                          # = uefi/u-boot/apply.sh (U-Boot at uefi/u-boot/BASE into u-boot/) + uefi/u-boot/build.sh redwood
```

`uefi/u-boot/build.sh` needs an AArch64 GCC (`aarch64-linux-gnu-gcc`, or Arm's `aarch64-none-elf-gcc` from
developer.arm.com; `CROSS_COMPILE=` to name another). The result is `u-boot/build-redwood/u-boot.bin`.

## Run

**With `xl`**, as always:

```sh
XL_FIRMWARE=u-boot ./xl enter        # without XL_FIRMWARE: Silicium when its image is built, else U-Boot
./xl ramdisk userland/system.apfs
./xl run
```

**One image, loader and payload inside:**

```sh
xnuloader/mkpayload.py -o payload.bin kernel=hakios-kernel ramdisk=userland/hello --cmdline "-v"
make u-boot-img UBOOT_PAYLOAD=payload.bin        # hakios-u-boot.img
fastboot boot hakios-u-boot.img
```

**U-Boot first, then anything over its fastboot** (no new boot image for a new kernel):

```sh
make u-boot-img                                   # no payload: U-Boot waits for the host
fastboot boot hakios-u-boot.img
tools/mkubootimg.py -p payload.bin -c x.hkub      # the container alone
fastboot stage x.hkub
fastboot oem run:"hakios load fastboot"
fastboot continue                                 # hakios boot
```

U-Boot's fastboot shows as Google's `18d1:4ee0` (the loader's own is `18d1:d00d`); `fastboot oem console` reads U-Boot's log.
Hold **volume down** while U-Boot starts for its menu (start, fastboot, what is in memory, a shell over USB serial): the volume
keys move, the power key chooses.

## What the screen says

U-Boot writes what it does on the phone's screen, each line marked **`[oosexgang]`**: what the hakios command took and where it put
it, and, once the loader talks to the host, the cable coming and going, every command of the host (the `getvar` polls of a run
aside), every download and upload with its size, progress and speed, and every `FAIL` the loader answers:

```
[oosexgang] hakios: payload of 66216 bytes at 0xf0000000
[oosexgang] usb: the host is connected
[oosexgang] host: upload
[oosexgang] sending 22 KiB
[oosexgang] sent 22 KiB in 4 ms (5500 KiB/s)
[oosexgang] host: oem screenshot
[oosexgang] host: upload
[oosexgang] sending 10125 KiB
[oosexgang]   30%
```

(lines from the phone's screen after `xl run` and `xl screenshot`)

`setenv hakios_status 0` (U-Boot's shell, or the environment file) turns it off. `hakios_usb_debug=1` shows much more: every
control request, endpoint and transfer of the USB device protocol, which is how the faults below were found.

## Try it in QEMU (nothing of Apple's)

```sh
make qemu-u-boot          # make qemu-open's kernel, loader and hello, booted by U-Boot: qemu/run-u-boot.sh
```

`qemu/run-u-boot.sh` puts an `HKUB` container at `0x50000000` (where a phone's ABL would have put the ramdisk) and boots
`u-boot/build-qemu/u-boot.bin` (`devices/qemu/u-boot/files/configs/hakios_qemu_defconfig`). What it printed (U-Boot 2026.07, the
prebuilt kernel and `hello`):

```
hakios: payload of 16825792 bytes at 0xf0000000
hakios: loader of 65536 bytes at 0x1bd510000
== XNULOADER v3 ==
platform qemu-virt
phys window f4000000 size c8000000, claim status 0
Hello from user space on Darwin!
kernel returned: log at f7bdc9c0, 21629 bytes
```

The same as under edk2 (`qemu/run.sh`), line for line where they are comparable. A bare payload with `loader.efi` in it was run the
same way and gave the same.

## On the phone: what is known and what is not

Run on a POCO X5 Pro 5G (`redwood`, bootloader unlocked, Android on slot b), from Windows' platform-tools, 29 October 2026:

- ABL starts the boot image with `fastboot boot`; U-Boot comes up on the panel, takes the size of RAM from ABL's device tree and
  brings UFS up (SK hynix, six LUNs).
- `./xl enter` (`XL_FIRMWARE` unset, no Silicium image): `xl: loader is up` in about 7 s. The loader finds the phone by its
  framebuffer (`platform redwood`), arms its watchdog, and its fastboot answers (`product: xnuloader`, `version: 0.5`).
- `./xl memmap`: every carve-out of the device tree reserved, the payload area as U-Boot's, and 5.4 GiB free above `0xF4000000`
  for the kernel's window.
- Ten `getvar` and 1 MiB `stage` in turn, all answered; the 16 MiB run payload in 0.55 s.
- `XL_RAMDISK=prebuilt/userland/hello ./xl run`: the prebuilt kernel boots, `Hello from user space on Darwin!`, `initproc` exits
  (that is what `hello` does) and the kernel returns: `result: returned 22037 0`. The loader's log is written to `logdump` and read
  back the same.
- `./xl screenshot`: the kernel's last screen, with U-Boot's `[oosexgang]` lines under it.
- The way through U-Boot's own fastboot (`stage`, `oem run:hakios load fastboot`, `continue`): U-Boot takes the container and
  starts the loader, but **the loader's USB then does not come up on the bus**: the controller, released by U-Boot's fastboot and
  taken again by the loader, does not connect. Until that is fixed, put the loader and the payload in the boot image (`xl enter`
  does).

What it took on the phone, each seen there first (`hakios_usb_debug=1` and photographs of the screen):

| Fault | Fix |
|---|---|
| The loader's descriptors (USB 2.10 and a BOS) promise Link Power Management, which U-Boot's DWC3 does not do: the device left the bus 3 s after it came | USB 2.0 and no BOS, as U-Boot's own gadgets |
| U-Boot's endpoint autoconfig ORs the endpoint number into the address it is given (`0x01` became `0x03`) and writes a full-speed packet size (64) | only the direction goes in; the caller's packet size (512) comes back |
| U-Boot's DWC3 starts a bulk transfer only on the XferNotReady the host's next token brings | it starts on queue, and the next one when one completes, as Linux's dwc3 |
| U-Boot's DWC3 clears a stall with a transfer started (WinUSB clears the halt of a pipe it opens) | the transfer ends first and starts again after, as Linux's dwc3 |
| `fastboot oem console`: every response dequeued a request that was not queued; the error went into the console it was sending, for ever | only a request in flight is dequeued |
| The loader's log buffer was not aligned to the block size U-Boot's block I/O wants | aligned (`xnuloader/loader.c`) |
| A download the host gave up half-way left the loader waiting for good | 20 s without a byte and it is given up |

If U-Boot hangs, the menu (volume down) and `fastboot oem console` are the ways in;
[U-Boot's own notes on phones](https://docs.u-boot.org/en/latest/board/qualcomm/phones.html) say how to bisect a hang.
- What the loader asks Qualcomm's UEFI for and U-Boot does not have: the charger protocols (the loader goes on without the battery
  reading) and `EFI_CPU_ARCH_PROTOCOL` (the loader's exception hook is not installed; a fault in the loader resets the phone instead of
  being logged).
