# Changelog

What changed, newest first. One line a change: the part of the system, then what is different for whoever runs it.
The commits say how; `docs/` says why.

## Sources

What a build is made of, and where each part is fetched from. Nothing of these is kept in the repositories but the
changes to it; the scripts named fetch the release and put the changes on.

- the kernel: Apple's [XNU](https://github.com/apple-oss-distributions/xnu), tag `xnu-8796.121.2` (the release the Darwin 16.5 user space was built from) -- `xnu/apply.sh`, the changes in `xnu/hakios.patch` and `xnu/files/`, the drivers in `drivers/`
- the data volume: Apple's [HFS Plus](https://github.com/apple-oss-distributions/hfs), tag `hfs-627.100.6` -- `xnu/hfs/hfs.sh`, the changes in `xnu/hfs/hfs.patch`
- the boot: [U-Boot](https://source.denx.de/u-boot/u-boot) `v2026.07` -- `hakios-uefi/u-boot/apply.sh`, the changes in `hakios-uefi/u-boot/hakios.patch` and `files/`; the device's part in `hakios-sm7325/u-boot/`
- the UEFI (the other way to start): [Project Silicium's Mu-Silicium](https://github.com/Project-Silicium/Mu-Silicium) at `e51cd877e31c` -- `hakios-uefi/apply.sh`
- the GPU's OpenGL ES: [Mesa](https://archive.mesa3d.org/) 25.0.7, the freedreno driver for the Adreno -- `tools/mk-mesa.sh`, the changes in `tools/mesa/`
- the build tools: Apple's [libdispatch](https://github.com/apple-oss-distributions/libdispatch), [libpthread](https://github.com/apple-oss-distributions/libpthread) and [AvailabilityVersions](https://github.com/apple-oss-distributions/AvailabilityVersions), [bootstrap_cmds](https://github.com/markmentovai/bootstrap_cmds), [unifdef](https://github.com/fanf2/unifdef), the [macOS 13.3 SDK](https://github.com/joseluisq/macosx-sdks) and the [iPhoneOS 16.5 SDK](https://github.com/theos/sdks) -- `xnu-deps/setup.sh`
- not fetched by anything here: the Darwin 16.5 user space, which you lay out from your own IPSW (`sysimage/`), and the phone's own firmware, pulled from its Android (`fw-*`)

## 2026-11-01

- u-boot: USB on the Pixel 8a: its eUSB2 PHY and repeater brought up (`eusb`), a DWC_usb31 no longer forced to SuperSpeed, the phone's serial number (`hakios serial`); U-Boot's fastboot answers over USB
- u-boot: `acpm`: Google's power firmware and its PMIC protocol, from U-Boot (the Pixel 8a's regulators)
- u-boot: `hakios trace` (registers once a second while U-Boot waits); `hakios wdt` keeps one watchdog as a net
- loader: under a firmware that gives the phone's serial number, the loader's USB device is the phone's (18d1:4ee0, that serial); a Samsung watchdog left counting is fed while a host talks to the loader, and only then
- loader, kernel: the kernel's other cores from a loader at EL2: CPU_ON goes to the loader's trampoline (`/chosen/el2-trampoline`), which takes each core down to EL1; checked in QEMU (EDK2 and U-Boot, GICv3 and GICv4) and on the Pixel 8a, eight cores
- u-boot: the instruction cache turned on where a boot loader before U-Boot left it off (Exynos' `enable_caches()`): on the Pixel 8a a run went from five minutes to fifty seconds
- u-boot: `hakios load` and the loader's start timed (seconds since the SoC's reset) in the console

## 2026-10-31

- u-boot: `hakios wdt` shows and stops Samsung watchdogs a bootloader left counting (the Pixel 8a: both clusters, which restarted it two minutes in)
- akita: the first boot on the Pixel 8a: U-Boot, the loader at EL2, the kernel at EL1, `hello`, and back to the loader
- u-boot: `hakios load` takes the container from behind U-Boot in the boot image's kernel, for a bootloader that never loads boot's ramdisk; U-Boot's load address is noted before it relocates
- u-boot: `hakios_pstore`: the console kept in a ramoops zone over a restart, read back from Linux's /sys/fs/pstore
- tools: `mkubootimg.py` makes an LZ4 kernel, the container in the kernel, and an unsigned AVB hash footer, as a device's `u-boot/boot-image` asks
- loader: started at EL2 (a firmware that enters its payload there, with or without VHE), it starts the kernel at EL1 and takes it back through HVC; PSCI by SMC there, one core only so far (checked in QEMU with `virtualization=on`)
- loader: a second phone, the Pixel 8a (`platform akita`), known by the `google,zuma` of the device tree the firmware hands over; QEMU's fw_cfg screen is set up on QEMU only
- loader: the timer's rate from the platform where the firmware leaves `CNTFRQ_EL0` unset; the GIC's redistributor region and stride from the platform (redwood's stride, 0x20000, as its tree says)
- kernel: the redistributors are walked as Linux walks them: by the device tree's `redistributor-stride`, else by `GICR_TYPER.VLPIS` (0x40000 a core with virtual LPIs)
- u-boot: `hakios load` finds the container behind the vendor ramdisks of a header version 4 bootloader; `tools/mkubootimg.py` makes the image a device's `u-boot/boot-image` asks for (header version 4, U-Boot as it is)
- u-boot: exynos-mobile builds without its eMMC set-up (`MISC_INIT_R` off), for a phone whose storage U-Boot cannot reach

## 2026-10-30

- repositories: `main` is the branch of `hakios-xnu-kernel` -- what was `hakios` and then `u-boot`; `hakios` is gone, `git clone` needs no `-b`
- site: a Changelog page, this file
- site: the commands to start with clone the public repository (`hakios-xnu-kernel`); `oosexgang/hakios` is private and cloned for nobody
- site: the documentation synced, with the guide to installing beside Android and the U-Boot page in the table of contents

## 2026-10-29

- u-boot: HakiOS's own U-Boot, started by the phone's bootloader from RAM; it starts the loader and the kernel from memory, nothing of the vendor's firmware in between
- u-boot: the loader's fastboot works under U-Boot (USB 2.0 without the link power states that dropped the cable after 3 s, endpoints and packet sizes as the loader asks, DWC3 transfers started when queued)
- u-boot: every download and every step says what it does on the screen, tagged `[oosexgang]`
- u-boot: `hakios sum` checks a flashed partition from the phone itself; `hakios reboot bootloader` gets to fastboot
- install: HakiOS beside Android: U-Boot in `boot_a`, Android in slot b untouched, a menu at every start (Android after 10 s, HakiOS (Darwin), fastboot, U-Boot's shell)
- install: choosing Android marks HakiOS's slot unbootable and lets the bootloader switch; U-Boot never changes which slot is active (changing it by hand once left every image refused with "Device Error")
- install: `tools/userdata-carve.py --wipe` makes `darwin` at the end of userdata; `tools/sparse-split.py` cuts an 8 GB image into pieces Windows fastboot can send
- storage: the data volume is HFS Plus in `darwin` after the system, about 12 GB, journaled, kept between starts; before, `/private/var` was a memory volume shown as 1 GB and empty at every start
- storage: HFS Plus built into the kernel; Apple's sources are fetched at `xnu/apply.sh`, only this port's changes are kept
- storage: the data volume's writes go straight to the flash and the cache is flushed when the journal asks (a flush and a read back after every write made the system crawl)
- storage: the loader takes any phone's `darwin` (name, type, unit, at least 20 GiB), not only the first phone's place and identifier
- storage: `tools/mkdatavol.py` makes the empty volume as a 50 KB image that leaves the system volume alone
- graphics: the installed image draws with the GPU (Mesa, `GL=1`): smooth, and the wallpaper drawn
- graphics: the system's other services and the processes that draw parts of the interface are in the image (`REST=1`): the wallpaper and the applications that need them
- graphics: frames at 120 a second (`fb_hz=120`), and `fb_measure=1` writes to the log how often the panel itself refreshes
- loader: nothing printed over the system's screen while it runs; the log still takes every line
- loader: its log written under U-Boot too; a download the computer gave up on no longer hangs it
- wifi: the Wi-Fi controller is found where the system looks for its address, so registering applications no longer waits 5 s for each (found by the ios-sapphire port)
- tree: Darwin, not iOS, in the sources, the scripts, the file names and the documentation; no file of Apple's in the repositories
- docs: installing beside Android, step by step (`docs/dual-boot.md`, in Russian too), and the U-Boot page with everything as it was done and checked
