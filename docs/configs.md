# Configurations: defconfigs, fragments and a `.config`

A Linux kernel is configured with a `.config` made from a `<device>_defconfig` (`make redwood_defconfig`), and an
Android kernel adds fragments on top (`merge_config.sh`). This kernel does the same, with one difference that is the
whole design: **its options are not compiled in.** XNU takes them from the boot-arguments, so a configuration here is
a file of `CONFIG_*` symbols that `tools/hakios-config.py` turns into the kernel's command line.

```
configs/defconfig             what every configuration starts from: the OPEN one (nothing of a phone's own, one core)
configs/redwood_defconfig     the POCO X5 Pro 5G (redwood, SM7325): what the project's own phone line uses
configs/open.config           fragment: the open profile (CONFIG_FIRMWARE="none")
configs/virt.config           fragment: QEMU's virt machine (virtio network)
configs/trace.config          fragment: log a line for each of the first frames on the network interfaces
configs/persist.config        fragment: keep /private/var across reboots (THIS WRITES TO THE PHONE, partition `darwin`)
```

## Use

```bash
make redwood_defconfig                  # .config = configs/defconfig + configs/redwood_defconfig
make redwood_defconfig FRAGMENTS=trace  # ... and a fragment on top
make virt.config                        # lay a fragment over the .config there is
make config                             # show it: symbols, the kernel's arguments, the firmware it wants
make savedefconfig                      # the .config as a minimal defconfig (what differs from configs/defconfig)
make phone                              # with a .config, ITS arguments are the kernel's
```

The same without make: `tools/hakios-config.py defconfig|merge|args|check|show|needs|savedefconfig|list|help`.
`tools/hakios-config.py help GPU` says what a symbol means, what it needs and what it requires. `make doctor`
checks the host.

## What a symbol is

`CONFIG_GPU=y` is the argument `gpu=1`; `CONFIG_CPUS=8` is `cpus=8`; `CONFIG_MODEM_BUFFER=0xa8000000` is
`modem_buffer=0xa8000000`. A symbol that is `# CONFIG_X is not set` is left out, and the kernel's own default applies.
Three symbols are facts about the build, not arguments: `CONFIG_DEVICE`, `CONFIG_SOC` (it goes into the kernel's version
string, see [versioning.md](versioning.md)) and `CONFIG_FIRMWARE`.

The table of symbols is the `SYMBOLS` list at the top of `tools/hakios-config.py`; it is the one place that says
which symbol is which argument.

## What is checked, and why

- **Types and ranges.** A bool is `y` or not set, a hex number starts with `0x`, `CONFIG_CPUS` is 1 to 8. A typo in
  a name gets a suggestion (`no symbol CONFIG_GPUU (did you mean CONFIG_GPU?)`).
- **Dependencies.** `CONFIG_VAR_STORE` requires `CONFIG_UFS_WRITE`, which requires `CONFIG_UFS=2`; `CONFIG_SOUND`
  requires `CONFIG_ADSP`; the modem requires the storage service.
- **Firmware.** `CONFIG_FIRMWARE="none"` is the open profile: *nothing of the phone's own is loaded.* The check refuses
  a configuration that says `none` and still asks for the GPU, Wi-Fi, audio, haptics, cameras or the modem, because each
  of those uses firmware pulled from the owner's own phone (`fw-*/`, never in this repository).
- **The command line fits.** The kernel's boot line holds 607 characters, and a longer one loses its tail without an
  error. The check counts the arguments, adds what a run adds itself (about 260) and refuses what cannot fit.

## `CONFIG_BOOTARGS`: Linux's `CONFIG_CMDLINE`

For an argument that has no symbol of its own: `CONFIG_BOOTARGS="fb_beat=2 loader_return_after=5"`. It is not an
escape from the checks. Every name must be one the source reads ([boot-args.md](boot-args.md), made by
`tools/boot-args.py` from the code, and checked in CI), so `fb_beet=2` is an error with a suggestion rather than a
silent no-op; an argument that a symbol already sets (`cpus=8`) is refused with the name of the symbol; a word is
`name=value` with no space, quote or backslash in it; and the words count towards the 607 characters.

## What is the same as Linux, and what is not

| | Linux / Android | here |
|---|---|---|
| a device | `arch/arm64/configs/<device>_defconfig` | `configs/<device>_defconfig` |
| a fragment | `kernel/configs/*.config`, `merge_config.sh` | `configs/*.config`, `make NAME.config` |
| the result | `.config`, compiled in | `.config`, becomes the boot-arguments |
| a minimal config | `make savedefconfig` | `make savedefconfig` |
| a command line baked in | `CONFIG_CMDLINE` | `CONFIG_BOOTARGS` (names checked) |
| an option that needs a blob | `CONFIG_EXTRA_FIRMWARE` | `CONFIG_FIRMWARE="device"` / `"none"`, and `needs` |
| the version | `uname -r` | `kern.hakios.version`, `kernel.ver=` |

## A new device

`tools/new-device.py` makes `devices/<name>/` from a phone's own device tree; its defconfig is a file of a few
lines, `configs/<name>_defconfig`, that starts from the open one and turns on only what that phone has been shown
to do.

## What a configuration does not do

It does not make a driver exist, and it does not make a driver work on a phone it was not written for. A symbol turns
on a part of the kernel that is already there; whether it works on a given phone is what [the device's page](../devices)
and the tests say, and the project does not claim a result it has not seen.
