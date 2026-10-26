# Drivers

The kernel's own drivers, by subsystem, the way Linux keeps `drivers/`. They are files of Apple's XNU tree
(`bsd/dev/`, `iokit/Kernel/`), kept here so that they can be found and reviewed by what they do.

| Where | Is |
|---|---|
| `drivers/map` | the list: for each file, its subsystem and the path it has in Apple's tree |
| `xnu/apply.sh` | lays Apple's tag, `xnu/hakios.patch`, `xnu/files/` and these drivers into one tree |
| `xnu/export.sh` | the reverse, run before committing; refuses a new `hakios.*` file that `drivers/map` does not list |
| `tools/drivers-check.py` | CI: the map and the directories agree, nothing is listed twice, nothing is kept in two places |

Adding a driver: put it in `drivers/<subsystem>/`, add a line to `drivers/map`, add its `bsd/conf/files` line to
the patch (as the others have), `python3 tools/drivers-check.py`. A driver that is started by a boot argument
has to be in `docs/boot-args.md` (`python3 tools/boot-args.py --check`).

## Subsystems

| Directory | Drivers | Boot arguments | What it is |
|---|---|---|---|
| `input/` | `hakios.goodix.c`, `hakios.input.c`, `hakios.buttons.c`, `hakios.virtio_input.c` | `touchscreen=` | the touch controller, the event queue behind `/dev/hakios-touch`, the buttons, QEMU's virtio input |
| `storage/` | `hakios.ufs.c`, `hakios.ufshc.c/.h`, `hakios.sparse.c`, `hakios.image.c`, `hakios.guard.c`, `hakios.apfs_check.c`, `hakios.virtio_blk.c` | `ufs=`, `ufs_write=`, `ufs_native=` | the phone's flash (UFS), the sparse and image devices, the checks of the APFS read path, QEMU's virtio-blk. **The only code of the kernel that may write to the phone**: see `tools/provenance.allow` |
| `gpu/` | `hakios.adreno.c`, `hakios.gpu.cpp` | `gpu=`, `adreno=` | the Adreno GPU bring-up and the IOKit object user space talks to |
| `camera/` | `hakios.camera.c`, `hakios.torch.c` | `cam=`, `torch=` | the camera sensors and ISP interface, the flashlight LED |
| `haptic/` | `hakios.haptic.c` | `haptic=` | the vibration motor |
| `audio/` | `hakios.adsp.c`, `hakios.audio.c`, `hakios.sound.c`, `hakios.speaker.c`, `hakios.mic.c`, `hakios.virtio_sound.c` | `adsp=`, `sound=`, `mic=` | the audio DSP loader, `/dev/hakios-audio`, the speaker amplifiers and their guard, the microphone, QEMU's virtio sound |
| `power/` | `hakios.battery.c/.cpp`, `hakios.cpufreq.c` | `cpufreq=`, `battmgr=` | battery readings (through the DSP), CPU frequency |
| `wifi/` | `hakios.wifi.c`, `hakios.wlfw.c`, `hakios.lx.c`, `hakios.sta.c`, `hakios.apple80211.c`, `hakios.wlan.cpp` | `wifi=`, `wifi_step=` | the Wi-Fi subsystem's bring-up, the WLAN firmware service, the Linux-driver layer (`wifi/lx/`) and the surface the iOS Wi-Fi service expects |
| `modem/` | `hakios.mpss.c`, `hakios.cell.c`, `hakios.qmi.c`, `hakios.nas.c`, `hakios.ipa.c` | `modem=` | the cellular modem path (research; not reviewed in `docs/driver-review/`) |
| `net/` | `hakios.net.c`, `hakios.virtio_net.c` | | the `enN` network interface and QEMU's virtio-net |
| `core/` | `hakios.version.c` | `hakios_kver=` | `kern.hakios.version` |

## Reviews

`docs/driver-review/` has one read-only audit per group (what was checked, what was found, how sure). They are
findings to verify, not a certificate: read the report before relying on a driver, and fix findings in the driver,
not in the report.

The kernel is built from the C and C++ files only; the drivers need no files that are not in this repository,
except the touch firmware header (`input/hakios.focaltech_fw.h`), which `tools/provenance.allow` marks as pending.
