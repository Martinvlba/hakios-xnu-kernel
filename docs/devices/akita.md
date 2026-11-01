<!-- Made by tools/docs-gen.py from devices/akita/device.conf. Do not edit: change that, then run tools/docs-gen.py. -->


# Pixel 8a (`akita`)

| | |
|---|---|
| Vendor | Google |
| SoC | `zuma` |
| Configuration | `make akita_defconfig` |
| Maintainer | Hilledkinged |
| Repository | [`devices/akita/`](https://github.com/oosexgang/hakios-xnu-kernel/tree/main/devices/akita) |

## What works

| Part | State | |
|---|---|---|
| display | partly | the bootloader's framebuffer (0xfac00000, 1080 x 2400), drawn in software by Darwin: ~10 frames a second |
| touch | works | Goodix GT9895 on SPI (HSI0 USI2), polled; taps, swipes, the app switcher (CONFIG_HAKIOS_AKITA_TOUCH, touchscreen=1) |
| buttons | partly | power, volume up, volume down: found, no false presses; a press not yet seen (CONFIG_HAKIOS_AKITA_KEYS, buttons=1) |
| cpu | partly | 8 of 9 cores (the X3 not used); no frequency control (ACPM's) |
| usb | partly | the development link only: xl through U-Boot and the loader (44-46 MB/s); nothing for Darwin |
| storage | partly | reads: U-Boot brings UFS up (HS gear 4) and the loader serves it; no darwin partition yet |
| gpu | no | Mali-G715: software drawing only |
| battery | no | behind the AoC and ACPM; the icon is empty, no charging while in HakiOS |
| audio | no | behind the AoC |
| wifi | no | Broadcom on PCIe |
| cellular | no | Samsung's modem |
| thermal | no | no throttling |

The state is what has been seen on this device, written down in `devices/akita/device.conf`; a part the table does not name has not been looked at.
