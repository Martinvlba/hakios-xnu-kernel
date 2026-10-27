# Device repositories

`devices/manifest.conf` names every device; `tools/hakios-device.py sync` clones each into `devices/<name>/`. `devices/qemu/` is
kept here: QEMU's machine, with none of a phone's drivers, the device every checkout of the common repository can build and run.
A device of your own: `tools/hakios-device.py new` ([Port a device](../docs/porting/index.md)).
