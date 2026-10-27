# UEFI for redwood

The POCO X5 Pro 5G's platform for HakiOS's UEFI (`uefi/` of the common repository: Mu-Silicium with HakiOS's changes and banner).
It is modelled on `lisa` (the same `KodiakPkg` SoC package).

What is here, in `files/`:

- `Platforms/Xiaomi/redwoodPkg`: the memory map and the driver list;
- `Resources/Configs/redwood.toml`, `Resources/DTBs/redwood.dtb` (the device tree from the phone's vendor_boot);
- `Binaries/redwood`: the vendor's UEFI drivers, **not tracked**: you make this directory yourself from your own phone
  (`uefi/README.md` says how); `files/Binaries/redwood/README.md` is kept: it lists the patches that were applied to them.

```sh
devices/redwood/uefi/apply.sh     # = uefi/apply.sh redwood
```

The build and the vendor's drivers: [uefi/README.md](../../../uefi/README.md).
