# UEFI for redwood

A port of [Project Silicium / Mu-Silicium](https://github.com/Project-Silicium/Mu-Silicium) to the
POCO X5 Pro 5G, modelled on `lisa` (the same `KodiakPkg` SoC package).
The image is built on your machine, from your own phone's firmware: the vendor's drivers that go into
it are not in this repository (see below).

What is here:

- `BASE`: the Mu-Silicium commit the port is made on;
- `mu-silicium.patch`: changes to existing files: `BootShim/BootShim.S` (copies our "XNUP"
  container behind the UEFI image to 0xF0000000) and `BootManagerLib` (starts the XNU loader from
  the container instead of a menu);
- `files/`: new files: `Platforms/Xiaomi/redwoodPkg` (memory map, driver list),
  `Silicon/.../BootManagerLib/XnuLoader.[ch]`, `Resources/Configs/redwood.toml`,
  `Resources/DTBs/redwood.dtb` (the device tree from the phone's vendor_boot), `Binaries/redwood`
  (the vendor's UEFI drivers: **not tracked**, you make this directory yourself, see below;
  `files/Binaries/redwood/README.md` is kept: it lists the patches that were applied to them).

## The vendor's drivers are not in the repository

A phone's UEFI is made of open code (Project Mu, EDK2) and of drivers that Qualcomm and Xiaomi wrote for
that phone: display, storage, USB, secure boot and more. They are proprietary and have not been
licensed for redistribution, so this repository carries neither them (`files/Binaries/redwood/*`) nor
the UEFI image built with them (`prebuilt/SILICIUM_UEFI.fd`). `tools/provenance-check.py` fails the
build if they come back; it runs on every push, and `tools/provenance.allow` says why each binary that
*is* here may be.

Without that image `mkxnuimg.py` stops with a message that points here. To get one, make both from
**your own phone**:

1. Get your phone's UEFI firmware: the vendor's `xbl` firmware, which is on the phone (with root, the
   `xbl` partition) and in the vendor's firmware package for your model and region.
   `files/Binaries/redwood/README.md` records the version the port was made on (`OS1.0.3.0.UMSMIXM`):
   another version may need other patches.
2. Extract the drivers from it with [UEFIReader](https://github.com/WOA-Project/UEFIReader) into
   `files/Binaries/redwood/`. Its README says no more than what the tool is for ("generate .inf
   payloads ... out of an existing UEFI volume", as of 2026-10-26); Mu-Silicium's documentation for a
   device describes the layout it expects. **The authors of this repository have not repeated this
   extraction since the drivers were taken out of the tree: if you do it, say what worked in an
   issue.**
3. Build as below. `mkxnuimg.py` then takes the image from `Mu-Silicium/`.

What you extract is the vendor's, from your own device: keep it on your disk and never commit it.

## Build

Packages as in Mu-Silicium's `setup_env.sh` (Arch: `git base-devel lld nasm clang llvm
aarch64-linux-gnu-gcc python python-pip python-virtualenv`).

```sh
devices/redwood/uefi/apply.sh          # clones Mu-Silicium (with submodules, ~3 GB) and applies the port
python3 -m venv venv && venv/bin/pip install -r Mu-Silicium/pip-requirements.txt
cd Mu-Silicium
source ../venv/bin/activate
python3 build_uefi.py -d redwood -r RELEASE     # -> Build/redwoodPkg/RELEASE_CLANGPDB/FV/SILICIUM_UEFI.fd
```

The `mu_nasm` dependency comes through NuGet and wants Mono; without Mono, point it at the
system nasm:

```sh
D=Mu_Basecore/BaseTools/Bin/mu_nasm_extdep
mkdir -p $D/Linux-x86-64
echo 'version: 20016.1.1' > $D/extdep_state.yaml
ln -s /usr/bin/nasm /usr/bin/ndisasm $D/Linux-x86-64/
```

After a build, `mkxnuimg.py` (called by `./xl enter`) takes the UEFI from `Mu-Silicium/`, and
`tools/update-prebuilt.sh` refreshes `devices/redwood/prebuilt`. `devices/redwood/uefi/export.sh`
is the reverse: it saves the changes from `Mu-Silicium/` here.
