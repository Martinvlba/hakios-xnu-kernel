#!/usr/bin/env python3
"""Start a port: make devices/NAME/ from a phone's own device tree, and say how far the SoC code carries.

usage: tools/new-device.py NAME [--fdt FILE | --adb] [--out DIR] [--force]

  NAME    the new directory under devices/ (lower case letters, digits and dashes: poco-x5-pro)
  --fdt   the phone's device tree, a file (adb pull /sys/firmware/fdt; the tree of the phone's own Android)
  --adb   pull it from the phone on adb now (as the shell user, then through su: on some ROMs the file is
          readable by root only)
  --out   where to make the directory (default: devices/NAME beside this tool); tests use it
  --force write into a directory that exists

What it makes, in devices/NAME/:
  soc.args    the soc_* boot-arguments tools/soc-args.py reads out of the tree: the addresses of the modem's,
              the IPA's and the radio's blocks, which the run scripts add to the kernel command line
  README.md   what the tree says the phone is, which arguments were found and which not, which of them
              differ from the reference phone's, and a checklist of what is the phone's own to port, with
              the files whose head says where their numbers came from
  pull-soc.sh pull the tree again and regenerate soc.args

It also compares the result with the reference phone's (devices/poco-x5-pro): a tree that gives the same
values means the kernel's built-in defaults already fit, and a port is then about the board (display,
touch, UEFI) and not the SoC. Nothing here changes the phone; the only things read from it are its tree.
"""
import argparse
import datetime
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOC_ARGS = os.path.join(HERE, "soc-args.py")
REFERENCE = "poco-x5-pro"
REFERENCE_DTB = os.path.join(ROOT, "devices", REFERENCE, "prebuilt", "device-tree.dtb")

# what is the phone's own, and the files whose head says where their numbers came from
CHECKLIST = [
    ("UEFI", "the firmware that starts the loader: devices/%s/uefi/ is a port of Mu-Silicium; the vendor's drivers are "
             "extracted from your own phone (uefi/README.md)" % REFERENCE),
    ("Display, buttons, touch", "xnu/files/bsd/dev/hakios.input.c, hakios.buttons.c, hakios.goodix.c (the reference phone's "
                                "controller; another phone has another)"),
    ("Storage", "hakios.ufs.c and hakios.ufshc.c (UFS); a phone with eMMC needs another driver"),
    ("CPU clocks", "hakios.cpufreq.c"),
    ("GPU", "hakios.adreno.c (Adreno 642L here: another generation needs its own firmware and registers)"),
    ("Wi-Fi", "hakios.wifi.c, hakios.lx.c and wifi/lx/ (the WCN6750 radio of the reference phone)"),
    ("Audio and battery", "hakios.adsp.c, hakios.audio.c, hakios.sound.c, hakios.speaker.c, hakios.mic.c, hakios.battery.c"),
    ("Cellular", "hakios.mpss.c, hakios.qmi.c, hakios.nas.c, hakios.ipa.c, hakios.cell.c; soc.args carries the addresses, "
                 "the IPA layout is for IPA 4.11 (docs/cellular.md)"),
    ("Haptics and cameras", "hakios.haptic.c, hakios.camera.c"),
]


def load_soc_args():
    spec = importlib.util.spec_from_file_location("soc_args", SOC_ARGS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(fdt_path):
    """The lines tools/soc-args.py prints for a tree, or an error message."""
    run = subprocess.run([sys.executable, SOC_ARGS, fdt_path], capture_output=True, text=True)
    if run.returncode != 0:
        return None, (run.stderr or run.stdout).strip()
    return [l for l in run.stdout.splitlines() if l.strip()], None


def text_prop(props, name):
    value = props.get(name, b"")
    return [s.decode(errors="replace") for s in value.split(b"\0") if s]


def pull_fdt(dest):
    adb = shutil.which("adb") or shutil.which("adb.exe")
    for candidate in ("C:/platform-tools/adb.exe", "/mnt/c/platform-tools/adb.exe"):
        adb = adb or (candidate if os.path.exists(candidate) else None)
    if not adb:
        sys.exit("new-device: adb not found; put platform-tools on PATH, or pass the tree with --fdt")

    def adb_run(*args):
        return subprocess.run([adb, *args], capture_output=True, text=True)

    if adb_run("pull", "/sys/firmware/fdt", dest).returncode == 0 and os.path.getsize(dest) > 0:
        return "adb pull /sys/firmware/fdt"
    # some ROMs make it readable by root only: copy it under su to a place the shell can read, pull it, remove it
    stage = "/data/local/tmp/hakios-fdt"
    adb_run("shell", "su -c 'cat /sys/firmware/fdt > %s && chmod 644 %s'" % (stage, stage))
    ok = adb_run("pull", stage, dest).returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0
    adb_run("shell", "su -c 'rm -f %s'" % stage)
    if not ok:
        sys.exit("new-device: could not read /sys/firmware/fdt over adb, with or without su")
    return "adb pull through su (the file is readable by root only on this ROM)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--fdt")
    source.add_argument("--adb", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        sys.exit("new-device: NAME is lower case letters, digits and dashes (like poco-x5-pro), got %r" % args.name)
    out = args.out or os.path.join(ROOT, "devices", args.name)
    if os.path.exists(out) and not args.force:
        sys.exit("new-device: %s exists (--force to write into it)" % out)

    tmp = tempfile.mkdtemp(prefix="new-device-")
    try:
        if args.adb:
            fdt = os.path.join(tmp, "fdt")
            how = pull_fdt(fdt)
        else:
            fdt = args.fdt
            how = "the file %s" % os.path.basename(fdt)
            if not os.path.isfile(fdt):
                sys.exit("new-device: no such file: %s" % fdt)
        lines, error = generate(fdt)
        if lines is None:
            sys.exit("new-device: %s\n  (is this the phone's own device tree? adb pull /sys/firmware/fdt)" % error)

        nodes = load_soc_args().nodes_of(open(fdt, "rb").read())
        root = nodes.get("", {})
        model = (text_prop(root, "model") or ["(the tree has no model)"])[0]
        compatible = text_prop(root, "compatible")
        reference, _ = generate(REFERENCE_DTB) if os.path.exists(REFERENCE_DTB) else (None, None)

        found = dict(l.split("=", 1) for l in lines)
        names = ["soc_ipa_base", "soc_gsi_base", "soc_ipcc", "soc_aoss", "soc_rsc", "soc_apps_smmu", "soc_cmd_db",
                 "soc_mpss_mem", "modem_ipa_fw_at", "soc_smem", "soc_imem", "soc_ipa_stream", "soc_ipa_stream_uc",
                 "soc_pas_mpss", "soc_pas_ipa", "soc_ipa_fw"]
        missing = [n for n in names if n not in found]
        ref = dict(l.split("=", 1) for l in reference) if reference else {}
        differs = [n for n in found if n in ref and ref[n] != found[n]]
        same_as_reference = bool(ref) and not missing and not differs and set(found) == set(ref)

        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "soc.args"), "w", newline="\n").write("\n".join(lines) + "\n")
        pull = os.path.join(ROOT, "devices", REFERENCE, "pull-soc.sh")
        if os.path.exists(pull):
            shutil.copyfile(pull, os.path.join(out, "pull-soc.sh"))
            os.chmod(os.path.join(out, "pull-soc.sh"), 0o755)

        rows = "\n".join("| `%s` | `%s` | %s |" % (n, found.get(n, "(not found)"),
                         "the same as the reference" if n in found and ref.get(n) == found[n] else
                         ("**differs from the reference** (`%s`)" % ref.get(n) if n in found and n in ref else
                          "**missing**: the tree has no node this was read from; see the table in devices/%s/README.md" % REFERENCE
                          if n not in found else "")) for n in names)
        verdict = (
            "**The tree gives the same %d values as the reference phone (`devices/%s`).** The kernel's built-in defaults "
            "already fit this SoC, so `soc.args` changes nothing here; what is left to port is the board (UEFI, display, "
            "touch, and the rest of the checklist below), not the SoC." % (len(found), REFERENCE)
            if same_as_reference else
            "**This tree differs from the reference phone's** in %d value(s) and lacks %d: %s. Those are this phone's own "
            "addresses; `soc.args` carries them to the kernel and the run scripts add the file to the command line." %
            (len(differs), len(missing), ", ".join("`%s`" % n for n in (differs + missing)) or "none") if ref else
            "No reference tree to compare with (devices/%s/prebuilt/device-tree.dtb is missing)." % REFERENCE)
        checklist = "\n".join("- [ ] **%s**: %s" % (a, b) for a, b in CHECKLIST)
        readme = """# Device: {model} ({name})

Made by `tools/new-device.py` on {date}, from {how}. Edit it as the port goes.

| What | Value |
|---|---|
| Board (the tree's `model`) | {model} |
| Compatible | `{compatible}` |

## The SoC addresses

{verdict}

| Argument | Value | Against the reference |
|---|---|---|
{rows}

`soc.args` is what `tools/soc-args.py` read out of the tree; `pull-soc.sh` makes it again from the phone on
adb. The run scripts add it to the kernel command line. Where each argument comes from in the tree:
`devices/{reference}/README.md`.

## What is this phone's own

The kernel, loader and user space know nothing about a phone: addresses, interrupts, rails and firmware
come from its tree and its Android. What is written for one phone is the list below. The head of each file
says where its numbers came from; that is the map of what to rewrite.

{checklist}

## Firmware

Pulled from your own phone, never committed: `devices/{reference}/pull-firmware.sh`, or
`pull-firmware-root.sh` where the firmware is readable by root only. Check with `make doctor`.

## Before you open a pull request

`tools/provenance-check.py` must pass: no vendor firmware, no UEFI drivers, no Apple files in the tree.
""".format(model=model, name=args.name, date=datetime.date.today().isoformat(), how=how,
           compatible=" ".join(compatible), verdict=verdict, rows=rows, checklist=checklist, reference=REFERENCE)
        open(os.path.join(out, "README.md"), "w", newline="\n").write(readme)

        print("new-device: %s" % model)
        print("new-device: %d of %d soc_* values read from the tree%s" % (len(found), len(names),
              "" if not missing else "; missing: " + ", ".join(missing)))
        print("new-device: %s" % ("the SAME values as the reference phone (%s): the SoC code already fits" % REFERENCE
              if same_as_reference else "differs from the reference in: %s" % (", ".join(differs + missing) or "nothing")))
        print("new-device: wrote %s" % out)
        for f in sorted(os.listdir(out)):
            print("    " + f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
