#!/usr/bin/env python3
"""The pages of the documentation that are made from the source, so that they cannot say something the source does not:

  docs/reference/drivers.md     every driver (drivers/*/conf, and the SoC and device repositories'): its symbol, files, what it
                                requires, which boot-arguments it reads, which configurations build it
  docs/devices/index.md         every device and SoC (devices/*/device.conf, socs/*/soc.conf): how far each part works
  docs/devices/<name>.md        one device's page

usage: tools/docs-gen.py            write them
       tools/docs-gen.py --check    exit 1 if one is not what the source makes (CI runs this)
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "https://github.com/oosexgang/hakios-xnu-kernel/blob/main/"
MARK = {"works": "works", "partial": "partly", "missing": "no"}
HEADER = "<!-- Made by tools/docs-gen.py from %s. Do not edit: change that, then run tools/docs-gen.py. -->\n\n"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drivers_page(hc, hd):
    readers = hc.arg_readers() or {}
    by_file = {}
    for arg, files in readers.items():
        for f in files:
            by_file.setdefault(f, set()).add(arg)
    configs = {}
    for kind, name in hd.present():
        if kind != "device":
            continue
        fields = hd.describe(kind, name)
        defconfig = hd.one(fields, "defconfig")
        try:
            config = hc.overlay({}, hc.resolve(ROOT, [defconfig]))
        except hc.ConfigError:
            continue
        configs[name] = hc.drivers_built(config)
    out = [HEADER % "drivers/*/conf and the SoC and device repositories' drivers",
           "# Drivers", "",
           "Every driver the kernel can be built with. `CONFIG_HAKIOS_<NAME>=y` in a defconfig builds it in; "
           "[Drivers in the configuration](../drivers-config.md) says how, and [Writing a driver](../porting/drivers.md) how to add one.", ""]
    out += ["## Always built", "", "| File | What |", "|---|---|"]
    for path, tags, sub, help_ in hc.STANDARD:
        out.append("| `%s` | %s%s |" % (os.path.basename(path), help_, " (with XNU's `%s`)" % " ".join(tags) if tags else ""))
    subs = []
    for d in hc.DRIVERS:
        key = (d.origin or "", d.sub)
        if key not in subs:
            subs.append(key)
    for origin, sub in subs:
        title = sub if not origin else "%s (%s)" % (sub, os.path.relpath(origin, ROOT))
        out += ["", "## %s" % title, ""]
        out += ["| Driver | Files | Requires | Boot arguments | %s|" % "".join("%s | " % n for n in configs),
                "|---|---|---|---|" + "---|" * len(configs)]
        for d in hc.DRIVERS:
            if (d.origin or "", d.sub) != (origin, sub):
                continue
            args = sorted({a for f, _t in d.files for a in by_file.get(os.path.basename(f), ())})
            out.append("| **`%s`**<br>%s | %s | %s | %s | %s|" % (
                d.symbol, d.help.replace("|", "\\|"), "<br>".join("`%s`" % os.path.basename(f) for f, _t in d.files),
                ", ".join("`HAKIOS_%s`" % r for r in d.requires) or "", ", ".join("`%s=`" % a for a in args[:8]) +
                (" and %d more" % (len(args) - 8) if len(args) > 8 else ""),
                "".join("%s | " % ("built" if built[d.symbol] else "") for built in configs.values())))
    out += ["", "The columns on the right are the devices' defconfigs: \"built\" means that defconfig builds the driver in.", ""]
    return "\n".join(out)


def repository_link(hd, name):
    """A device with a repository of its own (devices/manifest.conf) links there; one kept in this tree, here."""
    for kind, n, fields, _line in hd.manifest():
        repository = hd.one(fields, "repository")
        if kind == "device" and n == name and repository:
            url = repository[:-4] if repository.endswith(".git") else repository
            return "[%s](%s)" % (url.rsplit("/", 1)[-1], url)
    return "[`devices/%s/`](%sdevices/%s)" % (name, REPO.replace("/blob/", "/tree/"), name)


def device_page(hd, name, fields):
    out = [HEADER % "devices/%s/device.conf" % name, "# %s (`%s`)" % (hd.one(fields, "title"), name), ""]
    rows = [("Vendor", hd.one(fields, "vendor", "?")), ("SoC", "`%s`" % hd.one(fields, "soc", "?")),
            ("Configuration", "`make %s`" % hd.one(fields, "defconfig", "?")), ("Maintainer", hd.one(fields, "maintainer", "?")),
            ("Repository", repository_link(hd, name))]
    if hd.one(fields, "firmware"):
        rows.append(("Firmware", "`devices/%s/%s` pulls it from your own phone" % (name, hd.one(fields, "firmware"))))
    out += ["| | |", "|---|---|"] + ["| %s | %s |" % r for r in rows] + ["", "## What works", "", "| Part | State | |", "|---|---|---|"]
    for area, state, note in hd.statuses(fields):
        out.append("| %s | %s | %s |" % (area, MARK[state], note.replace("|", "\\|")))
    out += ["", "The state is what has been seen on this device, written down in `devices/%s/device.conf`; "
            "a part the table does not name has not been looked at." % name, ""]
    return "\n".join(out)


def devices_page(hd):
    devices = [(n, hd.describe("device", n)) for k, n in hd.present() if k == "device"]
    socs = [(n, hd.describe("soc", n)) for k, n in hd.present() if k == "soc"]
    areas = [a for a in hd.AREAS if any(a == x[0] for _n, f in devices for x in hd.statuses(f))]
    out = [HEADER % "devices/*/device.conf and socs/*/soc.conf", "# Devices", "",
           "Every device with a repository, and how far each part of it works. A device of yours: "
           "[Port a device](../porting/index.md).", "",
           "| Device | SoC | %s |" % " | ".join(areas), "|---|---|" + "---|" * len(areas)]
    short = {"works": "yes", "partial": "partly", "missing": "no"}
    for name, fields in devices:
        states = {a: s for a, s, _n in hd.statuses(fields)}
        out.append("| [%s](%s.md) | `%s` | %s |" % (hd.one(fields, "title"), name, hd.one(fields, "soc", "?"),
                                                   " | ".join(short.get(states.get(a), "") for a in areas)))
    out += ["", "An empty cell is a part nobody has looked at on that device yet.", "", "## SoC families", "", "| SoC | What | |", "|---|---|---|"]
    for name, fields in socs:
        carried = [d for k, d in hd.present() if k == "device" and os.path.isdir(os.path.join(ROOT, "devices", d, "soc"))
                   and hd.config_tool().soc_name(os.path.join(ROOT, "devices", d, "soc")) == name]
        where = "carried by %s (`devices/%s/soc/`)" % (", ".join(carried), carried[0]) if carried else "`socs/%s/`" % name
        out.append("| `%s` | %s | %s |" % (name, hd.one(fields, "title"), where))
    out.append("")
    return "\n".join(out)


def pages():
    hc = load("hakios_config", "hakios-config.py")
    hd = load("hakios_device", "hakios-device.py")
    made = {os.path.join("docs", "reference", "drivers.md"): drivers_page(hc, hd),
            os.path.join("docs", "devices", "index.md"): devices_page(hd)}
    for kind, name in hd.present():
        if kind == "device":
            made[os.path.join("docs", "devices", name + ".md")] = device_page(hd, name, hd.describe(kind, name))
    return made


def main():
    check = "--check" in sys.argv[1:]
    stale = 0
    for path, text in sorted(pages().items()):
        full = os.path.join(ROOT, path)
        have = open(full, encoding="utf-8").read() if os.path.isfile(full) else None
        if have == text:
            continue
        if check:
            print("docs-gen: %s is not what the source makes: run tools/docs-gen.py" % path)
            stale += 1
        else:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w", encoding="utf-8", newline="\n").write(text)
            print("docs-gen: %s written" % path)
    if check and not stale:
        print("docs-gen: the made pages are current")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
