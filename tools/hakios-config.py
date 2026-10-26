#!/usr/bin/env python3
"""HakiOS configuration: defconfigs, fragments, a .config and the kernel's boot-arguments made from it.

The way Linux does it (arch/arm64/configs/<device>_defconfig, scripts/kconfig/merge_config.sh, `make
savedefconfig`), for a kernel whose options are not compiled in but given on its command line:

  tools/hakios-config.py defconfig redwood            .config = configs/defconfig + configs/redwood_defconfig
  tools/hakios-config.py defconfig redwood open       ... and the fragment configs/open.config on top
  tools/hakios-config.py merge virt                   put a fragment over the .config there is
  tools/hakios-config.py args                         the kernel's boot-arguments for the .config
  tools/hakios-config.py check | show | needs         validate it; say what it is; which firmware it wants
  tools/hakios-config.py savedefconfig                the .config as a minimal defconfig (its differences from configs/defconfig)
  tools/hakios-config.py list | help SYMBOL           what configs/ holds; what a symbol means

`make redwood_defconfig` and `make config` call it.

A symbol CONFIG_GPU=y is the boot-argument gpu=1; CONFIG_CPUS=8 is cpus=8; CONFIG_MODEM_BUFFER=0xa8000000 is
modem_buffer=0xa8000000. A symbol that is "not set" is left out of the command line (the kernel's own default
applies). A few symbols are not arguments but facts about the build (CONFIG_DEVICE, CONFIG_SOC, CONFIG_FIRMWARE).

Where this differs from Linux's, on purpose:
  - CONFIG_FIRMWARE="none" is a profile that loads NOTHING of the phone's own (no GPU, Wi-Fi, audio or modem
    firmware pulled from the device): the open configuration. The check refuses an option that needs firmware in it.
  - the kernel's command line holds 607 characters; the check says how much of them the configuration takes, and
    refuses what cannot fit, because a command line that is too long loses its tail without an error.
"""
import argparse
import difflib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KERNEL_LINE = 607                   # BOOT_LINE_LENGTH 608 less the terminator: what the kernel's boot_args can hold
RUN_OVERHEAD = 260                  # what a run adds itself: the loader's own arguments, xl's, the runner's (measured: ~190-250)

# name, kind, boot-arg, one-line help, firmware directory it needs, what it requires
# kinds: bool (y/n, becomes arg=1), int, hex, str, and meta (a fact about the build, not an argument)
SYMBOLS = [
    ("DEVICE", "meta", None, "the directory of the phone under devices/ this configuration is for", None, None),
    ("SOC", "meta", None, "the SoC family (sc7280: SM7325, QCM6490); it is part of the kernel's version string", None, None),
    ("FIRMWARE", "meta", None, "\"device\": options may use firmware pulled from your own phone (fw-*/); \"none\": the open "
                               "configuration, nothing of the phone's own is loaded", None, None),
    ("CPUS", "int", "cpus", "CPU cores the kernel uses, 1 to 8; one if not set", None, None),
    ("BOOTCPU", "int", "bootcpu", "the core the kernel runs on, 0 to 7; 7 is the biggest core (the loader keeps core 0)", None, None),
    ("CPUFREQ", "bool", "cpufreq", "set the boot core to its top speed", None, None),
    ("UFS", "int", "ufs", "the phone's storage: 1 a survey, read only; 2 the loader's block service, through which the kernel "
                          "reads it (the modem's file systems, /private/var)", None, None),
    ("UFS_WRITE", "bool", "ufs_write", "let the kernel write, to the partition `darwin` and nowhere else (writes to the phone's "
                                       "storage: it is the one switch for it)", None, "UFS=2"),
    ("VAR_STORE", "bool", "var_store", "keep /private/var across reboots, in the partition `darwin`", None, "UFS_WRITE"),
    ("TOUCHSCREEN", "bool", "touchscreen", "the touch screen, and the power and volume buttons", None, None),
    ("GPU", "bool", "gpu", "rendering on Adreno (an image built with GL=1; the GPU's firmware from your phone)", "fw-gpu", None),
    ("ADSP", "bool", "adsp", "the audio DSP; the battery level in the UI and charging state come through it", "fw-audio", None),
    ("SOUND", "int", "sound", "the speakers (4 is what the README's phone line uses)", "fw-audio", "ADSP"),
    ("MIC", "int", "mic", "the microphone (4 is what the README's phone line uses)", "fw-audio", "ADSP"),
    ("WIFI", "bool", "wifi", "Wi-Fi: the radio, its driver, a connection (also fw-wifi/mac and fw-wifi/network)", "fw-wifi", None),
    ("HAPTIC", "int", "haptic", "the vibration motor (5 is what the README's phone line uses)", "fw-haptic", None),
    ("CAM", "int", "cam", "the cameras (5 is what the README uses)", "fw-camera", None),
    ("MODEM", "bool", "modem", "the modem beside the system: its firmware, file systems, IPA, SIM, NAS", "fw-modem", "UFS=2"),
    ("MODEM_AGAIN", "bool", "modem_again", "start the modem although a kernel before this one left it shut down", None, "MODEM"),
    ("MODEM_BUFFER", "hex", "modem_buffer", "the modem's RMTFS buffer, an address below 4 GB (0xa8000000 on the reference phone)", None, "MODEM"),
    ("NET_TRACE", "int", "net_trace", "a log line for each of the first N frames on the network interfaces", None, None),
    ("NET_NAMES", "bool", "net_names", "a log line for every name asked of a DNS server and every TCP connection begun "
                                        "(the names go to the log: for debugging only)", None, None),
    ("VIRTIO_NET", "bool", "virtio_net", "QEMU's virtio network device as en0 (QEMU only; its bus address means nothing on a phone)", None, None),
]
BY_NAME = {s[0]: s for s in SYMBOLS}
FIRMWARE_VALUES = ("device", "none")
RANGES = {"CPUS": (1, 8), "BOOTCPU": (0, 7), "UFS": (1, 2)}


class ConfigError(Exception):
    pass


# ---- reading and writing --------------------------------------------------------------------------------------

def parse(path):
    """A defconfig, a fragment or a .config as (set, unset): {NAME: value} and a set of NAMEs `# CONFIG_X is not set`.
    Linux's rules: CONFIG_NAME=value, strings in quotes, no comment after a value, `# CONFIG_NAME is not set` is the
    way to say a symbol is off (so that a fragment can switch off what an earlier file switched on). `=n` is accepted
    for a bool as the same thing."""
    values, unset = {}, set()
    for n, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.strip()
        m = re.fullmatch(r"#\s*CONFIG_([A-Z0-9_]+) is not set", line)
        if m:
            unset.add(m.group(1))
            values.pop(m.group(1), None)
            continue
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"CONFIG_([A-Z0-9_]+)=(.*)", line)
        if not m:
            raise ConfigError("%s:%d: want CONFIG_NAME=value or # CONFIG_NAME is not set, got %r" % (path, n, line))
        name, value = m.group(1), m.group(2)
        if name not in BY_NAME:
            close = difflib.get_close_matches(name, BY_NAME, n=1)
            raise ConfigError("%s:%d: no symbol CONFIG_%s%s" % (path, n, name, " (did you mean CONFIG_%s?)" % close[0] if close else ""))
        if value == "n" and BY_NAME[name][1] == "bool":
            unset.add(name)
            values.pop(name, None)
            continue
        values[name] = value
        unset.discard(name)
    return values, unset


def typed(name, value):
    """The value of a symbol as Python has it, or a ConfigError that says what is wrong."""
    kind = BY_NAME[name][1]
    if kind == "bool":
        if value != "y":
            raise ConfigError("CONFIG_%s is a bool: y, or not set (got %s)" % (name, value))
        return True
    if kind in ("int", "hex"):
        if kind == "hex" and not value.lower().startswith("0x"):
            raise ConfigError("CONFIG_%s is a hex number and starts with 0x (got %s)" % (name, value))
        try:
            number = int(value, 16 if value.lower().startswith("0x") else 10)
        except ValueError:
            raise ConfigError("CONFIG_%s wants a %s, got %s" % (name, "hex number (0x...)" if kind == "hex" else "number", value))
        low, high = RANGES.get(name, (0, None))
        if number < low or (high is not None and number > high):
            raise ConfigError("CONFIG_%s is %s to %s, got %s" % (name, low, high if high is not None else "any", value))
        return number
    if not (len(value) >= 2 and value[0] == '"' and value[-1] == '"'):
        raise ConfigError("CONFIG_%s is a string in quotes (got %s)" % (name, value))
    text = value[1:-1]
    if name == "FIRMWARE" and text not in FIRMWARE_VALUES:
        raise ConfigError('CONFIG_FIRMWARE is "device" or "none" (got "%s")' % text)
    if not re.fullmatch(r"[A-Za-z0-9._-]*", text):
        raise ConfigError("CONFIG_%s holds letters, digits, dots, dashes and underscores (got %s)" % (name, value))
    return text


def config_dir(root):
    return os.path.join(root, "configs")


def find_config(root, name):
    """configs/<name>, configs/<name>_defconfig or configs/<name>.config: the file for NAME."""
    base = os.path.basename(name)
    for candidate in (base, base + "_defconfig", base + ".config"):
        path = os.path.join(config_dir(root), candidate)
        if os.path.isfile(path):
            return path
    available = sorted(os.listdir(config_dir(root))) if os.path.isdir(config_dir(root)) else []
    raise ConfigError("no configs/%s, configs/%s_defconfig or configs/%s.config (there is: %s)" %
                      (base, base, base, ", ".join(available) or "nothing"))


def overlay(config, files):
    """`config` with the files laid over it, later ones winning; `is not set` in a file removes a symbol.
    The same for the first file on an empty config (defconfig) and for a fragment over an existing one (merge)."""
    result = dict(config)
    for path in files:
        values, unset = parse(path)
        for name in unset:
            result.pop(name, None)
        result.update(values)
    return result


def resolve(root, names):
    """The files for `defconfig NAME FRAGMENT...`: the base, the device's defconfig, then the fragments."""
    files = [find_config(root, "defconfig")]
    device = names[0] if names else "defconfig"
    if device not in ("defconfig", ""):
        files.append(find_config(root, device if device.endswith("_defconfig") else device + "_defconfig"))
    files += [find_config(root, f) for f in names[1:]]
    return files


def rel(base, path):
    """`path` relative to `base` for the messages; as it is when it cannot be (another drive on Windows)."""
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:
        return path.replace(os.sep, "/")


def write_config(config, path, sources):
    """The .config, in the form Linux's has: every symbol, set or `is not set`, in the order of the table."""
    lines = ["#",
             "# Made by tools/hakios-config.py from %s." % ", ".join(sources),
             "# Do not edit it: change a defconfig or a fragment in configs/, and make the .config again.",
             "#"]
    for name, kind, _arg, _help, _needs, _requires in SYMBOLS:
        if name in config:
            lines.append("CONFIG_%s=%s" % (name, config[name]))
        elif kind != "meta":
            lines.append("# CONFIG_%s is not set" % name)
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def kernel_ver(root, config_path):
    """The kernel.ver string this .config gives (tools/kernel-version.py reads CONFIG_SOC from it)."""
    out = subprocess.run([sys.executable, os.path.join(HERE, "kernel-version.py"), "--config", config_path, "--root", root],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


# ---- what a configuration means -------------------------------------------------------------------------------

def check(config):
    """(problems, boot-arguments) of a merged configuration: the problems are strings, empty if it is fine."""
    problems = []
    parsed = {}
    for name, value in config.items():
        try:
            parsed[name] = typed(name, value)
        except ConfigError as e:
            problems.append(str(e))
    firmware = parsed.get("FIRMWARE", "device")
    for name in parsed:
        _n, _kind, _arg, _help, needs, requires = BY_NAME[name]
        if needs and firmware == "none":
            problems.append('CONFIG_%s needs the phone\'s own firmware (%s/), and this configuration says CONFIG_FIRMWARE="none": '
                            "it loads nothing from the device" % (name, needs))
        if requires:
            wanted, _, value = requires.partition("=")
            have = parsed.get(wanted)
            if have is None or (value and str(have) != value):
                problems.append("CONFIG_%s requires CONFIG_%s%s" % (name, wanted, "=" + value if value else ""))
    args = boot_args(parsed)
    length = len(" ".join(args))
    if length > KERNEL_LINE - RUN_OVERHEAD:
        problems.append("the boot-arguments take %d characters; a run adds about %d of its own and the kernel's command line holds %d: "
                        "the end of it would be lost" % (length, RUN_OVERHEAD, KERNEL_LINE))
    return problems, args


def boot_args(parsed):
    """['cpus=8', 'gpu=1', ...] in the order of the symbol table."""
    args = []
    for name, kind, arg, _help, _needs, _requires in SYMBOLS:
        if name not in parsed or arg is None:
            continue
        value = parsed[name]
        if kind == "bool":
            args.append("%s=1" % arg)
        elif kind == "hex":
            args.append("%s=0x%x" % (arg, value))
        else:
            args.append("%s=%s" % (arg, value))
    return args


def load(path):
    if not os.path.isfile(path):
        raise ConfigError("no %s: make one with `tools/hakios-config.py defconfig NAME` (or `make NAME_defconfig`)" % path)
    return parse(path)[0]


def report(problems):
    for p in problems:
        print("hakios-config: PROBLEM: " + p)


# ---- commands ---------------------------------------------------------------------------------------------------

def cmd_defconfig(a):
    files = resolve(a.root, [a.name] + a.fragments)
    config = overlay({}, files)
    problems, args = check(config)
    write_config(config, a.output, [rel(a.root, f) for f in files])
    print("hakios-config: %s made from %s" % (rel(os.getcwd(), a.output), ", ".join(os.path.basename(f) for f in files)))
    print("hakios-config: %d characters of boot-arguments; kernel.ver=%s" % (len(" ".join(args)), kernel_ver(a.root, a.output)))
    report(problems)
    return 1 if problems else 0


def cmd_merge(a):
    files = [find_config(a.root, f) for f in a.fragments]
    config = overlay(load(a.config), files)
    problems, args = check(config)
    write_config(config, a.config, [".config"] + [rel(a.root, f) for f in files])
    print("hakios-config: %s merged into %s; %d characters of boot-arguments" %
          (", ".join(os.path.basename(f) for f in files), rel(os.getcwd(), a.config), len(" ".join(args))))
    report(problems)
    return 1 if problems else 0


def cmd_args(a):
    if a.defconfig:
        # the arguments of a defconfig and fragments, without making a .config (what `make qemu-open` asks for)
        config = overlay({}, resolve(a.root, [a.defconfig] + a.fragment))
    else:
        config = load(a.config)
    problems, args = check(config)
    if problems and not a.force:
        for p in problems:
            print("hakios-config: PROBLEM: " + p, file=sys.stderr)
        return 1
    print(" ".join(args))
    return 0


def cmd_check(a):
    config = load(a.config)
    problems, args = check(config)
    for p in problems:
        print("PROBLEM: " + p)
    print("%s: %d symbols set, %d characters of boot-arguments (the kernel holds %d; a run adds about %d)%s" % (
        rel(os.getcwd(), a.config), len(config), len(" ".join(args)), KERNEL_LINE, RUN_OVERHEAD,
        "" if not problems else ", %d problem(s)" % len(problems)))
    return 1 if problems else 0


def cmd_show(a):
    config = load(a.config)
    problems, args = check(config)
    given = {x.split("=", 1)[0]: x for x in args}
    print("HakiOS configuration %s, kernel.ver=%s" % (rel(os.getcwd(), a.config), kernel_ver(a.root, a.config)))
    for name, _kind, arg, _help, _needs, _requires in SYMBOLS:
        if name in config:
            print("  CONFIG_%-13s = %-14s %s" % (name, config[name], "-> " + given[arg] if arg in given else "(a fact about the build)"))
    print("boot-arguments: %s" % " ".join(args))
    print("firmware wanted from the phone: %s" % (", ".join(sorted({BY_NAME[n][4] for n in config if BY_NAME[n][4]})) or "none"))
    for p in problems:
        print("PROBLEM: " + p)
    return 1 if problems else 0


def cmd_needs(a):
    config = load(a.config)
    needed = sorted({BY_NAME[n][4] for n in config if BY_NAME[n][4]})
    firmware = typed("FIRMWARE", config["FIRMWARE"]) if "FIRMWARE" in config else "device"
    if firmware == "none" and needed:
        print('this configuration says CONFIG_FIRMWARE="none" and still sets options that need firmware: %s' % ", ".join(needed))
        return 1
    missing = 0
    for d in needed:
        path = os.path.join(a.root, d)
        have = os.path.isdir(path) and any(os.scandir(path))
        missing += not have
        print("%-12s %s" % (d + "/", "pulled" if have else "MISSING: devices/<name>/pull-firmware.sh (or pull-firmware-root.sh)"))
    if not needed:
        print("nothing of the phone's own is needed")
    return 1 if missing else 0


def cmd_savedefconfig(a):
    config = load(a.config)
    base = parse(find_config(a.root, "defconfig"))[0]
    lines = ["# A minimal defconfig: what this configuration sets that configs/defconfig does not, or sets differently."]
    for name, *_ in SYMBOLS:
        if name in config and base.get(name) != config[name]:
            lines.append("CONFIG_%s=%s" % (name, config[name]))
        elif name not in config and name in base:
            lines.append("# CONFIG_%s is not set" % name)
    print("\n".join(lines))
    return 0


def first_comment(path):
    """The first comment of a file, which says what it is (`# CONFIG_X is not set` is not a comment, it is a setting)."""
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") and not re.fullmatch(r"#\s*CONFIG_[A-Z0-9_]+ is not set\s*", line):
            return line.lstrip("# ").strip()
    return ""


def cmd_list(a):
    d = config_dir(a.root)
    names = sorted(os.listdir(d)) if os.path.isdir(d) else []
    print("defconfigs (tools/hakios-config.py defconfig NAME, or make NAME_defconfig):")
    for n in names:
        if n == "defconfig" or n.endswith("_defconfig"):
            print("  %-16s %s" % (n[:-len("_defconfig")] if n.endswith("_defconfig") else n, first_comment(os.path.join(d, n))))
    print("fragments (put over a configuration: tools/hakios-config.py merge NAME, or defconfig NAME FRAGMENT):")
    for n in names:
        if n.endswith(".config"):
            print("  %-16s %s" % (n[:-len(".config")], first_comment(os.path.join(d, n))))
    return 0


def cmd_help(a):
    name = a.symbol.upper()
    name = name[len("CONFIG_"):] if name.startswith("CONFIG_") else name
    if name not in BY_NAME:
        close = difflib.get_close_matches(name, BY_NAME, n=3)
        print("no symbol CONFIG_%s%s" % (name, " (did you mean %s?)" % ", ".join("CONFIG_" + c for c in close) if close else ""))
        return 1
    _n, kind, arg, help_, needs, requires = BY_NAME[name]
    print("CONFIG_%s  (%s)" % (name, kind))
    print("  " + help_)
    if arg:
        print("  boot-argument: %s=%s" % (arg, {"bool": "1", "hex": "0x...", "int": "N", "str": "..."}[kind]))
    if needs:
        print("  needs firmware pulled from your phone: %s/" % needs)
    if requires:
        print("  requires: CONFIG_%s" % requires)
    return 0


def main():
    # --root and --config are accepted before the command and after it
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help="the top of the repository (tests use another)")
    common.add_argument("--config", default=argparse.SUPPRESS, help="the .config to use (default: .config at the top)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("defconfig", parents=[common], help="make .config from a defconfig and fragments")
    p.add_argument("name")
    p.add_argument("fragments", nargs="*")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_defconfig)
    p = sub.add_parser("merge", parents=[common], help="put fragments over the .config")
    p.add_argument("fragments", nargs="+")
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("args", parents=[common], help="the kernel's boot-arguments")
    p.add_argument("--force", action="store_true", help="print them even if the check finds a problem")
    p.add_argument("--defconfig", help="the arguments of this defconfig (and --fragment) instead of the .config's")
    p.add_argument("--fragment", action="append", default=[], help="a fragment over it; more than one is allowed")
    p.set_defaults(fn=cmd_args)
    for name, fn, text in (("check", cmd_check, "validate the .config"), ("show", cmd_show, "say what the .config is"),
                           ("needs", cmd_needs, "which firmware the .config wants, and whether it is there"),
                           ("savedefconfig", cmd_savedefconfig, "the .config as a minimal defconfig"),
                           ("list", cmd_list, "the defconfigs and fragments in configs/")):
        sub.add_parser(name, parents=[common], help=text).set_defaults(fn=fn)
    p = sub.add_parser("help", parents=[common], help="what a symbol means")
    p.add_argument("symbol")
    p.set_defaults(fn=cmd_help)
    a = ap.parse_args()
    a.root = os.path.abspath(getattr(a, "root", ROOT))
    a.config = os.path.abspath(getattr(a, "config", os.path.join(a.root, ".config")))
    if getattr(a, "output", None) is None:
        a.output = a.config
    try:
        sys.exit(a.fn(a))
    except ConfigError as e:
        print("hakios-config: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
