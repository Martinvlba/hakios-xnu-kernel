#!/usr/bin/env python3
"""tools/hakios-config.py: what the configurations of configs/ give, and what the tool refuses.

  the real configs/   redwood gives exactly the arguments the project's phone line used before there were configurations
                      (so choosing it changes nothing); redwood + open loads nothing of the phone's own; persist adds the
                      writes to `darwin`; virt and trace do what they say; kernel.ver carries -sc7280
  refusals            an unknown symbol (with a suggestion), a firmware option in the open profile, a dependency that is
                      missing, a number out of range, a bad hex or string, a command line that cannot fit
  workflow            merge over an existing .config; `is not set` in a fragment switches off; savedefconfig round-trips;
                      --config and --root work after the command as well as before it
"""
import os
import shutil
import subprocess
import sys
import tempfile
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(HERE, "hakios-config.py")
failures = []

# what `make phone` passed before there were configurations: the runner's four, then the Makefile's ARGS
OLD_PHONE_LINE = ("bootcpu=7 cpus=8 cpufreq=1 ufs=2 touchscreen=1 gpu=1 adsp=1 sound=4 wifi=1 haptic=5 modem=1 modem_again=1 "
                  "modem_buffer=0xa8000000").split()


def check(label, ok, detail=""):
    print("%s %s" % ("ok  " if ok else "FAIL", label))
    if not ok:
        failures.append(label)
        if detail:
            print("     " + str(detail).strip().replace("\n", "\n     ")[:500])


def run(*args, root=ROOT):
    return subprocess.run([sys.executable, TOOL, "--root", root, *args], capture_output=True, text=True)


def args_of(out):
    return out.stdout.split()


work = tempfile.mkdtemp(prefix="hakios-config-test-")
try:
    cfg = os.path.join(work, "cfg")

    r = run("defconfig", "redwood", "-o", cfg)
    check("defconfig redwood: makes the .config", r.returncode == 0 and os.path.isfile(cfg), r.stdout + r.stderr)
    a = args_of(run("args", "--config", cfg))
    check("redwood: the same arguments as the project's phone line before configurations", sorted(a) == sorted(OLD_PHONE_LINE),
          "got %s\nwant %s" % (sorted(a), sorted(OLD_PHONE_LINE)))
    check("redwood: kernel.ver carries the SoC", "-sc7280-" in r.stdout, r.stdout)
    check("redwood: the .config is in Linux's form",
          "CONFIG_GPU=y" in open(cfg).read() and "# CONFIG_CAM is not set" in open(cfg).read() and 'CONFIG_FIRMWARE="device"' in open(cfg).read())
    check("redwood: check passes", run("check", "--config", cfg).returncode == 0)

    r = run("needs", "--config", cfg)
    check("redwood: needs says which firmware, and that it is missing here (none is pulled in a clean checkout)",
          "fw-gpu/" in r.stdout and "fw-modem/" in r.stdout, r.stdout)

    # the open profile over redwood
    o = os.path.join(work, "open")
    r = run("defconfig", "redwood", "open", "-o", o)
    check("redwood + open: passes", r.returncode == 0, r.stdout + r.stderr)
    check("redwood + open: nothing of the phone's own is loaded",
          sorted(args_of(run("args", "--config", o))) == sorted("cpus=8 bootcpu=7 cpufreq=1 ufs=2 touchscreen=1".split()),
          run("args", "--config", o).stdout)
    r = run("needs", "--config", o)
    check("redwood + open: needs nothing", r.returncode == 0 and "nothing of the phone's own" in r.stdout, r.stdout)

    # fragments
    p = os.path.join(work, "persist")
    shutil.copyfile(cfg, p)
    r = run("merge", "persist", "--config", p)
    check("merge persist: ufs_write and var_store appear", r.returncode == 0 and {"ufs_write=1", "var_store=1"} <= set(args_of(run("args", "--config", p))), r.stdout + r.stderr)
    check("merge persist: UFS stays 2, not duplicated", args_of(run("args", "--config", p)).count("ufs=2") == 1)
    v = os.path.join(work, "virt")
    run("defconfig", "defconfig", "virt", "-o", v)
    check("defconfig + virt: virtio_net, one core, no firmware", sorted(args_of(run("args", "--config", v))) == ["cpus=1", "virtio_net=1"],
          run("args", "--config", v).stdout)
    d = os.path.join(work, "base")
    run("defconfig", "defconfig", "-o", d)
    check("defconfig alone is the open one: cpus=1", args_of(run("args", "--config", d)) == ["cpus=1"], run("args", "--config", d).stdout)
    t = os.path.join(work, "trace")
    run("defconfig", "redwood", "trace", "-o", t)
    check("trace: net_trace=20", "net_trace=20" in args_of(run("args", "--config", t)))

    # the arguments straight from a defconfig, with no .config made (what `make qemu-open` asks for)
    check("args --defconfig defconfig: cpus=1", args_of(run("args", "--defconfig", "defconfig")) == ["cpus=1"], run("args", "--defconfig", "defconfig").stdout)
    check("args --defconfig redwood --fragment open: the open phone",
          sorted(args_of(run("args", "--defconfig", "redwood", "--fragment", "open"))) == sorted("cpus=8 bootcpu=7 cpufreq=1 ufs=2 touchscreen=1".split()))
    check("args --defconfig redwood: the same as the .config's", sorted(args_of(run("args", "--defconfig", "redwood"))) == sorted(a))

    # the order of the arguments is the table's: cpus first, as in the README's lines
    check("the arguments come in the order of the symbol table", args_of(run("args", "--config", cfg))[:3] == ["cpus=8", "bootcpu=7", "cpufreq=1"])

    # savedefconfig round trip: the minimal defconfig, made a defconfig of its own, gives the same arguments
    sd = run("savedefconfig", "--config", cfg)
    check("savedefconfig: drops what defconfig already has", "CONFIG_FIRMWARE=\"none\"" not in sd.stdout and "CONFIG_GPU=y" in sd.stdout, sd.stdout)
    fake = os.path.join(work, "root")
    shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(fake, "configs"))
    open(os.path.join(fake, "configs", "mine_defconfig"), "w").write(sd.stdout)
    shutil.copyfile(os.path.join(ROOT, "VERSION"), os.path.join(fake, "VERSION"))
    r = run("defconfig", "mine", "-o", os.path.join(work, "mine"), root=fake)
    check("savedefconfig round trip: same arguments", sorted(args_of(run("args", "--config", os.path.join(work, "mine"), root=fake))) == sorted(a), r.stdout + r.stderr)

    # `--config` and `--root` after the command too (checked above for --config); before it:
    check("--config before the command works too", run("--config", cfg, "check").returncode == 0)

    # refusals, one configuration each in a temporary configs/
    def refusal(label, text, want, base_text=None):
        d = os.path.join(work, "r%d" % len(os.listdir(work)))
        os.makedirs(os.path.join(d, "configs"))
        open(os.path.join(d, "configs", "defconfig"), "w").write(base_text if base_text is not None else 'CONFIG_FIRMWARE="device"\nCONFIG_CPUS=1\n')
        open(os.path.join(d, "configs", "bad_defconfig"), "w").write(text)
        shutil.copyfile(os.path.join(ROOT, "VERSION"), os.path.join(d, "VERSION"))
        r = run("defconfig", "bad", "-o", os.path.join(d, "out"), root=d)
        out = r.stdout + r.stderr
        check(label, r.returncode != 0 and want in out, "exit %d, want %r in:\n%s" % (r.returncode, want, out))

    refusal("an unknown symbol is refused, with a suggestion", "CONFIG_WIFFI=y\n", "did you mean CONFIG_WIFI")
    refusal("a line that is no setting is refused, with its number", "this is not a setting\n", "bad_defconfig:1")
    refusal("a firmware option in the open profile is refused", 'CONFIG_FIRMWARE="none"\nCONFIG_GPU=y\n', "needs the phone's own firmware")
    refusal("a missing dependency is refused (modem without ufs=2)", "CONFIG_MODEM=y\nCONFIG_UFS=1\n", "requires CONFIG_UFS=2")
    refusal("var_store without ufs_write is refused", "CONFIG_VAR_STORE=y\n", "requires CONFIG_UFS_WRITE")
    refusal("cpus out of range", "CONFIG_CPUS=9\n", "CONFIG_CPUS is 1 to 8")
    refusal("a hex that is not hex", "CONFIG_UFS=2\nCONFIG_MODEM=y\nCONFIG_MODEM_BUFFER=a8000000\n", "starts with 0x")
    refusal("a bool that is not y", "CONFIG_GPU=1\n", "is a bool: y, or not set")
    refusal("a string without quotes", "CONFIG_DEVICE=redwood\n", "string in quotes")
    refusal("a FIRMWARE that is neither", 'CONFIG_FIRMWARE="maybe"\n', '"device" or "none"')

    r = run("defconfig", "nosuchphone", "-o", os.path.join(work, "x"))
    check("an unknown defconfig lists what there is", r.returncode != 0 and "redwood_defconfig" in (r.stdout + r.stderr), r.stdout + r.stderr)
    r = run("args", "--config", os.path.join(work, "nothing"))
    check("no .config: says how to make one", r.returncode != 0 and "defconfig NAME" in (r.stdout + r.stderr), r.stdout + r.stderr)

    # `=n` for a bool is `is not set`; a later `is not set` removes what an earlier file set
    fake2 = os.path.join(work, "root2")
    os.makedirs(os.path.join(fake2, "configs"))
    shutil.copyfile(os.path.join(ROOT, "VERSION"), os.path.join(fake2, "VERSION"))
    open(os.path.join(fake2, "configs", "defconfig"), "w").write("CONFIG_CPUS=2\nCONFIG_CPUFREQ=y\n")
    open(os.path.join(fake2, "configs", "x_defconfig"), "w").write("CONFIG_CPUFREQ=n\nCONFIG_TOUCHSCREEN=y\n")
    open(os.path.join(fake2, "configs", "y.config"), "w").write("# CONFIG_TOUCHSCREEN is not set\n")
    out2 = os.path.join(work, "o2")
    run("defconfig", "x", "y", "-o", out2, root=fake2)
    check("=n and `is not set` switch a symbol off", args_of(run("args", "--config", out2, root=fake2)) == ["cpus=2"], run("args", "--config", out2, root=fake2).stdout)

    # the command line that cannot fit: the table is short, so shrink the limit and ask the checker
    spec = importlib.util.spec_from_file_location("hc", TOOL)
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    hc.KERNEL_LINE = 300
    problems, _ = hc.check({"CPUS": "8", "BOOTCPU": "7", "UFS": "2", "MODEM": "y", "MODEM_BUFFER": "0xa8000000", "FIRMWARE": '"device"'})
    check("a command line that cannot fit is a problem", any("the end of it would be lost" in p for p in problems), problems)

    # help, show, list
    r = run("help", "gpu")
    check("help GPU: the argument, the firmware", "gpu=1" in r.stdout and "fw-gpu/" in r.stdout, r.stdout)
    r = run("help", "CONFIG_MODEM_BUFFER")
    check("help takes CONFIG_ and says what it requires", r.returncode == 0 and "requires: CONFIG_MODEM" in r.stdout, r.stdout)
    r = run("help", "gpuu")
    check("help on an unknown symbol suggests", r.returncode != 0 and "did you mean CONFIG_GPU" in r.stdout, r.stdout)
    r = run("list")
    check("list: defconfigs and fragments", all(x in r.stdout for x in ("redwood", "open", "virt", "persist", "trace")), r.stdout)
finally:
    shutil.rmtree(work, ignore_errors=True)

if failures:
    sys.exit("hakios-config-test: %d FAILED" % len(failures))
print("hakios-config-test: all passed")
