#!/usr/bin/env python3
"""The 135 library functions: how far each one gets, and against what.

An air.visible function is not a shader. Metal links one into a caller at
pipeline time through the ABI its own metadata describes (air.stitching_info),
which this project does not implement -- so a third of the corpus had nothing
that could be dispatched and was counted only as "not an entry point".

air_to_nir's --library wraps one in a generated entry point instead: argument i
read from device buffer 0 at byte 16*i, the return value written to buffer 1
(wrap_library_function in air_to_nir.c). That makes a library function runnable
and comparable, and this script reports what each comparison is worth, because
they are not worth the same thing:

  LLVM            the function compiled for this machine by llc and called with
                  the same bytes. Independent of everything of ours, and the
                  strongest evidence here. Available only for a function that
                  calls no air.* at all: an air.* call is an undefined symbol
                  on the host, and defining one would put our own reading of
                  that operation on both sides of the comparison.
  folded vs run   our NIR constant-folded on this machine against our NIR run
                  by a Gallium driver. Not independent of the frontend -- a
                  mistranslation is on both sides -- but it does catch the
                  backend and the hardware disagreeing with the folder, which
                  is how the ir3 u32-to-f32 rounding bug was found.

Both sides read argument i from byte 16*i, so neither comparison can catch a
slot layout that is wrong in the same way twice. What they do catch is the two
sides disagreeing about which slot is which argument, because the fill pattern
makes every slot's bytes different. Apple's own air.arg_type_name for each
argument is checked against the LLVM signature before anything runs
(cpuref.py, check_against_metadata), which is the one part of the layout that
has a second source.

usage: libcheck.py [--list] [FILE.metallib...]
       (no files: portrait_filters.metallib, the corpus's library)
"""

import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AIR2NIR = os.environ.get(
    "AIR2NIR", os.path.join(ROOT, "build/mesa-host/build/src/compiler/air/air2nir"))
AIR_RUN = os.environ.get(
    "AIR_RUN",
    os.path.join(ROOT, "build/mesa-host/build/src/gallium/drivers/freedreno/air_run"))
CPUREF = os.path.join(ROOT, "tools/air/cpuref.py")

LIBS = ["sysimage/rootfs/System/Library/CoreImage/PortraitFilters.cifilter/"
        "portrait_filters.metallib"]

# The air.* operations whose result is exactly defined, by the family name in
# the declaration. The split matters: a function calling an approximate one
# cannot be compared bit-exactly against anything, here or anywhere, and saying
# so is not the same as saying it is wrong (docs/air.md section 4).
APPROX = {"fast_pow", "fast_powr", "fast_exp", "fast_exp2", "fast_exp10", "fast_log",
          "fast_log2", "fast_log10", "fast_sin", "fast_cos", "fast_tan", "fast_atan2",
          "fast_rsqrt", "fast_divide", "fast_asin", "fast_acos", "fast_atan",
          "fast_sinh", "fast_cosh", "fast_tanh"}


def report(lib):
    """air2nir --library --report: one row per module of the library."""
    out = subprocess.run([AIR2NIR, "--library", "--report", lib],
                         capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        if not line.startswith("module="):
            continue
        r = dict(kv.split("=", 1) for kv in line.split("\t") if "=" in kv)
        r["calls"] = [c for c in r.get("calls", "").split(",") if c]
        rows.append(r)
    return rows


def families(row):
    return {c.split(".")[1] for c in row["calls"] if c.startswith("air.")}


def folded(lib, name):
    """What our frontend folds the wrapper to, as 32-bit words."""
    out = subprocess.run([AIR2NIR, "--library", "--eval", lib, name],
                         capture_output=True, text=True)
    if out.returncode:
        return None, "air2nir --eval failed"
    for line in out.stdout.splitlines():
        if not line.startswith("  buffer"):
            continue
        if "did not fold" in line:
            return None, "the store did not fold to a constant"
        return [bits_of(v) for v in line.split()[1:]], None
    return None, "no buffer store in the folded shader"


def bits_of(text):
    """A float as printed by --eval, back to its 32 bits. NaN keeps no payload."""
    f = float(text)
    if f != f:
        return "nan"
    return struct.unpack("<I", struct.pack("<f", f))[0]


def ran(lib, name):
    """What a Gallium driver writes into buffer 1, as 32-bit words."""
    out = subprocess.run([AIR_RUN, "--library", lib, name],
                         capture_output=True, text=True)
    if out.returncode:
        last = (out.stderr or out.stdout).strip().splitlines()
        return None, (last[-1][:120] if last else "air_run failed")
    words, taking = [], False
    for line in out.stdout.splitlines():
        if line.startswith("air_run: device buffer 1 after"):
            taking = True
            continue
        if taking:
            m = re.match(r"\s*\[\s*\d+\]\s+\S+\s+0x([0-9a-f]+)", line)
            if not m:
                break
            words.append(int(m.group(1), 16))
    return (words, None) if words else (None, "air_run printed no buffer")


def same(a, b):
    """Word for word, with any NaN equal to any NaN and nothing else loose."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == "nan" or y == "nan":
            # a NaN's payload is not specified by Metal and not preserved by
            # every lowering, so two NaNs agree and a NaN beside a number does
            # not. Counting them apart is the caller's job.
            if not (is_nan(x) and is_nan(y)):
                return False
        elif x != y:
            return False
    return True


def is_nan(w):
    return w == "nan" or (isinstance(w, int) and (w & 0x7f800000) == 0x7f800000
                          and (w & 0x7fffff))


def llvm_agrees(lib, name):
    """cpuref.py --compare: LLVM on this machine against our folded NIR."""
    out = subprocess.run([sys.executable, CPUREF, "--compare", lib, name],
                         capture_output=True, text=True)
    text = out.stdout + out.stderr
    if "they agree" in text:
        return True, None
    if "they DIFFER" in text:
        return False, text.strip().splitlines()[-1][:120]
    return None, text.strip().splitlines()[-1][:120] if text.strip() else "no output"


def main(argv):
    want_list = "--list" in argv
    libs = [a for a in argv if not a.startswith("--")] or LIBS
    for prog in (AIR2NIR, AIR_RUN):
        if not os.path.exists(prog):
            print("libcheck: run tools/air/mk-host.sh first (%s)" % prog, file=sys.stderr)
            return 1

    rows = []
    for lib in libs:
        path = lib if os.path.isabs(lib) else os.path.join(ROOT, lib)
        if not os.path.exists(path):
            print("libcheck: missing %s" % lib, file=sys.stderr)
            return 1
        for r in report(path):
            r["path"] = path
            rows.append(r)

    wrapped = [r for r in rows if r.get("translate") == "ok"]
    refused = [r for r in rows if r.get("translate") != "ok"]

    plain, exact, approx = [], [], []
    for r in wrapped:
        fam = families(r)
        (plain if not fam else approx if fam & APPROX else exact).append(r)

    # what each one is worth, measured rather than assumed
    llvm_ok, llvm_bad, llvm_other = [], [], []
    for r in plain:
        ok, why = llvm_agrees(r["path"], r["module"])
        (llvm_ok if ok else llvm_bad if ok is False else llvm_other).append((r, why))

    run_ok, run_bad, no_fold, no_run, nan_rows = [], [], [], [], []
    for r in wrapped:
        theirs, why2 = ran(r["path"], r["module"])
        if theirs is None:
            no_run.append((r, why2))
            continue
        ours, why1 = folded(r["path"], r["module"])
        if ours is None:
            # it ran: what is missing is the host side of the comparison, not
            # the dispatch. Folding stops at a loop or at control flow the
            # folder cannot resolve, which says nothing about the shader.
            no_fold.append((r, why1))
            continue
        if any(is_nan(w) for w in ours):
            nan_rows.append(r)
        (run_ok if same(ours, theirs) else run_bad).append(
            (r, "folded %s, ran %s" % (ours, theirs)))

    print("libcheck: %d library functions in %d librar%s"
          % (len(rows), len(libs), "y" if len(libs) == 1 else "ies"))
    print("  %3d  wrapped in a generated entry point and translated" % len(wrapped))
    print("  %3d  refused: an argument the wrapper cannot pass" % len(refused))
    print()
    print("  of the %d wrapped, what a comparison can be worth:" % len(wrapped))
    print("  %3d  call no air.* at all, so LLVM on this machine is a full reference"
          % len(plain))
    print("  %3d  call only exactly defined air.*: no host definition for those yet"
          % len(exact))
    print("  %3d  call an approximate air.*, which nothing compares bit-exactly"
          % len(approx))
    print()
    print("  %3d  of the %d agree with LLVM bit for bit" % (len(llvm_ok), len(plain)))
    print("  %3d  DIFFER from LLVM" % len(llvm_bad))
    print("  %3d  could not be compared with LLVM" % len(llvm_other))
    print("  %3d  of the %d dispatched on softpipe and read their result back"
          % (len(run_ok) + len(run_bad) + len(no_fold), len(wrapped)))
    print("  %3d  of those match our own folding of the same shader" % len(run_ok))
    print("  %3d  run but DIFFER from our own folding" % len(run_bad))
    print("  %3d  ran, but our folding did not reduce to a constant, so there was "
          "nothing to compare" % len(no_fold))
    print("  %3d  could not be dispatched at all" % len(no_run))
    if nan_rows:
        print("  (%d of them return a NaN, where only the quiet bit is compared)"
              % len(nan_rows))

    if want_list:
        for r in refused:
            print("        refused  %-44s %s" % (r["module"], r.get("error", "")[:80]))
        for r, why in llvm_bad + run_bad + llvm_other + no_run + no_fold:
            print("        %-44s %s" % (r["module"], why))

    # the rows have to account for every module before the numbers are read
    if len(wrapped) + len(refused) != len(rows):
        print("libcheck: the rows add to %d and there are %d functions"
              % (len(wrapped) + len(refused), len(rows)), file=sys.stderr)
        return 1
    if len(run_ok) + len(run_bad) + len(no_fold) + len(no_run) != len(wrapped):
        print("libcheck: the run rows add to %d and %d were wrapped"
              % (len(run_ok) + len(run_bad) + len(no_fold) + len(no_run), len(wrapped)),
              file=sys.stderr)
        return 1
    if len(plain) + len(exact) + len(approx) != len(wrapped):
        print("libcheck: the reference rows add to %d and %d were wrapped"
              % (len(plain) + len(exact) + len(approx), len(wrapped)), file=sys.stderr)
        return 1
    print("libcheck: every function is in exactly one row of each group. "
          "Wrapping is not stitching: Metal's own ABI for these functions is not "
          "implemented, and nothing here is evidence about it.")
    return 1 if (llvm_bad or run_bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
