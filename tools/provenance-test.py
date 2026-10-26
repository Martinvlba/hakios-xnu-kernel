#!/usr/bin/env python3
"""tools/provenance-check.py on small repositories of its own: what it lets through and what it stops.

Every kind of violation has a case that must fail and the same case, explained in tools/provenance.allow,
that must pass; a clean tree passes; a `vendor` line is a tripwire that stays quiet while the files are
gone and fails when one comes back; a stale line is reported; a comment that only mentions a flash
command is not a flash command.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "provenance-check.py")
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64
MACHO = b"\xcf\xfa\xed\xfe\x07\x00\x00\x01" + b"not really a kernel"
FIRMWARE = ("static const unsigned char fw[] = {\n" + "0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,\n" * 250 + "};\n").encode()
failures = []


def repo(files, allow=""):
    d = tempfile.mkdtemp(prefix="provenance-test-")
    subprocess.run(["git", "init", "-q", d], check=True)
    files = dict(files)
    files["tools/provenance.allow"] = allow
    for name, data in files.items():
        path = os.path.join(d, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(data if isinstance(data, bytes) else data.encode())
    subprocess.run(["git", "-C", d, "add", "-A"], check=True, capture_output=True)
    return d


def check(label, files, allow, ok, expect=()):
    d = repo(files, allow)
    try:
        r = subprocess.run([sys.executable, CHECK, "--root", d], capture_output=True, text=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    out = r.stdout + r.stderr
    good = (r.returncode == 0) == ok and all(e in out for e in expect)
    print("%s %s" % ("ok  " if good else "FAIL", label))
    if not good:
        failures.append(label)
        print("     want exit %s and %r; got exit %d:\n     %s" % (0 if ok else 1, list(expect), r.returncode,
                                                              out.strip().replace("\n", "\n     ")[:500]))


SRC = {"src/ok.c": "int main(void) { return 0; }\n"}

check("a clean tree passes", SRC, "", True)

check("a screenshot, unexplained, fails", {**SRC, "docs/home.png": PNG}, "", False, ["docs/home.png"])
check("a screenshot, explained, passes", {**SRC, "docs/home.png": PNG}, "open docs/home.png my own drawing\n", True)

check("an unexplained Mach-O fails", {**SRC, "bin/stray": MACHO}, "", False, ["Mach-O"])
check("a Mach-O built from the tree passes", {**SRC, "bin/stray": MACHO}, "open bin/* built from src/ok.c\n", True)

for kind, data in (("an IPSW", b"PK\x03\x04" + b"\0" * 20), ("a font", b"\0\1\0\0" + b"\0" * 20)):
    name = "x.ipsw" if kind == "an IPSW" else "x.ttf"
    check("%s fails" % kind, {**SRC, name: data}, "", False, [name])

check("a path through a cryptex fails", {**SRC, "sysimage/cryptex/readme.txt": "hi\n"}, "", False, ["cryptex"])

flash_sh = {"tools/flash-it.sh": "#!/bin/sh\nfastboot flash boot x.img\n"}
check("a script that flashes fails", {**SRC, **flash_sh}, "", False, ["writes to the phone"])
check("a script that flashes, declared, passes with a warning", {**SRC, **flash_sh}, "flash tools/flash-it.sh a one-time job\n", True, ["declared exception"])
check("a flash command in a comment is not a flash command", {**SRC, "tools/note.sh": "# never run: fastboot flash boot x.img\necho hi\n"}, "", True)
check("erase and oem unlock are caught too", {**SRC, "tools/a.sh": "fastboot erase userdata\n", "tools/b.sh": "fastboot oem unlock\n"}, "", False, ["tools/a.sh", "tools/b.sh"])

loader = {"xnuloader/loader.c": "void f(void) { bio->write_blocks(bio, 0, 0, 1, b); }\n"}
check("the loader writing blocks, undeclared, fails", {**SRC, **loader}, "", False, ["xnuloader/loader.c"])
check("the loader writing blocks, declared, passes", {**SRC, **loader}, "flash xnuloader/loader.c its log to logdump\n", True, ["xnuloader/loader.c"])
check("a C file elsewhere may use the name", {**SRC, "src/other.c": "int write_blocks(int x);\n"}, "", True)

check("firmware compiled into a header fails", {**SRC, "src/fw.h": FIRMWARE}, "", False, ["src/fw.h"])
check("... explained as pending: passes, with a warning", {**SRC, "src/fw.h": FIRMWARE}, "pending src/fw.h waiting for the vendor\n", True, ["pending"])

vendor_tracked = {**SRC, "vendor/driver.efi": b"MZ\x90\0" + b"\0" * 32}
check("a vendor file tracked fails", vendor_tracked, "vendor vendor/* the vendor's\n", False, ["vendor (proprietary)"])
check("a vendor line with the files gone: quiet, no stale warning", SRC, "vendor vendor/* the vendor's\n", True)
check("an open line that matches nothing is reported as stale", SRC, "open nothing/here/* gone\n", True, ["stale"])

check("a bad status in the allowlist is an error", SRC, "maybe docs/x something\n", False, ["provenance.allow"])
check("the report lists a declared writer", {**SRC, **flash_sh}, "flash tools/flash-it.sh a one-time job\n", True)

if failures:
    sys.exit("provenance-test: %d FAILED" % len(failures))
print("provenance-test: all passed")
