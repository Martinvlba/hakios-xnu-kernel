#!/usr/bin/env python3
"""Check that drivers/map and the directories under drivers/ agree.

Rules (each one is a way the layout went wrong while it was being made):
  - every line of drivers/map is "<subsystem> <path in the XNU tree>", the subsystem is a known one, the path is
    in bsd/dev/ or iokit/Kernel/ and begins with "hakios.";
  - no file is listed twice, and no two files share a name (they would sit in one directory under it);
  - every listed file exists as drivers/<subsystem>/<name>;
  - every file under drivers/<subsystem>/ is listed (README.md excepted): an unlisted file would not reach the kernel;
  - xnu/files/ holds no driver any more (a copy there would be applied over the one in drivers/ or the other way round).
usage: tools/drivers-check.py [--root DIR]      exit 0 if all is well, 1 and the list of problems if not
"""
import os
import sys

SUBSYSTEMS = {"input", "storage", "gpu", "camera", "haptic", "audio", "power", "wifi", "modem", "net", "core"}
TREE_DIRS = ("bsd/dev/", "iokit/Kernel/")


def read_map(root):
    """[(subsystem, tree path)] from drivers/map, and the problems met on the way."""
    entries, problems = [], []
    path = os.path.join(root, "drivers", "map")
    if not os.path.isfile(path):
        return entries, ["drivers/map does not exist"]
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2:
                problems.append("drivers/map:%d: want '<subsystem> <path>', got %r" % (number, text))
                continue
            sub, tree = parts
            if sub not in SUBSYSTEMS:
                problems.append("drivers/map:%d: unknown subsystem %r (known: %s)" % (number, sub, " ".join(sorted(SUBSYSTEMS))))
            if not tree.startswith(TREE_DIRS) or not os.path.basename(tree).startswith("hakios."):
                problems.append("drivers/map:%d: %r is not a hakios.* file of bsd/dev/ or iokit/Kernel/" % (number, tree))
            entries.append((sub, tree))
    return entries, problems


def check(root):
    entries, problems = read_map(root)
    seen_tree, seen_name = {}, {}
    for sub, tree in entries:
        if tree in seen_tree:
            problems.append("%s is listed twice" % tree)
        seen_tree[tree] = sub
        name = os.path.basename(tree)
        if name in seen_name and seen_name[name] != tree:
            problems.append("%s and %s have the same name" % (seen_name[name], tree))
        seen_name[name] = tree
        if not os.path.isfile(os.path.join(root, "drivers", sub, name)):
            problems.append("drivers/%s/%s is listed but is not there" % (sub, name))
    listed = {(sub, os.path.basename(tree)) for sub, tree in entries}
    drivers = os.path.join(root, "drivers")
    for sub in sorted(os.listdir(drivers)) if os.path.isdir(drivers) else []:
        sub_dir = os.path.join(drivers, sub)
        if not os.path.isdir(sub_dir):
            continue
        if sub not in SUBSYSTEMS:
            problems.append("drivers/%s is not a known subsystem directory" % sub)
        for name in sorted(os.listdir(sub_dir)):
            if name != "README.md" and (sub, name) not in listed:
                problems.append("drivers/%s/%s is not in drivers/map, so it would never reach the kernel" % (sub, name))
    for tree in seen_tree:
        stale = os.path.join(root, "xnu", "files", tree)
        if os.path.exists(stale):
            problems.append("xnu/files/%s: a driver must not also be kept here" % tree)
    return entries, problems


def main(argv):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if len(argv) == 3 and argv[1] == "--root":
        root = argv[2]
    entries, problems = check(os.path.abspath(root))
    if problems:
        for p in problems:
            print("drivers-check: " + p)
        print("drivers-check: %d problem(s)" % len(problems))
        return 1
    print("drivers-check: %d drivers in %d subsystems, all listed, all present" % (
        len(entries), len({s for s, _ in entries})))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
