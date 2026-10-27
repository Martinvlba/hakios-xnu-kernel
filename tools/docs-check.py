#!/usr/bin/env python3
"""Every link of the documentation goes somewhere: docs/**/*.md and the README files, checked without anything to install.

  - a relative link names a file or directory of the repository that is there;
  - a link to a heading (page.md#heading, #heading) names a heading that page has (the anchor as GitHub and the website make it:
    lower case, spaces to dashes, punctuation dropped);
  - links inside code blocks are not links.

usage: tools/docs-check.py      exit 1 and the list if a link goes nowhere (CI runs it)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"^(```|~~~)")


def slug(heading):
    text = re.sub(r"`([^`]*)`", r"\1", heading.strip().lower())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def prose(path):
    """The lines of a page outside its code blocks."""
    out, fenced = [], False
    for line in open(path, encoding="utf-8"):
        if FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return out


def anchors(path, cache={}):
    if path not in cache:
        found = set()
        for line in prose(path):
            m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
            if m:
                base, n = slug(m.group(1)), 1
                name = base
                while name in found:
                    name, n = "%s-%d" % (base, n), n + 1
                found.add(name)
        cache[path] = found
    return cache[path]


def pages():
    out = [os.path.join(ROOT, n) for n in sorted(os.listdir(ROOT)) if n.startswith("README") and n.endswith(".md")]
    for d, _dirs, files in os.walk(os.path.join(ROOT, "docs")):
        out += [os.path.join(d, f) for f in sorted(files) if f.endswith(".md")]
    return out


def main():
    problems, links = [], 0
    for page in pages():
        shown = os.path.relpath(page, ROOT)
        for n, line in enumerate(prose(page), 1):
            for target in LINK.findall(line):
                if re.match(r"[a-z]+:", target):
                    continue
                links += 1
                path, _, anchor = target.partition("#")
                full = os.path.normpath(os.path.join(os.path.dirname(page), path)) if path else page
                if not os.path.exists(full):
                    problems.append("%s: a link to %s, which is not there" % (shown, target))
                elif anchor and full.endswith(".md") and anchor not in anchors(full):
                    problems.append("%s: a link to %s, and %s has no heading #%s" % (shown, target, os.path.relpath(full, ROOT), anchor))
    for p in problems:
        print("docs-check: " + p)
    print("docs-check: %d pages, %d links%s" % (len(pages()), links, ", %d go nowhere" % len(problems) if problems else ", every one goes somewhere"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
