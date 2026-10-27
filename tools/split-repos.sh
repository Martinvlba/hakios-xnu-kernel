#!/bin/bash
# Make the HakiOS repositories out of this one, the way UniXNU keeps its parts apart: the common repository, a SoC's, a device's,
# the UEFI. Each one keeps the history of its own files (git subtree split). This repository is not changed: the results are new
# clones in OUT, ready to push.
#
#   tools/split-repos.sh OUT [OWNER]        OWNER: where they will live on GitHub (default shimoku74)
#
#   OUT/hakios                  common: everything but the parts below; devices/manifest.conf names those by repository
#   OUT/hakios-sc7280-common    socs/sc7280
#   OUT/hakios-redwood          devices/redwood
#   OUT/hakios-uefi             uefi
#
# Commit messages lose their "Co-Authored-By: Claude ..." lines (the project's contributors are its people); authors and dates
# are kept as they are. Pushing is tools/publish-repos.sh's, with your own GitHub login.
set -e
SRC=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
OUT=${1:?usage: tools/split-repos.sh OUT [OWNER]}
OWNER=${2:-shimoku74}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[ -z "$(git -C "$SRC" status --porcelain)" ] || { echo "split-repos: commit first: the parts are made from what is committed" >&2; exit 1; }

PARTS="socs/sc7280:hakios-sc7280-common:soc:sc7280 devices/redwood:hakios-redwood:device:redwood uefi:hakios-uefi:uefi:hakios-uefi"

# the messages without Claude's trailer lines (and no blank line left hanging at the end)
clean_messages() {
	git -C "$1" -c advice.detachedHead=false filter-branch -f --msg-filter \
		"sed -E '/^[Cc]o-[Aa]uthored-[Bb]y: Claude/d' | sed -e :a -e '/^\n*\$/{\$d;N;ba' -e '}'" -- --all >/dev/null 2>&1
	git -C "$1" update-ref -d refs/original/refs/heads/main 2>/dev/null || true
	for ref in $(git -C "$1" for-each-ref --format='%(refname)' refs/original); do git -C "$1" update-ref -d "$ref"; done
	git -C "$1" reflog expire --expire=now --all
	git -C "$1" gc -q --prune=now
}

for part in $PARTS; do
	IFS=: read -r prefix name _kind _id <<<"$part"
	[ -e "$OUT/$name" ] && { echo "split-repos: $OUT/$name exists" >&2; exit 1; }
	echo "split-repos: $prefix -> $name"
	git -C "$SRC" subtree split -q --prefix="$prefix" -b "split/$name" >/dev/null
	git clone -q --no-local --branch "split/$name" "$SRC" "$OUT/$name"
	git -C "$SRC" branch -q -D "split/$name"
	git -C "$OUT/$name" branch -q -m main
	git -C "$OUT/$name" remote remove origin
	clean_messages "$OUT/$name"
done

echo "split-repos: the common repository -> hakios"
[ -e "$OUT/hakios" ] && { echo "split-repos: $OUT/hakios exists" >&2; exit 1; }
git clone -q --no-local "$SRC" "$OUT/hakios"
C=$OUT/hakios
git -C "$C" checkout -q -B main
git -C "$C" remote remove origin
for b in $(git -C "$C" for-each-ref --format='%(refname:short)' refs/heads); do [ "$b" = main ] || git -C "$C" branch -q -D "$b"; done
clean_messages "$C"
for part in $PARTS; do
	IFS=: read -r prefix name kind id <<<"$part"
	git -C "$C" rm -q -r "$prefix"
	rev=$(git -C "$OUT/$name" rev-parse HEAD)
	python3 - "$C/devices/manifest.conf" "$kind" "$id" "https://github.com/$OWNER/$name.git" "$rev" <<'PY'
import re, sys
path, kind, name, url, rev = sys.argv[1:]
text = open(path).read()
block = re.compile(r"^%s %s\n(?:\t.*\n?)*" % (kind, re.escape(name)), re.M)
new = "%s %s\n\trepository  %s\n\trevision    %s\n" % (kind, name, url, rev)
text, n = block.subn(new, text)
assert n == 1, (kind, name)
open(path, "w").write(text)
PY
done
# what sync clones into the tree is not the common repository's
python3 - "$C/.gitignore" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
text = text.replace("!/uefi/\n", "")
text += ("\n# The SoC, device and UEFI repositories that tools/hakios-device.py sync clones into place (devices/manifest.conf) are their own:\n"
         "/socs/*\n!/socs/README.md\n/devices/*\n!/devices/manifest.conf\n!/devices/README.md\n!/devices/qemu/\n")
open(path, "w").write(text)
PY
cat > "$C/socs/README.md" <<'EOF'
# SoC repositories

`tools/hakios-device.py sync` clones each SoC repository `devices/manifest.conf` names into `socs/<name>/`. A new one:
[How the repositories fit](../docs/porting/repositories.md).
EOF
cat > "$C/devices/README.md" <<'EOF'
# Device repositories

`devices/manifest.conf` names every device; `tools/hakios-device.py sync` clones each into `devices/<name>/`. `devices/qemu/` is
kept here: QEMU's machine, with none of a phone's drivers, the device every checkout of the common repository can build and run.
A device of your own: `tools/hakios-device.py new` ([Port a device](../docs/porting/index.md)).
EOF
git -C "$C" add -A
git -C "$C" commit -q -m "The SoC, device and UEFI repositories are their own: hakios-sc7280-common, hakios-redwood, hakios-uefi

devices/manifest.conf names them by repository and revision; tools/hakios-device.py sync clones them into socs/, devices/ and uefi/."
echo "split-repos: done in $OUT"
for r in hakios $(for p in $PARTS; do echo "$p" | cut -d: -f2; done); do
	printf '  %-22s %5s commits, %s\n' "$r" "$(git -C "$OUT/$r" rev-list --count HEAD)" "$(git -C "$OUT/$r" log -1 --format=%h)"
done
echo "split-repos: next: tools/publish-repos.sh $OUT"
