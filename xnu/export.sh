#!/bin/bash
# Save the kernel work as it is in the working tree, in the two forms apply.sh puts back:
#   hakios.patch   the changes to Apple's files (against the tag the work sits on)
#   files/          the files that are not in Apple's tree, at their paths
# usage: xnu/export.sh [KERNEL-TREE]     (default: $XNU_SRC, else xnu-src/xnu-8796.121.2)
# Run it before committing whenever the kernel tree has changed. Committed or not makes no
# difference there: everything is compared with the base commit.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${1:-${XNU_SRC:-$HERE/../xnu-src/xnu-8796.121.2}}
BASE=c54f35ca767986246321eb901baf8f5ff7923f6a           # tag xnu-8796.121.2
# never part of the work: build output and what editors, patch and python leave behind
# nor what tools/hakios-config.py kernel writes for one configuration (the drivers' choice)
JUNK='^BUILD/|^config/MASTER\.hakios$|^bsd/dev/hakios\.drivers\.h$|(^|/)__pycache__/|\.(orig|rej|pyc|o|swp)$|\.bak[0-9]*$|~$|(^|/)\.DS_Store$'

cd "$SRC"
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "export.sh: $SRC is not a clone of xnu at $BASE" >&2; exit 1; }

# no renames: a moved file is a deletion in the patch and a new file in files/
git -c core.abbrev=12 diff --binary --no-renames --no-color --no-ext-diff --diff-filter=MDT "$BASE" -- . > "$HERE/hakios.patch.new"
mv "$HERE/hakios.patch.new" "$HERE/hakios.patch"

rm -rf "$HERE/files.new"
mkdir "$HERE/files.new"
{ git diff --no-renames --name-only --diff-filter=A "$BASE" -- .; git ls-files --others --exclude-standard; } |
	sort -u | grep -Ev "$JUNK" > "$HERE/files.list"
while IFS= read -r f; do
	[ -f "$f" ] || [ -L "$f" ] || continue      # added in a commit, deleted since
	mkdir -p "$HERE/files.new/$(dirname "$f")"
	if [ "$(uname)" = Darwin ]; then cp -pP "$f" "$HERE/files.new/$f"; else cp -P --preserve=mode "$f" "$HERE/files.new/$f"; fi
done < "$HERE/files.list"

# The drivers of SoC and device repositories (tools/hakios-config.py external-files) are theirs, not this repository's: they are
# left out of files/ and are not refused as unlisted. Edit them in their repository; xnu/sync-drivers.sh copies them in.
EXTERNAL=$(python3 "$HERE/../tools/hakios-config.py" external-files | cut -f2)
if [ -n "$EXTERNAL" ]; then
	printf '%s\n' "$EXTERNAL" | while IFS= read -r tree; do rm -f "$HERE/files.new/$tree"; done
fi

# Apple's HFS Plus is not kept here: what this port changes in it goes into xnu/hfs/hfs.patch, and Apple's files are
# fetched again by xnu/apply.sh (xnu/hfs/hfs.sh). Its own files (bsd/hfs/hfs_builtin.c) stay in files/.
while read -r f; do rm -f "$HERE/files.new/bsd/hfs/$f"; done < "$HERE/hfs/core.list"
while read -r f; do rm -f "$HERE/files.new/bsd/hfs_encodings/$f"; done < "$HERE/hfs/encodings.list"
"$HERE/hfs/hfs.sh" export "$SRC"

# The drivers (drivers/map) are kept by subsystem in drivers/<subsystem>/, not under files/: take them out of
# files.new and put each one where the map says. A hakios.* file the map does not list is refused, not guessed at.
MAP="$HERE/../drivers/map"
[ -f "$MAP" ] || { echo "export.sh: no $MAP" >&2; exit 1; }
LISTED=$(awk '{ sub(/#.*/, "") } NF == 2 { print $2 }' "$MAP")
UNLISTED=$(cd "$HERE/files.new" && find bsd/dev iokit/Kernel -type f -name 'hakios.*' 2>/dev/null | sort |
	grep -vxF -f <(printf '%s\n' "$LISTED") || true)
if [ -n "$UNLISTED" ]; then
	echo "export.sh: these new drivers are not in drivers/map (add '<subsystem> <path>' for each, then run this again):" >&2
	echo "$UNLISTED" | sed 's/^/    /' >&2
	rm -rf "$HERE/files.new"; exit 1
fi
NDRIVERS=0
while read -r sub tree _; do
	case "$sub" in ''|'#'*) continue ;; esac
	if [ ! -f "$HERE/files.new/$tree" ]; then
		echo "export.sh: drivers/map lists $tree but the kernel tree has no such new file (deleted? take its line out of drivers/map)" >&2
		rm -rf "$HERE/files.new"; exit 1
	fi
	mkdir -p "$HERE/../drivers/$sub"
	mv "$HERE/files.new/$tree" "$HERE/../drivers/$sub/$(basename "$tree")"
	NDRIVERS=$((NDRIVERS + 1))
done < "$MAP"
find "$HERE/files.new" -type d -empty -delete 2>/dev/null || true
mkdir -p "$HERE/files.new"
rm -rf "$HERE/files" "$HERE/files.list"
mv "$HERE/files.new" "$HERE/files"

echo "xnu/hakios.patch: $(grep -c '^diff --git' "$HERE/hakios.patch") files changed, $(wc -c < "$HERE/hakios.patch") bytes"
echo "xnu/files: $(find "$HERE/files" -type f | wc -l) new files, $(du -sk "$HERE/files" | cut -f1) KB"
echo "drivers/: $NDRIVERS drivers"
