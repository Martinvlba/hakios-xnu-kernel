#!/bin/bash
# HFS Plus for the kernel (the data volume, bsd/hfs): Apple's sources are not kept here, as XNU's are not. They are
# fetched from Apple's own release, hfs-627.100.6, and what this port changes in them is xnu/hfs/hfs.patch; the
# files of its own (bsd/hfs/hfs_builtin.c) are in xnu/files like every other new file.
#
#   xnu/hfs/hfs.sh put TREE      Apple's files that core.list and encodings.list name into TREE's bsd/hfs and
#                                bsd/hfs_encodings, and hfs.patch over them (xnu/apply.sh does this)
#   xnu/hfs/hfs.sh export TREE   hfs.patch again from TREE (xnu/export.sh does this)
#
# env: HFS_URL  where to fetch from (default Apple's repository on GitHub; a clone that has the tag will do)
#      HFS_SRC  where the release is kept once fetched (default xnu-src/hfs-627.100.6 beside the kernel tree)
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=hfs-627.100.6
BASE=31d5ec238dfab485e78a9a30b8b92d13e8914f03
URL=${HFS_URL:-https://github.com/apple-oss-distributions/hfs.git}
SRC=${HFS_SRC:-$HERE/../../xnu-src/$TAG}
CMD=$1 TREE=$2
[ -n "$CMD" ] && [ -d "$TREE" ] || { echo "usage: xnu/hfs/hfs.sh put|export TREE" >&2; exit 1; }

if [ ! -d "$SRC/core" ]; then
	rm -rf "$SRC"
	git init -q "$SRC"
	git -C "$SRC" remote add origin "$URL"
	git -C "$SRC" fetch -q --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
	git -C "$SRC" checkout -q "$TAG"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$BASE" ] || { echo "hfs.sh: $SRC is not $TAG ($BASE)" >&2; exit 1; }

case "$CMD" in
put)
	mkdir -p "$TREE/bsd/hfs" "$TREE/bsd/hfs_encodings"
	while read -r f; do cp "$SRC/core/$f" "$TREE/bsd/hfs/$f"; done < "$HERE/core.list"
	while read -r f; do cp "$SRC/hfs_encodings/$f" "$TREE/bsd/hfs_encodings/$f"; done < "$HERE/encodings.list"
	git -C "$TREE" apply --whitespace=nowarn "$HERE/hfs.patch"
	echo "hfs.sh: $TAG: $(wc -l < "$HERE/core.list") + $(wc -l < "$HERE/encodings.list") files, $(grep -c '^+++ ' "$HERE/hfs.patch") changed"
	;;
export)
	: > "$HERE/hfs.patch.new"
	while read -r f; do
		diff -u --label "a/bsd/hfs/$f" --label "b/bsd/hfs/$f" "$SRC/core/$f" "$TREE/bsd/hfs/$f" >> "$HERE/hfs.patch.new" || true
	done < "$HERE/core.list"
	while read -r f; do
		diff -u --label "a/bsd/hfs_encodings/$f" --label "b/bsd/hfs_encodings/$f" "$SRC/hfs_encodings/$f" \
		    "$TREE/bsd/hfs_encodings/$f" >> "$HERE/hfs.patch.new" || true
	done < "$HERE/encodings.list"
	mv "$HERE/hfs.patch.new" "$HERE/hfs.patch"
	;;
*)
	echo "usage: xnu/hfs/hfs.sh put|export TREE" >&2; exit 1 ;;
esac
