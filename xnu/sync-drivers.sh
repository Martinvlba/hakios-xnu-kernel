#!/bin/bash
# Copy the drivers that were edited in drivers/ into the kernel tree, so that `make kernel` builds what is in
# the repository. A driver is copied only when the file in drivers/ is NEWER than the one in the tree: a driver
# that was edited in the tree (and not yet exported with xnu/export.sh) is never overwritten by an old copy.
# usage: xnu/sync-drivers.sh [KERNEL-TREE]       (default: $XNU_SRC, else xnu-src/xnu-8796.121.2)
# prints one line per file it copied, and "drivers: in step" when there was nothing to do.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${1:-${XNU_SRC:-$HERE/../xnu-src/xnu-8796.121.2}}
[ -d "$TREE" ] || { echo "sync-drivers: no kernel tree at $TREE (xnu/apply.sh makes it)" >&2; exit 1; }
[ -f "$HERE/../drivers/map" ] || { echo "sync-drivers: no drivers/map" >&2; exit 1; }
copied=0
while read -r sub tree _; do
	case "$sub" in ''|'#'*) continue ;; esac
	src="$HERE/../drivers/$sub/$(basename "$tree")"
	[ -f "$src" ] || { echo "sync-drivers: drivers/$sub/$(basename "$tree") is in drivers/map but missing" >&2; exit 1; }
	if [ ! -e "$TREE/$tree" ] || { [ "$src" -nt "$TREE/$tree" ] && ! cmp -s "$src" "$TREE/$tree"; }; then
		mkdir -p "$(dirname "$TREE/$tree")"
		cp -p "$src" "$TREE/$tree"
		echo "sync-drivers: drivers/$sub/$(basename "$tree") -> $tree"
		copied=$((copied + 1))
	fi
done < "$HERE/../drivers/map"
# ... and the drivers of the SoC and device repositories checked out (socs/*/drivers/, devices/*/drivers/: docs/porting/repositories.md)
while IFS="$(printf '\t')" read -r src tree; do
	[ -n "$src" ] || continue
	[ -f "$src" ] || { echo "sync-drivers: $src is named by its repository's conf but missing" >&2; exit 1; }
	if [ ! -e "$TREE/$tree" ] || { [ "$src" -nt "$TREE/$tree" ] && ! cmp -s "$src" "$TREE/$tree"; }; then
		mkdir -p "$(dirname "$TREE/$tree")"
		cp -p "$src" "$TREE/$tree"
		echo "sync-drivers: ${src#$HERE/../} -> $tree"
		copied=$((copied + 1))
	fi
done <<EOF2
$(python3 "$HERE/../tools/hakios-config.py" external-files)
EOF2
[ "$copied" -gt 0 ] || echo "sync-drivers: drivers in step with the kernel tree"
