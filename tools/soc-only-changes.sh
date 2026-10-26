#!/bin/bash
# soc_only_changes FILE: the lines of a soc.args that DIFFER from the kernel's built-in defaults.
#
# Sourced by tools/modem-cycle.sh and tools/modem-cycle-win.sh, which add the soc_* arguments of a port to
# the kernel's command line. The kernel carries the reference phone's values as defaults
# (devices/poco-x5-pro/soc.args holds them): an argument equal to its default changes nothing, and the
# command line has room for 607 characters, of which sixteen soc_* values take about 340. A run with all of
# them lost the end of its command line without a word (the loader now says so). A port that differs from the
# reference in two values sends two.
#
# CRs are dropped (a file checked out on Windows may have them); lines that are blank or comments are dropped.
soc_only_changes() {
	local file=$1 ref=${SOC_DEFAULTS:-devices/poco-x5-pro/soc.args}
	[ -f "$file" ] || return 0
	if [ -f "$ref" ]; then
		tr -d '\r' < "$file" | grep -vE '^[[:space:]]*(#|$)' | grep -vxF -f <(tr -d '\r' < "$ref") || true
	else
		tr -d '\r' < "$file" | grep -vE '^[[:space:]]*(#|$)' || true
	fi
}
