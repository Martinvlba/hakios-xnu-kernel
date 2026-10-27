#!/bin/bash
# Push the repositories tools/split-repos.sh made (and the website's) to GitHub, each as a PRIVATE repository of OWNER (the organization oosexgang by default).
#
#   tools/publish-repos.sh DIR [OWNER]          DIR: where split-repos.sh put them; OWNER: default oosexgang (the organization)
#
# It signs in as nobody: it uses YOUR login. With the GitHub CLI (`gh auth login` once), it makes each repository and pushes it;
# without it, make each one empty and private on github.com first (New repository, the same name, nothing in it), and git asks
# your credential manager when it pushes. Never put a token in this file, in a command line or in a chat.
set -e
DIR=${1:?usage: tools/publish-repos.sh DIR [OWNER]}
OWNER=${2:-oosexgang}
REPOS="hakios hakios-sc7280-common hakios-redwood hakios-uefi hakios-web"
for r in $REPOS; do
	[ -d "$DIR/$r/.git" ] || { echo "publish-repos: no $DIR/$r (tools/split-repos.sh makes the first four; the website is hakios-web)"; continue; }
	url="https://github.com/$OWNER/$r.git"
	if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
		gh repo view "$OWNER/$r" >/dev/null 2>&1 || gh repo create "$OWNER/$r" --private --description "$(head -1 "$DIR/$r/README.md" | sed 's/^# *//')"
	fi
	git -C "$DIR/$r" remote remove origin 2>/dev/null || true
	git -C "$DIR/$r" remote add origin "$url"
	git -C "$DIR/$r" push -u origin main
	echo "publish-repos: $r -> $url"
done
