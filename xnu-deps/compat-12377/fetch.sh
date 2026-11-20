#!/bin/bash
# The Apple-private headers xnu-12377 includes but no longer ships, that are published in an older XNU:
# TrustCache/ (EXTERNAL_HEADERS of xnu-8796.121.2, APSL). Fetched, not kept in git, like the rest of xnu-deps.
# The other headers here (CoreEntitlements/V2, CodeSignature) are HakiOS's own stand-ins.
set -e
cd "$(dirname "$0")/include"
[ -f TrustCache/API.h ] && exit 0
mkdir -p TrustCache
for f in API.h RawTypes.h Return.h Types.h TypesConfig.h; do
	curl -sfL -o TrustCache/$f https://raw.githubusercontent.com/apple-oss-distributions/xnu/xnu-8796.121.2/EXTERNAL_HEADERS/TrustCache/$f
done
echo "compat-12377: TrustCache headers fetched from xnu-8796.121.2"
