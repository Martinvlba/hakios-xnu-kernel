#!/bin/bash
# Save the redwood platform from Mu-Silicium/ into files/ here (uefi/export.sh runs this; HakiOS's own changes are uefi/'s).
set -e
cd "$(dirname "$(readlink -f "$0")")"
MU=${MU:-../../../Mu-Silicium}
rm -rf files
mkdir -p files/Resources/Configs files/Resources/DTBs files/Platforms/Xiaomi files/Binaries
cp -r $MU/Platforms/Xiaomi/redwoodPkg files/Platforms/Xiaomi/
cp $MU/Resources/Configs/redwood.toml files/Resources/Configs/
cp $MU/Resources/DTBs/redwood.dtb files/Resources/DTBs/
cp -r $MU/Binaries/redwood files/Binaries/
find files -name __pycache__ -prune -exec rm -rf {} +
echo "devices/redwood/uefi/export.sh: $(find files -type f | wc -l) files"
