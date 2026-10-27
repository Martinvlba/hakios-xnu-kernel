#!/bin/bash
# The redwood platform on HakiOS's UEFI: uefi/apply.sh (Mu-Silicium with HakiOS's changes and banner), with this phone's
# platform from files/ here. See README.md here for the vendor's drivers and the build.
exec "$(dirname "$(readlink -f "$0")")/../../../uefi/apply.sh" redwood
