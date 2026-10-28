What `xl` needs of Linux on a Mac: `flock`, GNU `timeout` and `stat` (MacPorts' gtimeout and gstat),
`sha256sum` with GNU's `--quiet` (for tools/mk-mesa.sh; it is `shasum -a 256` underneath), and Android's `fastboot` and `adb` (platform-tools, which `xl` finds here or in $PATH). Put the
platform-tools directory next to this file as `platform-tools/` (not in the repository).
r34 platform-tools (adb 34.0.5): the newer 37.0.1 adb server is killed within seconds on this Mac
