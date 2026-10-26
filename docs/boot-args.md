# Boot-arguments

Made by `tools/boot-args.py --markdown`; CI fails if it is out of date with the source (`tools/boot-args.py --check`).

Every argument that the kernel, the loader or user space of this port reads, and which file reads it. **314 arguments**: 265 read by the kernel, 14 by the loader, 44 by user space (one is read in more than one place).

This is a catalogue of what exists and who reads it, not a manual. What an argument means is in the head of the file that reads it. The stable switches have a symbol in the configuration (`tools/hakios-config.py help SYMBOL`, `docs/configs.md`); the rest are the knobs of bring-up (a driver's steps, pauses, traces, test modes) and can change with the driver. Any of them can be given in a configuration with `CONFIG_BOOTARGS="name=value ..."`, which checks that each name is in this list.

Layers: **K** the kernel, **L** the loader, **U** user space (through `sysctl kern.bootargs`).

## GPU and display (67)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `adreno` | K | `hakios.adreno.c` |  |  |
| `adreno_blind` | K | `hakios.adreno.c` |  |  |
| `adreno_dirty` | K | `hakios.adreno.c` |  |  |
| `adreno_fine` | K | `hakios.adreno.c` |  |  |
| `adreno_keep` | K | `hakios.adreno.c` |  |  |
| `adreno_only` | K | `hakios.adreno.c` |  |  |
| `adreno_pause` | K | `hakios.adreno.c` |  |  |
| `adreno_return` | K | `hakios.adreno.c` |  |  |
| `adreno_skip` | K | `hakios.adreno.c` |  |  |
| `adreno_speed` | K | `hakios.adreno.c` |  |  |
| `adreno_stream1` | K | `hakios.adreno.c` |  |  |
| `adreno_watch` | K | `hakios.adreno.c` |  |  |
| `display_probe` | K | `hakios.adreno.c` |  |  |
| `fb_beat` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fb_diff` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fb_filter` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fb_hz` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fb_queue` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fb_threads` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fbheight` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fbscale` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fbtrace` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `fbwidth` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `glblit` | U | `eagl.c` |  |  |
| `glcensus` | U | `eagl.c` |  |  |
| `glcopy` | U | `eagl.c` |  |  |
| `glenv` | U | `eagl.c` |  |  |
| `glfetch` | U | `eagl.c` |  |  |
| `glframes` | U | `eagl.c` |  |  |
| `glgroup` | U | `eagl.c` |  |  |
| `glnumbers` | U | `eagl.c` |  |  |
| `glplain` | U | `eagl.c` |  |  |
| `glprecision` | U | `eagl.c` |  |  |
| `glshaders` | U | `eagl.c` |  |  |
| `glshow` | U | `eagl.c` |  |  |
| `glshow_part` | U | `eagl.c` |  |  |
| `glshow_zoom` | U | `eagl.c` |  |  |
| `glskip` | U | `eagl.c` |  |  |
| `glslow` | U | `eagl.c` |  |  |
| `glsync` | U | `eagl.c` |  |  |
| `gltrace` | U | `eagl.c` |  |  |
| `gltrace_at` | U | `eagl.c` |  |  |
| `gltrace_corners` | U | `eagl.c` |  |  |
| `gltrace_draws` | U | `eagl.c` |  |  |
| `gltrace_frames` | U | `eagl.c` |  |  |
| `gltrace_function` | U | `eagl.c` |  |  |
| `gltrace_lit` | U | `eagl.c` |  |  |
| `gltrace_look` | U | `eagl.c` |  |  |
| `gltrace_place` | U | `eagl.c` |  |  |
| `gltrace_shaders` | U | `eagl.c` |  |  |
| `gltrace_uniforms` | U | `eagl.c` |  |  |
| `glwait` | U | `eagl.c` |  |  |
| `gpu` | K | `hakios.adreno.c` | `CONFIG_GPU` |  |
| `gpu_cached` | K | `hakios.gpu.cpp` |  |  |
| `gpu_exact` | K | `hakios.adreno.c` |  |  |
| `gpu_keep` | K | `hakios.gpu.cpp` |  |  |
| `gpu_log` | K | `hakios.adreno.c` |  |  |
| `rec_mb` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `rec_ring` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `rec_secs` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `rec_start` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `rec_step` | K | `MiniIOMobileFramebuffer.cpp` |  |  |
| `surf_clear` | K | `MiniIOSurface.cpp` |  |  |
| `surftrace` | K | `MiniIOSurface.cpp` |  |  |
| `surftrace_of` | K | `MiniIOSurface.cpp` |  |  |
| `touch_lat` | KU | `MiniIOMobileFramebuffer.cpp`, `eagl.c`, `hakios.goodix.c`, `hakios.gpu.cpp`, `hakios.input.c`, `inject.c`, `inject_sb.c` |  |  |
| `vblur` | U | `eagl.c` |  |  |

## Security stand-ins (1)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `privs` | K | `kern_minimac.c` |  |  |

## Wi-Fi (31)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `wifi` | K | `hakios.wifi.c` | `CONFIG_WIFI` |  |
| `wifi_again` | K | `hakios.wifi.c` |  |  |
| `wifi_ask` | K | `hakios.wifi.c` |  |  |
| `wifi_ba` | K | `lx_80211.c` |  | implied by wifi=1 |
| `wifi_cal` | K | `hakios.wlfw.c` |  |  |
| `wifi_dirty` | K | `hakios.wifi.c` |  |  |
| `wifi_file` | K | `hakios.wifi.c` |  |  |
| `wifi_fine` | K | `hakios.wifi.c` |  |  |
| `wifi_hold` | K | `lx_bus.c` |  |  |
| `wifi_irq` | K | `lx_bus.c` |  | implied by wifi=1 |
| `wifi_join` | K | `lx_bus.c` |  |  |
| `wifi_keep` | K | `hakios.wifi.c` |  | implied by wifi=1 |
| `wifi_listen` | K | `hakios.wifi.c` |  |  |
| `wifi_loud` | K | `hakios.wifi.c` |  |  |
| `wifi_lx` | K | `hakios.wifi.c`, `lx_bus.c` |  | implied by wifi=1 (as 2) |
| `wifi_lx_step` | K | `lx_base.c` |  |  |
| `wifi_mac` | K | `hakios.wifi.c` |  |  |
| `wifi_mission` | K | `hakios.wlfw.c` |  |  |
| `wifi_new` | K | `hakios.wifi.c` |  |  |
| `wifi_pause` | K | `hakios.wifi.c` |  |  |
| `wifi_quiet` | K | `lx_bus.c` |  |  |
| `wifi_radio_read` | K | `hakios.wifi.c` |  |  |
| `wifi_ready` | K | `hakios.wlfw.c` |  |  |
| `wifi_return` | K | `hakios.wifi.c` |  |  |
| `wifi_scan` | K | `lx_bus.c` |  |  |
| `wifi_smmu` | K | `hakios.wifi.c` |  | implied by wifi=1 (as 2) |
| `wifi_station` | K | `lx_bus.c` |  | implied by wifi=1 |
| `wifi_step` | K | `hakios.wifi.c` |  |  |
| `wifi_wait` | K | `hakios.wifi.c` |  |  |
| `wifi_watch` | K | `hakios.wifi.c` |  |  |
| `wifi_wide` | K | `lx_80211.c` |  | implied by wifi=1 |

## Cellular (53)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `modem` | K | `hakios.mpss.c` | `CONFIG_MODEM` |  |
| `modem_again` | K | `hakios.mpss.c` | `CONFIG_MODEM_AGAIN` |  |
| `modem_apn` | K | `hakios.mpss.c` |  |  |
| `modem_buffer` | K | `hakios.mpss.c` | `CONFIG_MODEM_BUFFER` |  |
| `modem_data_nobind` | K | `hakios.mpss.c` |  |  |
| `modem_data_off` | K | `hakios.mpss.c` |  |  |
| `modem_data_wait` | K | `hakios.mpss.c` |  |  |
| `modem_dpm_off` | K | `hakios.mpss.c` |  |  |
| `modem_ipa` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_bus_off` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_data_off` | K | `hakios.ipa.c` |  |  |
| `modem_ipa_data_trace` | K | `hakios.cell.c`, `hakios.ipa.c` |  |  |
| `modem_ipa_fw` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_fw_at` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_gsi_enable` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_item` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_no_probe` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_quiet` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_raw` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_routes` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_skip_uc` | K | `hakios.mpss.c` |  |  |
| `modem_ipa_smmu_off` | K | `hakios.mpss.c` |  |  |
| `modem_keep` | K | `hakios.mpss.c` |  |  |
| `modem_listen` | K | `hakios.mpss.c` |  |  |
| `modem_nas_wait` | K | `hakios.mpss.c` |  |  |
| `modem_new` | K | `hakios.mpss.c` |  |  |
| `modem_no_ipa_qmi` | K | `hakios.mpss.c` |  |  |
| `modem_no_nav` | K | `hakios.mpss.c` |  |  |
| `modem_no_tftp` | K | `hakios.mpss.c` |  |  |
| `modem_online` | K | `hakios.mpss.c` |  |  |
| `modem_pause` | K | `hakios.mpss.c` |  |  |
| `modem_return` | K | `hakios.mpss.c` |  |  |
| `modem_rmtfs_delay` | K | `hakios.mpss.c` |  |  |
| `modem_rmtfs_rw_fake` | K | `hakios.mpss.c` |  |  |
| `modem_rmtfs_rw_off` | K | `hakios.mpss.c` |  |  |
| `modem_share_off` | K | `hakios.mpss.c` |  |  |
| `modem_share_test` | K | `hakios.mpss.c` |  |  |
| `modem_sim_probe_off` | K | `hakios.mpss.c` |  |  |
| `modem_sim_switch_off` | K | `hakios.mpss.c` |  |  |
| `modem_step` | K | `hakios.mpss.c` |  |  |
| `modem_wait` | K | `hakios.mpss.c` |  |  |
| `modem_wda_off` | K | `hakios.mpss.c` |  |  |
| `soc_cmd_db` | K | `hakios.mpss.c` |  |  |
| `soc_gsi_base` | K | `hakios.ipa.c` |  |  |
| `soc_imem` | K | `hakios.mpss.c` |  |  |
| `soc_ipa_base` | K | `hakios.ipa.c` |  |  |
| `soc_ipa_fw` | K | `hakios.mpss.c` |  |  |
| `soc_ipa_stream` | K | `hakios.mpss.c` |  |  |
| `soc_ipa_stream_uc` | K | `hakios.mpss.c` |  |  |
| `soc_mpss_mem` | K | `hakios.mpss.c` |  |  |
| `soc_pas_ipa` | K | `hakios.mpss.c` |  |  |
| `soc_pas_mpss` | K | `hakios.mpss.c` |  |  |
| `soc_smem` | K | `hakios.mpss.c`, `hakios.socinfo.c` |  |  |

## Audio and battery (31)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `adsp` | K | `hakios.adsp.c` | `CONFIG_ADSP` |  |
| `adsp_again` | K | `hakios.adsp.c` |  |  |
| `adsp_apr_wait` | K | `hakios.adsp.c` |  |  |
| `adsp_keep` | K | `hakios.adsp.c` |  |  |
| `adsp_listen` | K | `hakios.adsp.c` |  |  |
| `adsp_new` | K | `hakios.adsp.c` |  |  |
| `adsp_pause` | K | `hakios.adsp.c` |  |  |
| `adsp_return` | K | `hakios.adsp.c` |  |  |
| `adsp_step` | K | `hakios.adsp.c` |  |  |
| `adsp_wait` | K | `hakios.adsp.c` |  |  |
| `battery` | KL | `hakios.adsp.c`, `hakios.battery.c`, `loader.c` |  |  |
| `battery_shared_pa` | K | `hakios.battery.c` |  | the loader (the address of its charger page) |
| `battmgr` | K | `hakios.adsp.c` |  |  |
| `mic` | K | `hakios.mic.c` | `CONFIG_MIC` |  |
| `mic_adc` | K | `hakios.mic.c` |  |  |
| `mic_input` | K | `hakios.mic.c` |  |  |
| `mic_main` | K | `hakios.mic.c` |  |  |
| `mic_secs` | K | `hakios.mic.c` |  |  |
| `mic_tap` | K | `hakios.mic.c` |  |  |
| `mic_test` | K | `hakios.mic.c` |  |  |
| `rec_sound` | K | `hakios.audio.c` |  |  |
| `rec_sound_fed` | K | `hakios.audio.c` |  |  |
| `rec_sound_mic` | K | `hakios.audio.c` |  |  |
| `sound` | K | `hakios.sound.c` | `CONFIG_SOUND` |  |
| `sound_below` | K | `hakios.sound.c` |  |  |
| `sound_gain` | K | `hakios.sound.c` |  |  |
| `sound_guard_as` | K | `hakios.sound.c` |  |  |
| `sound_pause` | K | `hakios.sound.c` |  |  |
| `sound_secs` | K | `hakios.sound.c` |  |  |
| `speaker_probe` | K | `hakios.speaker.c` |  |  |
| `virtio_sound` | K | `hakios.virtio_sound.c` |  |  |

## Cameras (13)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `cam` | KL | `hakios.camera.c`, `loader.c` | `CONFIG_CAM` |  |
| `cam_after` | K | `hakios.camera.c` |  |  |
| `cam_fake` | K | `hakios.camera.c` |  |  |
| `cam_fake_fps` | K | `hakios.camera.c` |  |  |
| `cam_frames` | K | `hakios.camera.c` |  |  |
| `cam_gain` | K | `hakios.camera.c` |  |  |
| `cam_keep_mb` | K | `hakios.camera.c` |  |  |
| `cam_mode` | K | `hakios.camera.c` |  |  |
| `cam_pattern` | K | `hakios.camera.c` |  |  |
| `cam_sensor` | K | `hakios.camera.c` |  |  |
| `cam_settle` | K | `hakios.camera.c` |  |  |
| `cam_shutter` | K | `hakios.camera.c` |  |  |
| `cam_vblank` | K | `hakios.camera.c` |  |  |

## Touch and buttons (8)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `buttons` | K | `hakios.buttons.c` |  |  |
| `buttons_steps` | K | `hakios.buttons.c` |  |  |
| `touch_flip` | K | `hakios.goodix.c` |  |  |
| `touch_poll_us` | K | `hakios.goodix.c` |  |  |
| `touch_rate` | K | `hakios.goodix.c` |  |  |
| `touch_spi_hz` | K | `hakios.goodix.c` |  |  |
| `touchscreen` | K | `hakios.buttons.c`, `hakios.goodix.c` | `CONFIG_TOUCHSCREEN` |  |
| `virtio_input` | K | `hakios.virtio_input.c` |  |  |

## Storage and APFS (43)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `apfs_cheap` | K | `apfs_vfsops.c` |  |  |
| `apfs_check` | K | `hakios.apfs_check.c` |  |  |
| `apfs_chunk` | K | `hakios.apfs_check.c` |  |  |
| `apfs_contents` | K | `hakios.apfs_check.c` |  |  |
| `apfs_depth` | K | `hakios.apfs_check.c` |  |  |
| `apfs_dev` | K | `hakios.apfs_check.c` |  |  |
| `apfs_hash` | K | `hakios.apfs_check.c` |  |  |
| `apfs_max` | K | `hakios.apfs_check.c` |  |  |
| `apfs_meta_cluster` | K | `apfs_vfsops.c` |  |  |
| `apfs_mnt` | K | `hakios.apfs_check.c` |  |  |
| `apfs_par` | K | `apfs_vfsops.c` |  |  |
| `apfs_root_c` | K | `apfs_vfsops.c` |  |  |
| `apfs_root_v` | K | `apfs_vfsops.c` |  |  |
| `apfs_round` | K | `apfs_vfsops.c` |  |  |
| `apfs_skip` | K | `hakios.apfs_check.c` |  |  |
| `apfs_sparse` | K | `hakios.sparse.c` |  |  |
| `apfs_store_len` | K | `hakios.apfs_check.c` |  |  |
| `apfs_store_note` | K | `hakios.apfs_check.c` |  |  |
| `apfs_store_seed` | K | `hakios.apfs_check.c` |  |  |
| `apfs_subtree` | K | `hakios.apfs_check.c` |  |  |
| `apfs_threads` | K | `hakios.apfs_check.c` |  |  |
| `apfs_ubc_read` | K | `apfs_vfsops.c` |  |  |
| `apfs_volume` | K | `hakios.apfs_check.c` |  |  |
| `apfs_volume_at` | K | `hakios.apfs_check.c` |  |  |
| `apfs_walkers` | K | `hakios.apfs_check.c` |  |  |
| `apfs_write_at` | K | `hakios.apfs_check.c` |  |  |
| `apfs_write_byte` | K | `hakios.apfs_check.c` |  |  |
| `apfs_write_len` | K | `hakios.apfs_check.c` |  |  |
| `apfs_write_path` | K | `hakios.apfs_check.c` |  |  |
| `ufs_bench` | K | `hakios.ufs.c` |  |  |
| `ufs_check` | K | `hakios.ufs.c` |  |  |
| `ufs_native` | KL | `hakios.ufs.c`, `loader.c` |  |  |
| `ufs_native_bench` | K | `hakios.ufs.c` |  |  |
| `ufs_native_buffer` | K | `hakios.ufshc.c` |  |  |
| `ufs_native_check` | K | `hakios.ufs.c` |  |  |
| `ufs_native_hang` | K | `hakios.ufshc.c` |  |  |
| `ufs_native_leave` | K | `hakios.ufshc.c` |  |  |
| `ufs_native_slots` | K | `hakios.ufshc.c` |  |  |
| `ufs_native_spin` | K | `hakios.ufshc.c` |  |  |
| `ufs_shared_pa` | K | `hakios.ufs.c` |  | the loader (the address of its block-service page) |
| `ufs_write` | KL | `hakios.ufs.c`, `loader.c` | `CONFIG_UFS_WRITE` |  |
| `var_save_ms` | K | `hakios.ufs.c` |  |  |
| `virtio_blk` | K | `hakios.virtio_blk.c` |  |  |

## Network (3)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `net_names` | K | `hakios.net.c` | `CONFIG_NET_NAMES` |  |
| `net_trace` | K | `hakios.net.c` | `CONFIG_NET_TRACE` |  |
| `virtio_net` | K | `hakios.virtio_net.c` | `CONFIG_VIRTIO_NET` |  |

## Haptics and torch (5)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `haptic` | K | `hakios.haptic.c` | `CONFIG_HAPTIC` |  |
| `haptic_effect` | K | `hakios.haptic.c` |  |  |
| `torch` | KL | `hakios.torch.c`, `loader.c` |  |  |
| `torch_max_s` | K | `hakios.torch.c` |  |  |
| `torch_secs` | K | `hakios.torch.c` |  |  |

## Chip, thermal and PMIC (10)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `pmic` | K | `hakios.pmic.c` |  |  |
| `pmic_list` | K | `hakios.pmic.c` |  |  |
| `soc_smem_size` | K | `hakios.socinfo.c` |  |  |
| `socinfo` | K | `hakios.socinfo.c` |  |  |
| `thermal` | K | `hakios.thermal.c` |  |  |
| `thermal_log` | K | `hakios.thermal.c` |  |  |
| `thermal_poll` | K | `hakios.thermal.c` |  |  |
| `thermal_stop` | K | `hakios.thermal.c` |  |  |
| `thermal_tm0` | K | `hakios.thermal.c` |  |  |
| `thermal_tm1` | K | `hakios.thermal.c` |  |  |

## Version (1)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `hakios_kver` | K | `hakios.version.c` |  | the loader (from the payload's kernel.ver) |

## CPU, memory and boot (23)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `applog` | K | `(patch)` |  |  |
| `applog_name` | K | `(patch)` |  |  |
| `bootcpu` | KL | `loader.c`, `loader_return.c` | `CONFIG_BOOTCPU` |  |
| `cpufreq` | K | `hakios.cpufreq.c` | `CONFIG_CPUFREQ` |  |
| `cpuload` | K | `hakios.cpufreq.c` |  |  |
| `ddr` | K | `hakios.cpufreq.c` |  |  |
| `ddr_rest` | K | `hakios.cpufreq.c` |  |  |
| `ddr_vote` | K | `hakios.cpufreq.c` |  |  |
| `epoch` | K | `GenericARMPE.cpp` |  | xl (the host's clock) |
| `hv_wdt_pet` | K | `loader_return.c` |  | the loader, when it has armed the watchdog (the runner also passes it) |
| `loader_return_after` | KL | `loader.c`, `loader_return.c` |  | xl (XL_SECS) |
| `machine_name` | K | `GenericARMPE.cpp` |  |  |
| `membench` | K | `hakios.cpufreq.c` |  |  |
| `rd` | KL | `(patch)`, `loader.c` |  |  |
| `regtrace` | K | `(patch)` |  |  |
| `return_cpu` | K | `loader_return.c` |  |  |
| `return_panic` | K | `loader_return.c` |  |  |
| `small_cores` | K | `(patch)` |  |  |
| `small_cores_after` | K | `GenericARMPE.cpp` |  |  |
| `syssleep` | K | `(patch)` |  |  |
| `systrace` | K | `(patch)`, `user_exception_trace.c` |  |  |
| `systrace_name` | K | `(patch)` |  |  |
| `xpcproxy_idversion` | K | `(patch)` |  |  |

## Debugging (6)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `ustacks` | K | `user_exception_trace.c` |  |  |
| `ustacks_at` | K | `user_exception_trace.c` |  |  |
| `ustacks_count` | K | `user_exception_trace.c` |  |  |
| `ustacks_ms` | K | `user_exception_trace.c` |  |  |
| `ustacks_threads` | K | `user_exception_trace.c` |  |  |
| `ustacks_tname` | K | `user_exception_trace.c` |  |  |

## Loader (6)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `cpumask` | L | `loader.c` |  |  |
| `cpus` | L | `loader.c` | `CONFIG_CPUS` |  |
| `gamut` | L | `loader.c` |  |  |
| `serial` | L | `loader.c` |  |  |
| `ufs` | L | `loader.c` | `CONFIG_UFS` |  |
| `ufs_probe` | L | `loader.c` |  |  |

## User space (13)

| Argument | Layer | Read by | Configuration | Set by |
|---|---|---|---|---|
| `autounlock` | U | `inject_sb.c` |  |  |
| `ca_vbl` | U | `inject.c` |  |  |
| `caenv` | U | `inject.c` |  |  |
| `cc_fade` | U | `inject_sb.c` |  |  |
| `gl` | U | `inject.c`, `inject_sb.c` |  |  |
| `glcompile` | U | `inject.c` |  |  |
| `glno` | U | `inject.c` |  |  |
| `glsb3` | U | `inject.c` |  |  |
| `glyes` | U | `inject.c` |  |  |
| `root_apfs` | U | `init.c` |  |  |
| `sb_delay` | U | `inject_sb.c` |  | xl (XL_SB_DELAY) |
| `switcher_wallpaper` | U | `inject_sb.c` |  |  |
| `var_store` | U | `init.c` | `CONFIG_VAR_STORE` |  |
