/*
 * HakiOS: HFS Plus built into the kernel (bsd/hfs, from Apple's hfs-627.100.6) rather than a kext, for the data
 * volume on the phone's flash.  FIFO, CONFIG_PROTECT and NAMEDSTREAMS are the kernel's own options (config/MASTER);
 * what is HFS's alone is decided here.
 */
#ifndef _hfs_config_
#define _hfs_config_
#define HFS_COMPRESSION 1       /* (the code does not build without it; decmpfs is the kernel's) */
#ifndef QUOTA
#define QUOTA 0
#endif
#define CONFIG_SECLUDED_RENAME 0
/* content protection's keys are AppleKeyStore's, which this kernel does not have (bsd/hfs/hfs_builtin.c) */
#undef CONFIG_PROTECT
#define CONFIG_PROTECT 0
#endif /* defined(_hfs_config_) */
