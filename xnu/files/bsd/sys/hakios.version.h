/*
 * The version of this kernel (drivers/core/hakios.version.c): sysctl kern.hakios.version and a line of the boot log.
 */
#ifndef _SYS_HAKIOS_VERSION_H_
#define _SYS_HAKIOS_VERSION_H_

#ifdef KERNEL
/* Read the boot-arg hakios_kver, keep it as kern.hakios.version and say it in the log. Called from bsd_init(). */
void hakios_version_init(void);
#endif /* KERNEL */

#endif /* _SYS_HAKIOS_VERSION_H_ */
