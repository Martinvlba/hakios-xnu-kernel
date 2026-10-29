/*
 * HakiOS: what HFS's kext had in its C++ part (hfs_iokit.cpp), for HFS built into the kernel.
 *
 * The kext registered the filesystem from its IOService's start(); built in, it is an entry of the static table
 * (bsd/vfs/vfs_conf.c) and hfs_init() does the rest.  The IOKit questions -- is the medium ejectable, which media
 * carry a given content UUID, the platform's serial number -- have no use for a data volume on the phone's own
 * flash, and the keys of content protection are AppleKeyStore's, which this kernel does not have: a volume made
 * without content protection never asks for them.
 */
#include <sys/types.h>
#include <sys/errno.h>
#include <stdbool.h>
#include <mach/kern_return.h>
#include "hfs.h"
#include "hfs_iokit.h"

#ifndef panic_on_assert
bool panic_on_assert = false;
#endif

int
hfs_is_ejectable(__unused const char *cdev_name)
{
	return 0;
}

void
hfs_iterate_media_with_content(__unused const char *content_uuid_cstring,
    __unused int (*func)(const char *bsd_name, const char *uuid_str, void *arg), __unused void *arg)
{
}

kern_return_t
hfs_get_platform_serial_number(__unused char *serial_number_str, __unused uint32_t len)
{
	return KERN_FAILURE;
}

#if CONFIG_PROTECT
int
hfs_unwrap_key(__unused aks_cred_t access, __unused const aks_wrapped_key_t wrapped_key_in,
    __unused aks_raw_key_t key_out)
{
	return ENOTSUP;
}

int
hfs_rewrap_key(__unused aks_cred_t access, __unused cp_key_class_t dp_class,
    __unused const aks_wrapped_key_t wrapped_key_in, __unused aks_wrapped_key_t wrapped_key_out)
{
	return ENOTSUP;
}

int
hfs_new_key(__unused aks_cred_t access, __unused cp_key_class_t dp_class, __unused aks_raw_key_t key_out,
    __unused aks_wrapped_key_t wrapped_key_out)
{
	return ENOTSUP;
}

int
hfs_backup_key(__unused aks_cred_t access, __unused const aks_wrapped_key_t wrapped_key_in,
    __unused aks_wrapped_key_t wrapped_key_out)
{
	return ENOTSUP;
}
#endif /* CONFIG_PROTECT */
