/*
 * libhakios.installd.dylib: code of ours inside installd (build.sh's INJECT), so that a PC can
 * install applications over the USB cable without a valid Apple code signature
 * (devices/akita/tools/usbmux-install.py).
 *
 * installd asks libmis (MISValidateSignatureAndCopyInfo) to verify the code signature of every
 * application it installs.  An application sideloaded as an .ipa without an Apple-issued
 * signature -- or without any signature at all -- fails with "ApplicationVerificationFailed
 * ... No code signature found".  The kernel here does not enforce code signatures
 * (kern_minimac.c accepts every ad-hoc signed binary as a platform binary, and unsigned
 * binaries are let through as well), but installd's own check is in user space, before the
 * binary is ever run.
 *
 * This library interposes MISValidateSignatureAndCopyInfo: every call returns success (0) with
 * an empty validation-info dictionary.  Nothing else of installd's is touched.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern int MISValidateSignatureAndCopyInfo(CFStringRef path, CFDictionaryRef options,
    CFDictionaryRef *info);

static void
say(const char *line, size_t length)
{
	int fd = open("/dev/console", O_WRONLY);

	if (fd >= 0) {
		(void)write(fd, line, length);
		close(fd);
	}
}

static int
hakios_MISValidateSignatureAndCopyInfo(CFStringRef path,
    __unused CFDictionaryRef options, CFDictionaryRef *info)
{
	char line[512], buf[256];
	const char *p = "(?)";
	int n;

	if (path != NULL && CFStringGetCString(path, buf, sizeof(buf), kCFStringEncodingUTF8)) {
		p = buf;
	}
	n = snprintf(line, sizeof(line),
	    "hakios: installd: MISValidateSignatureAndCopyInfo(%s): answered as valid\n", p);
	if (n > 0 && n < (int)sizeof(line)) {
		say(line, (size_t)n);
	}
	if (info != NULL) {
		*info = CFDictionaryCreate(kCFAllocatorDefault, NULL, NULL, 0,
		    &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	}
	return 0;
}

__attribute__((used)) static struct {
	const void *ours, *theirs;
} mis_interpose __attribute__((section("__DATA,__interpose"))) = {
	(const void *)hakios_MISValidateSignatureAndCopyInfo,
	(const void *)MISValidateSignatureAndCopyInfo
};
