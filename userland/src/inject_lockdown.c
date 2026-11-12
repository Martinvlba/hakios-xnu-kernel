/*
 * libhakios.lockdown.dylib: code of ours inside lockdownd (build.sh's INJECT), so that a PC can install applications
 * over the USB cable (devices/akita/tools/usbmux-install.py, ideviceinstaller, Xcode's way).
 *
 * lockdownd starts a service for a paired host only if its own table says AllowUnactivatedService -- AFC, the
 * notification proxy, the log -- or the phone is ACTIVATED. This one is not and cannot be (the activation chain wants
 * a Secure Enclave: mkjobs.py), so com.apple.mobile.installation_proxy, house_arrest, streaming_zip_conduit and
 * misagent were all "InvalidService" (the tester's phone, 11-13). lockdownd asks MobileActivation
 * (MAEGetActivationStateWithError) and compares the answer with kMAActivationStateActivated; here it is given that
 * answer, as SpringBoard is given it in inject_sb.c. Nothing else of lockdownd's is touched.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern CFStringRef MAEGetActivationStateWithError(CFErrorRef *error);
extern const CFStringRef kMAActivationStateActivated;

static CFStringRef
activated(CFErrorRef *error)
{
	static int said;

	if (error != NULL) {
		*error = NULL;
	}
	if (!said) {
		static const char line[] = "hakios: lockdownd: the activation state it asked for is answered \"Activated\" "
		    "(its services for a paired host: the installation proxy)\n";
		int fd = open("/dev/console", O_WRONLY);

		said = 1;
		if (fd >= 0) {
			(void)write(fd, line, sizeof(line) - 1);
			close(fd);
		}
	}
	return kMAActivationStateActivated;
}

__attribute__((used)) static struct {
	const void *ours, *theirs;
} activation_interpose __attribute__((section("__DATA,__interpose"))) = {
	(const void *)activated, (const void *)MAEGetActivationStateWithError
};
