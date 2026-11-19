/*
 * wifipw_test: what wifid does with a network's password when Settings joins it, done here the same
 * way: MobileWiFi's WiFiSecuritySetPassword() into the keychain, WiFiSecurityCopyPassword() back
 * (what Auto-Join asks before it joins a known network on its own), WiFiSecurityRemovePassword()
 * (what "Forget This Network" does). The network is a made-up one ("hakios-wifipw-test") with a
 * made-up password, neither printed in full. Signed with wifipw_test.entitlements: wifid's name
 * and its right to the system keychain, as the keychain answers a program by those.
 * MobileWiFi's own os_log lines ("__WiFiSecuritySetPassword:[...] Error result -25291") say the
 * keychain's answer; they are printed on the console as well (OS_ACTIVITY_DT_MODE).
 */
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SSID	CFSTR("hakios-wifipw-test")
#define SECRET	CFSTR("not-a-real-password")

static CFStringRef (*copy_password)(CFStringRef ssid);

static void
read_back(const char *when)
{
	CFStringRef pw = copy_password(SSID);

	printf("wifipw_test: %s: WiFiSecurityCopyPassword: %s\n", when,
	    pw == NULL ? "nothing" : CFEqual(pw, SECRET) ? "THE SAME PASSWORD" : "another password");
	if (pw != NULL)
		CFRelease(pw);
}

int
main(int argc, char **argv)
{
	void *mw;
	Boolean (*set_password)(CFStringRef ssid, CFStringRef password);
	Boolean (*remove_password)(CFStringRef ssid);
	CFMutableDictionaryRef q;
	CFTypeRef attrs = NULL;
	OSStatus status;

	(void)argc;
	if (getenv("OS_ACTIVITY_DT_MODE") == NULL) {
		/* os_log decides where it prints when the process starts */
		setenv("OS_ACTIVITY_DT_MODE", "YES", 1);
		execv(argv[0], argv);
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	mw = dlopen("/System/Library/PrivateFrameworks/MobileWiFi.framework/MobileWiFi", RTLD_NOW);
	set_password = mw ? dlsym(mw, "WiFiSecuritySetPassword") : NULL;
	copy_password = mw ? dlsym(mw, "WiFiSecurityCopyPassword") : NULL;
	remove_password = mw ? dlsym(mw, "WiFiSecurityRemovePassword") : NULL;
	if (set_password == NULL || copy_password == NULL || remove_password == NULL) {
		printf("wifipw_test: MobileWiFi or its functions not found: %s\n", dlerror());
		return 1;
	}
	if (argc > 1 && strcmp(argv[1], "keybag") == 0) {
		/* wifipw_test keybag: what the keybag says of the lock (what Auto-Join waits for before it reads a password) */
		void *mkb = dlopen("/System/Library/PrivateFrameworks/MobileKeyBag.framework/MobileKeyBag", RTLD_NOW);
		int (*since_boot)(void) = mkb ? dlsym(mkb, "MKBDeviceUnlockedSinceBoot") : NULL;
		int (*lock_state)(CFDictionaryRef) = mkb ? dlsym(mkb, "MKBGetDeviceLockState") : NULL;
		int (*formatted)(void) = mkb ? dlsym(mkb, "MKBDeviceFormattedForContentProtection") : NULL;

		printf("wifipw_test: keybag: MKBDeviceUnlockedSinceBoot %d, MKBGetDeviceLockState %d (0 unlocked, 1 locked, 2 locking, 3 no "
		    "passcode), formatted for content protection %d\n", since_boot ? since_boot() : -99, lock_state ? lock_state(NULL) : -99,
		    formatted ? formatted() : -99);
		return 0;
	}
	if (argc > 2 && strcmp(argv[1], "plist") == 0) {
		/* wifipw_test plist PATH: a property list file said as text (the known networks' records: no password is in them) */
		CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)argv[2], (CFIndex)strlen(argv[2]), false);
		CFReadStreamRef stream = url ? CFReadStreamCreateWithFile(NULL, url) : NULL;
		CFPropertyListRef list = NULL;

		if (stream != NULL && CFReadStreamOpen(stream)) {
			list = CFPropertyListCreateWithStream(NULL, stream, 0, kCFPropertyListImmutable, NULL, NULL);
		}
		if (list == NULL) {
			printf("wifipw_test: plist: %s could not be read\n", argv[2]);
		} else {
			CFStringRef text = CFCopyDescription(list);
			CFIndex n = CFStringGetMaximumSizeForEncoding(CFStringGetLength(text), kCFStringEncodingUTF8) + 1;
			char *c = malloc((size_t)n), *line, *save = NULL;

			CFStringGetCString(text, c, n, kCFStringEncodingUTF8);
			for (line = strtok_r(c, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save)) {
				printf("wifipw_test: plist: %.300s\n", line);
				usleep(1500);
			}
		}
		return 0;
	}
	if (argc > 1 && strcmp(argv[1], "keep") == 0) {
		/* wifipw_test keep: the made-up network's password kept and LEFT (then `check hakios-wifipw-test` after a restart) */
		printf("wifipw_test: keep: WiFiSecuritySetPassword: %s\n", set_password(SSID, SECRET) ? "yes" : "NO");
		read_back("keep");
		return 0;
	}
	if (argc > 2 && strcmp(argv[1], "check") == 0) {
		/* wifipw_test check NAME: is a password kept for that network (what Auto-Join asks)? Its length only. */
		CFStringRef name = CFStringCreateWithCString(NULL, argv[2], kCFStringEncodingUTF8);
		CFStringRef pw = copy_password(name);

		printf("wifipw_test: check: WiFiSecurityCopyPassword for the network given: %s", pw == NULL ? "NOTHING KEPT\n" : "a password of ");
		if (pw != NULL) {
			printf("%ld characters\n", (long)CFStringGetLength(pw));
			CFRelease(pw);
		}
		q = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
		CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
		CFDictionarySetValue(q, kSecAttrService, CFSTR("AirPort"));
		CFDictionarySetValue(q, kSecAttrSynchronizable, kSecAttrSynchronizableAny);
		CFDictionarySetValue(q, kSecMatchLimit, kSecMatchLimitAll);
		CFDictionarySetValue(q, kSecReturnAttributes, kCFBooleanTrue);
		status = SecItemCopyMatching(q, &attrs);
		printf("wifipw_test: check: every item of service AirPort: status %d, %ld item(s)\n", (int)status,
		    attrs != NULL && CFGetTypeID(attrs) == CFArrayGetTypeID() ? (long)CFArrayGetCount(attrs) : 0L);
		return 0;
	}
	read_back("before");
	printf("wifipw_test: WiFiSecuritySetPassword: %s\n", set_password(SSID, SECRET) ? "yes" : "NO");
	read_back("after");

	/* the item itself, as the keychain has it: wifid's service name, the network's name for account */
	q = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
	CFDictionarySetValue(q, kSecAttrService, CFSTR("AirPort"));
	CFDictionarySetValue(q, kSecAttrAccount, SSID);
	CFDictionarySetValue(q, kSecAttrSynchronizable, kSecAttrSynchronizableAny);
	CFDictionarySetValue(q, kSecReturnAttributes, kCFBooleanTrue);
	status = SecItemCopyMatching(q, &attrs);
	printf("wifipw_test: the item (service AirPort): %d\n", (int)status);
	if (attrs != NULL && CFGetTypeID(attrs) == CFDictionaryGetTypeID()) {
		CFTypeRef group = CFDictionaryGetValue(attrs, kSecAttrAccessGroup);
		CFTypeRef cls = CFDictionaryGetValue(attrs, kSecAttrAccessible);
		CFTypeRef sync = CFDictionaryGetValue(attrs, kSecAttrSynchronizable);
		char g[64] = "?", c[16] = "?";

		if (group != NULL && CFGetTypeID(group) == CFStringGetTypeID())
			CFStringGetCString(group, g, sizeof(g), kCFStringEncodingUTF8);
		if (cls != NULL && CFGetTypeID(cls) == CFStringGetTypeID())
			CFStringGetCString(cls, c, sizeof(c), kCFStringEncodingUTF8);
		printf("wifipw_test: its access group %s, protection class %s, synchronizable %s\n", g, c,
		    sync == NULL ? "?" : CFEqual(sync, kCFBooleanTrue) ? "yes" : "no");
	}
	if (attrs != NULL)
		CFRelease(attrs);
	CFRelease(q);

	printf("wifipw_test: WiFiSecurityRemovePassword: %s\n", remove_password(SSID) ? "yes" : "NO");
	read_back("after removing");
	return 0;
}
