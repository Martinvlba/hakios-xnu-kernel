/*
 * healthcheck (TODO 86 / 87): a one-shot device health report, for a first-boot self-test and for
 * diagnosing the open items. Reads sysctls, the /private/var filesystem type (persistence), the key
 * IOKit services the port brings up, and the battery; prints a report and a PASS/FAIL summary
 * (exit 0 = all ok, 1 = something failed). No arguments. Safe: reads only, changes nothing.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/sysctl.h>
#include <unistd.h>

static int failures, checks;

static void
say(bool ok, const char *what, const char *detail)
{
	checks++;
	failures += !ok;
	printf("[%s] %-26s %s\n", ok ? " ok " : "FAIL", what, detail ? detail : "");
	fflush(stdout);
}

static bool
sysctl_str(const char *name, char *buf, size_t len)
{
	size_t n = len;
	buf[0] = '\0';
	return sysctlbyname(name, buf, &n, NULL, 0) == 0;
}

static long
sysctl_num(const char *name)
{
	uint64_t v = 0;
	size_t n = sizeof(v);
	return sysctlbyname(name, &v, &n, NULL, 0) == 0 ? (long)v : -1;
}

static bool
service_present(const char *cls)
{
	io_service_t s = IOServiceGetMatchingService(0, IOServiceMatching(cls));
	if (s) {
		IOObjectRelease(s);
		return true;
	}
	return false;
}

int
main(void)
{
	char buf[1024];

	setvbuf(stdout, NULL, _IONBF, 0);
	printf("== hakios healthcheck (pid %d) ==\n", getpid());

	/* identity */
	if (sysctl_str("kern.version", buf, sizeof(buf))) {
		char *nl = strchr(buf, '\n');
		if (nl) {
			*nl = '\0';
		}
		printf("kernel : %s\n", buf);
	}
	if (sysctl_str("kern.bootargs", buf, sizeof(buf))) {
		printf("bootarg: %s\n", buf);
	}
	printf("cpus   : %ld   memory: %ld MB\n", sysctl_num("hw.ncpu"), sysctl_num("hw.memsize") / (1024 * 1024));

	/* persistence: /private/var must be HFS+ on the flash, not a memory filesystem wiped each boot */
	{
		struct statfs sfs;
		bool got = (statfs("/private/var", &sfs) == 0);
		bool persistent = got && strncmp(sfs.f_fstypename, "hfs", 3) == 0;
		say(persistent, "/private/var persistent", got ? sfs.f_fstypename : "statfs failed");
	}

	/* the key drivers the port brings up (those that register a matchable IOKit service) */
	say(service_present("HakiosGPU"), "GPU driver", NULL);
	say(service_present("HakiosUSBMux"), "usbmux driver", NULL);
	say(service_present("IOPMPowerSource"), "fuel gauge", NULL);

	/* battery level, from the fuel gauge's registry entry */
	{
		io_service_t bat = IOServiceGetMatchingService(0, IOServiceMatching("IOPMPowerSource"));
		if (bat) {
			CFMutableDictionaryRef props = NULL;
			if (IORegistryEntryCreateCFProperties(bat, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
				int cur = 0, max = 0;
				bool ext = false;
				CFNumberRef c = CFDictionaryGetValue(props, CFSTR("CurrentCapacity"));
				CFNumberRef m = CFDictionaryGetValue(props, CFSTR("MaxCapacity"));
				CFBooleanRef e = CFDictionaryGetValue(props, CFSTR("ExternalConnected"));
				if (c) {
					CFNumberGetValue(c, kCFNumberIntType, &cur);
				}
				if (m) {
					CFNumberGetValue(m, kCFNumberIntType, &max);
				}
				if (e) {
					ext = CFBooleanGetValue(e);
				}
				printf("battery: %d%%%s\n", max ? cur * 100 / max : cur, ext ? " (charging)" : "");
				CFRelease(props);
			}
			IOObjectRelease(bat);
		}
	}

	printf("== %d checks, %d failed ==\n", checks, failures);
	return failures ? 1 : 0;
}
