/*
 * First user-space program for the Darwin-on-redwood bring-up.
 *
 * A static arm64 Mach-O with no libc and no dyld: it talks to the kernel with
 * raw system calls. The kernel's mockfs presents the ramdisk (this very file)
 * as /sbin/launchd, so this runs as process 1.
 */
typedef unsigned long u64;

/* XNU arm64 system call: number in x16, arguments in x0.., "svc #0x80"; carry set on error */
static long
syscall3(long num, long a, long b, long c)
{
	register long x16 __asm__("x16") = num;
	register long x0 __asm__("x0") = a;
	register long x1 __asm__("x1") = b;
	register long x2 __asm__("x2") = c;
	__asm__ volatile ("svc #0x80" : "+r" (x0) : "r" (x16), "r" (x1), "r" (x2) : "memory", "cc");
	return x0;
}

/* the same with six arguments, for sysctlbyname(name, namelen, old, &oldlen, new, newlen) */
static long
syscall6(long num, long a, long b, long c, long d, long e, long f)
{
	register long x16 __asm__("x16") = num;
	register long x0 __asm__("x0") = a;
	register long x1 __asm__("x1") = b;
	register long x2 __asm__("x2") = c;
	register long x3 __asm__("x3") = d;
	register long x4 __asm__("x4") = e;
	register long x5 __asm__("x5") = f;
	__asm__ volatile ("svc #0x80" : "+r" (x0) : "r" (x16), "r" (x1), "r" (x2), "r" (x3), "r" (x4), "r" (x5) : "memory", "cc");
	return x0;
}

#define SYS_exit    1
#define SYS_write   4
#define SYS_open    5
#define SYS_getpid  20
#define SYS_getuid  24
#define SYS_select  93
#define SYS_sysctlbyname 274
#define O_RDWR      2

static unsigned
length(const char *s)
{
	unsigned n = 0;
	while (s[n]) {
		n++;
	}
	return n;
}

static void
put(int fd, const char *s)
{
	syscall3(SYS_write, fd, (long)s, length(s));
}

static void
put_number(int fd, long v)
{
	char buf[24];
	int i = sizeof(buf);
	buf[--i] = 0;
	do {
		buf[--i] = (char)('0' + v % 10);
		v /= 10;
	} while (v);
	put(fd, buf + i);
}

/* "name=value" for a string sysctl, or nothing at all if the kernel has no such name (the driver is off) */
static void
show_string(int fd, const char *name)
{
	char value[1100];
	u64 size = sizeof(value);

	if (syscall6(SYS_sysctlbyname, (long)name, length(name), (long)value, (long)&size, 0, 0) == 0) {
		put(fd, name);
		put(fd, "=");
		put(fd, value);
		put(fd, "\n");
	}
}

/* ... for an int sysctl (four bytes) */
static void
show_int(int fd, const char *name)
{
	int value = 0;
	u64 size = sizeof(value);

	if (syscall6(SYS_sysctlbyname, (long)name, length(name), (long)&value, (long)&size, 0, 0) == 0) {
		put(fd, name);
		put(fd, "=");
		if (value < 0) {
			put(fd, "-");
			value = -value;
		}
		put_number(fd, value);
		put(fd, "\n");
	}
}

/* hello_wait=N in the kernel's command line (kern.bootargs): N, or 0 */
static long
wait_seconds(void)
{
	char args[700];
	u64 size = sizeof(args);
	long n = 0;

	if (syscall6(SYS_sysctlbyname, (long)"kern.bootargs", 13, (long)args, (long)&size, 0, 0) != 0) {
		return 0;
	}
	for (char *c = args; *c; c++) {
		if (c[0] == 'h' && c[1] == 'e' && c[2] == 'l' && c[3] == 'l' && c[4] == 'o' && c[5] == '_' && c[6] == 'w' &&
		    c[7] == 'a' && c[8] == 'i' && c[9] == 't' && c[10] == '=') {
			for (c += 11; *c >= '0' && *c <= '9'; c++) {
				n = n * 10 + (*c - '0');
			}
			return n;
		}
	}
	return 0;
}

void
start(void)
{
	int fd = (int)syscall3(SYS_open, (long)"/dev/console", O_RDWR, 0);

	put(fd, "\nHello from user space on Darwin!\n");
	put(fd, "I am process ");
	put_number(fd, syscall3(SYS_getpid, 0, 0, 0));
	put(fd, ", uid ");
	put_number(fd, syscall3(SYS_getuid, 0, 0, 0));
	put(fd, ", console fd ");
	put_number(fd, fd);
	put(fd, ".\n");

	/*
	 * The kernel's own version: sysctl kern.hakios.version, which the loader gave the kernel on its command line
	 * (hakios_kver, from the payload's kernel.ver). The same string the banner shows under hello hackios-kernel!
	 */
	{
		char version[96];
		u64 size = sizeof(version);
		long failed = syscall6(SYS_sysctlbyname, (long)"kern.hakios.version", 19, (long)version, (long)&size, 0, 0);

		put(fd, "kernel.ver=");
		put(fd, failed ? "(sysctl kern.hakios.version not available)" : version);
		put(fd, "\n");
	}

	/* What the drivers that are on say about the chip, through the same door (a driver that is off has no such name) */
	show_string(fd, "kern.hakios.soc_name");
	show_int(fd, "kern.hakios.soc_id");
	show_int(fd, "kern.hakios.pmic_peripherals");
	show_string(fd, "kern.hakios.thermal_sensors");

	/*
	 * hello_wait=N: stay N seconds, and say once in five how warm the SoC is. Process 1 is not supposed to exit (doing so
	 * panics the kernel, which ends the test run), so this is also how a run is made to last.
	 */
	for (long second = 0, wait = wait_seconds(); second < wait; second++) {
		long tv[2] = { 1, 0 };

		syscall6(SYS_select, 0, 0, 0, 0, (long)tv, 0);
		if (second % 5 == 4) {
			show_int(fd, "kern.hakios.thermal_max_mc");
			show_string(fd, "kern.hakios.thermal_hottest");
		}
	}

	/* Process 1 is not supposed to exit; doing so panics the kernel, which ends the test run. */
	syscall3(SYS_exit, 0, 0, 0);
	for (;;) {
	}
}
