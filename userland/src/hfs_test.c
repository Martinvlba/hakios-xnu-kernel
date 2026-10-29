/*
 * hfs_test: HFS Plus, built into the kernel (bsd/hfs), on a disk of its own -- what the data volume on the phone's flash
 * will be asked to do, tried first in QEMU on a virtio disk.
 *
 * A static program with no libc, run as process 1 from a tiny APFS root that has /dev, /mnt and this as
 * /sbin/launchd (qemu/run-hfs-test.sh). It mounts the volume, reads its size, writes a megabyte, reads it back,
 * renames, counts its own boots in a file on the volume, unmounts, mounts again and checks everything is still
 * there. Every step prints one line; the last is "hfs_test: ALL OK" or the step that failed.
 */
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;

static long
sys6(long num, long a, long b, long c, long d, long e, long f)
{
	register long x16 __asm__("x16") = num;
	register long x0 __asm__("x0") = a;
	register long x1 __asm__("x1") = b;
	register long x2 __asm__("x2") = c;
	register long x3 __asm__("x3") = d;
	register long x4 __asm__("x4") = e;
	register long x5 __asm__("x5") = f;
	long carry;
	__asm__ volatile ("svc #0x80\n cset %1, cs" : "+r" (x0), "=r" (carry) : "r" (x16), "r" (x1), "r" (x2), "r" (x3),
	    "r" (x4), "r" (x5) : "memory", "cc");
	return carry ? -x0 : x0;
}
#define sys3(n, a, b, c) sys6(n, (long)(a), (long)(b), (long)(c), 0, 0, 0)

enum { SYS_exit = 1, SYS_read = 3, SYS_write = 4, SYS_open = 5, SYS_close = 6, SYS_unlink = 10, SYS_fsync = 95,
       SYS_rename = 128, SYS_mkdir = 136, SYS_unmount = 159, SYS_mount = 167, SYS_lseek = 199, SYS_statfs64 = 345 };
#define O_RDONLY 0
#define O_RDWR   2
#define O_CREAT  0x200
#define O_TRUNC  0x400

static unsigned long
slen(const char *s)
{
	unsigned long n = 0;
	while (s[n]) {
		n++;
	}
	return n;
}

static long console = -1;

static void
say(const char *s)
{
	sys3(SYS_write, console, s, slen(s));
}

static void
num(long v)
{
	char b[24];
	int i = 23;
	int neg = v < 0;
	b[i] = 0;
	if (neg) {
		v = -v;
	}
	do {
		b[--i] = '0' + v % 10;
		v /= 10;
	} while (v);
	if (neg) {
		b[--i] = '-';
	}
	say(b + i);
}

static void
fail(const char *what, long err)
{
	say("hfs_test: FAILED: ");
	say(what);
	say(" (");
	num(err);
	say(")\n");
	sys3(SYS_exit, 1, 0, 0);
}

/* struct hfs_mount_args as user space passes it (bsd/hfs/hfs_mount.h) */
struct hfs_args {
	const char *fspec;
	u32 uid, gid;
	u16 mask, pad;
	u32 encoding;
	int tz[2];
	int flags, journal_tbuffer_size, journal_flags, journal_disable;
};

static char buf[1 << 20], back[1 << 20];

static long
mount_hfs(void)
{
	struct hfs_args a = { .fspec = "/dev/rvblk0" };
	return sys6(SYS_mount, (long)"hfs", (long)"/mnt", 0, (long)&a, 0, 0);
}

static void
fill(char *p, u32 seed)
{
	for (u32 i = 0; i < sizeof(buf); i++) {
		seed = seed * 1103515245 + 12345;
		p[i] = (char)(seed >> 16);
	}
}

static int
same(const char *a, const char *b, unsigned long n)
{
	for (unsigned long i = 0; i < n; i++) {
		if (a[i] != b[i]) {
			return 0;
		}
	}
	return 1;
}

static long
read_all(const char *path, char *into, unsigned long n)
{
	long fd = sys3(SYS_open, path, O_RDONLY, 0), got = 0, r;
	if (fd < 0) {
		return fd;
	}
	while (got < (long)n && (r = sys3(SYS_read, fd, into + got, n - got)) > 0) {
		got += r;
	}
	sys3(SYS_close, fd, 0, 0);
	return got;
}

void
_start(void)
{
	long r, fd;
	unsigned char sfs[2168];

	console = sys3(SYS_open, "/dev/console", O_RDWR, 0);
	say("hfs_test: start\n");
	if ((r = mount_hfs()) < 0) {
		fail("mount hfs /dev/rvblk0 on /mnt", r);
	}
	say("hfs_test: mounted\n");

	if ((r = sys3(SYS_statfs64, "/mnt", sfs, 0)) < 0) {
		fail("statfs /mnt", r);
	}
	u32 bsize = *(u32 *)(sfs + 0);
	u64 blocks = *(u64 *)(sfs + 8), bfree = *(u64 *)(sfs + 16);
	say("hfs_test: statfs: type ");
	say((const char *)(sfs + 72));
	say(", ");
	num((long)(blocks * bsize >> 20));
	say(" MB, free ");
	num((long)(bfree * bsize >> 20));
	say(" MB\n");

	/* the boots this volume has seen: a counter in a file on it */
	char cnt[8] = { 0 };
	long boots = 0;
	if (read_all("/mnt/boots", cnt, 8) == 8) {
		boots = *(long *)cnt;
	}
	boots++;
	*(long *)cnt = boots;
	if ((fd = sys3(SYS_open, "/mnt/boots", O_RDWR | O_CREAT | O_TRUNC, 0644)) < 0 || sys3(SYS_write, fd, cnt, 8) != 8 ||
	    sys3(SYS_fsync, fd, 0, 0) < 0) {
		fail("write /mnt/boots", fd);
	}
	sys3(SYS_close, fd, 0, 0);
	say("hfs_test: this volume's boot number ");
	num(boots);
	say("\n");

	sys3(SYS_mkdir, "/mnt/d", 0755, 0);
	fill(buf, (u32)boots);
	if ((fd = sys3(SYS_open, "/mnt/d/f", O_RDWR | O_CREAT | O_TRUNC, 0644)) < 0) {
		fail("create /mnt/d/f", fd);
	}
	if ((r = sys3(SYS_write, fd, buf, sizeof(buf))) != (long)sizeof(buf)) {
		fail("write 1 MB", r);
	}
	if ((r = sys3(SYS_fsync, fd, 0, 0)) < 0) {
		fail("fsync", r);
	}
	sys3(SYS_close, fd, 0, 0);
	if ((r = read_all("/mnt/d/f", back, sizeof(back))) != (long)sizeof(back) || !same(buf, back, sizeof(buf))) {
		fail("read the megabyte back", r);
	}
	say("hfs_test: 1 MB written and read back the same\n");
	sys3(SYS_unlink, "/mnt/d/g", 0, 0);
	if ((r = sys3(SYS_rename, "/mnt/d/f", "/mnt/d/g", 0)) < 0) {
		fail("rename", r);
	}

	if ((r = sys3(SYS_unmount, "/mnt", 0, 0)) < 0) {
		fail("unmount", r);
	}
	if ((r = mount_hfs()) < 0) {
		fail("mount again", r);
	}
	for (unsigned long i = 0; i < sizeof(back); i++) {
		back[i] = 0;
	}
	if ((r = read_all("/mnt/d/g", back, sizeof(back))) != (long)sizeof(back) || !same(buf, back, sizeof(buf))) {
		fail("the megabyte after unmount and mount", r);
	}
	if ((r = sys3(SYS_unmount, "/mnt", 0, 0)) < 0) {
		fail("unmount again", r);
	}
	say("hfs_test: after unmount and mount, the same; boots counted ");
	num(boots);
	say("\nhfs_test: ALL OK\n");
	sys3(SYS_exit, 0, 0, 0);
}
