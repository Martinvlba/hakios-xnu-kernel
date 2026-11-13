/*
 * klogsave: the kernel's log kept in a file on the data partition while HakiOS runs (com.hakios.klogsave), for crashes
 * that take pstore and the klog partition with them (the tester's idea, 11-09: a panic after a USB replug whose log
 * Android's start overwrote). Twice a second the kernel's message buffer (proc_kmsgbuf, as dmesg reads it: oldest
 * first, as text -- NOT sysctl kern.msgbuf, which in this XNU is the buffer's size as an int: every file until 11-13
 * was one byte long) is looked at
 * and what is new appended to /private/var/mobile/Media/hakios-klog/kernel_YYYY-MM-DD_HH-MM-SS.log, synced each time
 * (under Media since 11-13: the tester uses the phone off the cable -- battery, CPU idle -- and the host fetches the
 * files over AFC when it is back, devices/akita/tools/usbmux-pull-klog.py; with battery_say=6 and cpuidle=1 the
 * kernel writes the battery's current and the idle counts there every 30 s).
 *
 * At its start it prints the end of the newest earlier file (the boot before: up to a moment before a panic, whose own
 * message comes after user space has stopped) to the console, where bootrun.sh's log catches it; files past the
 * newest 24 are removed.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define DIR_PATH        "/private/var/mobile/Media/hakios-klog"
#define KEEP            24
#define TAIL_SHOWN      8192
/* libproc's (proc_info PROC_INFO_CALL_KERNMSGBUF; root only): the bytes copied, or 0 with errno */
extern int proc_kmsgbuf(void *buffer, uint32_t buffersize);

#define MARK            256             /* the bytes written last, looked for in the buffer to find what is new */

static int
by_name(const void *a, const void *b)
{
	return strcmp(*(char *const *)a, *(char *const *)b);
}

/* the earlier files, oldest first (their names sort by time); the newest's end shown, the oldest beyond KEEP - 1 removed */
static void
earlier_files(void)
{
	DIR *d = opendir(DIR_PATH);
	struct dirent *e;
	char *names[256];
	int n = 0;

	if (d == NULL) {
		return;
	}
	while ((e = readdir(d)) != NULL && n < 256) {
		if (strncmp(e->d_name, "kernel_", 7) == 0) {
			names[n++] = strdup(e->d_name);
		}
	}
	closedir(d);
	qsort(names, (size_t)n, sizeof(names[0]), by_name);
	if (n > 0) {
		char path[512];
		snprintf(path, sizeof(path), "%s/%s", DIR_PATH, names[n - 1]);
		int fd = open(path, O_RDONLY);
		if (fd >= 0) {
			off_t size = lseek(fd, 0, SEEK_END);
			off_t from = size > TAIL_SHOWN ? size - TAIL_SHOWN : 0;
			char *buf = malloc(TAIL_SHOWN + 1);
			ssize_t got = buf ? pread(fd, buf, TAIL_SHOWN, from) : -1;
			if (got > 0) {
				buf[got] = 0;
				fprintf(stderr, "klogsave: the boot before's kernel log, %s (%lld bytes), its last %zd:\n", names[n - 1],
				    (long long)size, got);
				fwrite(buf, 1, (size_t)got, stderr);
				fprintf(stderr, "\nklogsave: (end of the boot before's kernel log)\n");
			}
			free(buf);
			close(fd);
		}
	}
	for (int i = 0; i < n; i++) {
		if (i < n - (KEEP - 1)) {
			char path[512];
			snprintf(path, sizeof(path), "%s/%s", DIR_PATH, names[i]);
			unlink(path);
		}
		free(names[i]);
	}
}

int
main(void)
{
	size_t cap = 0, len;
	int fd;
	char path[512], stamp[64];
	time_t now = time(NULL);
	struct tm tm;
	static char last[MARK];
	size_t last_len = 0;

	mkdir("/private/var/mobile/Media", 0755);
	mkdir(DIR_PATH, 0755);
	chmod(DIR_PATH, 0755);
	earlier_files();
	localtime_r(&now, &tm);
	strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &tm);
	snprintf(path, sizeof(path), "%s/kernel_%s.log", DIR_PATH, stamp);
	fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
	if (fd < 0) {
		fprintf(stderr, "klogsave: %s: %s\n", path, strerror(errno));
		return 1;
	}
	fprintf(stderr, "klogsave: the kernel's log into %s, twice a second\n", path);
	{
		int size = 0;
		size_t n = sizeof(size);
		/* (kern.msgbuf: the ring's size) */
		cap = sysctlbyname("kern.msgbuf", &size, &n, NULL, 0) == 0 && size > 0 ? (size_t)size + 2 : 1 << 20;
	}
	char *buf = malloc(cap + 1);
	if (buf == NULL) {
		return 1;
	}
	for (;;) {
		int got = proc_kmsgbuf(buf, (uint32_t)cap);
		static int said;
		if (got <= 0 && !said) {
			said = 1;
			fprintf(stderr, "klogsave: proc_kmsgbuf: %s (uid %d)\n", strerror(errno), (int)getuid());
		}
		len = got > 0 ? (size_t)got : 0;
		if (len > 0) {
			/* (oldest first; its closing NUL is dropped below) */
			size_t start = 0;
			while (start < len && buf[start] == 0) {
				start++;
			}
			size_t from = start;
			if (last_len != 0) {
				/* what follows the last bytes written; if they are gone (the ring went round), all of it */
				char *hit = NULL;
				for (size_t i = len >= last_len ? len - last_len + 1 : 0; i-- > start;) {
					if (memcmp(buf + i, last, last_len) == 0) {
						hit = buf + i;
						break;
					}
				}
				if (hit != NULL) {
					from = (size_t)(hit - buf) + last_len;
				} else {
					static const char lost[] = "\nklogsave: (the kernel's ring went round: some lines lost)\n";
					(void)write(fd, lost, sizeof(lost) - 1);
				}
			}
			if (from < len) {
				size_t n = len - from;
				while (n > 0 && buf[from + n - 1] == 0) {
					n--;
				}
				/* (proc_kmsgbuf closes a line the kernel is still writing with a newline of its own: the last
				 * newline is left for the next time, or the mark would not be found again and the whole ring
				 * written anew -- the 11-13 files had every line up to five times) */
				if (n > 0 && buf[from + n - 1] == '\n') {
					n--;
				}
				if (n > 0 && write(fd, buf + from, n) > 0) {
					fsync(fd);
					size_t end = from + n;
					last_len = end - start < MARK ? end - start : MARK;
					memcpy(last, buf + end - last_len, last_len);
				}
			}
		}
		usleep(500000);
	}
}
