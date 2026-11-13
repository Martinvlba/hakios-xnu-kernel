/*
 * memsay: where the kernel's memory is, every 20 s (com.hakios.memsay) -- the tester's Speedometer run of 11-13 took the
 * phone down with "wired 4655 MB" in the kernel's own report (817 MB four minutes before), the GPU's buffers and the
 * surfaces accounting for 500 of them. mach_memory_info (what zprint reads; root): the biggest of the kernel's
 * allocation sites (the VM tags: IOKit, page tables, kalloc...) and of its zones, one line each time, to the console
 * and into /private/var/mobile/Media/hakios-klog/wired_YYYY-MM-DD_HH-MM-SS.log beside klogsave's files (fetched by
 * devices/akita/tools/usbmux-pull-klog.py when the phone was off the cable).
 */
#include <fcntl.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* osfmk/mach_debug/zone_info.h (xnu-8796), which the iOS SDK does not carry */
struct hk_zone_name { char name[80]; };
struct hk_zone_info { uint64_t count, cur_size, max_size, elem_size, alloc_size, sum_size, exhaustible, collectable; };
struct hk_memory_info {
	uint64_t flags, site, size, free, largest, collectable_bytes, mapped, peak;
	uint16_t tag, zone, resv_a[2];
	uint64_t resv[3];
	char name[80];
};
extern kern_return_t memory_info_call(mach_port_t host, struct hk_zone_name **names, mach_msg_type_number_t *names_n,
    struct hk_zone_info **info, mach_msg_type_number_t *info_n, struct hk_memory_info **memory,
    mach_msg_type_number_t *memory_n) __asm("_mach_memory_info");

static const char *const tags[] = { "none", "osfmk", "bsd", "IOKit", "libkern", "OSKext", "kext", "ipc", "stacks", "cpu",
	                            "pmap", "page tables", "zones", "kalloc", "compressor", "compressed data", "phantom cache",
	                            "waitq", "diag", "log", "file", "mbuf", "ubc", "security", "mlock", "reason", "skywalk",
	                            "ltable", "hv", "kalloc data", "retired", "kalloc type", "triage", "recount" };
static const char *const counters[] = { "(managed)", "(reserved)", "(wired)", "(wired managed)", "(stolen)", "(lopage)",
	                                "(map kernel)", "(map zone)", "(map kalloc)", "(wired at boot)", "(boot stolen)",
	                                "(static kernelcache)", "(map kalloc data)", "(map kernel data)" };

struct row { uint64_t size; char name[96]; };

static int
by_size(const void *a, const void *b)
{
	const struct row *x = a, *y = b;
	return x->size < y->size ? 1 : x->size > y->size ? -1 : 0;
}

int
main(int argc, char **argv)
{
	int every = argc > 1 ? atoi(argv[1]) : 20, fd, said = 0;
	char path[256], stamp[64], line[4096];
	time_t now = time(NULL);
	struct tm tm;

	mkdir("/private/var/mobile/Media", 0755);
	mkdir("/private/var/mobile/Media/hakios-klog", 0755);
	localtime_r(&now, &tm);
	strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &tm);
	snprintf(path, sizeof(path), "/private/var/mobile/Media/hakios-klog/wired_%s.log", stamp);
	fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
	for (;;) {
		struct hk_zone_name *names = NULL;
		struct hk_zone_info *info = NULL;
		struct hk_memory_info *memory = NULL;
		mach_msg_type_number_t names_n = 0, info_n = 0, memory_n = 0;
		kern_return_t kr = memory_info_call(mach_host_self(), &names, &names_n, &info, &info_n, &memory, &memory_n);

		if (kr != KERN_SUCCESS) {
			if (!said) {
				said = 1;
				fprintf(stderr, "memsay: mach_memory_info: 0x%x (uid %d)\n", kr, (int)getuid());
			}
			sleep((unsigned)every);
			continue;
		}
		struct row *sites = calloc(memory_n + 1, sizeof(*sites)), *zones = calloc(info_n + 1, sizeof(*zones));
		unsigned ns = 0, nz = 0;
		struct timespec up;
		int at;

		for (unsigned i = 0; sites != NULL && i < memory_n; i++) {
			const struct hk_memory_info *m = &memory[i];
			unsigned type = (unsigned)(m->flags & 0xff);

			if (m->size == 0 || (m->flags & 0x800)) {       /* (a zone's own entry: with the zones) */
				continue;
			}
			sites[ns].size = m->size;
			if ((m->flags & 0x400) && m->name[0] != 0) {
				snprintf(sites[ns].name, sizeof(sites[ns].name), "%.60s", m->name);
			} else if (type == 3 && m->site < sizeof(counters) / sizeof(counters[0])) {
				snprintf(sites[ns].name, sizeof(sites[ns].name), "%s", counters[m->site]);
			} else if (type == 0 && m->site < sizeof(tags) / sizeof(tags[0])) {
				snprintf(sites[ns].name, sizeof(sites[ns].name), "%s", tags[m->site]);
			} else {
				snprintf(sites[ns].name, sizeof(sites[ns].name), "%s %llu", type == 1 ? "kmod" : type == 2 ? "kernel site" : "tag",
				    (unsigned long long)m->site);
			}
			ns++;
		}
		for (unsigned i = 0; zones != NULL && i < info_n && i < names_n; i++) {
			if (info[i].cur_size != 0) {
				zones[nz].size = info[i].cur_size;
				snprintf(zones[nz].name, sizeof(zones[nz].name), "%.60s", names[i].name);
				nz++;
			}
		}
		if (sites != NULL) {
			qsort(sites, ns, sizeof(*sites), by_size);
		}
		if (zones != NULL) {
			qsort(zones, nz, sizeof(*zones), by_size);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &up);
		at = snprintf(line, sizeof(line), "memsay: at %ld s, MB: ", (long)up.tv_sec);
		for (unsigned i = 0; i < ns && i < 16 && at < (int)sizeof(line) - 120; i++) {
			at += snprintf(line + at, sizeof(line) - (size_t)at, "%s %llu, ", sites[i].name, (unsigned long long)(sites[i].size >> 20));
		}
		at += snprintf(line + at, sizeof(line) - (size_t)at, "ZONES: ");
		for (unsigned i = 0; i < nz && i < 10 && at < (int)sizeof(line) - 120; i++) {
			at += snprintf(line + at, sizeof(line) - (size_t)at, "%s %llu, ", zones[i].name, (unsigned long long)(zones[i].size >> 20));
		}
		at += snprintf(line + at, sizeof(line) - (size_t)at, "\n");
		fputs(line, stderr);
		if (fd >= 0) {
			(void)write(fd, line, (size_t)at);
			fsync(fd);
		}
		free(sites);
		free(zones);
		vm_deallocate(mach_task_self(), (vm_address_t)names, names_n * sizeof(*names));
		vm_deallocate(mach_task_self(), (vm_address_t)info, info_n * sizeof(*info));
		vm_deallocate(mach_task_self(), (vm_address_t)memory, memory_n * sizeof(*memory));
		sleep((unsigned)every);
	}
}
