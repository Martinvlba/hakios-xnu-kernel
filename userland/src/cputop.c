/*
 * Which processes take the processors: every few seconds, the processor time each process was given since the
 * last look (user and system together), the most first, as one line for the run's log:
 *
 *   cputop: 25 s: 412% in all: backboardd[13] 96% SpringBoard[41] 88% WeatherWidget[131] 40% ...
 *
 * (100% is one core's whole time.) tools/frames.py says when the cores were all busy ("cores busy") while the
 * window server's frames came late; this says with whom it shared them. A process born since the last look counts
 * with all the time it has had.
 *
 *   usage: cputop [SECONDS [LOOKS]]     every SECONDS (default 5; fractions too, 0.1 the least -- each line then
 *                                       says the kernel's uptime, to be matched with a recording), LOOKS times
 *                                       (default: until the run ends)
 */
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* libproc's, which the iPhoneOS SDK does not declare (XNU's bsd/sys/proc_info.h, libsyscall/wrappers/libproc) */
#define PROC_ALL_PIDS           1
#define PROC_PIDTASKINFO        4
struct proc_taskinfo {
	uint64_t pti_virtual_size;
	uint64_t pti_resident_size;
	uint64_t pti_total_user;        /* in mach_absolute_time's units */
	uint64_t pti_total_system;
	uint64_t pti_threads_user;
	uint64_t pti_threads_system;
	int32_t pti_policy;
	int32_t pti_faults;
	int32_t pti_pageins;
	int32_t pti_cow_faults;
	int32_t pti_messages_sent;
	int32_t pti_messages_received;
	int32_t pti_syscalls_mach;
	int32_t pti_syscalls_unix;
	int32_t pti_csw;
	int32_t pti_threadnum;
	int32_t pti_numrunning;
	int32_t pti_priority;
};
int proc_listpids(uint32_t type, uint32_t typeinfo, void *buffer, int buffersize);
int proc_pidinfo(int pid, int flavor, uint64_t arg, void *buffer, int buffersize);
int proc_name(int pid, void *buffer, uint32_t buffersize);

#define MAX_PROCS       2048
#define SHOWN           10

struct sample {
	int pid;
	uint64_t time;                  /* user + system, mach_absolute_time's units */
};

static struct sample last[MAX_PROCS], now[MAX_PROCS];
static int last_count;

struct share {
	int pid;
	uint64_t time;
};

static int
by_time(const void *a, const void *b)
{
	const struct share *x = a, *y = b;

	return x->time < y->time ? 1 : x->time > y->time ? -1 : 0;
}

static uint64_t
time_before(int pid, int *found)
{
	for (int i = 0; i < last_count; i++) {
		if (last[i].pid == pid) {
			*found = 1;
			return last[i].time;
		}
	}
	*found = 0;
	return 0;
}

/*
 * With SECONDS under 1, each core's busy share since the look before too (11-10): which cores a burst of work went to --
 * the A510s (0-3) are to be used only when the faster ones are all busy (ml_pair_slow_processors), the X3 is 8.
 */
static void
cpu_shares(void)
{
	static natural_t was[16][CPU_STATE_MAX];
	static bool have;
	processor_cpu_load_info_t load;
	mach_msg_type_number_t words;
	natural_t n;

	if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &n, (processor_info_array_t *)&load, &words) != KERN_SUCCESS) {
		return;
	}
	if (have) {
		printf("cputop: cores busy %%:");
		for (natural_t c = 0; c < n && c < 16; c++) {
			natural_t busy = 0, all = 0;

			for (int k = 0; k < CPU_STATE_MAX; k++) {
				natural_t d = load[c].cpu_ticks[k] - was[c][k];

				all += d;
				busy += k == CPU_STATE_IDLE ? 0 : d;
			}
			printf("%s %u", c == 4 || c == 8 ? " |" : "", all ? 100 * busy / all : 0);
		}
		printf("\n");
	}
	for (natural_t c = 0; c < n && c < 16; c++) {
		memcpy(was[c], load[c].cpu_ticks, sizeof(was[c]));
	}
	have = true;
	vm_deallocate(mach_task_self(), (vm_address_t)load, words * sizeof(natural_t));
}

int
main(int argc, char **argv)
{
	double seconds = argc > 1 ? atof(argv[1]) : 5;
	unsigned looks = argc > 2 ? (unsigned)atoi(argv[2]) : 0;
	static int pids[MAX_PROCS];
	static struct share shares[MAX_PROCS];
	mach_timebase_info_data_t timebase;
	uint64_t start = mach_absolute_time(), before = start;

	if (seconds <= 0) {
		seconds = 5;
	} else if (seconds < 0.1) {
		seconds = 0.1;
	}
	mach_timebase_info(&timebase);
	setvbuf(stdout, NULL, _IOLBF, 0);
	for (unsigned look = 0; looks == 0 || look <= looks; look++) {
		int bytes = proc_listpids(PROC_ALL_PIDS, 0, pids, sizeof(pids)), count = 0, shown = 0;
		uint64_t at = mach_absolute_time(), all = 0;

		for (int i = 0; i < bytes / (int)sizeof(int) && count < MAX_PROCS; i++) {
			struct proc_taskinfo info;

			if (pids[i] <= 0 || proc_pidinfo(pids[i], PROC_PIDTASKINFO, 0, &info, sizeof(info)) != sizeof(info)) {
				continue;
			}
			now[count].pid = pids[i];
			now[count].time = info.pti_total_user + info.pti_total_system;
			count++;
		}
		if (look > 0) {
			int n = 0;

			for (int i = 0; i < count; i++) {
				int found;
				uint64_t then = time_before(now[i].pid, &found);

				if (now[i].time > then) {
					shares[n].pid = now[i].pid;
					shares[n].time = now[i].time - then;
					all += shares[n].time;
					n++;
				}
			}
			qsort(shares, (size_t)n, sizeof(shares[0]), by_time);
			/* the shares as percent of one core over the time between the looks (both in mach_absolute_time's units) */
			double span = (double)(at - before);
			if (seconds < 1) {
				/* (the kernel's uptime, as its log and a recording's frames say it: mach_absolute_time) */
				printf("cputop: at %.3f s: %.0f%% in all:", (double)at * timebase.numer / timebase.denom / 1e9,
				    span > 0 ? 100.0 * (double)all / span : 0);
			} else {
				printf("cputop: %llu s: %.0f%% in all:", (unsigned long long)((at - start) * timebase.numer / timebase.denom / 1000000000ull),
				    span > 0 ? 100.0 * (double)all / span : 0);
			}
			for (int i = 0; i < n && shown < SHOWN; i++) {
				char name[64] = "?";
				double percent = span > 0 ? 100.0 * (double)shares[i].time / span : 0;

				if (percent < 1) {
					break;
				}
				proc_name(shares[i].pid, name, sizeof(name));
				printf(" %s[%d] %.0f%%", name, shares[i].pid, percent);
				shown++;
			}
			printf("\n");
			if (seconds < 1) {
				cpu_shares();
			}
		}
		memcpy(last, now, sizeof(now[0]) * (size_t)count);
		last_count = count;
		before = at;
		if (looks != 0 && look == looks) {
			break;
		}
		usleep((useconds_t)(seconds * 1e6));
	}
	return 0;
}
