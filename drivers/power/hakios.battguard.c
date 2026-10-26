/*
 * A run nobody watches must not empty the battery: with boot-arg battery_stop=<percent>, hand control back to the boot loader
 * when the charge is at or below that for three looks in a row, the way drivers/power/hakios.thermal.c does for heat.
 *
 * Why. The phone is not charged while HakiOS runs (the kernel has no charging path yet: docs/install-safe.md), and a session
 * left going overnight ends with a flat battery, which is a phone that has gone off USB until somebody plugs it in. This is the
 * protection that costs nothing: at 15 percent the run ends, the loader has the log, and (xl with XL_REBOOT_AFTER=1) the phone
 * goes back to Android, which charges it.
 *
 * What it believes. It asks the battery driver's own numbers (sysctl kern.hakios.battery_soc, ..._present, ..._charger_present;
 * they come from the loader's charger page or, with adsp=1, from the audio DSP's battery service) and acts only on a reading that is
 * between 0 and 100 with a battery present and NO charger attached. An unknown charge (-1), no battery, or a charger on the cable
 * is never a reason to stop. It does not read any hardware itself.
 *
 * Status (what has and has not been seen): COMPILED ONLY. It was written while the phone was off USB, so it has not run on one;
 * whether the numbers it reads are filled in when a kernel runs without adsp=1 is not known (the battery driver's header says
 * where they come from; the first run that sets battery_stop will print its "battguard:" line and say). Not run in QEMU.
 *
 * Boot-args:
 *   battery_stop=<percent>   stop at this charge or below (0: never, the default; 1 to 99)
 *   battery_stop_poll=<s>    how often to look (5 to 600; 20)
 */
#include <kern/clock.h>
#include <kern/startup.h>
#include <kern/thread_call.h>
#include <libkern/libkern.h>
#include <mach/mach_types.h>
#include <pexpert/pexpert.h>
#include <sys/sysctl.h>
#include <sys/systm.h>

extern void loader_log_putc(char c);
extern void loader_return(void);
extern void hakios_var_save_on_return(void);
extern void hakios_adreno_stop(void);
extern void hakios_wifi_stop(void);
extern void hakios_adsp_stop(void);
extern int kernel_sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);

static int say(const char *format, ...) __printflike(1, 2);

static int
say(const char *format, ...)
{
	char line[300];
	va_list list;

	va_start(list, format);
	vsnprintf(line, sizeof(line), format, list);
	va_end(list);
	for (const char *c = line; *c != 0; c++) {
		loader_log_putc(*c);
	}
	return printf("%s", line);
}

static thread_call_t call;
static uint32_t stop_percent, poll_seconds = 20, looks, strikes, last_said_minute = ~0u;
static bool stopping;

static bool
read_int(const char *name, int *value)
{
	size_t size = sizeof(*value);

	return kernel_sysctlbyname(name, value, &size, NULL, 0) == 0 && size == sizeof(*value);
}

static void
tick(__unused thread_call_param_t p0, __unused thread_call_param_t p1)
{
	int soc = -1, present = 0, charger = 0;
	uint64_t deadline;
	bool known = read_int("kern.hakios.battery_soc", &soc) && read_int("kern.hakios.battery_present", &present) &&
	    read_int("kern.hakios.battery_charger_present", &charger) && soc >= 0 && soc <= 100 && present && !charger;
	uint32_t minute = (uint32_t)(++looks * poll_seconds / 300);       /* a line every five minutes, and the first */

	if (minute != last_said_minute || looks == 1) {
		last_said_minute = minute;
		say("battguard: charge %d%% (battery %s, charger %s); stop at or below %u%%%s\n", soc, present ? "present" : "not reported",
		    charger ? "attached: the guard waits" : "not attached", stop_percent, known ? "" : " -- not a reading it acts on");
	}
	strikes = (known && (uint32_t)soc <= stop_percent) ? strikes + 1 : 0;
	if (strikes >= 3 && !stopping) {
		stopping = true;
		say("\n--- battguard: the charge is %d%%, at or below battery_stop=%u three looks in a row: back to the loader ---\n", soc,
		    stop_percent);
		hakios_var_save_on_return();
		hakios_adreno_stop();
		hakios_wifi_stop();
		hakios_adsp_stop();
		loader_return();
	}
	clock_interval_to_deadline(poll_seconds, NSEC_PER_SEC, &deadline);
	thread_call_enter_delayed(call, deadline);
}

__startup_func
static void
battguard_init(void)
{
	uint32_t value;

	if (!PE_parse_boot_argn("battery_stop", &stop_percent, sizeof(stop_percent)) || stop_percent == 0 || stop_percent > 99) {
		return;
	}
	if (PE_parse_boot_argn("battery_stop_poll", &value, sizeof(value)) && value >= 5 && value <= 600) {
		poll_seconds = value;
	}
	say("battguard: on, looking every %u s; the run ends at %u%% or less (and only with a battery, a known charge and no charger)\n",
	    poll_seconds, stop_percent);
	uint64_t deadline;

	call = thread_call_allocate(tick, NULL);

	clock_interval_to_deadline(poll_seconds, NSEC_PER_SEC, &deadline);
	thread_call_enter_delayed(call, deadline);
}
STARTUP(EARLY_BOOT, STARTUP_RANK_LAST, battguard_init);
