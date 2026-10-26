/*
 * The SoC's temperature sensors: sysctl kern.hakios.thermal_*, a line in the log, and a stop for a phone that is
 * getting too hot to be left alone. Only with boot-arg thermal=1.
 *
 * What is here. The chip has two blocks of temperature sensors ("TSENS", dt: tsens@c222000 and tsens@c223000,
 * compatible qcom,tsens26xx, which Linux's tsens-v2.c drives): 15 sensors in the first (the CPU cores, the CPU
 * subsystem, the always-on block) and 12 in the second (the GPU, the neural processors, video, memory, the
 * modem, the camera). The blocks measure by themselves, all the time, and keep the last reading of each sensor in
 * a status word that anyone may read: bits [11:0] the temperature in tenths of a degree, signed, and bit 21 set
 * when there is a reading. THIS DRIVER ONLY READS those words (one 32-bit load each); it writes nothing, it
 * takes no interrupt and it sets no threshold. Calibration is the firmware's: the blocks are calibrated from the
 * fuses by the secure side before the operating system runs (Linux's tsens does it too, from the QFPROM, for its
 * own use of the raw codes; the status words used here are already degrees).
 *
 * What a temperature is worth. A reading is believed when it has its valid bit and lies between -40 and 160
 * degrees; anything else is a sensor that is not connected, or not yet measured, and is left out of the maximum.
 * The table of sensors (names, which block, which number, and the temperatures at which the vendor slows
 * things down and gives up) is the phone's own device tree's thermal-zones, made by tools/dt-thermal.py: the
 * numbers are the vendor's, not ours.
 *
 * What it does with them.
 *   - every thermal_poll milliseconds (default 2000) it reads all sensors;
 *   - sysctl kern.hakios.thermal_max_mc (the hottest, in millidegrees), thermal_hottest (its name),
 *     thermal_sensors ("cpu-0-0=41.2 ... " in degrees), thermal_reads;
 *   - a line in the log (and the loader's log) every thermal_log seconds (default 60) and at once when the
 *     hottest sensor crosses a multiple of 5 degrees upwards from 60;
 *   - the stop: when the hottest valid sensor reads at least thermal_stop degrees three times in a row (default
 *     105; thermal_stop=0 turns the stop off), it says so and hands control back to the boot loader the way
 *     "xl stop" does (the GPU, Wi-Fi and audio DSP given back first). The vendor starts to slow the CPU at 108 to
 *     112 degrees; 105 is below that on purpose, because nothing here slows anything down: the stop is the whole
 *     protection, and it is for a phone left running unattended. It is a stop of the run, not a repair: a hot phone
 *     is hot for minutes after.
 *
 * Status (what has and has not been seen): RUN ON THE PHONE (POCO X5 Pro, 2026-10-27, four runs): the status-word layout
 * (from Linux's tsens-v2.c as remembered, the same as drivers/gpu/hakios.adreno.c reads for the GPU's two sensors) is right:
 * 27 of 27 sensors have a reading, 33.3 to 39.5 degrees on a phone at rest, the CPU cores a degree or two above the rest, the
 * values move between runs; kern.hakios.thermal_* are read from user space (userland/src/hello.c with hello_wait=). NOT yet
 * seen: the stop (thermal_stop) firing, a phone under load, a reading above 40 degrees. The sensors are the device tree's; whether
 * the phone's own tree is the reference one (devices/redwood/prebuilt/device-tree.dtb) was not compared sensor by sensor.
 * Not run in QEMU, and thermal=1 must not be given there: it has no such blocks, and a load from an address it does not map faults.
 *
 * Boot-args:
 *   thermal=1          read the sensors
 *   thermal_poll=<ms>  how often (200 to 60000; 2000)
 *   thermal_log=<s>    a summary line this often (0: only on a rise; 60)
 *   thermal_stop=<C>   stop the run at this temperature (0: never; 105)
 *   thermal_tm0=<pa>, thermal_tm1=<pa>  the two blocks' readings, when this is not the reference phone
 *                      (0xc263000 and 0xc265000)
 */
#include <stdint.h>
#include <kern/clock.h>
#include <kern/startup.h>
#include <kern/thread_call.h>
#include <libkern/libkern.h>
#include <mach/mach_types.h>
#include <pexpert/pexpert.h>
#include <sys/sysctl.h>
#include <sys/systm.h>

extern vm_offset_t ml_io_map(vm_offset_t phys_addr, vm_size_t size);
extern void loader_log_putc(char c);
extern void loader_return(void);
extern void hakios_var_save_on_return(void);
extern void hakios_adreno_stop(void);
extern void hakios_wifi_stop(void);
extern void hakios_adsp_stop(void);

SYSCTL_DECL(_kern_hakios);

static int say(const char *format, ...) __printflike(1, 2);

static int
say(const char *format, ...)
{
	char line[700];                                 /* the line of all the sensors is the long one */
	va_list list;

	va_start(list, format);
	vsnprintf(line, sizeof(line), format, list);
	va_end(list);
	for (const char *c = line; *c != 0; c++) {
		loader_log_putc(*c);
	}
	return printf("%s", line);
}

#define TM_SIZE                 0x200
#define TM_STATUS(sensor)       (0xa0 + 4 * (sensor))           /* one word a sensor */
#define TM_VALID                (1u << 21)
#define TEMP_MIN_TENTHS         (-400)
#define TEMP_MAX_TENTHS         1600

/* made by tools/dt-thermal.py from the phone's device tree: name, block, sensor, slow down at, gives up at (millidegrees) */
static const struct sensor_info {
	const char *    name;
	uint32_t        tm;
	uint32_t        number;
	int32_t         slow_mc;
	int32_t         limit_mc;
} sensors[] = {
	{ "aoss-0", 0xc263000, 0, 115000, 125000 },
	{ "cpu-0-0", 0xc263000, 1, 110000, 125000 },
	{ "cpu-0-1", 0xc263000, 2, 110000, 125000 },
	{ "cpu-0-2", 0xc263000, 3, 110000, 125000 },
	{ "cpu-0-3", 0xc263000, 4, 110000, 125000 },
	{ "cpuss-0", 0xc263000, 5, 115000, 125000 },
	{ "cpuss-1", 0xc263000, 6, 115000, 125000 },
	{ "cpu-1-0", 0xc263000, 7, 108000, 125000 },
	{ "cpu-1-1", 0xc263000, 8, 108000, 125000 },
	{ "cpu-1-2", 0xc263000, 9, 108000, 125000 },
	{ "cpu-1-3", 0xc263000, 10, 108000, 125000 },
	{ "cpu-1-4", 0xc263000, 11, 108000, 125000 },
	{ "cpu-1-5", 0xc263000, 12, 108000, 125000 },
	{ "cpu-1-6", 0xc263000, 13, 108000, 125000 },
	{ "cpu-1-7", 0xc263000, 14, 108000, 125000 },
	{ "aoss-1", 0xc265000, 0, 115000, 125000 },
	{ "gpuss-0", 0xc265000, 1, 95000, 125000 },
	{ "gpuss-1", 0xc265000, 2, 95000, 125000 },
	{ "nspss-0", 0xc265000, 3, 100000, 125000 },
	{ "nspss-1", 0xc265000, 4, 100000, 125000 },
	{ "video", 0xc265000, 5, 100000, 125000 },
	{ "ddr", 0xc265000, 6, 115000, 125000 },
	{ "mdmss-0", 0xc265000, 7, 95000, 125000 },
	{ "mdmss-1", 0xc265000, 8, 95000, 125000 },
	{ "mdmss-2", 0xc265000, 9, 95000, 125000 },
	{ "mdmss-3", 0xc265000, 10, 95000, 125000 },
	{ "camera-0", 0xc265000, 11, 100000, 125000 },
};
#define NSENSORS        ((unsigned)(sizeof(sensors) / sizeof(sensors[0])))
#define REFERENCE_TM0   0xc263000u
#define REFERENCE_TM1   0xc265000u

static vm_offset_t tm_va[2];
static uint32_t tm_pa[2] = { REFERENCE_TM0, REFERENCE_TM1 };
static int32_t tenths[NSENSORS];                /* the last good reading; INT32_MIN: none yet */
static uint32_t poll_ms = 2000, log_seconds = 60, stop_celsius = 105;
static uint64_t last_log, reads;
static int strikes, last_step;
static bool stopping;

static int thermal_max_mc = -273000;
static char thermal_hottest[16] = "";
static char thermal_list[NSENSORS * 16 + 8];
static int thermal_reads_value;

SYSCTL_INT(_kern_hakios, OID_AUTO, thermal_max_mc, CTLFLAG_RD | CTLFLAG_LOCKED, &thermal_max_mc, 0,
    "the hottest valid temperature sensor of the SoC, in millidegrees (-273000: none read)");
SYSCTL_STRING(_kern_hakios, OID_AUTO, thermal_hottest, CTLFLAG_RD | CTLFLAG_LOCKED, thermal_hottest, 0, "the name of that sensor");
SYSCTL_STRING(_kern_hakios, OID_AUTO, thermal_sensors, CTLFLAG_RD | CTLFLAG_LOCKED, thermal_list, 0,
    "every sensor and its temperature in degrees: \"cpu-0-0=41.2 gpuss-0=38.0 ...\"");
SYSCTL_INT(_kern_hakios, OID_AUTO, thermal_reads, CTLFLAG_RD | CTLFLAG_LOCKED, &thermal_reads_value, 0, "how many times the sensors were read");

/* For other drivers (a GPU or modem that wants to back off): the hottest temperature, in millidegrees; -273000: not read. */
int
hakios_thermal_max_mc(void)
{
	return thermal_max_mc;
}

static thread_call_t call;

static void
text_of(char *out, size_t size, int32_t tenth)
{
	int whole = tenth / 10, frac = tenth % 10;

	snprintf(out, size, "%s%d.%d", (tenth < 0 && whole == 0) ? "-" : "", whole, frac < 0 ? -frac : frac);
}

static void
look(void)
{
	int32_t hottest = INT32_MIN;
	unsigned hot = 0, good = 0;
	char list[sizeof(thermal_list)], one[16];

	list[0] = 0;
	for (unsigned i = 0; i < NSENSORS; i++) {
		uint32_t status = *(volatile uint32_t *)(tm_va[sensors[i].tm == REFERENCE_TM0 ? 0 : 1] + TM_STATUS(sensors[i].number));
		int32_t value = (int32_t)(status << 20) >> 20;

		if (!(status & TM_VALID) || value < TEMP_MIN_TENTHS || value > TEMP_MAX_TENTHS) {
			continue;
		}
		tenths[i] = value;
		good++;
		if (value > hottest) {
			hottest = value;
			hot = i;
		}
		text_of(one, sizeof(one), value);
		if (strlen(list) + strlen(sensors[i].name) + strlen(one) + 3 < sizeof(list)) {
			strlcat(list, sensors[i].name, sizeof(list));
			strlcat(list, "=", sizeof(list));
			strlcat(list, one, sizeof(list));
			strlcat(list, " ", sizeof(list));
		}
	}
	reads++;
	thermal_reads_value = (int)reads;
	if (good == 0) {
		return;
	}
	strlcpy(thermal_list, list, sizeof(thermal_list));
	strlcpy(thermal_hottest, sensors[hot].name, sizeof(thermal_hottest));
	thermal_max_mc = hottest * 100;

	uint64_t now = mach_absolute_time(), seconds;
	int step = hottest >= 600 ? hottest / 50 : 0;           /* a step each 5 degrees from 60 */

	absolutetime_to_nanoseconds(now - last_log, &seconds);
	seconds /= NSEC_PER_SEC;
	if (step > last_step || last_log == 0 || (log_seconds != 0 && seconds >= log_seconds)) {
		text_of(one, sizeof(one), hottest);
		say("thermal: %u of %u sensors read; hottest %s at %s C; %s\n", good, NSENSORS, sensors[hot].name, one, thermal_list);
		last_log = now;
	}
	last_step = step;

	if (stop_celsius != 0 && hottest >= (int32_t)stop_celsius * 10) {
		strikes++;
	} else {
		strikes = 0;
	}
	if (strikes >= 3 && !stopping) {
		stopping = true;
		text_of(one, sizeof(one), hottest);
		say("\n--- thermal: %s reads %s C, at or above thermal_stop=%u three times in a row: back to the loader ---\n",
		    sensors[hot].name, one, stop_celsius);
		hakios_var_save_on_return();
		hakios_adreno_stop();
		hakios_wifi_stop();
		hakios_adsp_stop();
		loader_return();
	}
}

static void
tick(__unused thread_call_param_t p0, __unused thread_call_param_t p1)
{
	uint64_t deadline;

	look();
	clock_interval_to_deadline(poll_ms, NSEC_PER_MSEC, &deadline);
	thread_call_enter_delayed(call, deadline);
}

__startup_func
static void
thermal_init(void)
{
	uint32_t enabled = 0, value;

	if (!PE_parse_boot_argn("thermal", &enabled, sizeof(enabled)) || !enabled) {
		return;
	}
	for (unsigned i = 0; i < NSENSORS; i++) {
		tenths[i] = INT32_MIN;
	}
	(void)PE_parse_boot_argn("thermal_tm0", &tm_pa[0], sizeof(tm_pa[0]));
	(void)PE_parse_boot_argn("thermal_tm1", &tm_pa[1], sizeof(tm_pa[1]));
	if (PE_parse_boot_argn("thermal_poll", &value, sizeof(value)) && value >= 200 && value <= 60000) {
		poll_ms = value;
	}
	if (PE_parse_boot_argn("thermal_log", &value, sizeof(value)) && value <= 86400) {
		log_seconds = value;
	}
	if (PE_parse_boot_argn("thermal_stop", &value, sizeof(value)) && value <= 160) {
		stop_celsius = value;
	}
	for (unsigned b = 0; b < 2; b++) {
		tm_va[b] = ml_io_map(tm_pa[b], TM_SIZE);
		if (tm_va[b] == 0) {
			say("thermal: could not map the sensors' block at %#x; off\n", tm_pa[b]);
			return;
		}
	}
	say("thermal: %u sensors in two blocks (%#x, %#x), every %u ms; stop at %u C%s\n", NSENSORS, tm_pa[0], tm_pa[1], poll_ms,
	    stop_celsius, stop_celsius ? "" : " (off)");
	call = thread_call_allocate(tick, NULL);
	tick(NULL, NULL);
}
STARTUP(EARLY_BOOT, STARTUP_RANK_LAST, thermal_init);
