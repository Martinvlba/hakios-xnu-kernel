/*
 * thermal_state: iOS's thermal state from the kernel's sensors (job com.hakios.thermalstate).
 *
 * On an iPhone thermalmonitord reads Apple's own sensors and sets the thermal pressure level: the notify key
 * com.apple.system.thermalpressurelevel (OSThermalNotification.h, kOSThermalNotificationPressureLevelName), whose state
 * is an OSThermalPressureLevel -- 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping. Foundation's
 * NSProcessInfo.thermalState and its notification, and everything in the system that slows itself down when the phone
 * is hot, go by that key. thermalmonitord is not let through here (userland/mkjobs.py: its sensors are not there);
 * this sets the key in its place, from the hottest of the SoC's sensors the kernel reads (kern.hakios.thermal_max_mc:
 * the Pixel 8a's drivers/power/hakios.akita_thermal.c, the Qualcomm phones' drivers/power/hakios.thermal.c, both
 * thermal=1): below 75 C nominal, 75 moderate, 85 heavy, 95 trapping, 100 sleeping (where the Pixel 8a's kernel powers
 * the phone off anyway). A level is left 3 degrees below where it was entered, so that it does not flap. Every 5 s.
 * Without the sysctl (no thermal=1) it ends at once. Runs as root: only root may set a com.apple.system key.
 */
#include <notify.h>
#include <stdio.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define KEY "com.apple.system.thermalpressurelevel"

static const int enter_c[] = { 0, 75, 85, 95, 100 };
static const char *const level_names[] = { "nominal", "moderate", "heavy", "trapping", "sleeping" };

static int
level_for(int celsius, int now)
{
	int level = 0;

	for (int i = 4; i > 0; i--) {
		if (celsius >= enter_c[i] || (i <= now && celsius >= enter_c[i] - 3)) {
			level = i;
			break;
		}
	}
	return level;
}

int
main(void)
{
	int token, level = -1, mc;
	size_t size = sizeof(mc);

	if (sysctlbyname("kern.hakios.thermal_max_mc", &mc, &size, NULL, 0) != 0) {
		fprintf(stderr, "thermal_state: no kern.hakios.thermal_max_mc (thermal=1 not given): nothing to do\n");
		return 0;
	}
	if (notify_register_check(KEY, &token) != NOTIFY_STATUS_OK) {
		fprintf(stderr, "thermal_state: cannot register %s\n", KEY);
		return 1;
	}
	for (;;) {
		size = sizeof(mc);
		if (sysctlbyname("kern.hakios.thermal_max_mc", &mc, &size, NULL, 0) == 0 && mc > -100000) {
			int celsius = mc / 1000, now = level_for(celsius, level < 0 ? 0 : level);

			if (now != level) {
				uint32_t status = notify_set_state(token, (uint64_t)now);

				if (status == NOTIFY_STATUS_OK) {
					status = notify_post(KEY);
				}
				fprintf(stderr, "thermal_state: %d C: %s (level %d)%s\n", celsius, level_names[now], now,
				    status == NOTIFY_STATUS_OK ? "" : " -- notifyd refused it");
				level = now;
			}
		}
		sleep(5);
	}
}
