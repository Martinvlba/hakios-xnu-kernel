/*
 * The Pixel 8a's GNSS fixes for iOS (com.hakios.locfeed). The kernel's driver (devices/akita/drivers/gnss) reads the
 * receiver's fixes into kern.hakios.gnss_*; this hands each valid one to locationd through CoreLocation's simulation
 * path (the private CLSimulationManager: what Xcode's "simulate location" uses), so every application's CoreLocation
 * sees it. It also keeps the receiver's memory save (kern.hakios.gnss_nvm, the chip's own state: what it knows of the
 * satellites and where it was) in /var/db/hakios/gnss.nvm and gives it back at the next start, as Android's gnssd does
 * with its NVM files: a start is then not a cold one.
 *
 * Needs locationd running (mkjobs denies it unless ALLOW=locationd) and the simulation entitlement
 * (src/locfeed.entitlements, signed with ldid). THE POSITION IS NEVER WRITTEN TO THE LOG: counts only.
 */
#import <Foundation/Foundation.h>
#import <CoreLocation/CoreLocation.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <unistd.h>

#define NVM_PATH "/var/db/hakios/gnss.nvm"
#define NVM_SIZE 334

static int
kint(const char *name, int *v)
{
	size_t n = sizeof(*v);

	return sysctlbyname(name, v, &n, NULL, 0);
}

static void
nvm_restore(void)
{
	uint8_t b[NVM_SIZE];
	FILE *f = fopen(NVM_PATH, "rb");

	if (f == NULL) {
		fprintf(stderr, "locfeed: no saved receiver memory yet (a cold start)\n");
		return;
	}
	size_t got = fread(b, 1, sizeof(b), f);
	fclose(f);
	if (got != NVM_SIZE || sysctlbyname("kern.hakios.gnss_nvm", NULL, NULL, b, sizeof(b)) != 0) {
		fprintf(stderr, "locfeed: the saved receiver memory could not be given back (%zu bytes)\n", got);
		return;
	}
	fprintf(stderr, "locfeed: the receiver's memory from the last time given back\n");
}

static void
nvm_save(void)
{
	uint8_t b[NVM_SIZE];
	size_t n = sizeof(b);

	if (sysctlbyname("kern.hakios.gnss_nvm", b, &n, NULL, 0) != 0 || n != NVM_SIZE) {
		return;
	}
	mkdir("/var/db/hakios", 0755);
	FILE *f = fopen(NVM_PATH ".new", "wb");
	if (f == NULL) {
		return;
	}
	fwrite(b, 1, n, f);
	fclose(f);
	chmod(NVM_PATH ".new", 0600);
	rename(NVM_PATH ".new", NVM_PATH);
}

int
main(void)
{
	@autoreleasepool {
		int state = 0, seq = 0, last_seq = 0, saves = 0, last_saves = 0, fed = 0, said = 0;
		id manager = nil;

		if (kint("kern.hakios.gnss_state", &state) != 0) {
			fprintf(stderr, "locfeed: no GNSS driver (gnss=N not given): done\n");
			return 0;
		}
		nvm_restore();
		Class cls = NSClassFromString(@"CLSimulationManager");
		if (cls == nil) {
			fprintf(stderr, "locfeed: CoreLocation has no CLSimulationManager: fixes are kept to the kernel\n");
		} else {
			manager = [[cls alloc] init];
		}
		for (time_t last_save = time(NULL);; sleep(1)) {
			int valid = 0, lat = 0, lon = 0, acc = 0, sv = 0;

			if (kint("kern.hakios.gnss_fix_seq", &seq) != 0 || seq == last_seq) {
				goto housekeeping;
			}
			last_seq = seq;
			kint("kern.hakios.gnss_valid", &valid);
			if (!valid) {
				goto housekeeping;
			}
			kint("kern.hakios.gnss_lat", &lat);
			kint("kern.hakios.gnss_lon", &lon);
			kint("kern.hakios.gnss_acc", &acc);
			kint("kern.hakios.gnss_sv", &sv);
			if (manager != nil) {
				CLLocationCoordinate2D c = CLLocationCoordinate2DMake(lat * 1e-7, lon * 1e-7);
				/* (the accuracy field: centimetres as read against Android's fixes -- but it can come negative (11-20, -865
				 * at Darwin's first fix): CoreLocation takes a negative accuracy for "invalid", so 30 m then) */
				double h = acc > 0 && acc < 10000000 ? acc / 100.0 : 30.0;
				CLLocation *loc = [[CLLocation alloc] initWithCoordinate:c altitude:0 horizontalAccuracy:h
				    verticalAccuracy:-1 timestamp:[NSDate date]];
				@try {
					[manager performSelector:@selector(clearSimulatedLocations)];
					[manager performSelector:@selector(appendSimulatedLocation:) withObject:loc];
					[manager performSelector:@selector(flush)];
					if (fed == 0) {
						[manager performSelector:@selector(startLocationSimulation)];
					}
					fed++;
				} @catch (NSException *e) {
					if (said++ < 3) {
						fprintf(stderr, "locfeed: locationd refused the fix: %s\n", e.reason.UTF8String);
					}
				}
			}
			if (fed == 1 || fed % 300 == 0) {
				fprintf(stderr, "locfeed: %d fixes handed to locationd (the last: %d satellites, accuracy %d cm)\n", fed, sv, acc);
			}
housekeeping:
			if (time(NULL) - last_save >= 60 && kint("kern.hakios.gnss_nvm_saves", &saves) == 0 && saves != last_saves) {
				last_saves = saves;
				last_save = time(NULL);
				nvm_save();
			}
		}
	}
}
