/*
 * libhakios.bt.dylib: code of ours inside BlueTool (build.sh's INJECT), for Bluetooth on a phone whose controller the
 * kernel has already switched on and loaded (devices/akita/drivers/bluetooth with bt=2; drivers/bluetooth's BT_HOST).
 *
 * iOS's bluetoothd asks BlueTool, over the service com.apple.BlueTool, to "boot" and "init" the controller: on an
 * iPhone BlueTool loads Broadcom's firmware for that iPhone's chip over a serial port and sets it up from the device
 * tree. Here there is no such port and nothing left to load, and BlueTool would time out. Inside the real process the
 * service is ours instead and answers each request as done -- result 0, manufacturer 15 (Broadcom), a firmware name.
 * (Upstream's way on its MediaTek phone: branch mtk, src/inject.c's bluetool_serves.) Each request is said on the
 * console, as it came, so that what bluetoothd asks is seen.
 */
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

/* (XNU's sys/kern_control.h and sys/sys_domain.h, which the SDK this is built with does not have) */
#ifndef SYSPROTO_CONTROL
#define SYSPROTO_CONTROL        2
#endif
#ifndef PF_SYSTEM
#define PF_SYSTEM               AF_SYSTEM
#endif
struct ctl_info {
	uint32_t ctl_id;
	char ctl_name[96];
};
#define CTLIOCGINFO             _IOWR('N', 3, struct ctl_info)

/* (libxpc's, declared here: the SDK this is built with has no xpc/xpc.h) */
typedef void *xpc_object_t;
typedef void *xpc_connection_t;
extern const struct _xpc_type_s _xpc_type_connection, _xpc_type_dictionary;
extern xpc_connection_t xpc_connection_create_mach_service(const char *name, dispatch_queue_t targetq, uint64_t flags);
extern void xpc_connection_set_event_handler(xpc_connection_t connection, void (^handler)(xpc_object_t object));
extern void xpc_connection_resume(xpc_connection_t connection);
extern void xpc_connection_send_message(xpc_connection_t connection, xpc_object_t message);
extern const struct _xpc_type_s *xpc_get_type(xpc_object_t object);
extern xpc_object_t xpc_dictionary_create_reply(xpc_object_t original);
extern void xpc_dictionary_set_int64(xpc_object_t xdict, const char *key, int64_t value);
extern void xpc_dictionary_set_string(xpc_object_t xdict, const char *key, const char *string);
extern char *xpc_copy_description(xpc_object_t object);
extern void xpc_release(xpc_object_t object);

static void
say(const char *line)
{
	int fd = open("/dev/console", O_WRONLY);

	if (fd >= 0) {
		(void)write(fd, line, strlen(line));
		close(fd);
	}
}

__attribute__((constructor))
static void
bluetool_serves(void)
{
	const char *name = getprogname();

	if (name == NULL || strcmp(name, "BlueTool") != 0) {
		return;
	}
	xpc_connection_t listener = xpc_connection_create_mach_service("com.apple.BlueTool", NULL, 1 /* a listener */);

	say("hakios: BlueTool: com.apple.BlueTool is ours (the kernel has the controller on, its firmware loaded)\n");
	xpc_connection_set_event_handler(listener, ^(xpc_object_t peer) {
		if (xpc_get_type(peer) != &_xpc_type_connection) {
			return;
		}
		xpc_connection_set_event_handler((xpc_connection_t)peer, ^(xpc_object_t message) {
			if (xpc_get_type(message) != &_xpc_type_dictionary) {
				return;
			}
			char *text = xpc_copy_description(message), line[600];

			snprintf(line, sizeof(line), "hakios: BlueTool: asked %s\n", text != NULL ? text : "(?)");
			say(line);
			free(text);
			xpc_object_t reply = xpc_dictionary_create_reply(message);

			if (reply != NULL) {
				xpc_dictionary_set_int64(reply, "result", 0);
				xpc_dictionary_set_int64(reply, "manufacturer", 15);
				xpc_dictionary_set_string(reply, "firmware", "BCM4383A3");
				xpc_dictionary_set_string(reply, "output", "");
				xpc_connection_send_message((xpc_connection_t)peer, reply);
				xpc_release(reply);
			}
		});
		xpc_connection_resume((xpc_connection_t)peer);
	});
	xpc_connection_resume(listener);
	dispatch_main();
}

/*
 * IN bluetoothd (build.sh's INJECT names it too, 11-16): wait for the controller's transport before bluetoothd runs.
 * The registry's "bluetooth" service and its address are published a second into the boot (so that SpringBoard's
 * BluetoothManager attaches and Control Center's button is not "unavailable"), but the controller and the kernel
 * control com.apple.uart.sk.bluetooth come only ~45 s later (devices/akita drivers/bluetooth). A bluetoothd started in
 * between -- launchd starts it for SpringBoard within seconds -- found the address, went on to open the transport and
 * died "Bluetooth fatal error, reason 1201" five times, and every bluetoothd after that aborted ~30 s after connecting.
 * Here its start simply waits (up to two minutes) until the kernel control exists; a client's message waits in its
 * queue meanwhile.
 */
__attribute__((constructor))
static void
wait_for_transport(void)
{
	const char *name = getprogname();
	char line[160];

	if (name == NULL || strcmp(name, "bluetoothd") != 0) {
		return;
	}
	for (int tries = 0; tries < 480; tries++) {
		int fd = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL);
		struct ctl_info info;

		memset(&info, 0, sizeof(info));
		strlcpy(info.ctl_name, "com.apple.uart.sk.bluetooth", sizeof(info.ctl_name));
		if (fd >= 0 && ioctl(fd, CTLIOCGINFO, &info) == 0) {
			close(fd);
			if (tries != 0) {
				snprintf(line, sizeof(line), "hakios: bluetoothd waited %d.%d s for the controller's transport\n", tries / 4,
				    (tries % 4) * 25 / 10);
				say(line);
			}
			return;
		}
		if (fd >= 0) {
			close(fd);
		}
		usleep(250000);
	}
	say("hakios: bluetoothd: no controller transport after two minutes; started anyway\n");
}
