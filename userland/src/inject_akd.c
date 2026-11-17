/*
 * libhakios.akd.dylib, v2 (proposed replacement for hackios-kernel/userland/src/inject_akd.c; drop-in, same
 * build.sh line). Written 2026-11-17, offline, from the 20G81 (iOS 16.6.1, d16ap, arm64e) akd + AuthKit.
 *
 * WHY v2. v1 hooked only -[AKAnisetteProvisioningController anisetteDataWithError:] and
 * -fetchAnisetteDataAndProvisionIfNecessary:error: (AuthKit.framework, the CLIENT side). On 11-16 two native
 * sign-ins on the phone produced zero requests to the host anisette server and the same -42410 failure.
 * Reading akd itself (apple-id/tools/objc-methods.py + xref.py on the 20G81 akd) shows where the failing code
 * actually lives:
 *
 *   akd: AKAnisetteProvisioningService (the XPC service akd exports)  -> _helperServiceForDevice:
 *        -> AKNativeAnisetteService (class in the akd EXECUTABLE, not in AuthKit):
 *             -fetchAnisetteDataAndProvisionIfNecessary:withCompletion:   (async, block(AKAnisetteData*, NSError*))
 *             -_unsafe_anisetteDataForDSID:withError:                     ("Looking for Anisette data...")
 *             -_unsafe_provisionAnisetteWithCompletion:                   ("Kicking off Anisette provisioning....",
 *                                                                          xref at akd+0x21d64; ADIProvisioningEnd
 *                                                                          -> -42410)
 *
 *   AuthKit's AKAnisetteProvisioningController -fetchAnisetteDataAndProvisionIfNecessary:error: only forwards
 *   over XPC (synchronousRemoteObjectProxyWithErrorHandler:) to that service ("Remote Anisette service returned
 *   an error" is the controller relaying the native failure). Any request that reaches the native service by
 *   another route (AKSRPService/AKAppleIDSession in akd, or another process's controller over XPC) never passes
 *   the v1 hooks. v2 therefore hooks the BOTTOM of the chain, AKNativeAnisetteService, so every route is
 *   covered, and keeps the v1 controller hooks as a second net.
 *
 *   v1 also could not tell "not loaded" from "loaded, wrong hook": its only output went to /dev/console and was
 *   lost. v2 writes a log FILE, /var/mobile/Media/hakios-akd.log (AFC-visible: `afcclient get hakios-akd.log`),
 *   with a line at load time BEFORE the gate check, so one boot answers hypothesis A (AMFI/dyld refused the
 *   insert -> no file at all) vs B (file present, hooks listed, requests visible).
 *
 *   v2 fetches with a plain BSD-socket HTTP/1.0 GET instead of NSURLSession: no App Transport Security
 *   question for a cleartext http:// to a LAN IP, no run-loop/session state inside akd, a hard 3 s timeout.
 *
 * IDENTITY COHERENCE (path B, "present our freshly provisioned identity consistently", NOT another device's).
 *   Apple's anisette identity was provisioned on the host with that host identity's X-Mme-Device-Id and
 *   X-Apple-I-MD-LU (anisette-server's device.json). akd by default sends the phone's own UDID and
 *   SHA256(localUserUUID) next to it. v2 overrides those two request headers (and the same keys in the
 *   dictionary +ak_anisetteHeadersWithData: returns, which also feeds the GrandSlam cpd) with the values the
 *   anisette service returns, so mid/otp/LU/device-id all belong to the one identity. Switch off with
 *   boot-arg hakios_anisette_ids=0 (or "ids=0" in /redwood/etc/anisette-opts) to test the other way.
 *   X-MMe-Client-Info is left as the phone's unless hakios_anisette_ci=1 / "ci=1" (experiment knob).
 *
 * Gate, URL: unchanged from v1 (boot-arg hakios_appleid=1 or /redwood/etc/anisette-url exists; URL from
 * hakios_anisette_url=, the file, else http://127.0.0.1:6969/). On ANY failure every hook calls through to
 * Apple's implementation, so akd behaves exactly as without us (the clean -42410 failure).
 */
#include <arpa/inet.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

/* ---- logging: a file AFC can fetch, plus the console ---------------------------------------------- */

static int log_fd = -2;
static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;

static void
say(const char *format, ...)
{
	char line[1200];
	va_list ap;
	int n, h;
	struct timeval tv;
	struct tm tm;

	pthread_mutex_lock(&log_lock);
	if (log_fd == -2) {
		log_fd = open("/var/mobile/Media/hakios-akd.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
		if (log_fd < 0) {
			log_fd = open("/private/var/tmp/hakios-akd.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
		}
	}
	gettimeofday(&tv, NULL);
	localtime_r(&tv.tv_sec, &tm);
	h = snprintf(line, sizeof(line), "%02d:%02d:%02d.%03d akd[%d] ", tm.tm_hour, tm.tm_min, tm.tm_sec,
	    (int)(tv.tv_usec / 1000), getpid());
	va_start(ap, format);
	n = vsnprintf(line + h, sizeof(line) - (size_t)h, format, ap);
	va_end(ap);
	if (n >= 0) {
		n += h;
		if (n > (int)sizeof(line) - 1) {
			n = sizeof(line) - 1;
		}
		if (log_fd >= 0) {
			(void)write(log_fd, line, (size_t)n);
		}
		{
			int c = open("/dev/console", O_WRONLY | O_NOCTTY | O_NONBLOCK);

			if (c >= 0) {
				(void)write(c, "hakios: ", 8);
				(void)write(c, line + h, (size_t)(n - h));
				close(c);
			}
		}
	}
	pthread_mutex_unlock(&log_lock);
}

/* ---- small ObjC conveniences ---------------------------------------------------------------------- */

#define MSG0(ret, obj, sel)                 ((ret (*)(id, SEL))objc_msgSend)((id)(obj), sel_registerName(sel))
#define MSG1(ret, obj, sel, t0, a0)         ((ret (*)(id, SEL, t0))objc_msgSend)((id)(obj), sel_registerName(sel), (a0))
#define MSG2(ret, obj, sel, t0, a0, t1, a1) \
	((ret (*)(id, SEL, t0, t1))objc_msgSend)((id)(obj), sel_registerName(sel), (a0), (a1))

static id
nsstr(const char *s)
{
	Class cls = objc_getClass("NSString");

	if (cls == NULL || s == NULL) {
		return NULL;
	}
	return ((id (*)(Class, SEL, const char *))objc_msgSend)(cls, sel_registerName("stringWithUTF8String:"), s);
}

static bool
responds(id obj, SEL sel)
{
	return obj != NULL && ((BOOL (*)(id, SEL, SEL))objc_msgSend)(obj, sel_registerName("respondsToSelector:"), sel);
}

static const char *
to_cstr(id obj)
{
	SEL utf8 = sel_registerName("UTF8String");

	if (obj == NULL) {
		return NULL;
	}
	if (responds(obj, utf8)) {
		return ((const char *(*)(id, SEL))objc_msgSend)(obj, utf8);
	}
	{
		SEL string_value = sel_registerName("stringValue");

		if (responds(obj, string_value)) {
			id s = ((id (*)(id, SEL))objc_msgSend)(obj, string_value);

			if (s != NULL && responds(s, utf8)) {
				return ((const char *(*)(id, SEL))objc_msgSend)(s, utf8);
			}
		}
	}
	return NULL;
}

static id
dict_object(id dict, const char *key)
{
	id k = nsstr(key);

	if (dict == NULL || k == NULL) {
		return NULL;
	}
	return MSG1(id, dict, "objectForKey:", id, k);
}

/* ---- configuration: gate, URL, options ------------------------------------------------------------ */

static char bootargs[2048];

static void
read_bootargs(void)
{
	size_t size = sizeof(bootargs) - 1;

	if (sysctlbyname("kern.bootargs", bootargs, &size, NULL, 0) != 0) {
		bootargs[0] = '\0';
	}
}

/* KEY=VALUE as a word of SRC (boot-args or the opts file) -> out. true if found. */
static bool
word_value(const char *src, const char *key, char *out, size_t out_len)
{
	size_t klen = strlen(key);
	const char *p;

	for (p = src; (p = strstr(p, key)) != NULL; p += klen) {
		if ((p == src || p[-1] == ' ' || p[-1] == '\n') && p[klen] == '=') {
			const char *v = p + klen + 1;
			size_t i = 0;

			while (v[i] != '\0' && v[i] != ' ' && v[i] != '\n' && v[i] != '\r' && i < out_len - 1) {
				out[i] = v[i];
				i++;
			}
			out[i] = '\0';
			return i > 0;
		}
	}
	return false;
}

static bool
read_file_line(const char *path, char *out, size_t out_len)
{
	FILE *f = fopen(path, "r");
	size_t n;

	if (f == NULL) {
		return false;
	}
	if (fgets(out, (int)out_len, f) == NULL) {
		fclose(f);
		return false;
	}
	fclose(f);
	n = strlen(out);
	while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r' || out[n - 1] == ' ' || out[n - 1] == '\t')) {
		out[--n] = '\0';
	}
	return n > 0;
}

/* option NAME: boot-arg hakios_anisette_NAME=V, else "NAME=V" in /redwood/etc/anisette-opts, else dflt */
static bool
option(const char *name, bool dflt)
{
	char key[64], v[16], opts[512];

	snprintf(key, sizeof(key), "hakios_anisette_%s", name);
	if (word_value(bootargs, key, v, sizeof(v))) {
		return v[0] != '0';
	}
	if (read_file_line("/redwood/etc/anisette-opts", opts, sizeof(opts)) && word_value(opts, name, v, sizeof(v))) {
		return v[0] != '0';
	}
	return dflt;
}

static char service_url[512];
static bool opt_ids, opt_ci;
static bool observe_only;      /* v9: gate off, but install the passive logging hooks to watch native provisioning */

static void
resolve_url(void)
{
	if (word_value(bootargs, "hakios_anisette_url", service_url, sizeof(service_url))) {
		return;
	}
	if (read_file_line("/redwood/etc/anisette-url", service_url, sizeof(service_url))) {
		return;
	}
	snprintf(service_url, sizeof(service_url), "http://127.0.0.1:6969/");
}

static bool
gate_on(const char **why)
{
	char v[8];

	if (word_value(bootargs, "hakios_appleid", v, sizeof(v)) && v[0] != '0') {
		*why = "boot-arg hakios_appleid";
		return true;
	}
	if (access("/redwood/etc/anisette-url", F_OK) == 0) {
		*why = "/redwood/etc/anisette-url exists";
		return true;
	}
	return false;
}

/* ---- HTTP/1.0 GET over a plain socket, 3 s budget ------------------------------------------------- */

static int
connect_timeout(const char *host, const char *port, int ms)
{
	struct addrinfo hints, *res = NULL, *ai;
	int fd = -1;

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_NUMERICSERV;
	if (getaddrinfo(host, port, &hints, &res) != 0) {
		return -1;
	}
	for (ai = res; ai != NULL; ai = ai->ai_next) {
		struct pollfd pfd;
		int flags, err = 0;
		socklen_t len = sizeof(err);

		fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (fd < 0) {
			continue;
		}
		flags = fcntl(fd, F_GETFL, 0);
		fcntl(fd, F_SETFL, flags | O_NONBLOCK);
		if (connect(fd, ai->ai_addr, ai->ai_addrlen) != 0 && errno != EINPROGRESS) {
			close(fd);
			fd = -1;
			continue;
		}
		pfd.fd = fd;
		pfd.events = POLLOUT;
		if (poll(&pfd, 1, ms) != 1 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) {
			close(fd);
			fd = -1;
			continue;
		}
		fcntl(fd, F_SETFL, flags);
		{
			struct timeval tv = { ms / 1000, (ms % 1000) * 1000 };

			setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
			setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
		}
		{
			int one = 1;

			setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
		}
		break;
	}
	freeaddrinfo(res);
	return fd;
}

/* GET url -> body (malloc'd, NUL-terminated) or NULL. Only http://HOST[:PORT][/PATH]. */
static char *
http_get(const char *url, int *status_out)
{
	char host[256] = "", port[8] = "80", path[256] = "/", req[640];
	const char *p, *h_end, *slash;
	char *buf = NULL, *body;
	size_t cap = 8192, got = 0;
	int fd, status = 0;
	ssize_t n;

	if (strncmp(url, "http://", 7) != 0) {
		return NULL;
	}
	p = url + 7;
	slash = strchr(p, '/');
	h_end = slash != NULL ? slash : p + strlen(p);
	{
		const char *colon = memchr(p, ':', (size_t)(h_end - p));
		size_t hl = (size_t)((colon != NULL ? colon : h_end) - p);

		if (hl == 0 || hl >= sizeof(host)) {
			return NULL;
		}
		memcpy(host, p, hl);
		host[hl] = '\0';
		if (colon != NULL) {
			size_t pl = (size_t)(h_end - colon - 1);

			if (pl == 0 || pl >= sizeof(port)) {
				return NULL;
			}
			memcpy(port, colon + 1, pl);
			port[pl] = '\0';
		}
	}
	if (slash != NULL) {
		snprintf(path, sizeof(path), "%s", slash);
	}
	fd = connect_timeout(host, port, 3000);
	if (fd < 0) {
		return NULL;
	}
	n = snprintf(req, sizeof(req), "GET %s HTTP/1.0\r\nHost: %s:%s\r\nAccept: application/json\r\n"
	    "Connection: close\r\n\r\n", path, host, port);
	if (write(fd, req, (size_t)n) != n) {
		close(fd);
		return NULL;
	}
	buf = malloc(cap + 1);
	if (buf == NULL) {
		close(fd);
		return NULL;
	}
	while ((n = read(fd, buf + got, cap - got)) > 0) {
		got += (size_t)n;
		if (got == cap) {
			char *nb;

			if (cap >= 65536) {
				break;
			}
			cap *= 2;
			nb = realloc(buf, cap + 1);
			if (nb == NULL) {
				break;
			}
			buf = nb;
		}
	}
	close(fd);
	buf[got] = '\0';
	if (sscanf(buf, "HTTP/%*s %d", &status) != 1) {
		status = 0;
	}
	if (status_out != NULL) {
		*status_out = status;
	}
	body = strstr(buf, "\r\n\r\n");
	if (status != 200 || body == NULL) {
		free(buf);
		return NULL;
	}
	body += 4;
	memmove(buf, body, strlen(body) + 1);
	return buf;
}

/* ---- the anisette service ------------------------------------------------------------------------- */

/* identity values that do not change per request (device id, LU, client info): kept after the first good fetch */
static pthread_mutex_t ident_lock = PTHREAD_MUTEX_INITIALIZER;
static id ident_device_id, ident_lu, ident_client_info;     /* retained NSStrings, or NULL */
static unsigned long fetch_count, fetch_fail;

static void
remember_identity(id json)
{
	pthread_mutex_lock(&ident_lock);
	if (ident_device_id == NULL) {
		const char *a = to_cstr(dict_object(json, "X-Mme-Device-Id"));
		const char *b = to_cstr(dict_object(json, "X-Apple-I-MD-LU"));
		const char *c = to_cstr(dict_object(json, "X-MMe-Client-Info"));

		if (a != NULL && b != NULL) {
			ident_device_id = MSG0(id, nsstr(a), "retain");
			ident_lu = MSG0(id, nsstr(b), "retain");
			ident_client_info = c != NULL ? MSG0(id, nsstr(c), "retain") : NULL;
			/* lengths only: the values identify the provisioned identity and stay out of logs */
			say("identity from service: device-id %zu chars, LU %zu chars, client-info %s\n", strlen(a),
			    strlen(b), c != NULL ? "present" : "absent");
		}
	}
	pthread_mutex_unlock(&ident_lock);
}

/* the service's header dictionary (+0, autoreleased in the caller's pool) or NULL */
static id
fetch_json(const char *why)
{
	int status = 0;
	char *body = http_get(service_url, &status);
	id data, json;

	fetch_count++;
	if (body == NULL) {
		fetch_fail++;
		say("fetch (%s): %s failed (HTTP %d); calling through\n", why, service_url, status);
		return NULL;
	}
	data = ((id (*)(Class, SEL, const void *, unsigned long))objc_msgSend)(objc_getClass("NSData"),
	    sel_registerName("dataWithBytes:length:"), body, strlen(body));
	free(body);
	json = ((id (*)(Class, SEL, id, unsigned long, void *))objc_msgSend)(objc_getClass("NSJSONSerialization"),
	    sel_registerName("JSONObjectWithData:options:error:"), data, 0UL, NULL);
	if (json == NULL || !MSG1(BOOL, json, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		fetch_fail++;
		say("fetch (%s): answer was not a JSON object; calling through\n", why);
		return NULL;
	}
	remember_identity(json);
	return json;
}

static int
set_string(id obj, const char *setter, const char *ivar_name, id json, const char *key)
{
	const char *c = to_cstr(dict_object(json, key));
	id s;
	SEL sel;

	if (c == NULL || (s = nsstr(c)) == NULL) {
		return -1;
	}
	sel = sel_registerName(setter);
	if (responds(obj, sel)) {
		((void (*)(id, SEL, id))objc_msgSend)(obj, sel, s);
		return (int)strlen(c);
	}
	{
		Ivar iv = class_getInstanceVariable(object_getClass(obj), ivar_name);

		if (iv != NULL) {
			id old = object_getIvar(obj, iv);

			object_setIvar(obj, iv, MSG0(id, s, "copy"));
			if (old != NULL) {
				MSG0(void, old, "release");
			}
			return (int)strlen(c);
		}
	}
	return -1;
}

static void
set_routing_info(id obj, id json, const char *key)
{
	const char *c = to_cstr(dict_object(json, key));
	unsigned long long n;
	SEL sel;

	if (c == NULL) {
		return;
	}
	n = strtoull(c, NULL, 10);
	sel = sel_registerName("setRoutingInfo:");
	if (responds(obj, sel)) {
		((void (*)(id, SEL, unsigned long long))objc_msgSend)(obj, sel, n);
		return;
	}
	{
		Ivar iv = class_getInstanceVariable(object_getClass(obj), "_routingInfo");

		if (iv != NULL) {
			*(unsigned long long *)((char *)obj + ivar_getOffset(iv)) = n;
		}
	}
}

/* AKAnisetteData from the service, +1 retained (caller releases), or NULL. 20G81 ivars: _machineID,
 * _oneTimePassword, _routingInfo (ULL), _deviceDescription, _deviceSerialNumber, _machineData... */
static id
anisette_retained(const char *why)
{
	void *pool = objc_autoreleasePoolPush();
	Class cls = objc_getClass("AKAnisetteData");
	id json, obj = NULL;
	int mid, otp;

	json = cls != NULL ? fetch_json(why) : NULL;
	if (json != NULL) {
		obj = MSG0(id, MSG0(id, cls, "alloc"), "init");
		if (obj != NULL) {
			mid = set_string(obj, "setMachineID:", "_machineID", json, "X-Apple-I-MD-M");
			otp = set_string(obj, "setOneTimePassword:", "_oneTimePassword", json, "X-Apple-I-MD");
			set_string(obj, "setDeviceSerialNumber:", "_deviceSerialNumber", json, "X-Apple-I-SRL-NO");
			if (opt_ci) {
				set_string(obj, "setDeviceDescription:", "_deviceDescription", json, "X-MMe-Client-Info");
			}
			set_routing_info(obj, json, "X-Apple-I-MD-RINFO");
			if (mid < 0 || otp < 0) {
				say("fetch (%s): JSON lacks machineID (%d) or otp (%d); calling through\n", why, mid, otp);
				MSG0(void, obj, "release");
				obj = NULL;
			} else {
				const char *o = to_cstr(dict_object(json, "X-Apple-I-MD"));
				unsigned ck = 0;
				if (o != NULL) { for (const char *q = o; *q; q++) { ck = ck * 31u + (unsigned char)*q; } }
				say("anisette served (%s): mid %d B, otp %d B (otp_ck=%04x) [#%lu]\n", why, mid, otp, ck & 0xffff, fetch_count);
			}
		}
	}
	objc_autoreleasePoolPop(pool);
	return obj;
}

/* ---- hooks: AKNativeAnisetteService (akd executable) -- the bottom of every anisette route ---------- */

typedef void (^data_cb)(id data, id error);
typedef void (^bool_cb)(BOOL ok, id error);

static void (*o_native_fetch)(id, SEL, BOOL, data_cb);
static void
h_native_fetch(id self, SEL cmd, BOOL provision, data_cb completion)
{
	id data = anisette_retained("native fetchAnisetteDataAndProvisionIfNecessary:withCompletion:");

	if (data == NULL) {
		o_native_fetch(self, cmd, provision, completion);
		return;
	}
	if (completion != NULL) {
		completion(data, NULL);
	}
	MSG0(void, data, "release");
}

static void (*o_native_legacy)(id, SEL, id, data_cb);
static void
h_native_legacy(id self, SEL cmd, id dsid, data_cb completion)
{
	id data = anisette_retained("native legacyAnisetteDataForDSID:withCompletion:");

	if (data == NULL) {
		o_native_legacy(self, cmd, dsid, completion);
		return;
	}
	if (completion != NULL) {
		completion(data, NULL);
	}
	MSG0(void, data, "release");
}

static id (*o_native_unsafe)(id, SEL, id, id *);
static id
h_native_unsafe(id self, SEL cmd, id dsid, id *error)
{
	id data = anisette_retained("native _unsafe_anisetteDataForDSID:withError:");

	if (data == NULL) {
		return o_native_unsafe(self, cmd, dsid, error);
	}
	if (error != NULL) {
		*error = NULL;
	}
	return MSG0(id, data, "autorelease");
}

static id (*o_native_unsafe_ri)(id, SEL, id, id *);
static id
h_native_unsafe_ri(id self, SEL cmd, id dsid, id *error)
{
	id data = anisette_retained("native _unsafe_anisetteDataWithRoutingInfoForDSID:withError:");

	if (data == NULL) {
		return o_native_unsafe_ri(self, cmd, dsid, error);
	}
	if (error != NULL) {
		*error = NULL;
	}
	return MSG0(id, data, "autorelease");
}

/* provisioning is the host's job (anisette-server, already provisioned): report success iff it answers */
static void (*o_native_provision)(id, SEL, bool_cb);
static void
h_native_provision(id self, SEL cmd, bool_cb completion)
{
	id data = anisette_retained("native provisionAnisetteWithCompletion:");

	if (data == NULL) {
		o_native_provision(self, cmd, completion);
		return;
	}
	MSG0(void, data, "release");
	say("provisioning request answered from the service (no on-device ADI provisioning)\n");
	if (completion != NULL) {
		completion(YES, NULL);
	}
}

/* ---- hooks: AuthKit client side (v1's, kept as a second net) -------------------------------------- */

static id (*o_ctl_with_error)(id, SEL, id *);
static id
h_ctl_with_error(id self, SEL cmd, id *error)
{
	id data = anisette_retained("controller anisetteDataWithError:");

	if (data == NULL) {
		return o_ctl_with_error(self, cmd, error);
	}
	if (error != NULL) {
		*error = NULL;
	}
	return MSG0(id, data, "autorelease");
}

static id (*o_ctl_fetch_error)(id, SEL, BOOL, id *);
static id
h_ctl_fetch_error(id self, SEL cmd, BOOL provision, id *error)
{
	id data = anisette_retained("controller fetchAnisetteDataAndProvisionIfNecessary:error:");

	if (data == NULL) {
		return o_ctl_fetch_error(self, cmd, provision, error);
	}
	if (error != NULL) {
		*error = NULL;
	}
	return MSG0(id, data, "autorelease");
}

static void (*o_ctl_fetch_cb)(id, SEL, BOOL, data_cb);
static void
h_ctl_fetch_cb(id self, SEL cmd, BOOL provision, data_cb completion)
{
	id data = anisette_retained("controller fetchAnisetteDataAndProvisionIfNecessary:withCompletion:");

	if (data == NULL) {
		o_ctl_fetch_cb(self, cmd, provision, completion);
		return;
	}
	if (completion != NULL) {
		completion(data, NULL);
	}
	MSG0(void, data, "release");
}

/* ---- hooks: identity coherence on requests (NSMutableURLRequest(AuthKit)) ------------------------- */

static bool
identity(id *device_id, id *lu, id *client_info)
{
	if (ident_device_id == NULL) {
		void *pool = objc_autoreleasePoolPush();

		(void)fetch_json("identity");
		objc_autoreleasePoolPop(pool);
	}
	pthread_mutex_lock(&ident_lock);
	*device_id = ident_device_id;
	*lu = ident_lu;
	*client_info = ident_client_info;
	pthread_mutex_unlock(&ident_lock);
	return *device_id != NULL;
}

static void
set_header(id request, id value, const char *field)
{
	if (value != NULL) {
		MSG2(void, request, "setValue:forHTTPHeaderField:", id, value, id, nsstr(field));
	}
}

static void (*o_add_udid)(id, SEL);
static void
h_add_udid(id self, SEL cmd)
{
	id dev, lu, ci;

	o_add_udid(self, cmd);
	if (opt_ids && identity(&dev, &lu, &ci)) {
		set_header(self, dev, "X-Mme-Device-Id");
	}
}

static void (*o_add_lu)(id, SEL);
static void
h_add_lu(id self, SEL cmd)
{
	id dev, lu, ci;

	o_add_lu(self, cmd);
	if (opt_ids && identity(&dev, &lu, &ci)) {
		set_header(self, lu, "X-Apple-I-MD-LU");
	}
}

static void (*o_add_ci)(id, SEL);
static void
h_add_ci(id self, SEL cmd)
{
	id dev, lu, ci;

	o_add_ci(self, cmd);
	if (opt_ci && identity(&dev, &lu, &ci)) {
		set_header(self, ci, "X-MMe-Client-Info");
	}
}

/* +[NSMutableURLRequest ak_anisetteHeadersWithData:] -> NSDictionary; also the source of the GrandSlam cpd */
static id (*o_headers_with_data)(id, SEL, id);
static id
h_headers_with_data(id self, SEL cmd, id data)
{
	id dict = o_headers_with_data(self, cmd, data);
	id dev, lu, ci, m;

	if (dict == NULL || !(opt_ids || opt_ci) || !identity(&dev, &lu, &ci)) {
		return dict;
	}
	m = MSG0(id, dict, "mutableCopy");
	if (opt_ids) {
		MSG2(void, m, "setObject:forKey:", id, lu, id, nsstr("X-Apple-I-MD-LU"));
		MSG2(void, m, "setObject:forKey:", id, dev, id, nsstr("X-Mme-Device-Id"));
	}
	if (opt_ci && ci != NULL) {
		MSG2(void, m, "setObject:forKey:", id, ci, id, nsstr("X-MMe-Client-Info"));
	}
	return MSG0(id, m, "autorelease");
}

/* ---- installation --------------------------------------------------------------------------------- */

/* ---- v3: read what Apple's GrandSlam actually returns (akd makes that request itself, outside the anisette hooks) ---- */

/* +[NSPropertyListSerialization propertyListWithData:options:format:error:]: akd parses the GsService2 response as a
 * plist. A GrandSlam reply carries a "Status" dict {ec: error code, em: message, hsc: http-ish}, either at the top or
 * under "Response". Logged only when that Status is present, so akd's many other plists stay quiet. */
static id (*o_plist)(Class, SEL, id, unsigned long, void *, id *);

/* v9: dump a plist dict's keys with value class and length, masking long/secret values -- used to print the shape
 * of the midStart/midFinishProvisioning request and response bodies without logging the base64 attestation blobs. */
static void
dump_plist_keys(id dict, const char *label)
{
	id keys = MSG0(id, dict, "allKeys");
	unsigned long n = keys != NULL ? ((unsigned long (*)(id, SEL))objc_msgSend)(keys, sel_registerName("count")) : 0;

	say("%s: %lu keys\n", label, n);
	for (unsigned long i = 0; i < n; i++) {
		id k = ((id (*)(id, SEL, unsigned long))objc_msgSend)(keys, sel_registerName("objectAtIndex:"), i);
		const char *ks = to_cstr(k);
		id v = MSG1(id, dict, "objectForKey:", id, k);
		const char *cn = v != NULL ? class_getName(object_getClass(v)) : "nil";
		const char *vs = to_cstr(v);

		if (ks == NULL) {
			continue;
		}
		if (vs != NULL && strlen(vs) <= 24) {
			say("  %s[%s] = %s\n", label, ks, vs);          /* short scalar: a code, flag, RINFO, em string */
		} else if (vs != NULL) {
			say("  %s[%s] = <%s, %d B>\n", label, ks, cn, (int)strlen(vs));
		} else if (v != NULL && MSG1(BOOL, v, "isKindOfClass:", Class, objc_getClass("NSData"))) {
			say("  %s[%s] = <NSData, %lu B>\n", label, ks,
			    ((unsigned long (*)(id, SEL))objc_msgSend)(v, sel_registerName("length")));
		} else {
			say("  %s[%s] = (%s)\n", label, ks, cn);        /* nested dict / array */
		}
	}
}

/* v9: the mid provisioning bodies are plists {Response:{spim|ptm|tk|X-Apple-I-MD-RINFO|Status:{ec,em,...}}}. When a
 * parsed plist is one (a provisioning marker, or a Status that is NOT the GrandSlam SRP one), dump the Response shape
 * and its Status -- this is the midFinishProvisioning 401 body akd otherwise prints as <private>, i.e. WHY it is
 * refused. GrandSlam SRP responses (sp/spd/c/M2) are left to log_gs_status. */
static void
log_provisioning(id dict)
{
	id resp, status;
	bool has_prov, is_gs;

	if (dict == NULL || !MSG1(BOOL, dict, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		return;
	}
	resp = dict_object(dict, "Response");
	if (resp == NULL || !MSG1(BOOL, resp, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		return;
	}
	has_prov = dict_object(resp, "spim") != NULL || dict_object(resp, "ptm") != NULL ||
	    dict_object(resp, "tk") != NULL || dict_object(resp, "cpim") != NULL ||
	    dict_object(resp, "X-Apple-I-MD-RINFO") != NULL;
	is_gs = dict_object(resp, "sp") != NULL || dict_object(resp, "spd") != NULL ||
	    dict_object(resp, "c") != NULL || dict_object(resp, "M2") != NULL;
	status = dict_object(resp, "Status");
	if (!has_prov && !(status != NULL && !is_gs)) {
		return;
	}
	say("--- mid provisioning response (has_prov=%d) ---\n", has_prov);
	dump_plist_keys(resp, "prov-resp");
	if (status != NULL && MSG1(BOOL, status, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		dump_plist_keys(status, "prov-status");
	}
}

static void
log_gs_status(id dict)
{
	id status, resp;

	if (dict == NULL || !MSG1(BOOL, dict, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		return;
	}
	status = dict_object(dict, "Status");
	if (status == NULL) {
		resp = dict_object(dict, "Response");
		if (resp != NULL && MSG1(BOOL, resp, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
			status = dict_object(resp, "Status");
		}
	}
	if (status == NULL || !MSG1(BOOL, status, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		return;
	}
	{
		const char *ec = to_cstr(dict_object(status, "ec"));
		const char *em = to_cstr(dict_object(status, "em"));
		const char *hsc = to_cstr(dict_object(status, "hsc"));
		const char *au = to_cstr(dict_object(status, "au"));
		id rd = resp != NULL ? resp : dict;      /* the Response dict (or the root) to scan for stage markers */
		/* which SRP/stage fields are present (names only, no values): tells init vs complete vs 2FA vs downstream */
		static const char *keys[] = { "sp", "c", "s", "B", "i", "iteration", "salt", "spd", "np", "M2", "sk", "sc",
		    "au", "ec", "GsIdmsToken", "adsid", "prk", "t" };
		char present[256]; int n = 0;
		const char *op = to_cstr(dict_object(rd, "o"));

		for (unsigned k = 0; k < sizeof(keys) / sizeof(keys[0]) && n < (int)sizeof(present) - 16; k++) {
			if (dict_object(rd, keys[k]) != NULL) {
				n += snprintf(present + n, sizeof(present) - (size_t)n, "%s%s", n ? "," : "", keys[k]);
			}
		}
		present[n] = '\0';
		say("GrandSlam Status: ec=%s hsc=%s%s%s em=%s | op=%s fields=[%s]\n", ec ? ec : "?", hsc ? hsc : "-",
		    au ? " au=" : "", au ? au : "", em ? em : "(none)", op ? op : "?", present);
	}
}

static id
h_plist(Class self, SEL cmd, id data, unsigned long opts, void *fmt, id *err)
{
	id r = o_plist(self, cmd, data, opts, fmt, err);

	log_gs_status(r);
	log_provisioning(r);
	return r;
}

/* +[NSError errorWithDomain:code:userInfo:]: the auth failure ("cannot be completed at this time") is an NSError;
 * logged only for the Apple-ID / AuthKit / GrandSlam domains, so it is the sign-in error, not every NSError akd makes. */
static id (*o_nserror)(Class, SEL, id, long, id);

/* v8: log akd's full o=init cpd (every key), masking only real secrets, so RINFO/client-info/flags can be diffed
 * field-by-field against a known-good AltStore/SideStore init request -- the hsc=500 may be one malformed/extra field. */
static bool
cpd_secret(const char *k)
{
	return k != NULL && (strcmp(k, "X-Apple-I-MD") == 0 || strcmp(k, "X-Apple-I-MD-M") == 0 ||
	    strcmp(k, "X-Apple-I-MD-LU") == 0 || strstr(k, "tkn") != NULL || strstr(k, "oken") != NULL ||
	    strstr(k, " assword") != NULL);
}

static void
dump_cpd(id cpd)
{
	id keys = MSG0(id, cpd, "allKeys");
	unsigned long n = keys != NULL ? ((unsigned long (*)(id, SEL))objc_msgSend)(keys, sel_registerName("count")) : 0;

	say("init cpd: %lu keys\n", n);
	for (unsigned long i = 0; i < n; i++) {
		id k = ((id (*)(id, SEL, unsigned long))objc_msgSend)(keys, sel_registerName("objectAtIndex:"), i);
		const char *ks = to_cstr(k);
		id v = MSG1(id, cpd, "objectForKey:", id, k);
		const char *vs = to_cstr(v);
		const char *cn = v != NULL ? class_getName(object_getClass(v)) : "nil";

		if (ks == NULL) {
			continue;
		}
		if (cpd_secret(ks)) {
			say("  cpd[%s] = <%s, %d B>\n", ks, cn, vs ? (int)strlen(vs) : 0);
		} else if (vs != NULL) {
			say("  cpd[%s] = %s\n", ks, vs);
		} else {
			say("  cpd[%s] = (%s)\n", ks, cn);
		}
	}
}

/* +[NSPropertyListSerialization dataWithPropertyList:format:options:error:]: akd serializes its GsService2 request body
 * as a plist. When that plist is a GrandSlam request (has a "Request" dict with an "o"), log the operation so each
 * response Status can be tied to init / complete / a follow-up. Everything else akd serializes stays quiet. */
static id (*o_ser)(Class, SEL, id, unsigned long, unsigned long, id *);

static id
h_ser(Class self, SEL cmd, id plist, unsigned long fmt, unsigned long opts, id *err)
{
	if (plist != NULL && MSG1(BOOL, plist, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
		id req = dict_object(plist, "Request");
		const char *op = NULL;

		if (req != NULL && MSG1(BOOL, req, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
			op = to_cstr(dict_object(req, "o"));
		}
		if (op == NULL) {
			op = to_cstr(dict_object(plist, "o"));     /* some requests put it at the top */
		}
		/* v9: the midStart/midFinishProvisioning request bodies have no "o"; log their shape (finish carries cpim) */
		if (op == NULL && req != NULL &&
		    (dict_object(req, "cpim") != NULL || dict_object(req, "spim") != NULL)) {
			say("--- mid provisioning request ---\n");
			dump_plist_keys(req, "prov-req");
		}
		if (op != NULL) {
			/* what akd actually transmits in this request's cpd: the client-time and OTP it is using right now.
			 * If the time is current at o=init, a stale/cached one-time-password is ruled out as the -80044 cause. */
			id cpd = req != NULL ? dict_object(req, "cpd") : NULL;
			const char *ct = NULL, *o = NULL;
			unsigned ck = 0;

			if (cpd != NULL && MSG1(BOOL, cpd, "isKindOfClass:", Class, objc_getClass("NSDictionary"))) {
				ct = to_cstr(dict_object(cpd, "X-Apple-I-Client-Time"));
				o = to_cstr(dict_object(cpd, "X-Apple-I-MD"));
				if (o != NULL) { for (const char *q = o; *q; q++) { ck = ck * 31u + (unsigned char)*q; } }
			}
			say("GrandSlam request: o=%s cpd_client_time=%s cpd_otp_ck=%04x\n", op, ct ? ct : "(none)", ck & 0xffff);
			/* v7: akd caches anisette and sends a ~30 s-stale OTP/time at init, and the request's cpd is immutable.
			 * To rule staleness out for good: fetch fresh now, deep-copy plist > Request > cpd into mutable dicts,
			 * overwrite the time-sensitive triple (OTP, machine-data, client-time) with the fresh matched set, and
			 * serialize THAT copy. Identity fields (MID-LU, RINFO) are left as akd set them. */
			if (cpd != NULL && req != NULL && !observe_only && strcmp(op, "init") == 0) {
				dump_cpd(cpd);
				id fresh = fetch_json("init-refresh");

				if (fresh != NULL) {
					id ncpd = MSG1(id, objc_getClass("NSMutableDictionary"), "dictionaryWithDictionary:", id, cpd);
					id nreq = MSG1(id, objc_getClass("NSMutableDictionary"), "dictionaryWithDictionary:", id, req);
					id nplist = MSG1(id, objc_getClass("NSMutableDictionary"), "dictionaryWithDictionary:", id, plist);
					static const char *ks[] = { "X-Apple-I-MD", "X-Apple-I-MD-M", "X-Apple-I-Client-Time" };
					int done = 0;

					if (ncpd != NULL && nreq != NULL && nplist != NULL) {
						for (unsigned k = 0; k < sizeof(ks) / sizeof(ks[0]); k++) {
							id v = dict_object(fresh, ks[k]);
							if (v != NULL) { MSG2(void, ncpd, "setObject:forKey:", id, v, id, nsstr(ks[k])); done++; }
						}
						MSG2(void, nreq, "setObject:forKey:", id, ncpd, id, nsstr("cpd"));
						MSG2(void, nplist, "setObject:forKey:", id, nreq, id, nsstr("Request"));
						const char *nt = to_cstr(dict_object(fresh, "X-Apple-I-Client-Time"));
						say("  init cpd refreshed via copy: %d fields, new client_time=%s\n", done, nt ? nt : "?");
						return o_ser(self, cmd, nplist, fmt, opts, err);
					}
					say("  init cpd refresh: mutable copy failed; left as-is\n");
				} else {
					say("  init cpd refresh: fresh fetch failed; left as-is\n");
				}
			}
		}
	}
	return o_ser(self, cmd, plist, fmt, opts, err);
}

static id
h_nserror(Class self, SEL cmd, id domain, long code, id info)
{
	const char *d = to_cstr(domain);

	if (d != NULL && (strstr(d, "AppleIDAuthentication") || strstr(d, "AuthKit") || strstr(d, "AKAuthentication") ||
	    strstr(d, "GSAuth") || strstr(d, "AKAppleID") || strstr(d, "GrandSlam"))) {
		say("NSError %s code %ld\n", d, code);
	}
	return o_nserror(self, cmd, domain, code, info);
}

static int hooked, wanted;

static bool
hook(const char *class_name, bool meta, const char *selector, IMP replacement, IMP *orig)
{
	Class cls = objc_getClass(class_name);
	Method m;

	wanted++;
	if (cls != NULL && meta) {
		cls = object_getClass((id)cls);
	}
	/* for a class method cls is already the metaclass, whose instance methods are the class methods */
	m = cls != NULL ? class_getInstanceMethod(cls, sel_registerName(selector)) : NULL;
	if (m == NULL) {
		say("  %c[%s %s] not found; not hooked\n", meta ? '+' : '-', class_name, selector);
		return false;
	}
	*orig = method_setImplementation(m, replacement);
	hooked++;
	say("  hooked %c[%s %s]\n", meta ? '+' : '-', class_name, selector);
	return true;
}

static void
install(void)
{
	hooked = wanted = 0;
	/* bottom of the chain: akd's own native service */
	hook("AKNativeAnisetteService", false, "fetchAnisetteDataAndProvisionIfNecessary:withCompletion:",
	    (IMP)h_native_fetch, (IMP *)&o_native_fetch);
	hook("AKNativeAnisetteService", false, "legacyAnisetteDataForDSID:withCompletion:",
	    (IMP)h_native_legacy, (IMP *)&o_native_legacy);
	hook("AKNativeAnisetteService", false, "_unsafe_anisetteDataForDSID:withError:",
	    (IMP)h_native_unsafe, (IMP *)&o_native_unsafe);
	hook("AKNativeAnisetteService", false, "_unsafe_anisetteDataWithRoutingInfoForDSID:withError:",
	    (IMP)h_native_unsafe_ri, (IMP *)&o_native_unsafe_ri);
	hook("AKNativeAnisetteService", false, "provisionAnisetteWithCompletion:",
	    (IMP)h_native_provision, (IMP *)&o_native_provision);
	/* client side (AuthKit) */
	hook("AKAnisetteProvisioningController", false, "anisetteDataWithError:",
	    (IMP)h_ctl_with_error, (IMP *)&o_ctl_with_error);
	hook("AKAnisetteProvisioningController", false, "fetchAnisetteDataAndProvisionIfNecessary:error:",
	    (IMP)h_ctl_fetch_error, (IMP *)&o_ctl_fetch_error);
	hook("AKAnisetteProvisioningController", false, "fetchAnisetteDataAndProvisionIfNecessary:withCompletion:",
	    (IMP)h_ctl_fetch_cb, (IMP *)&o_ctl_fetch_cb);
	/* identity coherence on outgoing requests */
	hook("NSMutableURLRequest", false, "ak_addDeviceUDIDHeader", (IMP)h_add_udid, (IMP *)&o_add_udid);
	hook("NSMutableURLRequest", false, "ak_addLocalUserUUIDHashHeader", (IMP)h_add_lu, (IMP *)&o_add_lu);
	hook("NSMutableURLRequest", false, "ak_addClientInfoHeader", (IMP)h_add_ci, (IMP *)&o_add_ci);
	hook("NSMutableURLRequest", true, "ak_anisetteHeadersWithData:", (IMP)h_headers_with_data,
	    (IMP *)&o_headers_with_data);
	/* v3: see Apple's GrandSlam answer and the sign-in error */
	hook("NSPropertyListSerialization", true, "propertyListWithData:options:format:error:", (IMP)h_plist, (IMP *)&o_plist);
	hook("NSPropertyListSerialization", true, "dataWithPropertyList:format:options:error:", (IMP)h_ser, (IMP *)&o_ser);
	hook("NSError", true, "errorWithDomain:code:userInfo:", (IMP)h_nserror, (IMP *)&o_nserror);
	say("%d of %d hooks installed\n", hooked, wanted);
}

/* v9: passive hooks only -- see what akd's own native adid provisioning does, injecting nothing. */
static void
install_observe(void)
{
	hooked = wanted = 0;
	hook("NSPropertyListSerialization", true, "propertyListWithData:options:format:error:", (IMP)h_plist, (IMP *)&o_plist);
	hook("NSPropertyListSerialization", true, "dataWithPropertyList:format:options:error:", (IMP)h_ser, (IMP *)&o_ser);
	hook("NSError", true, "errorWithDomain:code:userInfo:", (IMP)h_nserror, (IMP *)&o_nserror);
	say("observe-only: %d of %d passive hooks (native adid provisioning left untouched)\n", hooked, wanted);
}

__attribute__((constructor))
static void
hakios_akd_init(void)
{
	const char *name = getprogname();
	const char *why = "";

	if (name == NULL || strcmp(name, "akd") != 0) {
		return;
	}
	read_bootargs();
	/* logged BEFORE the gate: the file's existence alone proves dyld inserted us into akd */
	say("libhakios.akd v9 loaded (uid %d)\n", getuid());
	if (!gate_on(&why)) {
		char v[8];

		if ((word_value(bootargs, "hakios_akd_observe", v, sizeof(v)) && v[0] != '0') ||
		    access("/redwood/etc/akd-observe", F_OK) == 0) {
			observe_only = true;
			say("observe-only mode: logging akd's native adid provisioning (no anisette injected)\n");
			install_observe();
			return;
		}
		say("gate off (no hakios_appleid boot-arg, no /redwood/etc/anisette-url): native behaviour\n");
		return;
	}
	resolve_url();
	opt_ids = option("ids", true);
	opt_ci = option("ci", false);
	say("Apple ID anisette ON (%s); service %s; ids=%d ci=%d\n", why, service_url, opt_ids, opt_ci);
	install();
}
