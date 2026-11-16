/*
 * libhakios.akd.dylib: code of ours inside Apple's akd (AuthKit daemon), so that iOS's native Apple ID
 * sign-in works on this phone (build.sh's INJECT).
 *
 * The phone's own akd speaks Apple's current GrandSlam protocol fine and reaches gsa.apple.com; the one
 * thing it cannot do is make Anisette. On an iPhone the Anisette headers (X-Apple-I-MD-M, X-Apple-I-MD
 * and so on, which GrandSlam uses to prove the request comes from a real Apple device) are produced by
 * the ADI libraries, which are backed by FairPlay and a real device identity. There is none here, so
 * AuthKit's provisioning fails and sign-in stops:
 *
 *   akd(AuthKit): authkit/anisette-for-request -> authkit/fetch-anisette -> BEGIN FetchAnisette
 *   akd: Kicking off Anisette provisioning....
 *   akd: Provisioning failed. No Anisette for you today! Error: AKAnisetteError Code=-8004
 *        { NSUnderlyingError = AKUnderlyingADIErrors Code=-42410 }   (-42410: native ADI has no identity)
 *
 * This library supplies the Anisette instead, from a small HTTP service of ours (the sanctioned way
 * AltStore/SideStore use: Apple's own ADI libraries + a provisioned identity run elsewhere, serving the
 * X-Apple-I-* / X-Mme-* headers as JSON). We hook AuthKit's synchronous Anisette fetch,
 * -[AKAnisetteProvisioningController anisetteDataWithError:], and return an AKAnisetteData built from the
 * headers our service hands back -- so AuthKit signs the GrandSlam request with valid Anisette and never
 * reaches the native ADI that fails with -42410. (fetchAnisetteDataAndProvisionIfNecessary:error:, the
 * "fetch-anisette" step above, is hooked the same way.)
 *
 * On ANY doubt -- the gate off, the service unreachable, the class or method not found, the JSON malformed
 * -- the hook calls through to the original implementation, so akd behaves exactly as it does today (the
 * clean -42410 failure). A malformed AKAnisetteData would be worse than that failure.
 *
 * Gate (so normal boots are untouched): the boot-arg hakios_appleid=1 (read from sysctl kern.bootargs, the
 * same mechanism inject_sb.c uses), or, as a fallback, the existence of /redwood/etc/anisette-url.
 *
 * Model: src/inject_sb.c and src/inject_lockdown.c (plain-C dylib, __attribute__((constructor)), say() to
 * /dev/console, ObjC classes reached through the runtime; APIs it needs declared minimally against the SDK).
 */
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

/* objc runtime autorelease-pool scope (not declared by objc/runtime.h in this SDK) */
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

/* where what we say goes: /dev/console if akd's sandbox lets us open it, else its standard error */
static int say_fd = -1;

static void
say(const char *format, ...)
{
	char line[1024];
	va_list ap;
	int n;

	if (say_fd < 0) {
		say_fd = open("/dev/console", O_WRONLY | O_NOCTTY);
		if (say_fd < 0) {
			say_fd = 2;
		}
	}
	va_start(ap, format);
	n = vsnprintf(line, sizeof(line), format, ap);
	va_end(ap);
	if (n < 0) {
		return;
	}
	if (n > (int)sizeof(line) - 1) {
		n = sizeof(line) - 1;
	}
	(void)write(say_fd, line, (size_t)n);
}

/* ---- small ObjC conveniences (plain C through objc_msgSend) -------------------------------------- */

#define MSG0(ret, obj, sel)                 ((ret (*)(id, SEL))objc_msgSend)((id)(obj), sel_registerName(sel))
#define MSG1(ret, obj, sel, t0, a0)         ((ret (*)(id, SEL, t0))objc_msgSend)((id)(obj), sel_registerName(sel), (a0))

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

/* a C string for an id that is an NSString (UTF8String) or an NSNumber (stringValue -> UTF8String); the
 * pointer is the autorelease pool's and is used at once by the caller */
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
	return ((id (*)(id, SEL, id))objc_msgSend)(dict, sel_registerName("objectForKey:"), k);
}

/*
 * Set an NSString field of the AKAnisetteData from JSON[key]: through the setter if the object has it
 * (so its copy/retain semantics are honoured), else straight into the object ivar if that is present,
 * else nothing. Never throws (no KVC): a field this build of AKAnisetteData does not have is simply
 * skipped. Returns the length of the C value set, or -1 if nothing was set.
 */
static int
set_string(id obj, const char *setter, const char *ivar_name, id json, const char *key)
{
	id value = dict_object(json, key);
	const char *c = to_cstr(value);
	id s;
	SEL sel;

	if (c == NULL) {
		return -1;
	}
	s = nsstr(c);
	if (s == NULL) {
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
			/* an object ivar: keep a copy, as the (copy) property would */
			id copy = MSG0(id, s, "copy");

			object_setIvar(obj, iv, copy);
			return (int)strlen(c);
		}
	}
	return -1;
}

/*
 * Set the scalar routingInfo (unsigned long long) from JSON[key] (a decimal string or number): through
 * setRoutingInfo: if present, else into the _routingInfo ivar. Skipped if the field is absent.
 */
static void
set_routing_info(id obj, id json, const char *key)
{
	id value = dict_object(json, key);
	const char *c = to_cstr(value);
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

/*
 * Build an AKAnisetteData from the header dictionary our service returned. The real ivars of this build's
 * AKAnisetteData (iOS 16.6.1, 20G81 arm64e; from the AuthKit library in the dyld shared cache) are
 * _machineID, _oneTimePassword, _routingInfo (unsigned long long), _deviceDescription, _deviceSerialNumber,
 * _machineData, _machineDataStyle, _machineDataSyncState -- there is no localUserID or deviceUniqueIdentifier
 * field on AKAnisetteData itself (those are request-level headers), so set_string for them simply finds no
 * setter/ivar and does nothing. Returns a +0 (autoreleased) object, as a non-alloc/new/copy method returns.
 */
static id
build_anisette(id json)
{
	Class cls = objc_getClass("AKAnisetteData");
	id obj;
	int mid_len, otp_len;

	if (cls == NULL) {
		say("hakios: akd: AKAnisetteData class not found; calling through\n");
		return NULL;
	}
	obj = MSG0(id, cls, "alloc");
	obj = MSG0(id, obj, "init");
	if (obj == NULL) {
		return NULL;
	}
	mid_len = set_string(obj, "setMachineID:", "_machineID", json, "X-Apple-I-MD-M");
	otp_len = set_string(obj, "setOneTimePassword:", "_oneTimePassword", json, "X-Apple-I-MD");
	set_string(obj, "setDeviceDescription:", "_deviceDescription", json, "X-MMe-Client-Info");
	set_string(obj, "setDeviceSerialNumber:", "_deviceSerialNumber", json, "X-Apple-I-SRL-NO");
	/* not AKAnisetteData fields on this build -- tried defensively, skipped cleanly if absent */
	set_string(obj, "setLocalUserID:", "_localUserID", json, "X-Apple-I-MD-LU");
	set_string(obj, "setDeviceUniqueIdentifier:", "_deviceUniqueIdentifier", json, "X-Mme-Device-Id");
	set_routing_info(obj, json, "X-Apple-I-MD-RINFO");
	if (mid_len < 0 || otp_len < 0) {
		/* without a machine ID and a one-time password the object cannot sign a request: refuse it */
		say("hakios: akd: anisette JSON missing machineID (%d) or otp (%d); calling through\n", mid_len, otp_len);
		MSG0(void, obj, "release");
		return NULL;
	}
	say("hakios: akd: installed anisette -- machineID %d bytes, otp %d bytes\n", mid_len, otp_len);
	return MSG0(id, obj, "autorelease");
}

/* ---- our anisette service ------------------------------------------------------------------------ */

/* a boot-arg's value, "KEY=VALUE" at the start of kern.bootargs or after a space, into out (word up to
 * the next space). true if found. */
static bool
boot_arg_value(const char *key, char *out, size_t out_len)
{
	char args[2048] = "";
	size_t size = sizeof(args) - 1;
	size_t klen = strlen(key);
	char *p;

	if (sysctlbyname("kern.bootargs", args, &size, NULL, 0) != 0) {
		return false;
	}
	for (p = args; (p = strstr(p, key)) != NULL; p += klen) {
		if ((p == args || p[-1] == ' ') && p[klen] == '=') {
			const char *v = p + klen + 1;
			size_t i = 0;

			while (v[i] != '\0' && v[i] != ' ' && i < out_len - 1) {
				out[i] = v[i];
				i++;
			}
			out[i] = '\0';
			return i > 0;
		}
	}
	return false;
}

/* where our anisette service is: boot-arg hakios_anisette_url=, else /redwood/etc/anisette-url (trimmed),
 * else the default. */
static void
resolve_url(char *out, size_t out_len)
{
	FILE *f;

	if (boot_arg_value("hakios_anisette_url", out, out_len) && out[0] != '\0') {
		return;
	}
	f = fopen("/redwood/etc/anisette-url", "r");
	if (f != NULL) {
		char line[1024] = "";

		if (fgets(line, sizeof(line), f) != NULL) {
			size_t n = strlen(line);

			while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ' ||
			    line[n - 1] == '\t')) {
				line[--n] = '\0';
			}
			if (n > 0) {
				snprintf(out, out_len, "%s", line);
				fclose(f);
				return;
			}
		}
		fclose(f);
	}
	snprintf(out, out_len, "http://127.0.0.1:6969/");
}

/*
 * Fetch the anisette headers from our service and build an AKAnisetteData, synchronously, with a short
 * timeout. The one-time password is time-based, so this runs on every call (never cached). Returns a +0
 * AKAnisetteData, or NULL on any failure (the caller then calls through to the native ADI).
 */
static id
fetch_anisette(void)
{
	char url[1024] = "";
	void *pool = objc_autoreleasePoolPush();
	id result = NULL;
	__block id body = NULL;      /* NSData, retained inside the completion block (runs on another thread) */
	dispatch_semaphore_t done;
	id nsurl, request, session, task, json;
	Class NSURL = objc_getClass("NSURL");
	Class NSURLRequest = objc_getClass("NSMutableURLRequest");
	Class NSURLSession = objc_getClass("NSURLSession");
	Class NSJSONSerialization = objc_getClass("NSJSONSerialization");

	if (NSURL == NULL || NSURLRequest == NULL || NSURLSession == NULL || NSJSONSerialization == NULL) {
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	resolve_url(url, sizeof(url));
	nsurl = MSG1(id, NSURL, "URLWithString:", id, nsstr(url));
	if (nsurl == NULL) {
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	request = MSG1(id, NSURLRequest, "requestWithURL:", id, nsurl);
	if (request == NULL) {
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	MSG1(void, request, "setTimeoutInterval:", double, 3.0);
	session = MSG0(id, NSURLSession, "sharedSession");
	if (session == NULL) {
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	done = dispatch_semaphore_create(0);
	{
		void (^handler)(id, id, id) = ^(id data, __unused id response, __unused id error) {
			if (data != NULL) {
				body = MSG0(id, data, "retain");
			}
			dispatch_semaphore_signal(done);
		};
		task = ((id (*)(id, SEL, id, void *))objc_msgSend)(session,
		    sel_registerName("dataTaskWithRequest:completionHandler:"), request, (void *)handler);
	}
	if (task == NULL) {
		dispatch_release(done);
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	MSG0(void, task, "resume");
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC)) != 0) {
		say("hakios: akd: anisette service %s timed out; calling through\n", url);
		dispatch_release(done);
		objc_autoreleasePoolPop(pool);
		return NULL;      /* the retained body (if any) leaks on timeout only; that path is rare */
	}
	dispatch_release(done);
	if (body == NULL) {
		say("hakios: akd: anisette service %s returned nothing; calling through\n", url);
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	json = ((id (*)(Class, SEL, id, unsigned long, void *))objc_msgSend)(NSJSONSerialization,
	    sel_registerName("JSONObjectWithData:options:error:"), body, 0UL, NULL);
	MSG0(void, body, "release");
	if (json == NULL || !((BOOL (*)(id, SEL, Class))objc_msgSend)(json, sel_registerName("isKindOfClass:"),
	    objc_getClass("NSDictionary"))) {
		say("hakios: akd: anisette service %s: answer was not a JSON object; calling through\n", url);
		objc_autoreleasePoolPop(pool);
		return NULL;
	}
	result = build_anisette(json);
	/* keep the result alive across the pool pop, then hand it back autoreleased */
	if (result != NULL) {
		result = MSG0(id, result, "retain");
	}
	objc_autoreleasePoolPop(pool);
	if (result != NULL) {
		result = MSG0(id, result, "autorelease");
	}
	return result;
}

/* ---- the hooks ----------------------------------------------------------------------------------- */

/* -[AKAnisetteProvisioningController anisetteDataWithError:] -> AKAnisetteData *  (the request-signing fetch) */
static id (*orig_anisette_with_error)(id, SEL, void *);

static id
hook_anisette_with_error(id self, SEL _cmd, void *error_out)
{
	id data = fetch_anisette();

	if (data != NULL) {
		if (error_out != NULL) {
			*(void **)error_out = NULL;
		}
		return data;
	}
	return orig_anisette_with_error(self, _cmd, error_out);
}

/* -[AKAnisetteProvisioningController fetchAnisetteDataAndProvisionIfNecessary:error:] -> AKAnisetteData * */
static id (*orig_fetch_and_provision)(id, SEL, BOOL, void *);

static id
hook_fetch_and_provision(id self, SEL _cmd, BOOL provision, void *error_out)
{
	id data = fetch_anisette();

	if (data != NULL) {
		if (error_out != NULL) {
			*(void **)error_out = NULL;
		}
		return data;
	}
	return orig_fetch_and_provision(self, _cmd, provision, error_out);
}

/* replace -[CLASS SELECTOR] with REPLACEMENT, saving the original into *orig. Returns true if hooked. */
static bool
hook_method(const char *class_name, const char *selector, IMP replacement, IMP *orig)
{
	Class cls = objc_getClass(class_name);
	Method method = cls != NULL ? class_getInstanceMethod(cls, sel_registerName(selector)) : NULL;

	if (method == NULL) {
		say("hakios: akd: -[%s %s] not found; not hooked (native behaviour kept)\n", class_name, selector);
		return false;
	}
	*orig = method_setImplementation(method, replacement);
	say("hakios: akd: hooked -[%s %s]\n", class_name, selector);
	return true;
}

/*
 * Is the Apple ID gate on? Boot-arg hakios_appleid=1 (read from sysctl kern.bootargs, as the rest of the
 * userland reads its arguments), or the fallback file /redwood/etc/anisette-url exists. STATE: the primary
 * gate is the boot-arg; the file is the fallback, and it doubles as the service URL.
 */
static bool
gate_on(const char **why)
{
	char args[2048] = "";
	size_t size = sizeof(args) - 1;

	if (sysctlbyname("kern.bootargs", args, &size, NULL, 0) == 0) {
		char *p;

		for (p = args; (p = strstr(p, "hakios_appleid=")) != NULL; p += 15) {
			if ((p == args || p[-1] == ' ') && p[15] != '0') {
				*why = "boot-arg hakios_appleid=1";
				return true;
			}
		}
	}
	if (access("/redwood/etc/anisette-url", F_OK) == 0) {
		*why = "/redwood/etc/anisette-url exists";
		return true;
	}
	return false;
}

__attribute__((constructor))
static void
hakios_akd_init(void)
{
	const char *name = getprogname();
	const char *why = "";
	char url[1024] = "";

	if (name == NULL || strcmp(name, "akd") != 0) {
		return;           /* injected only into akd (build.sh INJECT); nothing else is touched */
	}
	if (!gate_on(&why)) {
		return;           /* a normal boot: do nothing, akd behaves exactly as Apple's */
	}
	resolve_url(url, sizeof(url));
	say("hakios: akd: Apple ID anisette on (%s); service %s\n", why, url);
	hook_method("AKAnisetteProvisioningController", "anisetteDataWithError:",
	    (IMP)hook_anisette_with_error, (IMP *)&orig_anisette_with_error);
	hook_method("AKAnisetteProvisioningController", "fetchAnisetteDataAndProvisionIfNecessary:error:",
	    (IMP)hook_fetch_and_provision, (IMP *)&orig_fetch_and_provision);
}
