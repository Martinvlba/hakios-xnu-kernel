/*
 * libhakios.log.dylib: a stock Darwin daemon's os_log lines on its stderr (the console), for finding out why it does
 * what it does where nothing reads the system log. Inserted into the jobs named with it in INJECT (userland/build.sh;
 * off by default: INJECT_LOG="wifid" adds it for a run). Every log level is answered "enabled", each call's format
 * and arguments (strings, numbers, objects by their description) are printed once per distinct line (digits apart),
 * at most 6000 lines. Only calls made from the program itself and from the libraries whose path has one of the
 * words of the environment variable HAKIOS_LOG_FROM (default: every image) are printed.
 *
 * A diagnostic: nothing of the daemon's behaviour is changed but the cost of making the lines.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <os/log.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static bool
wanted_image(void *dso)
{
	static const void *yes[24], *no[96];
	static int n_yes, n_no;
	const char *words = getenv("HAKIOS_LOG_FROM");
	Dl_info where = { 0 };
	bool want = true;

	for (int i = 0; i < n_yes; i++) {
		if (yes[i] == dso) {
			return true;
		}
	}
	for (int i = 0; i < n_no; i++) {
		if (no[i] == dso) {
			return false;
		}
	}
	if (words != NULL && words[0] != 0) {
		char copy[256], *word, *save = NULL;

		want = false;
		if (dladdr(dso, &where) != 0 && where.dli_fname != NULL) {
			snprintf(copy, sizeof(copy), "%s", words);
			for (word = strtok_r(copy, " ,", &save); word != NULL; word = strtok_r(NULL, " ,", &save)) {
				want |= strstr(where.dli_fname, word) != NULL;
			}
		}
	}
	if (want && n_yes < 24) {
		yes[n_yes++] = dso;
	} else if (!want && n_no < 96) {
		no[n_no++] = dso;
	}
	return want;
}

static void
said(void *dso, os_log_type_t type, const char *format, const uint8_t *buf, uint32_t size)
{
	static uint64_t seen[6000];
	static int count;
	static int busy;
	char line[1700];
	int at;
	uint64_t hash = 1469598103934665603ULL;
	struct timespec now;

	if (format == NULL || count >= 6000 || busy || !wanted_image(dso)) {
		return;
	}
	busy = 1;       /* (a description below may log itself) */
	at = snprintf(line, sizeof(line), "(%d) %.200s", (int)type, format);
	/* the buffer: summary, count, then {descriptor, size, bytes} for each argument */
	if (buf != NULL && size >= 2) {
		uint32_t off = 2;

		for (int i = 0; i < buf[1] && off + 2 <= size && at < (int)sizeof(line) - 80; i++) {
			uint8_t kind = buf[off] >> 4, length = buf[off + 1];

			if (off + 2 + length > size) {
				break;
			}
			if (kind == 2 && length == 8) {
				const char *text = NULL;

				memcpy(&text, buf + off + 2, 8);
				at += snprintf(line + at, sizeof(line) - (size_t)at, " | %.1200s", text ? text : "(null)");
			} else if (kind == 4 && length == 8) {
				CFTypeRef object = NULL;

				memcpy(&object, buf + off + 2, 8);
				if (object == NULL) {
					at += snprintf(line + at, sizeof(line) - (size_t)at, " | (nil)");
				} else {
					CFStringRef description = CFCopyDescription(object);
					char text[220] = "?";

					if (description != NULL) {
						CFStringGetCString(description, text, sizeof(text), kCFStringEncodingUTF8);
						CFRelease(description);
					}
					for (char *c = text; *c != 0; c++) {
						if (*c == '\n') {
							*c = ' ';
						}
					}
					at += snprintf(line + at, sizeof(line) - (size_t)at, " | %s", text);
				}
			} else if (kind == 0 && length == 4) {
				int32_t v;

				memcpy(&v, buf + off + 2, 4);
				at += snprintf(line + at, sizeof(line) - (size_t)at, " | %d", v);
			} else if (kind == 0 && length == 8) {
				long long v;

				memcpy(&v, buf + off + 2, 8);
				at += snprintf(line + at, sizeof(line) - (size_t)at, " | %lld", v);
			}
			off += 2u + length;
		}
	}
	for (const char *c = line; *c != 0; c++) {
		if (*c < '0' || *c > '9') {
			hash = (hash ^ (uint8_t)*c) * 1099511628211ULL;
		}
	}
	for (int i = 0; i < count; i++) {
		if (seen[i] == hash) {
			busy = 0;
			return;
		}
	}
	seen[count++] = hash;
	clock_gettime(CLOCK_MONOTONIC, &now);
	fprintf(stderr, "hakios: log: %s[%d] @%ld.%01ld %s\n", getprogname(), getpid(), (long)now.tv_sec, now.tv_nsec / 100000000, line);
	busy = 0;
}

extern void _os_log_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_error_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_fault_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_debug_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);

#define LOG_WATCHED(name)                                                                                              \
	static void watched##name(void *dso, os_log_t log, os_log_type_t type, const char *format, uint8_t *buf,       \
	    uint32_t size)                                                                                             \
	{                                                                                                              \
		said(dso, type, format, buf, size);                                                                    \
		name(dso, log, type, format, buf, size);                                                               \
	}                                                                                                              \
	__attribute__((used)) static struct {                                                                          \
		const void *ours, *theirs;                                                                             \
	} interpose##name __attribute__((section("__DATA,__interpose"))) = { (const void *)watched##name,              \
		                                                             (const void *)name };

static bool
every_level(__unused os_log_t log, __unused os_log_type_t type)
{
	return true;
}

__attribute__((used)) static struct {
	const void *ours, *theirs;
} interpose_enabled __attribute__((section("__DATA,__interpose"))) = { (const void *)every_level, (const void *)os_log_type_enabled };

/*
 * The calls made from inside the shared cache (a daemon's frameworks: most of wifid's lines) are bound when the cache
 * is built and pass the interposing above by: for them the logging function itself is entered through a jump written
 * over its first four instructions (a private copy of the page), which are kept in a trampoline that goes on into it;
 * and "is this level enabled" is answered yes for everybody the same way.
 */
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <sys/mman.h>

typedef void (*log_impl)(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
static log_impl impl_trampoline;

static bool
jump_written(uintptr_t at, void *to)
{
	uintptr_t target = (uintptr_t)__builtin_ptrauth_strip(to, 0);
	uint32_t code[4];
	vm_address_t page = at & ~(vm_address_t)(vm_page_size - 1);

	code[0] = 0x58000050;           /* ldr x16, #8 */
	code[1] = 0xd61f0200;           /* br x16 */
	memcpy(&code[2], &target, 8);
	if (vm_protect(mach_task_self(), page, 2 * vm_page_size, FALSE, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) != KERN_SUCCESS) {
		return false;
	}
	memcpy((void *)at, code, sizeof(code));
	vm_protect(mach_task_self(), page, 2 * vm_page_size, FALSE, VM_PROT_READ | VM_PROT_EXECUTE);
	sys_icache_invalidate((void *)at, sizeof(code));
	return true;
}

static void
impl_hooked(void *dso, os_log_t log, os_log_type_t type, const char *format, uint8_t *buf, uint32_t size)
{
	said(dso, type, format, buf, size);
	impl_trampoline(dso, log, type, format, buf, size);
}

__attribute__((constructor)) static void
hook_the_cache(void)
{
	uintptr_t impl = (uintptr_t)__builtin_ptrauth_strip(dlsym(RTLD_DEFAULT, "_os_log_impl"), 0);
	uintptr_t enabled = (uintptr_t)__builtin_ptrauth_strip(dlsym(RTLD_DEFAULT, "os_log_type_enabled"), 0);
	uint32_t first[4], *trampoline;
	uintptr_t back;

	if (getenv("HAKIOS_LOG_NO_HOOK") != NULL || impl == 0) {
		return;
	}
	memcpy(first, (void *)impl, sizeof(first));
	for (int i = 0; i < 4; i++) {           /* (nothing there may depend on where it runs) */
		uint32_t w = first[i];

		if ((w & 0x1f000000) == 0x10000000 || (w & 0x7c000000) == 0x14000000 || (w & 0xff000010) == 0x54000000 ||
		    (w & 0x7e000000) == 0x34000000 || (w & 0x7e000000) == 0x36000000 || (w & 0x3b000000) == 0x18000000) {
			fprintf(stderr, "hakios: log: %s: _os_log_impl begins %08x %08x %08x %08x: not hooked\n", getprogname(), first[0], first[1],
			    first[2], first[3]);
			return;
		}
	}
	trampoline = mmap(NULL, 0x4000, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	if (trampoline == MAP_FAILED) {
		return;
	}
	memcpy(trampoline, first, sizeof(first));
	trampoline[4] = 0x58000050;
	trampoline[5] = 0xd61f0200;
	back = impl + 16;
	memcpy(&trampoline[6], &back, 8);
	if (mprotect(trampoline, 0x4000, PROT_READ | PROT_EXEC) != 0) {
		return;
	}
	sys_icache_invalidate(trampoline, 32);
	impl_trampoline = (log_impl)__builtin_ptrauth_sign_unauthenticated((void *)trampoline, 0, 0);
	if (!jump_written(impl, (void *)impl_hooked)) {
		return;
	}
	/*
	 * "Is this level enabled" answered yes for the cache's own callers too (their debug and info lines): the jump is
	 * written over the function only where four instructions of it are there to write over -- a plain branch at its
	 * start is followed once, a return within the first four leaves it alone (HAKIOS_LOG_ALL_LEVELS=0 too).
	 */
	if (enabled != 0 && getenv("HAKIOS_LOG_DEFAULT_LEVELS") == NULL) {
		uint32_t w[4];
		bool fits = true;

		memcpy(w, (void *)enabled, sizeof(w));
		if ((w[0] & 0xfc000000) == 0x14000000) {        /* b somewhere: there */
			int32_t offset = (int32_t)(w[0] << 6) >> 6;

			enabled += (intptr_t)offset * 4;
			memcpy(w, (void *)enabled, sizeof(w));
		}
		for (int i = 0; i < 4; i++) {
			fits &= w[i] != 0xd65f03c0 && w[i] != 0xd65f0fff && w[i] != 0xd65f0bff && (w[i] & 0xfc000000) != 0x14000000;
		}
		fprintf(stderr, "hakios: log: %s: os_log_type_enabled begins %08x %08x %08x %08x: %s\n", getprogname(), w[0], w[1], w[2], w[3],
		    fits ? "answered yes for everybody" : "left alone");
		if (fits) {
			(void)jump_written(enabled, (void *)every_level);
		}
	}
	fprintf(stderr, "hakios: log: %s[%d]: the logging function hooked in the shared cache\n", getprogname(), getpid());
}

/* (_os_log_impl is not interposed as well: dlsym would then answer with our own function, and the jump was written
 * over that -- wifid died of it, 11-19; the hook above sees every caller anyway) */
LOG_WATCHED(_os_log_error_impl)
LOG_WATCHED(_os_log_fault_impl)
LOG_WATCHED(_os_log_debug_impl)
