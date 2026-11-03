/*
 * ptp_responder: the phone's photos for a computer over USB, the way an iPhone shows them -- a PTP camera (ISO 15740)
 * with one read-only storage, "Internal Storage", holding /private/var/mobile/Media/DCIM (job com.hakios.ptp). On an
 * iPhone ptpcamerad does this through Apple's USB stack; here the kernel gives the PTP interface's endpoints as
 * /dev/hakios-ptp (the Pixel 8a: devices/akita/drivers/usb/hakios.akita_ptp.c).
 *
 * Only for a computer the user trusts, as on an iPhone: lockdownd posts com.apple.mobile.lockdown.trusted_ptp_attached
 * (and trusted_host_attached) when a paired computer is attached, host_detached when it goes. Until then a session
 * opens but shows no storage, and anything asked of an object is refused (AccessDenied); the trust goes with the
 * computer, and with the USB link.
 *
 * Operations: GetDeviceInfo, OpenSession, CloseSession, GetStorageIDs, GetStorageInfo, GetObjectHandles, GetObjectInfo,
 * GetObject, GetPartialObject; anything else OperationNotSupported. The objects are DCIM's folders and files, read again
 * at every OpenSession and when the root is listed. A data phase goes out so that every transfer but the last is a
 * multiple of 512 bytes, and a zero-length packet ends one that is (the computer reads to the first short packet).
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <notify.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DEVICE          "/dev/hakios-ptp"
#define MEDIA           "/private/var/mobile/Media"
#define STORAGE_ID      0x00010001u
#define MAX_OBJECTS     8192
#define STAGE           (64 * 1024)

/* container types, operations, responses (ISO 15740) */
enum { T_COMMAND = 1, T_DATA = 2, T_RESPONSE = 3 };
enum {
	OP_GET_DEVICE_INFO = 0x1001, OP_OPEN_SESSION = 0x1002, OP_CLOSE_SESSION = 0x1003, OP_GET_STORAGE_IDS = 0x1004,
	OP_GET_STORAGE_INFO = 0x1005, OP_GET_OBJECT_HANDLES = 0x1007, OP_GET_OBJECT_INFO = 0x1008, OP_GET_OBJECT = 0x1009,
	OP_GET_PARTIAL_OBJECT = 0x101b,
};
enum {
	RC_OK = 0x2001, RC_GENERAL_ERROR = 0x2002, RC_SESSION_NOT_OPEN = 0x2003, RC_OPERATION_NOT_SUPPORTED = 0x2005,
	RC_INVALID_STORAGE_ID = 0x2008, RC_INVALID_OBJECT_HANDLE = 0x2009, RC_ACCESS_DENIED = 0x200f,
	RC_SESSION_ALREADY_OPEN = 0x201e,
};

struct object {
	uint32_t parent;                /* 0: in the storage's root */
	int dir;
	uint64_t size;
	time_t mtime;
	char name[256];
	char path[1024];
};

static struct object objects[MAX_OBJECTS + 1];   /* handle n is objects[n]; 0 unused */
static uint32_t nobjects;
static int fd = -1, session_open, trusted;
static int tok_ptp = -1, tok_host = -1, tok_detached = -1;

/* ---- building the answers ---------------------------------------------------------------------------------- */

struct buf {
	uint8_t *p;
	size_t n, cap;
};

static void
put(struct buf *b, const void *d, size_t n)
{
	if (b->n + n > b->cap) {
		b->cap = (b->n + n) * 2 + 256;
		b->p = realloc(b->p, b->cap);
	}
	memcpy(b->p + b->n, d, n);
	b->n += n;
}
static void put8(struct buf *b, uint8_t v) { put(b, &v, 1); }
static void put16(struct buf *b, uint16_t v) { put(b, &v, 2); }         /* little-endian, as the phone is */
static void put32(struct buf *b, uint32_t v) { put(b, &v, 4); }
static void put64(struct buf *b, uint64_t v) { put(b, &v, 8); }

/* a PTP string: the count of UTF-16 units with the terminating 0, then the units; an empty one is a single 0 */
static void
putstr(struct buf *b, const char *s)
{
	size_t n = strlen(s);

	if (n == 0) {
		put8(b, 0);
		return;
	}
	if (n > 254) {
		n = 254;
	}
	put8(b, (uint8_t)(n + 1));
	for (size_t i = 0; i < n; i++) {
		put16(b, (uint8_t)s[i]);        /* (file names are ASCII on a camera roll: IMG_0001.JPG) */
	}
	put16(b, 0);
}

static void
put16array(struct buf *b, const uint16_t *v, uint32_t n)
{
	put32(b, n);
	for (uint32_t i = 0; i < n; i++) {
		put16(b, v[i]);
	}
}

static void
putdate(struct buf *b, time_t t)
{
	char s[32];
	struct tm tm;

	gmtime_r(&t, &tm);
	strftime(s, sizeof(s), "%Y%m%dT%H%M%S", &tm);
	putstr(b, s);
}

/* ---- sending ----------------------------------------------------------------------------------------------- */

/* everything of one phase goes through here: whole 512-byte multiples out until the end, then the rest */
struct out {
	uint8_t stage[STAGE];
	size_t n;
	uint64_t total;
	int failed;
};

static void
out_flush(struct out *o, int last)
{
	size_t n = last ? o->n : o->n - o->n % 512;

	if (n > 0 && !o->failed && write(fd, o->stage, n) != (ssize_t)n) {
		o->failed = 1;
	}
	memmove(o->stage, o->stage + n, o->n - n);
	o->n -= n;
	if (last && !o->failed && o->total % 512 == 0 && write(fd, o->stage, 0) != 0) {
		o->failed = 1;          /* the zero-length packet that ends a phase of whole packets */
	}
}

static void
out_put(struct out *o, const void *d, size_t n)
{
	const uint8_t *p = d;

	o->total += n;
	while (n > 0) {
		size_t room = STAGE - o->n, k = n < room ? n : room;

		memcpy(o->stage + o->n, p, k);
		o->n += k;
		p += k;
		n -= k;
		if (o->n == STAGE) {
			out_flush(o, 0);
		}
	}
}

static void
header(uint8_t h[12], uint32_t length, uint16_t type, uint16_t code, uint32_t transaction)
{
	memcpy(h, &length, 4);
	memcpy(h + 4, &type, 2);
	memcpy(h + 6, &code, 2);
	memcpy(h + 8, &transaction, 4);
}

static void
send_data(uint16_t code, uint32_t transaction, const struct buf *b)
{
	static struct out o;
	uint8_t h[12];

	o.n = 0;
	o.total = 0;
	o.failed = 0;
	header(h, (uint32_t)(12 + b->n), T_DATA, code, transaction);
	out_put(&o, h, 12);
	out_put(&o, b->p, b->n);
	out_flush(&o, 1);
}

static void
respond(uint16_t code, uint32_t transaction, const uint32_t *params, int nparams)
{
	uint8_t c[12 + 20];

	header(c, (uint32_t)(12 + 4 * nparams), T_RESPONSE, code, transaction);
	memcpy(c + 12, params, 4 * nparams);
	if (write(fd, c, 12 + 4 * nparams) < 0) {
		return;
	}
	if ((12 + 4 * nparams) % 512 == 0) {
		(void)write(fd, c, 0);
	}
}

/* ---- the objects ------------------------------------------------------------------------------------------- */

static void
scan_dir(const char *path, uint32_t parent, int depth)
{
	DIR *d = opendir(path);
	struct dirent *e;

	if (d == NULL) {
		return;
	}
	while ((e = readdir(d)) != NULL && nobjects < MAX_OBJECTS) {
		struct object *o;
		struct stat st;
		char full[1024];

		if (e->d_name[0] == '.') {
			continue;
		}
		snprintf(full, sizeof(full), "%s/%s", path, e->d_name);
		if (stat(full, &st) != 0 || (!S_ISDIR(st.st_mode) && !S_ISREG(st.st_mode))) {
			continue;
		}
		o = &objects[++nobjects];
		o->parent = parent;
		o->dir = S_ISDIR(st.st_mode);
		o->size = o->dir ? 0 : (uint64_t)st.st_size;
		o->mtime = st.st_mtime;
		strlcpy(o->name, e->d_name, sizeof(o->name));
		strlcpy(o->path, full, sizeof(o->path));
		if (o->dir && depth < 4) {
			scan_dir(full, nobjects, depth + 1);
		}
	}
	closedir(d);
}

static void
scan(void)
{
	struct object *o;
	struct stat st;

	nobjects = 0;
	if (stat(MEDIA "/DCIM", &st) != 0) {
		return;
	}
	o = &objects[++nobjects];               /* DCIM itself, in the root, as an iPhone shows it */
	o->parent = 0;
	o->dir = 1;
	o->size = 0;
	o->mtime = st.st_mtime;
	strlcpy(o->name, "DCIM", sizeof(o->name));
	strlcpy(o->path, MEDIA "/DCIM", sizeof(o->path));
	scan_dir(MEDIA "/DCIM", 1, 0);
}

static uint16_t
format_of(const struct object *o)
{
	const char *dot = strrchr(o->name, '.');

	if (o->dir) {
		return 0x3001;                  /* Association: a folder */
	}
	if (dot == NULL) {
		return 0x3000;
	}
	if (!strcasecmp(dot, ".jpg") || !strcasecmp(dot, ".jpeg")) {
		return 0x3801;                  /* EXIF/JPEG */
	}
	if (!strcasecmp(dot, ".png")) {
		return 0x380b;
	}
	if (!strcasecmp(dot, ".gif")) {
		return 0x3807;
	}
	if (!strcasecmp(dot, ".tif") || !strcasecmp(dot, ".tiff")) {
		return 0x380d;
	}
	if (!strcasecmp(dot, ".mov")) {
		return 0x300d;                  /* QuickTime */
	}
	if (!strcasecmp(dot, ".mp4") || !strcasecmp(dot, ".m4v")) {
		return 0x300b;                  /* MPEG */
	}
	return 0x3000;                          /* undefined: HEIC, AAE and the rest, given as they are */
}

/* ---- the trust ---------------------------------------------------------------------------------------------- */

static void
look_at_trust(void)
{
	int posted = 0;

	if (tok_detached >= 0 && notify_check(tok_detached, &posted) == NOTIFY_STATUS_OK && posted && trusted) {
		trusted = 0;
		fprintf(stderr, "ptp_responder: the computer went: no access\n");
	}
	posted = 0;
	if (tok_ptp >= 0 && notify_check(tok_ptp, &posted) == NOTIFY_STATUS_OK && posted && !trusted) {
		trusted = 1;
		fprintf(stderr, "ptp_responder: a trusted computer is attached: the photos are shown\n");
	}
	posted = 0;
	if (tok_host >= 0 && notify_check(tok_host, &posted) == NOTIFY_STATUS_OK && posted && !trusted) {
		trusted = 1;
		fprintf(stderr, "ptp_responder: a trusted computer is attached (host): the photos are shown\n");
	}
}

/* ---- the operations ----------------------------------------------------------------------------------------- */

static void
device_info(uint32_t transaction)
{
	static const uint16_t ops[] = { OP_GET_DEVICE_INFO, OP_OPEN_SESSION, OP_CLOSE_SESSION, OP_GET_STORAGE_IDS,
		                        OP_GET_STORAGE_INFO, OP_GET_OBJECT_HANDLES, OP_GET_OBJECT_INFO, OP_GET_OBJECT,
		                        OP_GET_PARTIAL_OBJECT };
	static const uint16_t formats[] = { 0x3000, 0x3001, 0x3801, 0x3807, 0x380b, 0x380d, 0x300b, 0x300d };
	struct buf b = { 0 };

	put16(&b, 100);                         /* standard version 1.00 */
	put32(&b, 0);                           /* no vendor extension */
	put16(&b, 0);
	putstr(&b, "");
	put16(&b, 0);                           /* functional mode: standard */
	put16array(&b, ops, sizeof(ops) / sizeof(ops[0]));
	put16array(&b, NULL, 0);                /* events */
	put16array(&b, NULL, 0);                /* device properties */
	put16array(&b, NULL, 0);                /* capture formats: none, it is not a camera here */
	put16array(&b, formats, sizeof(formats) / sizeof(formats[0]));
	putstr(&b, "Apple Inc.");
	putstr(&b, "iPhone");
	putstr(&b, "16.5");
	putstr(&b, "");                         /* serial number: lockdownd's, not ours to give */
	send_data(OP_GET_DEVICE_INFO, transaction, &b);
	free(b.p);
	respond(RC_OK, transaction, NULL, 0);
}

static void
storage_info(uint32_t transaction, uint32_t id)
{
	struct buf b = { 0 };
	struct statfs fs;

	if (!trusted || id != STORAGE_ID) {
		respond(RC_INVALID_STORAGE_ID, transaction, NULL, 0);
		return;
	}
	if (statfs(MEDIA, &fs) != 0) {
		memset(&fs, 0, sizeof(fs));
	}
	put16(&b, 0x0003);                      /* fixed RAM */
	put16(&b, 0x0002);                      /* a hierarchical file system */
	put16(&b, 0x0001);                      /* read only, without deletion */
	put64(&b, (uint64_t)fs.f_blocks * fs.f_bsize);
	put64(&b, (uint64_t)fs.f_bavail * fs.f_bsize);
	put32(&b, 0xffffffff);
	putstr(&b, "Internal Storage");
	putstr(&b, "");
	send_data(OP_GET_STORAGE_INFO, transaction, &b);
	free(b.p);
	respond(RC_OK, transaction, NULL, 0);
}

static void
object_handles(uint32_t transaction, const uint32_t *p)
{
	struct buf b = { 0 };
	uint32_t storage = p[0], parent = p[2], count = 0, at;

	if (!trusted || (storage != STORAGE_ID && storage != 0xffffffff)) {
		respond(trusted ? RC_INVALID_STORAGE_ID : RC_ACCESS_DENIED, transaction, NULL, 0);
		return;
	}
	if (parent == 0xffffffff || nobjects == 0) {
		scan();                         /* the root asked for: what is there now */
	}
	put32(&b, 0);
	at = 0;
	for (uint32_t h = 1; h <= nobjects; h++) {
		/* parent 0xffffffff: the root's; 0: all of them; else that folder's */
		if ((parent == 0xffffffff && objects[h].parent == 0) || parent == 0 || objects[h].parent == parent) {
			put32(&b, h);
			count++;
		}
	}
	memcpy(b.p + at, &count, 4);
	send_data(OP_GET_OBJECT_HANDLES, transaction, &b);
	free(b.p);
	respond(RC_OK, transaction, NULL, 0);
}

static void
object_info(uint32_t transaction, uint32_t h)
{
	struct buf b = { 0 };
	const struct object *o;

	if (!trusted) {
		respond(RC_ACCESS_DENIED, transaction, NULL, 0);
		return;
	}
	if (h == 0 || h > nobjects) {
		respond(RC_INVALID_OBJECT_HANDLE, transaction, NULL, 0);
		return;
	}
	o = &objects[h];
	put32(&b, STORAGE_ID);
	put16(&b, format_of(o));
	put16(&b, 0x0001);                      /* protection: read only */
	put32(&b, o->size > 0xffffffffULL ? 0xffffffff : (uint32_t)o->size);
	put16(&b, 0);                           /* no thumbnail */
	put32(&b, 0);
	put32(&b, 0);
	put32(&b, 0);
	put32(&b, 0);                           /* image size and depth: not looked into */
	put32(&b, 0);
	put32(&b, 0);
	put32(&b, o->parent);
	put16(&b, o->dir ? 0x0001 : 0);         /* a generic folder */
	put32(&b, 0);
	put32(&b, 0);
	putstr(&b, o->name);
	putdate(&b, o->mtime);
	putdate(&b, o->mtime);
	putstr(&b, "");
	send_data(OP_GET_OBJECT_INFO, transaction, &b);
	free(b.p);
	respond(RC_OK, transaction, NULL, 0);
}

/* an object's bytes from offset on, count of them (all: ~0), as one data phase */
static void
object_data(uint16_t code, uint32_t transaction, uint32_t h, uint64_t offset, uint64_t count)
{
	static struct out o;
	static uint8_t chunk[STAGE];
	const struct object *ob;
	uint8_t hdr[12];
	uint64_t left;
	int f;

	if (!trusted) {
		respond(RC_ACCESS_DENIED, transaction, NULL, 0);
		return;
	}
	if (h == 0 || h > nobjects || objects[h].dir) {
		respond(RC_INVALID_OBJECT_HANDLE, transaction, NULL, 0);
		return;
	}
	ob = &objects[h];
	f = open(ob->path, O_RDONLY);
	if (f < 0) {
		respond(RC_GENERAL_ERROR, transaction, NULL, 0);
		return;
	}
	left = offset >= ob->size ? 0 : ob->size - offset;
	if (count < left) {
		left = count;
	}
	o.n = 0;
	o.total = 0;
	o.failed = 0;
	header(hdr, 12 + left > 0xffffffffULL ? 0xffffffff : (uint32_t)(12 + left), T_DATA, code, transaction);
	out_put(&o, hdr, 12);
	lseek(f, (off_t)offset, SEEK_SET);
	while (left > 0 && !o.failed) {
		ssize_t n = read(f, chunk, left < sizeof(chunk) ? (size_t)left : sizeof(chunk));

		if (n <= 0) {
			memset(chunk, 0, sizeof(chunk));        /* the file shrank: the promised length still goes */
			n = left < sizeof(chunk) ? (ssize_t)left : (ssize_t)sizeof(chunk);
		}
		out_put(&o, chunk, (size_t)n);
		left -= (uint64_t)n;
	}
	out_flush(&o, 1);
	close(f);
	if (code == OP_GET_PARTIAL_OBJECT) {
		uint32_t sent = (uint32_t)(o.total - 12);

		respond(RC_OK, transaction, &sent, 1);
	} else {
		respond(RC_OK, transaction, NULL, 0);
	}
}

static void
command(uint16_t code, uint32_t transaction, const uint32_t *p, int nparams)
{
	look_at_trust();
	if (code == OP_GET_DEVICE_INFO) {
		device_info(transaction);
		return;
	}
	if (code == OP_OPEN_SESSION) {
		if (session_open) {
			respond(RC_SESSION_ALREADY_OPEN, transaction, NULL, 0);
			return;
		}
		session_open = 1;
		scan();
		fprintf(stderr, "ptp_responder: a session (%u objects in DCIM; %s)\n", nobjects,
		    trusted ? "a trusted computer" : "not trusted yet: no storage shown");
		respond(RC_OK, transaction, NULL, 0);
		return;
	}
	if (!session_open) {
		respond(RC_SESSION_NOT_OPEN, transaction, NULL, 0);
		return;
	}
	switch (code) {
	case OP_CLOSE_SESSION:
		session_open = 0;
		respond(RC_OK, transaction, NULL, 0);
		break;
	case OP_GET_STORAGE_IDS: {
		struct buf b = { 0 };

		put32(&b, trusted ? 1 : 0);     /* untrusted: no storage at all, as an iPhone that was not trusted */
		if (trusted) {
			put32(&b, STORAGE_ID);
		}
		send_data(code, transaction, &b);
		free(b.p);
		respond(RC_OK, transaction, NULL, 0);
		break;
	}
	case OP_GET_STORAGE_INFO:
		storage_info(transaction, nparams > 0 ? p[0] : 0);
		break;
	case OP_GET_OBJECT_HANDLES: {
		uint32_t q[3] = { nparams > 0 ? p[0] : 0, nparams > 1 ? p[1] : 0, nparams > 2 ? p[2] : 0 };

		object_handles(transaction, q);
		break;
	}
	case OP_GET_OBJECT_INFO:
		object_info(transaction, nparams > 0 ? p[0] : 0);
		break;
	case OP_GET_OBJECT:
		object_data(code, transaction, nparams > 0 ? p[0] : 0, 0, ~0ULL);
		break;
	case OP_GET_PARTIAL_OBJECT:
		object_data(code, transaction, nparams > 0 ? p[0] : 0, nparams > 1 ? p[1] : 0, nparams > 2 ? p[2] : 0);
		break;
	default:
		respond(RC_OPERATION_NOT_SUPPORTED, transaction, NULL, 0);
		break;
	}
}

int
main(void)
{
	static uint8_t in[STAGE];
	size_t have = 0;

	notify_register_check("com.apple.mobile.lockdown.trusted_ptp_attached", &tok_ptp);
	notify_register_check("com.apple.mobile.lockdown.trusted_host_attached", &tok_host);
	notify_register_check("com.apple.mobile.lockdown.host_detached", &tok_detached);
	look_at_trust();                        /* (the first check of each only arms it) */
	trusted = 0;
	for (;;) {
		if (fd < 0) {
			fd = open(DEVICE, O_RDWR);
			if (fd < 0) {
				sleep(2);               /* the node comes a few seconds into the kernel's start */
				continue;
			}
		}
		ssize_t n = read(fd, in + have, sizeof(in) - have);

		if (n <= 0) {
			if (session_open || trusted) {
				fprintf(stderr, "ptp_responder: the computer is gone (%s)\n", n < 0 ? strerror(errno) : "end");
			}
			session_open = 0;
			trusted = 0;
			have = 0;
			sleep(1);               /* ENXIO while no computer has the interface configured */
			continue;
		}
		have += (size_t)n;
		while (have >= 12) {
			uint32_t length, transaction, params[5] = { 0 };
			uint16_t type, code;

			memcpy(&length, in, 4);
			memcpy(&type, in + 4, 2);
			memcpy(&code, in + 6, 2);
			memcpy(&transaction, in + 8, 4);
			if (length < 12 || length > sizeof(in)) {
				have = 0;               /* not a container we can follow: start again with the next */
				break;
			}
			if (have < length) {
				break;
			}
			if (type == T_COMMAND) {
				int nparams = (int)((length - 12) / 4);

				if (nparams > 5) {
					nparams = 5;
				}
				memcpy(params, in + 12, 4 * (size_t)nparams);
				command(code, transaction, params, nparams);
			}                               /* a data phase from the computer: nothing here takes one */
			memmove(in, in + length, have - length);
			have -= length;
		}
	}
}
