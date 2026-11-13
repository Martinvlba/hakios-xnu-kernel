/*
 * MiniAppleKeyStore: a stand-in for Apple's AppleKeyStore (the kernel side of the keybag,
 * which on real hardware talks to the Secure Enclave), as seen by MobileKeyBag.framework
 * and libaks.
 *
 * A SOFTWARE KEYBAG (2026-11-13). There is no Secure Enclave here, so the keys are the kernel's own:
 *   - a key for every protection class (32 random bytes each), made the first time the store is asked for anything
 *     and kept in a file on the data volume (kAKSBagPath), each wrapped (AES key wrap, RFC 3394);
 *   - without a passcode every class key is wrapped by the bag's "device key", which lies in the same file: the
 *     wraps then keep nothing from whoever can read that file -- they are what makes the answers real ones;
 *   - with a passcode (kAKSChangeSecret, from keybagd when Settings sets one) the classes that iOS ties to the
 *     passcode -- A, B, C and the keychain's WhenUnlocked, AfterFirstUnlock and their ThisDeviceOnly kinds -- are
 *     wrapped by a key made from it: PBKDF2-HMAC-SHA256 over a salt, then HMAC'd with the device key. The device is
 *     LOCKED after a start until the passcode is given (kAKSUnlockDevice); locking it again (kAKSLockDevice) drops
 *     the keys of the classes that are only for an unlocked device, and keeps the after-first-unlock ones;
 *   - wrong passcodes are counted in the file and bring iOS's waits (from the sixth: 1, 5, 15, 60 minutes).
 * What a key of a class wraps (kAKSWrapKey: the keychain's item keys) is wrapped with RFC 3394 too -- 8 bytes more
 * than the key, which is what callers size their buffers for. Items wrapped by the stand-in this replaces (a mark
 * and a fixed pattern) are still unwrapped.
 * NOT hardware-bound: the device key is a file's content. Trusty's KeyMint and Gatekeeper are where it belongs on
 * the Pixel 8a; the file's layout leaves room for that (the device key is the only thing to move).
 * Bags other than the system's, escrow and backup bags, the stash, personas: refused, logged once.
 *
 * The protocol was worked out from the user-space side of Darwin userland 16.5: selectors, and the
 * DER-encoded answers ("SET of SEQUENCE { UTF8String key, INTEGER value }").
 */

#include <IOKit/IOLib.h>
#include <IOKit/IOKitKeys.h>
#include <IOKit/IOService.h>
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <sys/systm.h>
#include <sys/proc.h>
#include <sys/fcntl.h>
#include <sys/kauth.h>
#include <sys/vnode.h>
#include <kern/clock.h>
#include <libkern/crypto/aes.h>
#include <libkern/crypto/sha2.h>
#include <pexpert/pexpert.h>

#include "MiniDrivers.h"

extern "C" {
extern struct vnode * rootvnode;
extern void read_random(void * buffer, u_int numBytes);
}

enum {
	kAKSOpenCheck           = 0x00,
	kAKSCreateBag           = 0x02,         /* in: 6 scalars; out: a handle */
	kAKSSaveBag             = 0x03,         /* in: handle; out: the bag, to be kept by keybagd */
	kAKSUnloadBag           = 0x04,
	kAKSSetSystem           = 0x05,         /* in: handle, ...: this bag is the system's */
	kAKSLoadBag             = 0x06,         /* in: a bag saved before; out: a handle */
	kAKSGetLockState        = 0x07,         /* in: handle; out: state flags */
	kAKSLockDevice          = 0x08,
	kAKSUnlockDevice        = 0x09,
	kAKSWrapKey             = 0x0a,         /* in: handle, class; a key; out: the class, the key wrapped */
	kAKSUnwrapKey           = 0x0b,         /* in: handle, class; a wrapped key; out: the key */
	kAKSUnlockBag           = 0x0c,
	kAKSGetSystem           = 0x0e,         /* out: the handle of the system keybag */
	kAKSChangeSecret        = 0x0f,         /* in: handle, option, 0xff, old pointer, old length, new pointer, new length,
	                                         * (the Secure Element's: pointer, length), 2 more; out: the generation */
	kAKSGetDeviceState      = 0x11,         /* in: handle; out: DER */
	kAKSGetBagUUID          = 0x17,         /* out: 16 bytes */
	kAKSAssertHold          = 0x1a,
	kAKSAssertDrop          = 0x1b,
	kAKSSetConfiguration    = 0x1e,
	kAKSGetConfiguration    = 0x1f,         /* in: handle; out: DER (a big buffer, so out of line) */
	kAKSLockDeviceNow       = 0x20,
	kAKSGetExtendedState    = 0x23,         /* as kAKSGetDeviceState */
	kAKSGeneration          = 0x28,
	kAKSVerifyPassword      = 0x2a,         /* in: handle; a structure: count, then (length, bytes padded to 4) each */
	kAKSChangeSecretEpilogue = 0x69,
	kAKSAssertPromote       = 0x64,
	kAKSAssertConsume       = 0x65,
	kAKSNotifyStateSet      = 0x66,         /* "the first-unlock notification has been posted" */
	kAKSNotifyStateIsSet    = 0x67,
	kAKSPrewarm             = 0x73,
};

/* state flags (libaks' keybag_state): bit 0 locked, bit 1 no passcode, bit 2 unlocked since the start */
#define kAKSStateLocked         0x1
#define kAKSStateNoPasscode     0x2
#define kAKSStateBeenUnlocked   0x4
/* lock state: 0 unlocked, 1 locked, 2 locking, 3 disabled (no passcode) */
#define kAKSLockStateUnlocked   0
#define kAKSLockStateLocked     1
#define kAKSLockStateDisabled   3
/* what the client makes of an answer (MobileKeyBag's convertAKSErrorToMKB, 16.7.16): kIOReturnNotReadable is "the
 * passcode is wrong" (-3), kIOReturnNotPermitted "the device is locked" (-2), 0xe007c008 the wait after wrong ones (-14) */
#define kAKSReturnBadPasscode   kIOReturnNotReadable
#define kAKSReturnLocked        kIOReturnNotPermitted
#define kAKSReturnBackoff       ((IOReturn) 0xe007c008)

static uint64_t gAKSLogged[4];          /* selectors already reported */
static bool     gAKSNotifyState;

/* * * * the keybag * * * */

#define kAKSBagDirectory        "/private/var/db"
#define kAKSBagPath             "/private/var/db/hakios-keybag.bin"
#define kAKSBagMagic            0x31424b48      /* "HKB1" */
#define kAKSClasses             32
#define kAKSIterations          100000
#define kAKSMostPasscode        256

struct AKSBagFile {
	uint32_t magic, version;
	uint32_t flags;                 /* bit 0: a passcode is set */
	uint32_t iterations;
	uint32_t generation;
	uint32_t failed;                /* wrong passcodes in a row */
	uint64_t backoffUntil;          /* the calendar's seconds before which no passcode is tried */
	uint8_t  uuid[16];
	uint8_t  salt[32];
	uint8_t  deviceKey[32];
	uint8_t  check[40];             /* the device key wrapped by the passcode's key: what tells a right passcode */
	uint8_t  wrapped[kAKSClasses][40];
};

static IOLock *         gAKSLock;
static AKSBagFile       gBag;
static bool             gBagThere;              /* read from the file, or made */
static bool             gBagDirty;              /* not in the file as it is here */
static uint8_t          gClassKey[kAKSClasses][32];
static uint32_t         gClassHave;             /* a bit a class whose key is in gClassKey */
static bool             gAKSLocked, gAKSBeenUnlocked;
/* what user space waits for (MobileKeyBag's MKBEventsRegister: a general-interest notification on the AppleKeyStore
 * service; its aksNotificationCB takes two message types, 0xe3ffd00d and 0xe3ffd00f, and has the client ask for the
 * state again): sent after every change of the lock state or the passcode -- without them Settings went on showing
 * "Turn Passcode On" after one was set, and the lock screen did not ask for it until the next start (11-13) */
#define kAKSMessageLockState    0xe3ffd00d
#define kAKSMessageOther        0xe3ffd00f      /* (taken to be "unlocked for the first time") */
static IOService *      gAKSService;
static bool             gAKSChanged;            /* to be told (set with the lock held, sent without it) */
static bool             gAKSFirstUnlock;        /* ... and that this was the first unlock since the start */

/* the classes whose keys need the passcode (when there is one): files' A, B and C; the keychain's WhenUnlocked (6),
 * AfterFirstUnlock (7), their ThisDeviceOnly kinds (9, 10) and WhenPasscodeSet (12) */
#define kAKSPasscodeClasses     ((1u << 1) | (1u << 2) | (1u << 3) | (1u << 6) | (1u << 7) | (1u << 9) | (1u << 10) | (1u << 12))
/* ... and of those, the ones a locked device keeps once it has been unlocked: C, AfterFirstUnlock; B too (its real
 * kind is a public key that can always wrap; with one key for both ways it stays) */
#define kAKSKeptWhenLocked      ((1u << 2) | (1u << 3) | (1u << 7) | (1u << 10))

static void
aksHmac(const uint8_t * key, size_t keyLength, const uint8_t * a, size_t aLength, const uint8_t * b, size_t bLength,
    uint8_t out[32])
{
	uint8_t block[64], inner[32];
	SHA256_CTX ctx;

	bzero(block, sizeof(block));
	if (keyLength > sizeof(block)) {
		SHA256_Init(&ctx);
		SHA256_Update(&ctx, key, keyLength);
		SHA256_Final(block, &ctx);
	} else {
		memcpy(block, key, keyLength);
	}
	for (size_t i = 0; i < sizeof(block); i++) {
		block[i] ^= 0x36;
	}
	SHA256_Init(&ctx);
	SHA256_Update(&ctx, block, sizeof(block));
	SHA256_Update(&ctx, a, aLength);
	if (bLength != 0) {
		SHA256_Update(&ctx, b, bLength);
	}
	SHA256_Final(inner, &ctx);
	for (size_t i = 0; i < sizeof(block); i++) {
		block[i] ^= 0x36 ^ 0x5c;
	}
	SHA256_Init(&ctx);
	SHA256_Update(&ctx, block, sizeof(block));
	SHA256_Update(&ctx, inner, sizeof(inner));
	SHA256_Final(out, &ctx);
	bzero(block, sizeof(block));
	bzero(inner, sizeof(inner));
}

/* PBKDF2-HMAC-SHA256, one block of output (32 bytes) */
static void
aksPbkdf2(const uint8_t * passcode, size_t length, const uint8_t salt[32], uint32_t iterations, uint8_t out[32])
{
	static const uint8_t one[4] = { 0, 0, 0, 1 };
	uint8_t u[32];

	aksHmac(passcode, length, salt, 32, one, sizeof(one), u);
	memcpy(out, u, 32);
	for (uint32_t i = 1; i < iterations; i++) {
		aksHmac(passcode, length, u, sizeof(u), NULL, 0, u);
		for (unsigned k = 0; k < 32; k++) {
			out[k] ^= u[k];
		}
	}
	bzero(u, sizeof(u));
}

/* AES key wrap (RFC 3394) with a 256-bit key: length a multiple of 8, 16 to 64; the output 8 bytes more */
static void
aksWrap(const uint8_t kek[32], const uint8_t * in, size_t length, uint8_t * out)
{
	aes_encrypt_ctx ctx;
	uint8_t a[8], block[16];
	size_t n = length / 8;

	aes_encrypt_key256(kek, &ctx);
	memset(a, 0xa6, sizeof(a));
	memmove(out + 8, in, length);
	for (unsigned j = 0; j < 6; j++) {
		for (size_t i = 1; i <= n; i++) {
			uint64_t t = (uint64_t) n * j + i;

			memcpy(block, a, 8);
			memcpy(block + 8, out + 8 * i, 8);
			aes_encrypt(block, block, &ctx);
			memcpy(a, block, 8);
			for (unsigned k = 0; k < 8; k++) {
				a[7 - k] ^= (uint8_t) (t >> (8 * k));
			}
			memcpy(out + 8 * i, block + 8, 8);
		}
	}
	memcpy(out, a, 8);
	bzero(&ctx, sizeof(ctx));
	bzero(block, sizeof(block));
}

static bool
aksUnwrap(const uint8_t kek[32], const uint8_t * in, size_t length, uint8_t * out)
{
	aes_decrypt_ctx ctx;
	uint8_t a[8], block[16], bad = 0;
	size_t n = length / 8 - 1;

	aes_decrypt_key256(kek, &ctx);
	memcpy(a, in, 8);
	memmove(out, in + 8, 8 * n);
	for (int j = 5; j >= 0; j--) {
		for (size_t i = n; i >= 1; i--) {
			uint64_t t = (uint64_t) n * (unsigned) j + i;

			for (unsigned k = 0; k < 8; k++) {
				a[7 - k] ^= (uint8_t) (t >> (8 * k));
			}
			memcpy(block, a, 8);
			memcpy(block + 8, out + 8 * (i - 1), 8);
			aes_decrypt(block, block, &ctx);
			memcpy(a, block, 8);
			memcpy(out + 8 * (i - 1), block + 8, 8);
		}
	}
	for (unsigned k = 0; k < 8; k++) {
		bad |= a[k] ^ 0xa6;
	}
	bzero(&ctx, sizeof(ctx));
	bzero(block, sizeof(block));
	if (bad != 0) {
		bzero(out, 8 * n);
	}
	return bad == 0;
}

static uint64_t
aksNow(void)
{
	clock_sec_t seconds;
	clock_usec_t microseconds;

	clock_get_calendar_microtime(&seconds, &microseconds);
	return (uint64_t) seconds;
}

/* the bag into its file (the data volume may not be there yet, or not writable: tried again at the next request) */
static void
aksSave(void)
{
	vfs_context_t context = vfs_context_kernel();
	vnode_t vnode = NULLVP;
	int left = 0, error;

	gBagDirty = true;
	error = vnode_open(kAKSBagPath, O_CREAT | O_TRUNC | FWRITE, 0600, 0, &vnode, context);
	if (error == 0) {
		error = vn_rdwr(UIO_WRITE, vnode, (caddr_t) &gBag, (int) sizeof(gBag), 0, UIO_SYSSPACE, IO_SYNC,
		    vfs_context_ucred(context), &left, vfs_context_proc(context));
		vnode_close(vnode, FWRITE, context);
	}
	if (error == 0 && left == 0) {
		gBagDirty = false;
	} else {
		static unsigned said;

		if (said++ < 5) {
			IOLog("MiniAppleKeyStore: the keybag is NOT in %s (error %d, %d bytes left): kept in memory, tried again\n",
			    kAKSBagPath, error, left);
		}
	}
}

/* the bag: read from its file, or made (with the lock held). false: the data volume is not there yet. */
static bool
aksBag(void)
{
	vfs_context_t context = vfs_context_kernel();
	vnode_t vnode = NULLVP;
	int left = 0, error;

	if (gBagThere) {
		if (gBagDirty) {
			aksSave();
		}
		return true;
	}
	if (rootvnode == NULLVP || vnode_lookup(kAKSBagDirectory, 0, &vnode, context) != 0) {
		return false;
	}
	vnode_put(vnode);
	if (vnode_lookup(kAKSBagPath, 0, &vnode, context) == 0) {
		error = vn_rdwr(UIO_READ, vnode, (caddr_t) &gBag, (int) sizeof(gBag), 0, UIO_SYSSPACE, 0,
		    vfs_context_ucred(context), &left, vfs_context_proc(context));
		vnode_put(vnode);
		if (error == 0 && left == 0 && gBag.magic == kAKSBagMagic && gBag.version == 1) {
			bool passcode = (gBag.flags & 1) != 0;

			gClassHave = 0;
			for (unsigned c = 0; c < kAKSClasses; c++) {
				if ((!passcode || !(kAKSPasscodeClasses & (1u << c))) &&
				    aksUnwrap(gBag.deviceKey, gBag.wrapped[c], 40, gClassKey[c])) {
					gClassHave |= 1u << c;
				}
			}
			gAKSLocked = passcode;
			gAKSBeenUnlocked = !passcode;
			gBagThere = true;
			IOLog("MiniAppleKeyStore: the keybag of %s: %s, generation %u, %u wrong passcodes so far\n", kAKSBagPath,
			    passcode ? "a passcode is set -- LOCKED until it is given" : "no passcode", gBag.generation, gBag.failed);
			return true;
		}
		IOLog("MiniAppleKeyStore: %s is not a keybag of ours (error %d, %d bytes left, magic %#x): a new one is made\n",
		    kAKSBagPath, error, left, gBag.magic);
	}
	bzero(&gBag, sizeof(gBag));
	gBag.magic = kAKSBagMagic;
	gBag.version = 1;
	gBag.iterations = kAKSIterations;
	read_random(gBag.uuid, sizeof(gBag.uuid));
	read_random(gBag.deviceKey, sizeof(gBag.deviceKey));
	for (unsigned c = 0; c < kAKSClasses; c++) {
		read_random(gClassKey[c], sizeof(gClassKey[c]));
		aksWrap(gBag.deviceKey, gClassKey[c], 32, gBag.wrapped[c]);
	}
	gClassHave = ~0u;
	gAKSLocked = false;
	gAKSBeenUnlocked = true;
	gBagThere = true;
	IOLog("MiniAppleKeyStore: a new keybag (no passcode), kept in %s\n", kAKSBagPath);
	aksSave();
	return true;
}

/* the key a passcode gives: PBKDF2 over the bag's salt, then HMAC'd with the device key */
static void
aksPasscodeKey(const uint8_t * passcode, size_t length, uint8_t out[32])
{
	uint8_t derived[32];

	aksPbkdf2(passcode, length, gBag.salt, gBag.iterations, derived);
	aksHmac(gBag.deviceKey, sizeof(gBag.deviceKey), derived, sizeof(derived), NULL, 0, out);
	bzero(derived, sizeof(derived));
}

/* a passcode tried against the bag's (with the lock held): kIOReturnSuccess and its key, or why not -- wrong ones
 * are counted, and from the sixth each brings a wait (iOS's: 1, 5, 15, 60 minutes) */
static IOReturn
aksTry(const uint8_t * passcode, size_t length, uint8_t key[32])
{
	static const uint32_t waits[] = { 60, 300, 900, 3600 };
	uint8_t device[32];
	uint64_t now = aksNow();

	if (!(gBag.flags & 1)) {
		return length == 0 ? kIOReturnSuccess : kAKSReturnBadPasscode;
	}
	if (gBag.backoffUntil > now && gBag.backoffUntil - now <= 3600) {
		return kAKSReturnBackoff;
	}
	aksPasscodeKey(passcode, length, key);
	if (aksUnwrap(key, gBag.check, 40, device) && memcmp(device, gBag.deviceKey, 32) == 0) {
		bzero(device, sizeof(device));
		if (gBag.failed != 0 || gBag.backoffUntil != 0) {
			gBag.failed = 0;
			gBag.backoffUntil = 0;
			aksSave();
		}
		return kIOReturnSuccess;
	}
	bzero(key, 32);
	gBag.failed++;
	gBag.backoffUntil = gBag.failed >= 6 ? now + waits[gBag.failed - 6 < 3 ? gBag.failed - 6 : 3] : 0;
	aksSave();
	IOLog("MiniAppleKeyStore: a wrong passcode (%u in a row%s)\n", gBag.failed, gBag.backoffUntil != 0 ? "; a wait now" : "");
	return kAKSReturnBadPasscode;
}

/* every class key of the passcode's kind out of the bag, with the passcode's key */
static void
aksOpenClasses(const uint8_t key[32])
{
	for (unsigned c = 0; c < kAKSClasses; c++) {
		if ((kAKSPasscodeClasses & (1u << c)) && aksUnwrap(key, gBag.wrapped[c], 40, gClassKey[c])) {
			gClassHave |= 1u << c;
		}
	}
}

/* aks_tell=N (tests; see where the messages go out): bit 0 the first unlock told, bit 1 every lock and unlock */
static uint32_t
aksTell(void)
{
	static int tell = -1;
	uint32_t v = 0;

	if (tell < 0) {
		tell = PE_parse_boot_argn("aks_tell", &v, sizeof(v)) ? (int) v : 0;
	}
	return (uint32_t) tell;
}

static IOReturn
aksUnlock(const uint8_t * passcode, size_t length)
{
	uint8_t key[32];
	IOReturn ret = aksTry(passcode, length, key);

	if (ret == kIOReturnSuccess && (gBag.flags & 1)) {
		aksOpenClasses(key);
		if (gAKSLocked) {
			IOLog("MiniAppleKeyStore: unlocked%s\n", gAKSBeenUnlocked ? "" : " (the first time since the start)");
		}
		/* (NOT TOLD of an unlock or a lock it asked for itself -- see where the messages go out) */
		if (aksTell() & 1) {
			gAKSChanged |= !gAKSBeenUnlocked;
			gAKSFirstUnlock |= !gAKSBeenUnlocked;
		}
		if (aksTell() & 2) {
			gAKSChanged |= gAKSLocked;
		}
		gAKSLocked = false;
		gAKSBeenUnlocked = true;
	}
	bzero(key, sizeof(key));
	return ret;
}

static void
aksLock(void)
{
	if (!(gBag.flags & 1) || gAKSLocked) {
		return;
	}
	for (unsigned c = 0; c < kAKSClasses; c++) {
		if ((kAKSPasscodeClasses & ~kAKSKeptWhenLocked & (1u << c))) {
			bzero(gClassKey[c], sizeof(gClassKey[c]));
			gClassHave &= ~(1u << c);
		}
	}
	gAKSLocked = true;
	if (aksTell() & 2) {
		gAKSChanged = true;
	}
	IOLog("MiniAppleKeyStore: locked\n");
}

/* the passcode set, changed or taken away (length 0); the old one has to be right */
static IOReturn
aksChangeSecret(const uint8_t * oldPasscode, size_t oldLength, const uint8_t * newPasscode, size_t newLength)
{
	uint8_t key[32];
	IOReturn ret = aksTry(oldPasscode, oldLength, key);

	if (ret != kIOReturnSuccess) {
		return ret;
	}
	if (gBag.flags & 1) {
		aksOpenClasses(key);            /* (every key in hand before it is wrapped anew) */
	}
	if ((gClassHave & kAKSPasscodeClasses) != kAKSPasscodeClasses) {
		bzero(key, sizeof(key));
		return kIOReturnInternalError;
	}
	if (newLength != 0) {
		read_random(gBag.salt, sizeof(gBag.salt));
		gBag.iterations = kAKSIterations;
		aksPasscodeKey(newPasscode, newLength, key);
		aksWrap(key, gBag.deviceKey, 32, gBag.check);
		gBag.flags |= 1;
	} else {
		memcpy(key, gBag.deviceKey, sizeof(key));
		bzero(gBag.check, sizeof(gBag.check));
		bzero(gBag.salt, sizeof(gBag.salt));
		gBag.flags &= ~1u;
	}
	for (unsigned c = 0; c < kAKSClasses; c++) {
		if (kAKSPasscodeClasses & (1u << c)) {
			aksWrap(key, gClassKey[c], 32, gBag.wrapped[c]);
		}
	}
	bzero(key, sizeof(key));
	gBag.generation++;
	gBag.failed = 0;
	gBag.backoffUntil = 0;
	gAKSLocked = false;
	gAKSBeenUnlocked = true;
	gAKSChanged = true;
	aksSave();
	IOLog("MiniAppleKeyStore: %s (generation %u)\n", newLength != 0 ? (oldLength != 0 ? "the passcode changed" : "a passcode set") :
	    "the passcode taken away", gBag.generation);
	return kIOReturnSuccess;
}

static uint32_t
aksStateFlags(void)
{
	if (!gBagThere || !(gBag.flags & 1)) {
		return kAKSStateNoPasscode | kAKSStateBeenUnlocked;
	}
	return (gAKSLocked ? kAKSStateLocked : 0) | (gAKSBeenUnlocked ? kAKSStateBeenUnlocked : 0);
}

/* a caller's bytes by their address in its memory (how libaks hands passcodes over), kAKSMostPasscode at most */
static IOReturn
aksCopyIn(uint64_t address, uint64_t length, uint8_t * to)
{
	if (length == 0) {
		return kIOReturnSuccess;
	}
	if (address == 0 || length > kAKSMostPasscode) {
		return kIOReturnBadArgument;
	}
	return copyin((user_addr_t) address, to, (vm_size_t) length) == 0 ? kIOReturnSuccess : kIOReturnVMError;
}

/* Append SEQUENCE { UTF8String key, INTEGER value }; returns the new length. */
static size_t
derPut(uint8_t * buf, size_t length, const char * key, uint32_t value)
{
	size_t keyLength = strlen(key);
	unsigned bytes = value < 0x80 ? 1 : value < 0x8000 ? 2 : value < 0x800000 ? 3 : value < 0x80000000u ? 4 : 5;

	buf[length++] = 0x30;
	buf[length++] = (uint8_t) (keyLength + 4 + bytes);
	buf[length++] = 0x0c;
	buf[length++] = (uint8_t) keyLength;
	memcpy(buf + length, key, keyLength);
	length += keyLength;
	buf[length++] = 0x02;
	buf[length++] = (uint8_t) bytes;
	for (unsigned i = bytes; i > 0; i--) {
		buf[length++] = i == 5 ? 0 : (uint8_t) (value >> (8 * (i - 1)));
	}
	return length;
}

/* The answer to "device state": what the bag says (asked with the lock held). */
static size_t
deviceState(uint8_t * buf)
{
	bool passcode = gBagThere && (gBag.flags & 1);
	uint64_t now = aksNow();
	uint32_t wait = passcode && gBag.backoffUntil > now && gBag.backoffUntil - now <= 3600 ? (uint32_t) (gBag.backoffUntil - now) : 0;
	size_t length = 2;

	length = derPut(buf, length, "sb", wait);       /* backoff */
	length = derPut(buf, length, "sr", 0);
	length = derPut(buf, length, "ss", aksStateFlags());
	length = derPut(buf, length, "sas", 0);         /* assertion set */
	length = derPut(buf, length, "sfa", passcode ? gBag.failed : 0);        /* failed attempts */
	length = derPut(buf, length, "sgs", 0);         /* generation state */
	length = derPut(buf, length, "sls", !passcode ? kAKSLockStateDisabled : gAKSLocked ? kAKSLockStateLocked : kAKSLockStateUnlocked);
	length = derPut(buf, length, "sgpe", 0);        /* grace period enabled */
	length = derPut(buf, length, "srcd", 0);        /* recovery countdown */
	buf[0] = 0x31;
	buf[1] = (uint8_t) (length - 2);
	return length;
}

static size_t
configuration(uint8_t * buf)
{
	size_t length = 2;

	length = derPut(buf, length, "bd", 0);          /* backoff delay */
	length = derPut(buf, length, "gp", 0);          /* grace period */
	length = derPut(buf, length, "mua", 10);        /* maximum unlock attempts */
	buf[0] = 0x31;
	buf[1] = (uint8_t) (length - 2);
	return length;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

class AppleKeyStore : public IOService
{
	OSDeclareDefaultStructors(AppleKeyStore);

public:
	virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
};

OSDefineMetaClassAndStructors(AppleKeyStore, IOService);

bool
AppleKeyStore::start(IOService * provider)
{
	if (!IOService::start(provider)) {
		return false;
	}
	setProperty(kIOUserClientClassKey, "AppleKeyStoreUserClient");
	setName("AppleKeyStore");
	registerService();
	return true;
}

class AppleKeyStoreUserClient : public IOUserClient
{
	OSDeclareDefaultStructors(AppleKeyStoreUserClient);

	static IOReturn answer(IOExternalMethodArguments * args, const uint8_t * bytes, size_t length);

public:
	virtual IOReturn clientClose(void) APPLE_KEXT_OVERRIDE;
	virtual IOReturn registerNotificationPort(mach_port_t port, UInt32 type, io_user_reference_t refCon) APPLE_KEXT_OVERRIDE;
	virtual IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments * args,
	    IOExternalMethodDispatch * dispatch, OSObject * target, void * reference) APPLE_KEXT_OVERRIDE;
};

OSDefineMetaClassAndStructors(AppleKeyStoreUserClient, IOUserClient);

IOReturn
AppleKeyStoreUserClient::clientClose(void)
{
	if (!isInactive()) {
		terminate();
	}
	return kIOReturnSuccess;
}

/* keybagd asks to be told about keys to back up and bags to save; there never are any. */
IOReturn
AppleKeyStoreUserClient::registerNotificationPort(mach_port_t port, UInt32 type, io_user_reference_t refCon)
{
	releaseNotificationPort(port);
	return kIOReturnSuccess;
}

/*
 * kAKSWrapKey and kAKSUnwrapKey. A wrapped key is 8 bytes longer than the key, as with AES key
 * wrap (RFC 3394), which is what callers size their buffers for: a mark, the class, and the key
 * mixed with a fixed pattern. Not encryption: there is no secret here to encrypt with.
 */
static const uint8_t kWrapMark[4] = { 'H', 'K', 'w', 'r' };

static void
wrapMix(uint8_t * bytes, size_t length)
{
	for (size_t i = 0; i < length; i++) {
		bytes[i] ^= (uint8_t)(0x5a + 37 * i);
	}
}

/* A structure answer; a big buffer on the other side arrives as a memory descriptor. */
IOReturn
AppleKeyStoreUserClient::answer(IOExternalMethodArguments * args, const uint8_t * bytes, size_t length)
{
	if (args->structureOutputDescriptor != NULL) {
		IOMemoryDescriptor * out = args->structureOutputDescriptor;
		IOReturn ret;

		if (out->getLength() < length) {
			return kIOReturnNoSpace;
		}
		ret = out->prepare();
		if (ret == kIOReturnSuccess) {
			out->writeBytes(0, bytes, length);
			out->complete();
			args->structureOutputDescriptorSize = (uint32_t) length;
		}
		return ret;
	}
	if (args->structureOutput == NULL || args->structureOutputSize < length) {
		return kIOReturnNoSpace;
	}
	memcpy(args->structureOutput, bytes, length);
	args->structureOutputSize = (uint32_t) length;
	return kIOReturnSuccess;
}

IOReturn
AppleKeyStoreUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments * args,
    IOExternalMethodDispatch * dispatch, OSObject * target, void * reference)
{
	static const uint8_t bagUUID[16] = { 0x72, 0x65, 0x64, 0x77, 0x6f, 0x6f, 0x64, 0x00, 0x6b, 0x65, 0x79, 0x62, 0x61, 0x67, 0x00, 0x01 };
	uint8_t der[128];
	uint64_t * out = args->scalarOutput;

	/* boot-arg aks_say=1: every request said -- who, the selector, the scalars, the structure's size and first bytes
	 * (what keybagd and MobileKeyBag ask for while a passcode is set, 11-13) */
	static int say = -1;
	if (say < 0) {
		uint32_t v = 0;

		say = PE_parse_boot_argn("aks_say", &v, sizeof(v)) && v != 0;
	}
	if (say) {
		char procname[32] = "?", hex[2 * 48 + 1] = "";
		const uint8_t * in = (const uint8_t *) args->structureInput;

		proc_selfname(procname, sizeof(procname));
		for (uint32_t i = 0; in != NULL && i < args->structureInputSize && i < 48; i++) {
			snprintf(hex + 2 * i, 3, "%02x", in[i]);
		}
		char scalars[12 * 20] = "";

		for (uint32_t i = 0, at = 0; i < args->scalarInputCount && i < 12; i++) {
			at += snprintf(scalars + at, sizeof(scalars) - at, "%s0x%llx", i ? " " : "", args->scalarInput[i]);
		}
		IOLog("MiniAppleKeyStore: say: %s[%d] selector 0x%x; scalars %u in [%s], %u out; "
		    "structure %u in%s%s (%s), %u out%s\n", procname, proc_selfpid(), selector, args->scalarInputCount, scalars,
		    args->scalarOutputCount, args->structureInputSize, hex[0] ? " " : "", hex,
		    args->structureInputDescriptor != NULL ? "out of line" : "inline",
		    args->structureOutputDescriptor != NULL ? (uint32_t) args->structureOutputDescriptor->getLength() : args->structureOutputSize,
		    args->structureOutputDescriptor != NULL ? " (out of line)" : "");
	}

	/* (the bag: read or made at the first request that finds the data volume; until then the answers are those of a
	 * device without a passcode, and nothing is wrapped) */
	IOReturn ret = kIOReturnSuccess;
	bool there;

	IOLockLock(gAKSLock);
	there = aksBag();
	switch (selector) {
	case kAKSAssertHold:
		/* A hold on the bag's keys (in: the kind, the handle, a time): NOT GIVEN WHILE THE BAG IS LOCKED. SpringBoard
		 * asks for one of its grace-period kind (2) when the phone wakes after the side button's lock ("Checking if
		 * passcode locked") and takes a yes for "still within the grace period": with every hold granted the lock
		 * screen let the phone be opened by a swipe, a passcode set and the bag locked (11-13; sb_authsay=1 showed
		 * its state "Revoking" with that assertion valid). There is no grace period here yet: locked is locked. */
		if (there && (gBag.flags & 1) && gAKSLocked) {
			ret = kAKSReturnLocked;
		}
		break;

	case kAKSOpenCheck:
	case kAKSAssertDrop:
	case kAKSAssertPromote:
	case kAKSAssertConsume:
	case kAKSPrewarm:
	case kAKSUnloadBag:
	case kAKSSetSystem:
	case kAKSSetConfiguration:
	case kAKSChangeSecretEpilogue:
	case kAKSUnlockBag:
		break;

	case kAKSLockDevice:
	case kAKSLockDeviceNow:
		if (there) {
			aksLock();
		}
		break;

	case kAKSUnlockDevice: {
		/* in: handle, the passcode's address in the caller, its length */
		uint8_t passcode[kAKSMostPasscode];
		uint64_t length = args->scalarInputCount >= 3 ? args->scalarInput[2] : 0;

		if (!there) {
			break;
		}
		ret = aksCopyIn(args->scalarInputCount >= 3 ? args->scalarInput[1] : 0, length, passcode);
		if (ret == kIOReturnSuccess) {
			ret = aksUnlock(passcode, (size_t) length);
		}
		bzero(passcode, sizeof(passcode));
		break;
	}

	case kAKSVerifyPassword: {
		/* a structure: the count of items, then each as its length and its bytes padded to four; the first is the
		 * passcode */
		const uint8_t * in = (const uint8_t *) args->structureInput;
		uint32_t count = 0, length = 0;
		uint8_t key[32];

		if (!there) {
			break;
		}
		if (in == NULL || args->structureInputSize < 8) {
			ret = kIOReturnBadArgument;
			break;
		}
		memcpy(&count, in, 4);
		memcpy(&length, in + 4, 4);
		if (count < 1 || length > kAKSMostPasscode || 8 + (uint64_t) length > args->structureInputSize) {
			ret = kIOReturnBadArgument;
			break;
		}
		ret = aksTry(in + 8, length, key);
		bzero(key, sizeof(key));
		break;
	}

	case kAKSChangeSecret: {
		uint8_t before[kAKSMostPasscode], after[kAKSMostPasscode];

		if (!there) {
			ret = kIOReturnNotReady;
			break;
		}
		if (args->scalarInputCount < 7) {
			ret = kIOReturnBadArgument;
			break;
		}
		ret = aksCopyIn(args->scalarInput[3], args->scalarInput[4], before);
		if (ret == kIOReturnSuccess) {
			ret = aksCopyIn(args->scalarInput[5], args->scalarInput[6], after);
		}
		if (ret == kIOReturnSuccess) {
			ret = aksChangeSecret(before, (size_t) args->scalarInput[4], after, (size_t) args->scalarInput[6]);
		}
		bzero(before, sizeof(before));
		bzero(after, sizeof(after));
		if (ret == kIOReturnSuccess && args->scalarOutputCount >= 1) {
			out[0] = gBag.generation;
			args->scalarOutputCount = 1;
		}
		break;
	}

	case kAKSGetLockState:
		if (args->scalarOutputCount < 1) {
			ret = kIOReturnBadArgument;
			break;
		}
		out[0] = aksStateFlags();
		args->scalarOutputCount = 1;
		break;

	case kAKSGetSystem:
	case kAKSCreateBag:
	case kAKSLoadBag:
		if (args->scalarOutputCount >= 1) {
			out[0] = 0;
			args->scalarOutputCount = 1;
		}
		break;

	case kAKSGeneration:
		if (args->scalarOutputCount >= 1) {
			out[0] = there ? gBag.generation : 0;
			args->scalarOutputCount = 1;
		}
		break;

	case kAKSGetDeviceState:
	case kAKSGetExtendedState:
		ret = answer(args, der, deviceState(der));
		break;

	case kAKSGetConfiguration:
		ret = answer(args, der, configuration(der));
		break;

	case kAKSGetBagUUID:
	case kAKSSaveBag:
		/* (what keybagd keeps of a bag is its name: the bag itself is the kernel's file) */
		ret = answer(args, there ? gBag.uuid : bagUUID, sizeof(bagUUID));
		break;

	case kAKSNotifyStateSet:
		gAKSNotifyState = true;
		break;

	case kAKSWrapKey: {
		uint8_t wrapped[8 + 64];
		size_t length = args->structureInputSize;
		unsigned c = args->scalarInputCount >= 2 ? (unsigned) args->scalarInput[1] & (kAKSClasses - 1) : 0;

		if (args->structureInput == NULL || length == 0 || length > 64 || args->scalarInputCount < 2) {
			ret = kIOReturnBadArgument;
			break;
		}
		if (there && (length & 7) == 0 && length >= 16) {
			if (!(gClassHave & (1u << c))) {
				ret = kAKSReturnLocked;
				break;
			}
			aksWrap(gClassKey[c], (const uint8_t *) args->structureInput, length, wrapped);
		} else {
			/* (no bag yet, or a length key wrap has no form for: the mark and the pattern, as before) */
			memcpy(wrapped, kWrapMark, sizeof(kWrapMark));
			wrapped[4] = (uint8_t) args->scalarInput[1];
			wrapped[5] = wrapped[6] = wrapped[7] = 0;
			memcpy(wrapped + 8, args->structureInput, length);
			wrapMix(wrapped + 8, length);
		}
		if (args->scalarOutputCount >= 1) {
			out[0] = args->scalarInput[1];
			args->scalarOutputCount = 1;
		}
		ret = answer(args, wrapped, 8 + length);
		bzero(wrapped, sizeof(wrapped));
		break;
	}

	case kAKSUnwrapKey: {
		uint8_t key[64];
		const uint8_t * wrapped = (const uint8_t *) args->structureInput;
		size_t length = args->structureInputSize;
		unsigned c = args->scalarInputCount >= 2 ? (unsigned) args->scalarInput[1] & (kAKSClasses - 1) : 0;

		if (wrapped == NULL || length <= 8 || length > 8 + sizeof(key)) {
			ret = kIOReturnBadArgument;
			break;
		}
		if (memcmp(wrapped, kWrapMark, sizeof(kWrapMark)) == 0 && wrapped[5] == 0 && wrapped[6] == 0 && wrapped[7] == 0) {
			/* (wrapped before there was a bag) */
			memcpy(key, wrapped + 8, length - 8);
			wrapMix(key, length - 8);
		} else if (!there || (length & 7) != 0 || length < 24) {
			ret = kIOReturnBadArgument;
			break;
		} else if (!(gClassHave & (1u << c))) {
			ret = kAKSReturnLocked;
			break;
		} else if (!aksUnwrap(gClassKey[c], wrapped, length, key)) {
			static unsigned said;

			if (said++ < 10) {
				char procname[32] = "?";

				proc_selfname(procname, sizeof(procname));
				IOLog("MiniAppleKeyStore: %s[%d]: a key of class %u does not unwrap (%zu bytes): not this bag's\n", procname,
				    proc_selfpid(), c, length);
			}
			ret = kIOReturnNotPermitted;
			break;
		}
		ret = answer(args, key, length - 8);
		bzero(key, sizeof(key));
		break;
	}

	case kAKSNotifyStateIsSet:
		if (args->scalarOutputCount < 1) {
			ret = kIOReturnBadArgument;
			break;
		}
		out[0] = gAKSNotifyState;
		args->scalarOutputCount = 1;
		break;

	default:
		ret = kIOReturnUnsupported;
		break;
	}
	bool tell = gAKSChanged, first = gAKSFirstUnlock;

	gAKSChanged = gAKSFirstUnlock = false;
	IOLockUnlock(gAKSLock);
	/* (0xe3ffd00d is SpringBoard's "the keybag's status changed" (SBFMobileKeyBag's block: event 1), 0xe3ffd00f its
	 * "unlocked for the first time" (event 0). TOLD OF: a passcode set, changed or taken away, and the first unlock.
	 * NOT of each lock and unlock, which SpringBoard asks for itself: told in the middle of its own side-button lock
	 * -- two tries, both messages and the first alone -- it let the phone be opened without the passcode afterwards,
	 * the kernel's bag locked all the while; without the message it asks for it, as it should. Why is not known.) */
	if (tell && gAKSService != NULL) {
		gAKSService->messageClients(kAKSMessageLockState);
		if (first) {
			gAKSService->messageClients(kAKSMessageOther);
		}
	}
	if (say && ret != kIOReturnSuccess) {
		IOLog("MiniAppleKeyStore: say: selector 0x%x answered %#x\n", selector, ret);
	}
	if (ret != kIOReturnUnsupported) {
		return ret;
	}

	if (selector < 256 && !(gAKSLogged[selector / 64] & (1ULL << (selector % 64)))) {
		char procname[32] = "?";

		gAKSLogged[selector / 64] |= 1ULL << (selector % 64);
		proc_selfname(procname, sizeof(procname));
		IOLog("MiniAppleKeyStore: %s[%d] selector 0x%x not implemented (scalars %u in, %u out; structure %u in, %u out)\n",
		    procname, proc_selfpid(), selector, args->scalarInputCount, args->scalarOutputCount, args->structureInputSize,
		    args->structureOutputSize);
	}
	return kIOReturnUnsupported;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void
MiniAppleKeyStoreStart(void)
{
	/* user space looks for it by class, and by the path IOService:/IOResources/AppleKeyStore */
	IOService * resources = IOService::getResourceService();
	AppleKeyStore * store = OSTypeAlloc(AppleKeyStore);

	gAKSLock = IOLockAlloc();
	{
		/* keybagd keeps its copy of a bag encrypted by a key in the effaceable storage (AppleEffaceableStorage, a
		 * NAND's own area on an iPhone), unless the device tree's /defaults says "no-effaceable-storage" -- then it
		 * writes the bag as it is ("Saving SystemBag with NO EFFACEABLE ROLL"). There is none here: said so. */
		IORegistryEntry * defaults = IORegistryEntry::fromPath("/defaults", gIODTPlane);
		IORegistryEntry * root = IORegistryEntry::fromPath("/", gIODTPlane);
		const uint32_t one = 1;
		OSData * yes = OSData::withBytes(&one, sizeof(one));

		if (defaults == NULL && root != NULL) {
			defaults = OSTypeAlloc(IORegistryEntry);
			if (defaults != NULL && defaults->init()) {
				defaults->setName("defaults");
				defaults->attachToParent(root, gIODTPlane);
			} else {
				OSSafeReleaseNULL(defaults);
			}
		}
		if (defaults != NULL && yes != NULL) {
			defaults->setProperty("no-effaceable-storage", yes);
		}
		IOLog("MiniAppleKeyStore: the device tree's /defaults %s no-effaceable-storage\n",
		    defaults != NULL && yes != NULL ? "says" : "COULD NOT BE GIVEN");
		OSSafeReleaseNULL(yes);
		OSSafeReleaseNULL(root);
		OSSafeReleaseNULL(defaults);
	}
	{
		/* the two primitives against their documents' own examples (RFC 4231's second case; RFC 3394, 4.6), said
		 * once: a keybag built on a wrong one would wrap and unwrap happily and be worth nothing */
		static const uint8_t kek[32] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
			                         0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f };
		static const uint8_t data[16] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff };
		static const uint8_t wrapped[24] = { 0x64, 0xe8, 0xc3, 0xf9, 0xce, 0x0f, 0x5b, 0xa2, 0x63, 0xe9, 0x77, 0x79, 0x05, 0x81, 0x8a, 0x2a,
			                             0x93, 0xc8, 0x19, 0x1e, 0x7d, 0x6e, 0x8a, 0xe7 };
		static const uint8_t mac[32] = { 0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24, 0x26, 0x08, 0x95, 0x75, 0xc7,
			                         0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec, 0x38, 0x43 };
		uint8_t w[24], back[16], h[32];
		bool unwraps, tampered;

		aksWrap(kek, data, sizeof(data), w);
		unwraps = aksUnwrap(kek, w, sizeof(w), back);
		aksHmac((const uint8_t *) "Jefe", 4, (const uint8_t *) "what do ya want for nothing?", 28, NULL, 0, h);
		w[9] ^= 1;
		tampered = aksUnwrap(kek, w, sizeof(w), h + 0) /* (h is done with) */;
		aksHmac((const uint8_t *) "Jefe", 4, (const uint8_t *) "what do ya want ", 16, (const uint8_t *) "for nothing?", 12, h);
		IOLog("MiniAppleKeyStore: key wrap %s (RFC 3394), unwrap %s, a changed one %s, HMAC-SHA256 %s (RFC 4231)\n",
		    memcmp(w + 16, wrapped + 16, 8) == 0 && memcmp(w, wrapped, 9) == 0 ? "as the example" : "NOT AS THE EXAMPLE",
		    unwraps && memcmp(back, data, sizeof(data)) == 0 ? "gives the key back" : "DOES NOT GIVE THE KEY BACK",
		    tampered ? "IS TAKEN" : "is refused", memcmp(h, mac, sizeof(mac)) == 0 ? "as the example" : "NOT AS THE EXAMPLE");
	}

	if (resources == NULL || store == NULL || !store->init() || !store->attach(resources)) {
		OSSafeReleaseNULL(store);
		IOLog("MiniAppleKeyStore: cannot create the service\n");
		return;
	}
	if (!store->start(resources)) {
		store->detach(resources);
	} else {
		gAKSService = store;            /* (kept: never ended) */
		store->retain();
	}
	store->release();
}
