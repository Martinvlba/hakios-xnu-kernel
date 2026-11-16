/*
 * HakiOS PATH 2, step 1: the infrastructure for architected user pointer authentication.
 *
 * The cores have ARMv8.3 FEAT_PAuth (QARMA5 address auth, generic auth) and firmware lets EL1
 * write the key registers (step 0, hakios_pac_probe() in cpu.c). This kernel is ARM64, not
 * arm64e: it has no pac/aut instructions of its own, so it only MANAGES the keys for EL0:
 *
 *   - which images run PAC-on (boot-arg policy, hakios_pac_exec_wants_pac(), asked by exec);
 *   - the ids a task's keys come from (task->rop_pid / jop_pid, Apple's fields; set by the
 *     ml_task_set_*_pid() routines in machine_routines.c);
 *   - the keys themselves and SCTLR_EL1.En{IA,IB,DA,DB} at context switch (hakios_pac_switch()).
 *
 * Compiled in with `options HAKIOS_USER_PAC` (config/MASTER, in KERNEL_BASE of
 * config/MASTER.arm64.bcm2837). At run time everything hangs off boot-arg hakios_pac:
 *
 *   hakios_pac=0 (or absent)  off. No key register and no SCTLR_EL1 bit is ever written; every
 *                             image is PAC-off exactly as before this file existed.
 *   hakios_pac=1              arm64e images whose name is in hakios_pac_proc=a,b,c run PAC-on.
 *   hakios_pac=2              every arm64e image runs PAC-on.
 *
 * Key assignment (Apple's user ABI):
 *   APIAKey, APDAKey   <- jop_pid   process-independent ("A") keys: shared-cache and dyld
 *                                   fixups are signed with them
 *   APIBKey, APDBKey   <- rop_pid   process-dependent ("B") keys: return addresses; random per
 *                                   exec, inherited by fork
 *   APGAKey            <- one per-boot key for every thread
 *
 * NOT done here (steps 2-4; a PAC-on process cannot work end to end before them): signing the
 * dyld-pager / shared-cache fixups, pmap_sign_user_ptr(), and signing the thread state the
 * kernel hands to or takes from user space. Until then a PAC-on process faults as soon as it
 * authenticates a pointer the kernel should have signed.
 */

#include <mach/mach_types.h>
#include <kern/kern_types.h>
#include <kern/task.h>
#include <kern/thread.h>
#include <kern/startup.h>
#include <kern/misc_protos.h>
#include <kern/debug.h>
#include <machine/machine_routines.h>
#include <arm/cpu_data_internal.h>
#include <arm64/proc_reg.h>
#include <arm64/machine_machdep.h>
#include <arm64/hakios_pac.h>
#include <pexpert/pexpert.h>
#include <vm/vm_map.h>
#include <vm/pmap.h>
#include <stdarg.h>
#include <string.h>

#if HAKIOS_USER_PAC

/*
 * The compiler refuses the names of the key registers in arm64 mode, hence the generic
 * encodings (all op0=3 op1=0 CRn=2):
 */
#define APIAKEYLO       "S3_0_c2_c1_0"
#define APIAKEYHI       "S3_0_c2_c1_1"
#define APIBKEYLO       "S3_0_c2_c1_2"
#define APIBKEYHI       "S3_0_c2_c1_3"
#define APDAKEYLO       "S3_0_c2_c2_0"
#define APDAKEYHI       "S3_0_c2_c2_1"
#define APDBKEYLO       "S3_0_c2_c2_2"
#define APDBKEYHI       "S3_0_c2_c2_3"
#define APGAKEYLO       "S3_0_c2_c3_0"
#define APGAKEYHI       "S3_0_c2_c3_1"
#define ID_AA64ISAR1    "S3_0_c0_c6_1"
#define ID_AA64ISAR2    "S3_0_c0_c6_2"

/*
 * SCTLR_EL1.EnIA (31) | EnIB (30) | EnDA (27) | EnDB (13). (proc_reg.h's SCTLR_PAC*_ENABLED only
 * exist with HAS_APPLE_PAC.)
 */
#define HAKIOS_PAC_SCTLR_EN     ((1ULL << 31) | (1ULL << 30) | (1ULL << 27) | (1ULL << 13))

/* (the key numbers, HAKIOS_PAC_KEY_*, are in hakios_pac.h) */

struct hakios_pac_key {
	uint64_t lo;
	uint64_t hi;
};

/* Written once, in hakios_pac_init() (TUNABLES phase, one CPU, no user threads), then only read. */
int hakios_pac_mode = HAKIOS_PAC_MODE_OFF;
static bool hakios_pac_has_ga;                                  /* FEAT_PAuth generic auth: APGAKey exists */
static uint64_t hakios_pac_default_jop;                         /* the per-boot JOP id (ml_default_jop_pid) */
static struct hakios_pac_key hakios_pac_salt[HAKIOS_PAC_NKEYS]; /* per-boot, per-key; [GA] is the GA key */
static char hakios_pac_proc[256];                               /* boot-arg hakios_pac_proc=a,b,c */

/*
 * The 128-bit key for one key register from a task's 64-bit id.
 *
 * PHASE A, WEAK, bring-up only: key = per-boot per-key salt XOR (id, mix(id)). mix() is the
 * splitmix64 finalizer, a public bijection: this separates the keys of different ids and of the
 * different registers, but it is NOT a KDF. Whoever learns one process's key and the two ids knows
 * the salt and with it every other process's key for that register; nothing here is secret from
 * the kernel's own memory either. Replace with a keyed PRF (e.g. AES/SipHash under a boot secret)
 * before this is relied on for security.
 */
static inline uint64_t
hakios_pac_mix(uint64_t x)
{
	x ^= x >> 30;
	x *= 0xbf58476d1ce4e5b9ULL;
	x ^= x >> 27;
	x *= 0x94d049bb133111ebULL;
	x ^= x >> 31;
	return x;
}

static inline struct hakios_pac_key
hakios_pac_derive(int which, uint64_t id)
{
	struct hakios_pac_key k;

	k.lo = hakios_pac_salt[which].lo ^ id;
	k.hi = hakios_pac_salt[which].hi ^ hakios_pac_mix(id + 0x9e3779b97f4a7c15ULL);
	return k;
}

/*
 * Apple's interfaces (machine_routines.h). On Apple hardware these are the ids of kernel_task;
 * here the JOP one is also every user task's jop_pid (ml_task_set_jop_pid*, Phase A). Both are 0
 * while hakios_pac is 0, and nothing reads them then.
 */
uint64_t
ml_default_rop_pid(void)
{
	return 0;
}

uint64_t
ml_default_jop_pid(void)
{
	return hakios_pac_default_jop;
}

/*
 * exec (exec_mach_imgact): does this image run PAC-on? `name` is the last path component (what
 * becomes p_comm), not NUL-terminated. Never true with hakios_pac=0 or for a non-arm64e image.
 */
bool
hakios_pac_exec_wants_pac(const char *name, size_t namelen, bool is_arm64e)
{
	if (hakios_pac_mode == HAKIOS_PAC_MODE_OFF || !is_arm64e) {
		return false;
	}
	if (hakios_pac_mode == HAKIOS_PAC_MODE_ALL) {
		return true;
	}
	if (hakios_pac_mode != HAKIOS_PAC_MODE_NAMED || name == NULL || namelen == 0) {
		return false;
	}
	/* an exact match with one element of the comma-separated list */
	const char *p = hakios_pac_proc;
	while (*p != '\0') {
		const char *e = p;
		while (*e != '\0' && *e != ',') {
			e++;
		}
		if ((size_t)(e - p) == namelen && strncmp(p, name, namelen) == 0) {
			return true;
		}
		p = (*e == ',') ? e + 1 : e;
	}
	return false;
}

/*
 * Bring this CPU's key registers and SCTLR_EL1.En* in line with `thread`, a user thread that is
 * about to run on it. Interrupts off; hakios_pac_mode != 0. Writes only what the per-CPU cache
 * says differs.
 *
 *   PAC-on thread   (arm64e image selected by the policy: ARM_MACHINE_THREAD_DISABLE_USER_JOP
 *                   clear): its A and B keys loaded, all four En bits set.
 *   PAC-off thread  (everything else): all four En bits clear; the key registers are left as
 *                   they are (with En clear the address-auth instructions are NOPs / strip-less
 *                   no-ops at EL0, so the keys are unobservable).
 *
 * APGAKey has no enable bit (pacga always works), so it is made the same per-boot value on every
 * CPU for every user thread, PAC-on or not, the first time a user thread is switched to.
 *
 * The enable bits act at EL1 too. That is harmless only because this kernel executes no
 * pac* / aut* / ret-auth instruction (xpaclri, which it does use, ignores SCTLR and the keys).
 */
static void
hakios_pac_load(cpu_data_t *cdp, thread_t thread)
{
	uint8_t state = cdp->hakios_pac_state;
	bool changed = false;
	const bool on = (thread->machine.arm_machine_flags & ARM_MACHINE_THREAD_DISABLE_USER_JOP) == 0;

	if (!(state & HAKIOS_PAC_CPU_GA_VALID)) {
		if (hakios_pac_has_ga) {
			__builtin_arm_wsr64(APGAKEYLO, hakios_pac_salt[HAKIOS_PAC_KEY_GA].lo);
			__builtin_arm_wsr64(APGAKEYHI, hakios_pac_salt[HAKIOS_PAC_KEY_GA].hi);
			changed = true;
		}
		state |= HAKIOS_PAC_CPU_GA_VALID;
	}

	if (on) {
		const uint64_t jop = thread->machine.jop_pid;
		const uint64_t rop = thread->machine.rop_pid;
		const bool valid = (state & HAKIOS_PAC_CPU_KEYS_VALID) != 0;
		struct hakios_pac_key k;

		if (!valid || cdp->jop_key != jop) {
			k = hakios_pac_derive(HAKIOS_PAC_KEY_IA, jop);
			__builtin_arm_wsr64(APIAKEYLO, k.lo);
			__builtin_arm_wsr64(APIAKEYHI, k.hi);
			k = hakios_pac_derive(HAKIOS_PAC_KEY_DA, jop);
			__builtin_arm_wsr64(APDAKEYLO, k.lo);
			__builtin_arm_wsr64(APDAKEYHI, k.hi);
			cdp->jop_key = jop;
			changed = true;
		}
		if (!valid || cdp->rop_key != rop) {
			k = hakios_pac_derive(HAKIOS_PAC_KEY_IB, rop);
			__builtin_arm_wsr64(APIBKEYLO, k.lo);
			__builtin_arm_wsr64(APIBKEYHI, k.hi);
			k = hakios_pac_derive(HAKIOS_PAC_KEY_DB, rop);
			__builtin_arm_wsr64(APDBKEYLO, k.lo);
			__builtin_arm_wsr64(APDBKEYHI, k.hi);
			cdp->rop_key = rop;
			changed = true;
		}
		state |= HAKIOS_PAC_CPU_KEYS_VALID;
	}

	/* after a reset the cache knows nothing about SCTLR_EL1: look once rather than assume */
	if (!(state & HAKIOS_PAC_CPU_SCTLR_KNOWN)) {
		const uint64_t en = __builtin_arm_rsr64("SCTLR_EL1") & HAKIOS_PAC_SCTLR_EN;

		state |= HAKIOS_PAC_CPU_SCTLR_KNOWN;
		if (en == HAKIOS_PAC_SCTLR_EN) {
			state |= HAKIOS_PAC_CPU_SCTLR_EN;
		} else if (en == 0) {
			state &= ~HAKIOS_PAC_CPU_SCTLR_EN;
		} else {
			/* a mix: say the opposite of what is wanted, so that the write below happens */
			state = on ? (state & ~HAKIOS_PAC_CPU_SCTLR_EN) : (state | HAKIOS_PAC_CPU_SCTLR_EN);
		}
	}
	if (on != ((state & HAKIOS_PAC_CPU_SCTLR_EN) != 0)) {
		uint64_t sctlr = __builtin_arm_rsr64("SCTLR_EL1");

		if (on) {
			sctlr |= HAKIOS_PAC_SCTLR_EN;
			state |= HAKIOS_PAC_CPU_SCTLR_EN;
		} else {
			sctlr &= ~HAKIOS_PAC_SCTLR_EN;
			state &= ~HAKIOS_PAC_CPU_SCTLR_EN;
		}
		__builtin_arm_wsr64("SCTLR_EL1", sctlr);
		changed = true;
	}

	cdp->hakios_pac_state = state;

	if (changed) {
		/*
		 * The exception return to EL0 is itself a context synchronization event while
		 * SCTLR_EL1.EOS is set (or FEAT_ExS is absent), which would make this unnecessary;
		 * it is kept so that correctness does not depend on EOS, and it is only reached
		 * when a register was actually written.
		 */
		__asm__ volatile ("isb" ::: "memory");
	}
}

/* The context-switch hook's out-of-line part: hakios_pac_mode != 0, interrupts off. */
void
hakios_pac_switch_slow(thread_t new)
{
	/* kernel threads never run at EL0: leave whatever the last user thread loaded */
	if (new->machine.upcb == NULL) {
		return;
	}
	/* the outgoing thread is still current and still owns this CPU's cpu_data */
	hakios_pac_load(getCpuDatap(), new);
}

/*
 * A thread's PAC mode or ids were changed (ml_thread_set_disable_user_jop / ml_thread_set_jop_pid).
 * exec does that to a new thread that has not run, and the next switch to it loads the keys; this
 * covers the case where the thread is the one running, which would otherwise return to EL0 with
 * the old state.
 */
void
hakios_pac_thread_changed(thread_t thread)
{
	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return;
	}
	if (thread != current_thread() || thread->machine.upcb == NULL) {
		return;
	}
	const boolean_t istate = ml_set_interrupts_enabled(FALSE);
	hakios_pac_load(getCpuDatap(), thread);
	(void)ml_set_interrupts_enabled(istate);
}

/*
 * This CPU's key registers and SCTLR_EL1 were (or are about to be found) reset: the core was
 * powered down through PSCI, or is coming up for the first time. Forget everything; the next
 * switch to a user thread reloads. A plain store, done even with hakios_pac=0 (nothing reads it).
 */
void
hakios_pac_cpu_invalidate(cpu_data_t *cdp)
{
	cdp->hakios_pac_state = 0;
	cdp->rop_key = 0;
	cdp->jop_key = 0;
}

/*
 * ---------------------------------------------------------------------------------------------
 * Step 2: signing and authenticating user pointers from EL1.
 *
 * Apple's pmap_sign_user_ptr() (osfmk/arm/pmap/pmap.c, HAS_APPLE_PAC) switches the A keys to the
 * target's with ml_enable_user_jop_key(), runs one pac instruction with interrupts off and
 * switches back. The same here with the architected registers:
 *
 *   1. mask interrupts (IRQ and FIQ): nothing else may run on this CPU, in particular no
 *      context switch (which reprograms the same registers) and no return to EL0;
 *   2. KEY: if the per-CPU cache says the register already holds the key of pac_id (the current
 *      thread is PAC-on with that id: the common case for a fault in the process itself),
 *      nothing; else read the register's two halves, and write the derived key;
 *   3. ENABLE: read SCTLR_EL1; if the key's En bit is clear, set that one bit;
 *   4. isb (only if 2 or 3 wrote), then the pac instruction(s);
 *   5. write back exactly what 2 and 3 read, isb (only if something was written);
 *   6. restore the interrupt state.
 *
 * Everything is put back to the value READ from the register, not to what the cache believes, so
 * the per-CPU cache (hakios_pac_state, jop_key, rop_key) is neither consulted for the restore nor
 * changed: whatever it said before is exactly as true afterwards. Restoring (rather than merely
 * invalidating the cache) is required: the current thread may be a PAC-on user thread in a system
 * call that returns to EL0 without a context switch.
 *
 * The interrupts-off window is: at most 4 mrs + 6 msr + 2 isb + 3 pac/xpac instructions and a
 * key derivation (a dozen arithmetic instructions) -- well under a microsecond, no loop, no call
 * that can block or take a lock.
 *
 * While the En bit is set at EL1 the only pac-family instructions executed are the ones below
 * (this kernel is ARM64 and has no others; xpaclri does not depend on SCTLR).
 *
 * NEVER an aut* instruction: these cores have FPAC (ID_AA64ISAR1_EL1.APA == 5), so a failed
 * authentication at EL1 is a synchronous exception in the kernel, i.e. a panic. Authentication is
 * therefore strip (xpaci/xpacd: no key, no enable, cannot fault), sign the stripped pointer again
 * and compare with what was given. That is exactly the architected definition of AUT* (Auth()
 * computes AddPAC(strip(ptr)) and compares), so it accepts and rejects the same pointers.
 *
 * Same PAC as EL0? The PAC of a pointer depends on: the key, the modifier, the pointer's bits
 * below the PAC field, and where the PAC field is, which AddPAC() takes from the translation
 * regime of the pointer's half (bit 55): TCR_EL1.T0SZ (bottom of the field) and TCR_EL1.TBI0 /
 * TBID0 (whether bits 63:56 belong to it). EL0 and EL1 share that regime (HCR_EL2.{E2H,TGE} are
 * not both set under the loader), and none of it depends on the exception level. So a user
 * pointer signed here has the PAC EL0 computes PROVIDED TCR_EL1 has, at this moment, the T0SZ /
 * TBI0 / TBID0 the target process runs with. On this kernel TCR_EL1 is one constant
 * (TCR_EL1_BOOT; no __ARM_MIXED_PAGE_SIZE__, so pmap_switch never rewrites it), TBI0 = 1 and
 * TBID0 = 0 (TCR_TBID0_ENABLE is 0 without HAS_APPLE_PAC): the PAC of code AND data pointers is
 * bits 54:(64-T0SZ), the top byte is ignored and preserved for both. Apple's arm64e sets TBID0
 * (code pointers get 8 more PAC bits); if TCR_TBID0_ENABLE is ever turned on, do it in the
 * constant so that it holds for every thread and at EL1 too, and nothing here changes.
 * ---------------------------------------------------------------------------------------------
 */

/* SCTLR_EL1's enable bit for one address key */
static inline uint64_t
hakios_pac_sctlr_bit(int key)
{
	switch (key) {
	case HAKIOS_PAC_KEY_IA: return 1ULL << 31;      /* EnIA */
	case HAKIOS_PAC_KEY_IB: return 1ULL << 30;      /* EnIB */
	case HAKIOS_PAC_KEY_DA: return 1ULL << 27;      /* EnDA */
	default:                return 1ULL << 13;      /* EnDB */
	}
}

static inline struct hakios_pac_key
hakios_pac_key_read(int key)
{
	struct hakios_pac_key k;

	switch (key) {
	case HAKIOS_PAC_KEY_IA:
		k.lo = __builtin_arm_rsr64(APIAKEYLO);
		k.hi = __builtin_arm_rsr64(APIAKEYHI);
		break;
	case HAKIOS_PAC_KEY_IB:
		k.lo = __builtin_arm_rsr64(APIBKEYLO);
		k.hi = __builtin_arm_rsr64(APIBKEYHI);
		break;
	case HAKIOS_PAC_KEY_DA:
		k.lo = __builtin_arm_rsr64(APDAKEYLO);
		k.hi = __builtin_arm_rsr64(APDAKEYHI);
		break;
	default:
		k.lo = __builtin_arm_rsr64(APDBKEYLO);
		k.hi = __builtin_arm_rsr64(APDBKEYHI);
		break;
	}
	return k;
}

static inline void
hakios_pac_key_write(int key, struct hakios_pac_key k)
{
	switch (key) {
	case HAKIOS_PAC_KEY_IA:
		__builtin_arm_wsr64(APIAKEYLO, k.lo);
		__builtin_arm_wsr64(APIAKEYHI, k.hi);
		break;
	case HAKIOS_PAC_KEY_IB:
		__builtin_arm_wsr64(APIBKEYLO, k.lo);
		__builtin_arm_wsr64(APIBKEYHI, k.hi);
		break;
	case HAKIOS_PAC_KEY_DA:
		__builtin_arm_wsr64(APDAKEYLO, k.lo);
		__builtin_arm_wsr64(APDAKEYHI, k.hi);
		break;
	default:
		__builtin_arm_wsr64(APDBKEYLO, k.lo);
		__builtin_arm_wsr64(APDBKEYHI, k.hi);
		break;
	}
}

/*
 * The instructions. `volatile` (a side effect the compiler cannot see through) keeps each where
 * it is written relative to the msr / isb around it. The modifier is never sp ("r" excludes it).
 */
__attribute__((always_inline))
static inline uint64_t
hakios_pac_insn_sign(int key, uint64_t ptr, uint64_t mod)
{
	switch (key) {
	case HAKIOS_PAC_KEY_IA:
		__asm__ volatile (".arch_extension pauth\n\tpacia %0, %1" : "+r"(ptr) : "r"(mod));
		break;
	case HAKIOS_PAC_KEY_IB:
		__asm__ volatile (".arch_extension pauth\n\tpacib %0, %1" : "+r"(ptr) : "r"(mod));
		break;
	case HAKIOS_PAC_KEY_DA:
		__asm__ volatile (".arch_extension pauth\n\tpacda %0, %1" : "+r"(ptr) : "r"(mod));
		break;
	default:
		__asm__ volatile (".arch_extension pauth\n\tpacdb %0, %1" : "+r"(ptr) : "r"(mod));
		break;
	}
	return ptr;
}

/* xpaci / xpacd: replace the PAC field by copies of bit 55. No key, no enable bit, never faults. */
__attribute__((always_inline))
static inline uint64_t
hakios_pac_insn_strip(int key, uint64_t ptr)
{
	if (key == HAKIOS_PAC_KEY_IA || key == HAKIOS_PAC_KEY_IB) {
		__asm__ volatile (".arch_extension pauth\n\txpaci %0" : "+r"(ptr));
	} else {
		__asm__ volatile (".arch_extension pauth\n\txpacd %0" : "+r"(ptr));
	}
	return ptr;
}

/*
 * The common part of sign and auth. Returns pac<key>(ptr, disc) under pac_id's key; when
 * `stripped` is not NULL, ptr is stripped first (inside the same window) and *stripped gets the
 * stripped pointer. hakios_pac_mode != 0, key is one of the four address keys.
 */
static uint64_t
hakios_pac_compute(uint64_t ptr, int key, uint64_t disc, uint64_t pac_id, uint64_t *stripped)
{
	const bool jop = (key == HAKIOS_PAC_KEY_IA || key == HAKIOS_PAC_KEY_DA);
	const uint64_t en = hakios_pac_sctlr_bit(key);
	const struct hakios_pac_key want = hakios_pac_derive(key, pac_id);      /* pure: outside the window */
	struct hakios_pac_key saved = { 0, 0 };
	bool key_written = false, sctlr_written = false;
	uint64_t res;

	const boolean_t istate = ml_set_interrupts_enabled(FALSE);
	const cpu_data_t *cdp = getCpuDatap();

	/* 2. the key */
	if (!((cdp->hakios_pac_state & HAKIOS_PAC_CPU_KEYS_VALID) &&
	    (jop ? cdp->jop_key : cdp->rop_key) == pac_id)) {
		saved = hakios_pac_key_read(key);
		if (saved.lo != want.lo || saved.hi != want.hi) {
			hakios_pac_key_write(key, want);
			key_written = true;
		}
	}

	/* 3. the enable bit */
	const uint64_t sctlr = __builtin_arm_rsr64("SCTLR_EL1");
	if ((sctlr & en) == 0) {
		__builtin_arm_wsr64("SCTLR_EL1", sctlr | en);
		sctlr_written = true;
	}

	/* 4. */
	if (key_written || sctlr_written) {
		__asm__ volatile ("isb" ::: "memory");
	}
	if (stripped != NULL) {
		ptr = hakios_pac_insn_strip(key, ptr);
		*stripped = ptr;
	}
	res = hakios_pac_insn_sign(key, ptr, disc);

	/* 5. back to what was read */
	if (sctlr_written) {
		__builtin_arm_wsr64("SCTLR_EL1", sctlr);
	}
	if (key_written) {
		hakios_pac_key_write(key, saved);
	}
	if (key_written || sctlr_written) {
		__asm__ volatile ("isb" ::: "memory");
	}

	(void)ml_set_interrupts_enabled(istate);
	return res;
}

static inline bool
hakios_pac_key_is_addr(int key)
{
	return key == HAKIOS_PAC_KEY_IA || key == HAKIOS_PAC_KEY_IB ||
	       key == HAKIOS_PAC_KEY_DA || key == HAKIOS_PAC_KEY_DB;
}

uint64_t
hakios_pac_sign_user_ptr(uint64_t ptr, int key, uint64_t disc, uint64_t pac_id)
{
	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return ptr;
	}
	if (!hakios_pac_key_is_addr(key)) {
		panic("hakios_pac_sign_user_ptr: key %d", key);
	}
	return hakios_pac_compute(ptr, key, disc, pac_id, NULL);
}

/*
 * The stripped pointer if signed_ptr is what pac<key>(stripped, disc) gives under pac_id's key.
 * Otherwise the stripped pointer made non-canonical the way a pre-FPAC AUT* does: error code 1
 * (A keys) or 2 (B keys), here XORed into bits 54:53 AND 62:61 -- the two places the architecture
 * uses with and without top-byte-ignore -- so that the result faults at EL0 as a branch target
 * and as a data address under any TCR_EL1.TBI0/TBID0, and can never equal a valid pointer.
 */
uint64_t
hakios_pac_auth_user_ptr(uint64_t signed_ptr, int key, uint64_t disc, uint64_t pac_id)
{
	uint64_t stripped = signed_ptr;

	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return signed_ptr;
	}
	if (!hakios_pac_key_is_addr(key)) {
		panic("hakios_pac_auth_user_ptr: key %d", key);
	}
	const uint64_t resigned = hakios_pac_compute(signed_ptr, key, disc, pac_id, &stripped);

	if (resigned == signed_ptr) {
		return stripped;
	}
	return stripped ^ HAKIOS_PAC_POISON((key == HAKIOS_PAC_KEY_IA || key == HAKIOS_PAC_KEY_DA) ? 1 : 2);
}

/* Step 4: ptrauth_strip() of a user pointer. xpaci/xpacd need no key and no SCTLR_EL1 bit. */
uint64_t
hakios_pac_strip_user_ptr(uint64_t ptr, int key)
{
	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return ptr;
	}
	if (!hakios_pac_key_is_addr(key)) {
		panic("hakios_pac_strip_user_ptr: key %d", key);
	}
	return hakios_pac_insn_strip(key, ptr);
}

/*
 * Boot self-test, called once from kernel_bootstrap_thread() right after interrupts are enabled
 * (the EARLY_BOOT startup phase still runs with them off): boot CPU, no user thread exists yet.
 * Each line goes to the kernel log and to the loader's log (klogsave).
 *
 * PASS for a key means, with a fake user pointer P, a blended modifier and a made-up id:
 *   - sign(P) twice gives the same value S, and S != P (a PAC was inserted);
 *   - auth(S) == P;
 *   - auth(S with PAC bit 54 flipped) == P ^ poison -- and the kernel is still alive;
 *   - auth(S) with another modifier is poisoned (*);
 * and the summary line: SCTLR_EL1 and the four key registers read the same after as before.
 * Also printed, not judged: S under another id (*).
 * (*) a PAC is only 64 - T0SZ - 9 bits wide, so two unrelated PACs do collide once in a while;
 * the first sign retries with another modifier when the PAC is all zeroes, and a wrong-modifier
 * collision is reported as "disc?" without failing the key.
 */
#define HAKIOS_PAC_TEST_PTR     0x0000000123456780ULL
#define HAKIOS_PAC_TEST_SLOT    0x000000016f001238ULL   /* the "storage address" blended into the modifier */
#define HAKIOS_PAC_TEST_ID      0x48414b494f533254ULL

static void
hakios_pac_selftest_log(const char *line)
{
	extern void loader_log_line(const char *line);

	printf("%s", line);
	loader_log_line(line);
}

static bool
hakios_pac_selftest_key(int key, const char *name)
{
	const uint64_t p = HAKIOS_PAC_TEST_PTR;
	const uint64_t id = HAKIOS_PAC_TEST_ID;
	uint64_t disc16 = 0xc0de, disc, s1;
	char line[256];

	for (int tries = 0;; tries++) {
		disc = hakios_pac_blend(HAKIOS_PAC_TEST_SLOT, disc16);
		s1 = hakios_pac_sign_user_ptr(p, key, disc, id);
		if (s1 != p || tries == 7) {
			break;
		}
		disc16++;       /* an all-zero PAC: legitimate but useless here */
	}
	const uint64_t s2 = hakios_pac_sign_user_ptr(p, key, disc, id);
	const uint64_t ok = hakios_pac_auth_user_ptr(s1, key, disc, id);
	const uint64_t bad = hakios_pac_auth_user_ptr(s1 ^ (1ULL << 54), key, disc, id);
	const uint64_t wdisc = hakios_pac_auth_user_ptr(s1, key, disc ^ (1ULL << 48), id);
	const uint64_t other = hakios_pac_sign_user_ptr(p, key, disc, id + 1);
	const uint64_t poison = p ^ HAKIOS_PAC_POISON((key == HAKIOS_PAC_KEY_IA || key == HAKIOS_PAC_KEY_DA) ? 1 : 2);

	const bool pass = (s1 != p) && (s1 == s2) && (ok == p) && (bad == poison);

	snprintf(line, sizeof(line),
	    "hakios_pac: selftest %s ptr=0x%llx disc=0x%llx sign=0x%llx resign=0x%llx auth_ok=0x%llx auth_bad=0x%llx auth_wrongdisc=0x%llx%s sign_otherid=0x%llx %s\n",
	    name, p, disc, s1, s2, ok, bad, wdisc, wdisc == poison ? "" : " (disc?)", other,
	    pass ? "PASS" : "FAIL");
	hakios_pac_selftest_log(line);
	return pass;
}

void
hakios_pac_selftest(void)
{
	static const struct {
		int key;
		const char *name;
	} keys[] = {
		{ HAKIOS_PAC_KEY_IA, "IA" }, { HAKIOS_PAC_KEY_DA, "DA" },
		{ HAKIOS_PAC_KEY_IB, "IB" }, { HAKIOS_PAC_KEY_DB, "DB" },
	};
	struct hakios_pac_key before[4], after[4];
	char line[256];
	bool pass = true, restored = true;

	if (hakios_pac_mode == HAKIOS_PAC_MODE_OFF) {
		return;
	}

	/* one CPU for the before/after comparison of its registers */
	disable_preemption();
	const int cpu = cpu_number();
	const uint64_t sctlr0 = __builtin_arm_rsr64("SCTLR_EL1");
	for (int i = 0; i < 4; i++) {
		before[i] = hakios_pac_key_read(keys[i].key);
	}
	enable_preemption();

	for (int i = 0; i < 4; i++) {
		pass = hakios_pac_selftest_key(keys[i].key, keys[i].name) && pass;
	}

	/*
	 * Only meaningful if still on the same CPU and nothing switched to a user thread in
	 * between; at this point of the boot there is one CPU and no user thread.
	 */
	disable_preemption();
	const int cpu1 = cpu_number();
	const uint64_t sctlr1 = __builtin_arm_rsr64("SCTLR_EL1");
	for (int i = 0; i < 4; i++) {
		after[i] = hakios_pac_key_read(keys[i].key);
	}
	enable_preemption();
	if (cpu1 == cpu) {
		restored = (sctlr0 == sctlr1);
		for (int i = 0; i < 4; i++) {
			restored = restored && before[i].lo == after[i].lo && before[i].hi == after[i].hi;
		}
	}

	const uint64_t tcr = __builtin_arm_rsr64("TCR_EL1");
	snprintf(line, sizeof(line),
	    "hakios_pac: selftest cpu %d TCR_EL1=0x%llx (T0SZ=%u TBI0=%u TBID0=%u) SCTLR_EL1 0x%llx -> 0x%llx, keys %s: %s\n",
	    cpu, tcr, (unsigned)(tcr & 0x3f), (unsigned)((tcr >> 37) & 1), (unsigned)((tcr >> 51) & 1),
	    sctlr0, sctlr1,
	    cpu1 != cpu ? "not compared (moved cpu)" : restored ? "restored" : "NOT RESTORED",
	    (pass && restored) ? "PASS" : "FAIL");
	hakios_pac_selftest_log(line);
}

/*
 * Boot-args and the per-boot secrets. TUNABLES phase: boot CPU only, long before the first user
 * thread and before task_init() asks for ml_default_jop_pid().
 */
__startup_func
static void
hakios_pac_init(void)
{
	int mode = 0;

	if (!PE_parse_boot_argn("hakios_pac", &mode, sizeof(mode)) || mode == HAKIOS_PAC_MODE_OFF) {
		return;         /* off: hakios_pac_mode stays 0, no register is read or written */
	}
	if (mode != HAKIOS_PAC_MODE_NAMED && mode != HAKIOS_PAC_MODE_ALL) {
		return;         /* an unknown value is off, not "all" */
	}

	/*
	 * Address authentication must exist (boot CPU; step 0 found every core alike): APA (bits
	 * 7:4) or API (11:8) of ID_AA64ISAR1_EL1, or APA3 (15:12) of ID_AA64ISAR2_EL1. Without it
	 * the key registers are UNDEFINED, so stay off. Generic auth (GPA 27:24, GPI 31:28, GPA3
	 * ISAR2 11:8) decides whether APGAKey may be touched.
	 */
	const uint64_t isar1 = __builtin_arm_rsr64(ID_AA64ISAR1);
	const uint64_t isar2 = __builtin_arm_rsr64(ID_AA64ISAR2);

	if (((isar1 >> 4) & 0xff) == 0 && ((isar2 >> 12) & 0xf) == 0) {
		return;
	}
	hakios_pac_has_ga = ((isar1 >> 24) & 0xff) != 0 || ((isar2 >> 8) & 0xf) != 0;

	hakios_pac_proc[0] = '\0';
	(void)PE_parse_boot_arg_str("hakios_pac_proc", hakios_pac_proc, (int)sizeof(hakios_pac_proc));

	for (int i = 0; i < HAKIOS_PAC_NKEYS; i++) {
		hakios_pac_salt[i].lo = early_random();
		hakios_pac_salt[i].hi = early_random();
	}
	hakios_pac_default_jop = early_random();

	hakios_pac_mode = mode;         /* last: everything above is in place */
}
STARTUP(TUNABLES, STARTUP_RANK_MIDDLE, hakios_pac_init);

/*
 * Step 3 helpers for the two fixup pagers (see hakios_pac.h).
 */
bool
hakios_pac_task_signs(task_t task)
{
	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return false;
	}
	return task != TASK_NULL && task != kernel_task &&
	       task->map != VM_MAP_NULL && task->map->pmap != NULL &&
	       !task->map->pmap->disable_jop;
}

void
hakios_pac_task_ids(task_t task, uint64_t *jop_id, uint64_t *rop_id)
{
	*jop_id = task->jop_pid;
	*rop_id = task->rop_pid;
}

void
hakios_pac_task_name(task_t task, char *buf, size_t bufsize, int *pid)
{
	extern char *proc_name_address(void *p);
	void *p = (task != TASK_NULL) ? get_bsdtask_info(task) : NULL;

	strlcpy(buf, p != NULL ? proc_name_address(p) : "?", bufsize);
	*pid = p != NULL ? proc_pid((struct proc *)p) : -1;
}

void
hakios_pac_log(const char *fmt, ...)
{
	extern void loader_log_line(const char *line);
	extern int vsnprintf(char *, size_t, const char *, va_list) __printflike(3, 0);
	char line[320];
	va_list ap;

	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return;
	}
	va_start(ap, fmt);
	vsnprintf(line, sizeof(line), fmt, ap);
	va_end(ap);
	printf("%s", line);
	loader_log_line(line);
}

/* One line in the log once the console can take it, and only when the feature is on. */
__startup_func
static void
hakios_pac_report(void)
{
	if (hakios_pac_mode != HAKIOS_PAC_MODE_OFF) {
		printf("hakios_pac: user pointer authentication ON, mode %d (%s%s), generic key %s\n",
		    hakios_pac_mode,
		    hakios_pac_mode == HAKIOS_PAC_MODE_ALL ? "every arm64e image" : "hakios_pac_proc=",
		    hakios_pac_mode == HAKIOS_PAC_MODE_ALL ? "" : hakios_pac_proc,
		    hakios_pac_has_ga ? "managed" : "absent");
	}
}
STARTUP(EARLY_BOOT, STARTUP_RANK_MIDDLE, hakios_pac_report);

#endif /* HAKIOS_USER_PAC */
