/*
 * HakiOS PATH 2: architected (ARMv8.3 FEAT_PAuth) pointer authentication for EL0, managed by a
 * kernel that is itself plain ARM64 and never signs or authenticates its own pointers.
 * See osfmk/arm64/hakios_pac.c.
 */
#ifndef _ARM64_HAKIOS_PAC_H_
#define _ARM64_HAKIOS_PAC_H_

#if HAKIOS_USER_PAC

#include <kern/kern_types.h>
#include <kern/thread.h>

/* boot-arg hakios_pac: 0 off (default), 1 only the processes named in hakios_pac_proc=, 2 every arm64e image */
#define HAKIOS_PAC_MODE_OFF     0
#define HAKIOS_PAC_MODE_NAMED   1
#define HAKIOS_PAC_MODE_ALL     2

/* cpu_data_t.hakios_pac_state: what is known about this CPU's PAC registers. 0 = nothing. */
#define HAKIOS_PAC_CPU_KEYS_VALID       0x01    /* APIA/APDA hold jop_key's keys, APIB/APDB rop_key's */
#define HAKIOS_PAC_CPU_GA_VALID         0x02    /* APGA holds the per-boot generic key */
#define HAKIOS_PAC_CPU_SCTLR_KNOWN      0x04    /* HAKIOS_PAC_CPU_SCTLR_EN reflects SCTLR_EL1 */
#define HAKIOS_PAC_CPU_SCTLR_EN         0x08    /* SCTLR_EL1.En{IA,IB,DA,DB} are all set (else all clear) */

/* 0 until the TUNABLES startup phase, and for good when boot-arg hakios_pac is absent or 0 */
extern int hakios_pac_mode;

struct cpu_data;

extern void hakios_pac_switch_slow(thread_t new);
extern void hakios_pac_thread_changed(thread_t thread);
extern void hakios_pac_cpu_invalidate(struct cpu_data *cdp);

/*
 * Step 2: the signing primitives. The address-auth key registers, numbered like <ptrauth.h>'s
 * ptrauth_key (asia 0, asib 1, asda 2, asdb 3) so that a ptrauth_key can be passed as is.
 */
typedef enum {
	HAKIOS_PAC_KEY_IA = 0,          /* APIAKey <- jop id */
	HAKIOS_PAC_KEY_IB = 1,          /* APIBKey <- rop id */
	HAKIOS_PAC_KEY_DA = 2,          /* APDAKey <- jop id */
	HAKIOS_PAC_KEY_DB = 3,          /* APDBKey <- rop id */
	HAKIOS_PAC_KEY_GA = 4,          /* APGAKey: per boot, not valid for the two functions below */
	HAKIOS_PAC_NKEYS
} hakios_pac_key_t;

/*
 * Sign / authenticate a USER pointer (bit 55 clear) the way EL0 would with the key that pac_id
 * (a task's jop_pid for IA/DA, rop_pid for IB/DB) derives to. `disc` is the complete 64-bit
 * modifier (already blended where the ABI blends). With hakios_pac=0 both return their argument.
 * auth returns the stripped pointer, or on a mismatch a non-canonical (poisoned) one; it never
 * traps. Callable from any context; interrupts are masked for a few dozen instructions.
 */
extern uint64_t hakios_pac_sign_user_ptr(uint64_t ptr, int key, uint64_t disc, uint64_t pac_id);
extern uint64_t hakios_pac_auth_user_ptr(uint64_t signed_ptr, int key, uint64_t disc, uint64_t pac_id);

/*
 * Step 4: xpaci (IA/IB) / xpacd (DA/DB) on a user pointer: the PAC field replaced by copies of bit 55.
 * No key and no enable bit are involved. With hakios_pac=0 returns its argument.
 */
extern uint64_t hakios_pac_strip_user_ptr(uint64_t ptr, int key);

/*
 * Step 4 (thread state: osfmk/arm64/status.c): the values of Apple's ptrauth_string_discriminator()
 * for the strings the user thread-state ABI uses (<mach/arm/_structs.h>, libsystem_platform's
 * _sigtramp, xnu's own status.c). This compiler has no __builtin_ptrauth_string_discriminator in
 * arm64 mode; the numbers are what `clang -target arm64e-apple-ios` gives for it (same run gives
 * "recover" = 0x1e02 = proc_reg.h's PAC_DISCRIMINATOR_RECOVER).
 */
#define HAKIOS_PAC_DISC_PC      0x7481ULL       /* ptrauth_string_discriminator("pc") */
#define HAKIOS_PAC_DISC_LR      0x77d3ULL       /* ptrauth_string_discriminator("lr") */
#define HAKIOS_PAC_DISC_SP      0xcbedULL       /* ptrauth_string_discriminator("sp") */
#define HAKIOS_PAC_DISC_FP      0x4517ULL       /* ptrauth_string_discriminator("fp") */
#define HAKIOS_PAC_DISC_NONCE   0xb743ULL       /* ptrauth_string_discriminator("nonce") */
#define HAKIOS_PAC_DISC_UCTX    0xd4a6ULL       /* ptrauth_string_discriminator("uctx") */

/* What a failed hakios_pac_auth_user_ptr() XORs into the stripped pointer: see hakios_pac.c. */
#define HAKIOS_PAC_POISON(code) (((uint64_t)(code) << 53) | ((uint64_t)(code) << 61))

/* ptrauth_blend_discriminator(): the low 48 bits of the storage address under a 16-bit discriminator. */
static inline uint64_t
hakios_pac_blend(uint64_t addr, uint64_t disc16)
{
	return (addr & 0x0000ffffffffffffULL) | (disc16 << 48);
}

/*
 * Step 3 (signed fixups: osfmk/vm/vm_dyld_pager.c, vm_shared_region*.c).
 *
 * hakios_pac_task_signs(): does this task run PAC-on, i.e. must the pointers the kernel fixes up
 * for it be signed? (hakios_pac != 0 and its pmap's disable_jop clear.) Never true with hakios_pac=0.
 * hakios_pac_task_ids(): the ids its A (jop) and B (rop) keys derive from.
 * hakios_pac_log(): one line to the kernel log and to the loader log (boot klogsave). Prints nothing
 * with hakios_pac=0. The line must end in '\n'.
 */
struct task;
extern bool hakios_pac_task_signs(struct task *task);
extern void hakios_pac_task_ids(struct task *task, uint64_t *jop_id, uint64_t *rop_id);
extern void hakios_pac_task_name(struct task *task, char *buf, size_t bufsize, int *pid);
extern void hakios_pac_log(const char *fmt, ...) __printflike(1, 2);

/* Boot self-test of the above; call once, interrupts enabled. Does nothing with hakios_pac=0. */
extern void hakios_pac_selftest(void);

/*
 * Context switch hook (machine_switch_pmap_and_extended_context): interrupts off, the outgoing
 * thread still owns this CPU's cpu_data. With hakios_pac=0 this is one load and one branch.
 */
static inline void
hakios_pac_switch(thread_t old __unused, thread_t new)
{
	if (__probable(hakios_pac_mode == HAKIOS_PAC_MODE_OFF)) {
		return;
	}
	hakios_pac_switch_slow(new);
}

#endif /* HAKIOS_USER_PAC */
#endif /* _ARM64_HAKIOS_PAC_H_ */
