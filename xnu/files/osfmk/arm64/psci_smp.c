/*
 * More than one processor on generic (non-Apple) ARMv8 boards.
 *
 * On Apple hardware the other cores are powered up by closed drivers. Here
 * the firmware does it, through the standard power interface of ARM systems
 * (PSCI): CPU_ON starts a core at an address of the caller's choice, CPU_OFF
 * stops the calling core for good. The address we give is the kernel's own
 * reset vector (LowResetVectorBase in start.s), so that a new core takes the
 * path Apple's cores take: it finds its cpu data by its MPIDR, turns the MMU
 * on and comes out in arm_init_cpu(). The calls are made with SMC or, under
 * a hypervisor that implements them itself, with HVC; the boot loader says
 * which (/chosen/psci-method).
 *
 * The platform expert (iokit/Kernel/GenericARMPE.cpp) registers the cores of
 * the device tree and starts them here, one after the other. Signals between
 * processors are interrupts of the GIC (gic_v3_irq.c).
 *
 * A new core begins with its MMU and caches off, and what it reads then it
 * reads from memory. On real hardware that is not the same as what the
 * running processors see: under Qualcomm's hypervisor (the phone this was
 * written for) a line that is dirty in another core's cache is simply not
 * there yet. QEMU has no cache and shows none of this. So before CPU_ON
 * everything that core will read before its MMU is on is written back to
 * memory (psci_smp_start_cpu()), and what it writes in that state is read
 * only after throwing away the reader's cached copy.
 *
 * The other half is the way out. The kernel ends by returning to the boot
 * loader (loader_return.c), and the loader gets back only what it started:
 * one core, the boot processor's. psci_smp_stop_others() makes every other
 * core switch itself off first, wherever it is.
 */
#include <mach/mach_types.h>
#include <kern/clock.h>
#include <kern/cpu_number.h>
#include <kern/debug.h>
#include <kern/kalloc.h>
#include <kern/misc_protos.h>
#include <libkern/kernel_mach_header.h>
#include <machine/atomic.h>
#include <arm/caches_internal.h>
#include <arm/cpu_data_internal.h>
#include <arm/cpu_internal.h>
#include <arm/machine_routines.h>
#include <arm64/proc_reg.h>
#include <arm64/psci_smp.h>
#include <pexpert/pexpert.h>
#include <pexpert/device_tree.h>
#include <pexpert/arm64/board_config.h>
#include <vm/pmap.h>

#if defined(GENERIC_PSCI_SMP)

#define PSCI_CPU_OFF            0x84000002u
#define PSCI_CPU_ON             0xC4000003u
#define PSCI_AFFINITY_INFO      0xC4000004u
#define PSCI_AFFINITY_OFF       1

#define PSCI_SMP_START_SECONDS  5       /* how long a started core is waited for */
#define PSCI_SMP_STOP_SECONDS   3       /* ... and the cores that were told to stop */

extern void loader_return(void);
extern void loader_log_putc(char c);
extern unsigned long gVirtBase, gPhysBase;

/* The firmware is called with HVC, not SMC. (start.s reads this too, at its physical address.) */
uint32_t psci_smp_hvc;
/*
 * Where a new core goes when the firmware starts it at EL2 (a loader that was itself started at EL2: the Pixel 8a):
 * the loader's trampoline (/chosen/el2-trampoline), which takes it down to EL1 and on to the reset vector, given to
 * it as CPU_ON's context. 0: the reset vector itself, as before.
 */
static uint64_t psci_smp_el2_trampoline;

/*
 * Left by a core that took an exception between its first instruction and the kernel's vectors
 * (Lsecondary_early_exception in start.s). That core writes it with its cache off; it has a
 * cache line to itself so that looking at memory does not disturb anything else.
 */
struct psci_smp_start_fault {
	uint64_t valid, mpidr, esr, elr, far, lr, sp, unused;
} psci_smp_start_fault __attribute__((aligned(64)));
_Static_assert(sizeof(struct psci_smp_start_fault) == 64, "one cache line");

static uint32_t psci_smp_cpu_stage[MAX_CPUS];   /* PSCI_SMP_CPU_*, by cpu number */
static uint32_t psci_smp_started;               /* a bit per core that CPU_ON was granted for */

static int      psci_smp_stop_owner = -1;       /* the processor that began the return to the boot loader */
static bool     psci_smp_boot_wanted;           /* it was not the boot processor: that one is to finish */
static bool     psci_smp_boot_busy;             /* ... and has begun to */
static bool     psci_smp_stopping;              /* every other core switches itself off */

static int64_t
psci_call(uint64_t function, uint64_t arg1, uint64_t arg2, uint64_t arg3)
{
	register uint64_t x0 __asm__("x0") = function;
	register uint64_t x1 __asm__("x1") = arg1;
	register uint64_t x2 __asm__("x2") = arg2;
	register uint64_t x3 __asm__("x3") = arg3;

	if (psci_smp_hvc) {
		__asm__ volatile ("hvc #0" : "+r" (x0), "+r" (x1), "+r" (x2), "+r" (x3) : :
                    "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17");
	} else {
		__asm__ volatile ("smc #0" : "+r" (x0), "+r" (x1), "+r" (x2), "+r" (x3) : :
                    "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17");
	}
	return (int64_t)x0;
}

static const char *
psci_error(int64_t status)
{
	static const char *const names[] = {
		"success", "not supported", "invalid parameters", "denied", "already on", "on pending",
		"internal failure", "not present", "disabled", "invalid address",
	};

	return (status <= 0 && status > -(int64_t)(sizeof(names) / sizeof(names[0]))) ? names[-status] : "?";
}

void
psci_smp_init(void)
{
	DTEntry chosen;
	void const *prop;
	unsigned int size;

	if (SecureDTLookupEntry(NULL, "/chosen", &chosen) == kSuccess &&
	    SecureDTGetProperty(chosen, "psci-method", &prop, &size) == kSuccess &&
	    size >= 3 && strncmp((const char *)prop, "hvc", 3) == 0) {
		psci_smp_hvc = 1;
	}
	if (SecureDTLookupEntry(NULL, "/chosen", &chosen) == kSuccess &&
	    SecureDTGetProperty(chosen, "el2-trampoline", &prop, &size) == kSuccess && size == sizeof(uint64_t)) {
		psci_smp_el2_trampoline = *(uint64_t const *)prop;
		printf("psci: the other cores start at EL2: through the loader's trampoline at 0x%llx\n",
		    psci_smp_el2_trampoline);
	}
	psci_idle_init();
}

/* A processor other than the boot processor says how far it has come. */
void
psci_smp_cpu_reached(uint32_t stage)
{
	os_atomic_store(&psci_smp_cpu_stage[cpu_number()], stage, release);
}

/* ... all the way: from GenericARMCPU::initCPU(), where the scheduler takes it. */
void
psci_smp_cpu_running(void)
{
	psci_smp_cpu_reached(PSCI_SMP_CPU_RUNNING);
}

static void
psci_smp_clean_segment(const char *name)
{
	unsigned long size = 0;
	vm_offset_t start = (vm_offset_t)getsegdatafromheader(&_mh_execute_header, name, &size);

	if (start != 0 && size != 0) {
		CleanPoC_DcacheRegion(start, size);
	}
}

/*
 * Start a registered processor: the platform's half of cpu_start(). Waits until the core has
 * joined the scheduler, and says what became of it if it has not: on a phone the log is all
 * there is to go by.
 */
kern_return_t
psci_smp_start_cpu(int cpu)
{
	cpu_data_t *cdp = cpu_datap(cpu);
	uint32_t phys_id = cdp->cpu_phys_id, stage;
	uint64_t entry = ml_static_vtop((vm_offset_t)&LowResetVectorBase);
	uint64_t now, deadline;
	int64_t status;

	/*
	 * What that core reads with its MMU off: the code and the page tables of start.s, and what
	 * the reset vector finds its way by (ResetHandlerData, CpuDataEntries, the boot arguments,
	 * cpu_ttep, psci_smp_hvc ...), all of it in the kernel's own segments; and the cpu data of
	 * every processor, which the reset vector goes through looking for its own.
	 * Clean, not flush: the processors that are running keep using all of it.
	 */
	psci_smp_start_fault.valid = 0;
	psci_smp_clean_segment("__TEXT");
	psci_smp_clean_segment("__DATA_CONST");
	psci_smp_clean_segment("__TEXT_EXEC");
	psci_smp_clean_segment("__DATA");
	psci_smp_clean_segment("__BOOTDATA");
	for (int other = 0; other <= ml_get_max_cpu_number(); other++) {
		if (cpu_datap(other) != NULL) {
			CleanPoC_DcacheRegion((vm_offset_t)cpu_datap(other), sizeof(cpu_data_t));
		}
	}
	__builtin_arm_dsb(DSB_SY);

	os_atomic_store(&psci_smp_cpu_stage[cpu], PSCI_SMP_CPU_ASKED, release);
	if (psci_smp_el2_trampoline != 0) {
		status = psci_call(PSCI_CPU_ON, phys_id, psci_smp_el2_trampoline, entry);
	} else {
		status = psci_call(PSCI_CPU_ON, phys_id, entry, 0);
	}
	if (status != 0) {
		os_atomic_store(&psci_smp_cpu_stage[cpu], 0, release);
		printf("psci: cpu %d (MPIDR 0x%x) was not started: the firmware answers %lld (%s)\n",
		    cpu, phys_id, status, psci_error(status));
		return KERN_FAILURE;
	}
	os_atomic_or(&psci_smp_started, 1u << cpu, relaxed);

	clock_interval_to_deadline(PSCI_SMP_START_SECONDS, NSEC_PER_SEC, &deadline);
	do {
		stage = os_atomic_load(&psci_smp_cpu_stage[cpu], acquire);
		if (stage >= PSCI_SMP_CPU_RUNNING) {
			return KERN_SUCCESS;
		}
		/* memory, not this processor's copy of it */
		FlushPoC_DcacheRegion((vm_offset_t)&psci_smp_start_fault, sizeof(psci_smp_start_fault));
		if (psci_smp_start_fault.valid != 0) {
			break;
		}
		delay_for_interval(1, NSEC_PER_MSEC);
		now = mach_absolute_time();
	} while (now < deadline);

	if (psci_smp_start_fault.valid != 0) {
		struct psci_smp_start_fault *f = &psci_smp_start_fault;
		uint64_t to_virtual = gVirtBase - gPhysBase;

		printf("psci: cpu %d (MPIDR 0x%llx) took an exception on its way in and has switched itself off: "
		    "ESR 0x%llx ELR 0x%llx FAR 0x%llx LR 0x%llx SP 0x%llx (ELR and LR as kernel addresses: 0x%llx 0x%llx)\n",
		    cpu, f->mpidr, f->esr, f->elr, f->far, f->lr, f->sp, f->elr + to_virtual, f->lr + to_virtual);
		os_atomic_andnot(&psci_smp_started, 1u << cpu, relaxed);        /* nothing of it is left to stop */
	} else {
		printf("psci: cpu %d (MPIDR 0x%x) was started %d seconds ago and has not joined: %s\n",
		    cpu, phys_id, PSCI_SMP_START_SECONDS,
		    stage >= PSCI_SMP_CPU_IN_KERNEL ? "it got as far as arm_init_cpu()" : "no sign of it since the firmware took the request");
	}
	return KERN_FAILURE;
}

/*
 * The calling processor switches itself off. Its interrupted work, if any, is abandoned where
 * it was: only for the end of the system.
 */
void
psci_smp_cpu_off(void)
{
	__builtin_arm_wsr("DAIFSet", DAIFSC_ALL);
	/* leave nothing behind that would interrupt a core that is no longer there */
	__builtin_arm_wsr64("CNTV_CTL_EL0", CNTV_CTL_EL0_IMASKED);
	gic_v3_irq_cpu_shutdown();
	os_atomic_store(&psci_smp_cpu_stage[cpu_number()], PSCI_SMP_CPU_OFF, release);
	__builtin_arm_dsb(DSB_SY);

	(void) psci_call(PSCI_CPU_OFF, 0, 0, 0);
	for (;;) {
		__builtin_arm_wfi();    /* the firmware refused: stay out of the way all the same */
	}
}

static void
psci_smp_note(const char *what, int cpu)
{
	static const char prefix[] = "\npsci: cpu ";

	for (const char *c = prefix; *c; c++) {
		loader_log_putc(*c);
	}
	loader_log_putc((char)('0' + cpu));
	for (const char *c = what; *c; c++) {
		loader_log_putc(*c);
	}
}

/* Tell every core but this one to switch itself off (psci_smp_stop_here() is where they hear it). */
static void
psci_smp_stop_rest(void)
{
	int me = cpu_number(), max_cpu = ml_get_max_cpu_number();

	os_atomic_store(&psci_smp_stopping, true, release);
	__builtin_arm_dsb(DSB_SY);
	for (int cpu = 0; cpu <= max_cpu; cpu++) {
		cpu_data_t *cdp = cpu_datap(cpu);

		if (cpu != me && cdp != NULL && (cpu == master_cpu || (psci_smp_started & (1u << cpu)))) {
			gic_v3_irq_signal_stop(cdp->cpu_phys_id);
		}
	}
	/* for those that wait with their interrupts off (wait_while_mp_kdp_trap(), after a panic) */
	__builtin_arm_sev();
}

/*
 * The system is going back to the boot loader (loader_return()). Returns on the boot processor
 * only, once the other cores are off -- or have been given up on. On any other processor it
 * does not return: that one hands the job to the boot processor, which the loader's side of
 * the return belongs to, and switches itself off like the rest.
 */
void
psci_smp_stop_others(void)
{
	int me, max_cpu;
	uint64_t now, deadline;

	if (psci_smp_started == 0) {
		return;                 /* there is only this one */
	}
	__builtin_arm_wsr("DAIFSet", DAIFSC_ALL);
	me = cpu_number();
	max_cpu = ml_get_max_cpu_number();
	if (!os_atomic_cmpxchg(&psci_smp_stop_owner, -1, me, acq_rel) && psci_smp_stop_owner != me && me != master_cpu) {
		psci_smp_cpu_off();     /* another processor is at it already */
	}

	if (me != master_cpu) {
		/*
		 * Nobody is stopped before the boot processor has answered: a core that is switched
		 * off while it holds a lock the boot processor is waiting for would take the return
		 * with it.
		 */
		psci_smp_note(" ends the system: the return to the boot loader is the boot processor's to make\n", me);
		os_atomic_store(&psci_smp_boot_wanted, true, release);
		__builtin_arm_dsb(DSB_SY);
		gic_v3_irq_signal_stop(cpu_datap(master_cpu)->cpu_phys_id);
		__builtin_arm_sev();
		now = mach_absolute_time();
		nanoseconds_to_absolutetime(PSCI_SMP_STOP_SECONDS * NSEC_PER_SEC, &deadline);
		deadline += now;
		while (!os_atomic_load(&psci_smp_stopping, acquire) && now < deadline) {
			now = mach_absolute_time();
		}
		if (!os_atomic_load(&psci_smp_stopping, acquire)) {
			/*
			 * It does not hear (stuck with its interrupts off). There is no return without
			 * it; a loader that watches from its own core will find the kernel silent and
			 * take the log as it is, and so this goes into the log.
			 */
			psci_smp_note(" cannot return to the boot loader and the boot processor does not answer: all cores stop here\n", me);
			psci_smp_stop_rest();
		}
		psci_smp_cpu_off();
	}

	psci_smp_stop_rest();
	now = mach_absolute_time();
	nanoseconds_to_absolutetime(PSCI_SMP_STOP_SECONDS * NSEC_PER_SEC, &deadline);
	deadline += now;
	for (int cpu = 0; cpu <= max_cpu; cpu++) {
		cpu_data_t *cdp = cpu_datap(cpu);

		if (cpu == me || cdp == NULL || !(psci_smp_started & (1u << cpu))) {
			continue;
		}
		while (now < deadline) {
			if (os_atomic_load(&psci_smp_cpu_stage[cpu], acquire) == PSCI_SMP_CPU_OFF) {
				/* it has said it is going; the firmware knows when it has gone (if it can tell at all) */
				int64_t state = psci_call(PSCI_AFFINITY_INFO, cdp->cpu_phys_id, 0, 0);

				if (state == PSCI_AFFINITY_OFF || state < 0) {
					break;
				}
			}
			now = mach_absolute_time();
		}
		if (os_atomic_load(&psci_smp_cpu_stage[cpu], acquire) != PSCI_SMP_CPU_OFF &&
		    psci_call(PSCI_AFFINITY_INFO, cdp->cpu_phys_id, 0, 0) != PSCI_AFFINITY_OFF) {
			psci_smp_note(" did not stop: it is left running\n", cpu);
		}
	}
}

/*
 * Where a processor learns that the system is ending: the interrupt psci_smp_stop_rest() sends,
 * and the loop in which the processors wait while another one panics (they have their
 * interrupts off there). Comes back if there is nothing of the kind going on.
 */
void
psci_smp_stop_here(void)
{
	if (psci_smp_started == 0) {
		return;
	}
	if (cpu_number() != master_cpu) {
		if (os_atomic_load(&psci_smp_stopping, acquire)) {
			psci_smp_cpu_off();
		}
		return;
	}
	if (os_atomic_load(&psci_smp_boot_wanted, acquire) && !psci_smp_boot_busy) {
		psci_smp_boot_busy = true;
		loader_return();        /* stops the rest (psci_smp_stop_others()); back only if no loader listens */
		for (;;) {
			__builtin_arm_wfi();
		}
	}
}

/*
 * IDLE IN A POWER-DOWN STATE (11-12, devices/akita TODO 27; boot-arg cpuidle=1, off by default). A core with nothing
 * to do waits in WFI, clock-gated but powered. The firmware offers more (the Pixel 8a's device tree: one state,
 * "c2", arm,psci-suspend-param 0x10000 -- the core powered down, its registers lost; back in ~0.2 ms, worth it from
 * ~2-3.5 ms): PSCI CPU_SUSPEND with an entry point. Such a core wakes through the reset vector like a core just
 * started (the loader's EL2 trampoline first, as CPU_ON's), finds its cpu data, and XNU's own way back from a
 * non-retention idle takes it on: cpu_machine_idle_init() set its reset handler to resume_idle_cpu, so
 * arm_init_idle_cpu() -> cpu_idle_exit(TRUE) -> Idle_load_context(), back where the idle thread was
 * (psci_idle_resumed() in between: the GIC's CPU interface and the core's timer set again).
 * The core's own timer stops in that state (local-timer-stop): it would not wake for its next deadline. So it goes
 * only when a platform driver has given a wake that keeps running (psci_idle_set_wake: the Pixel's MCT, armed for the
 * deadline), and only when the deadline is further than the state is worth (cpuidle_us=, 3000).
 * cpuidle_cpus=MASK: which cores (default every one but the boot processor).
 */
#define PSCI_CPU_SUSPEND        0xC4000001u

static uint32_t psci_idle_on;
static uint32_t psci_idle_state = 0x10000;
static uint32_t psci_idle_cpus = ~1u;
static uint64_t psci_idle_min_abs, psci_idle_margin_abs, psci_idle_max_abs;
static psci_idle_arm_t psci_idle_arm;
static psci_idle_disarm_t psci_idle_disarm;
static struct {
	uint64_t tries, downs, back, refused;
} __attribute__((aligned(64))) psci_idle_stat[MAX_CPUS];
/* cpuidle_probe=1 (11-12: 0x10000 refused, -2): after 50 refusals in a row the next of these, the first one taken kept */
static const uint32_t psci_idle_candidates[] = { 0x10000, 0x10001, 0x10002, 0x10003, 0x1010022, 0x0 };
static uint32_t psci_idle_probe, psci_idle_candidate;
static volatile uint32_t psci_idle_taken = ~0u;  /* the state the firmware took first (~0: none yet) */
static int64_t psci_idle_last_error;           /* the firmware's last refusal (11-12: every call refused at first) */
static uint32_t psci_idle_refusals_in_a_row;   /* ... and after 1000 in a row it is not asked for a second (WFI meanwhile) */
static uint64_t psci_idle_ask_again_abs;       /* (11-12 evening: once permitted, the firmware still cancels a call while an
                                                * SGI is pending at the core's redistributor or its secure timer runs --
                                                * Android counts those "rejected" -- so a run of refusals is no reason to
                                                * stop for good) */
static struct {
	uint32_t pending, enabled;                 /* the core's SGIs/PPIs right after its last refusal */
} psci_idle_refusal[MAX_CPUS];

extern int setPop(uint64_t time);
extern void gic_v3_irq_cpu_resume(void);
extern void gic_v3_irq_cpu_sgi_ppi(uint32_t *pending, uint32_t *enabled);

void
psci_idle_init(void)
{
	uint32_t us = 3000;

	PE_parse_boot_argn("cpuidle", &psci_idle_on, sizeof(psci_idle_on));
	PE_parse_boot_argn("cpuidle_state", &psci_idle_state, sizeof(psci_idle_state));
	PE_parse_boot_argn("cpuidle_cpus", &psci_idle_cpus, sizeof(psci_idle_cpus));
	PE_parse_boot_argn("cpuidle_us", &us, sizeof(us));
	PE_parse_boot_argn("cpuidle_probe", &psci_idle_probe, sizeof(psci_idle_probe));
	if (psci_idle_probe) {
		psci_idle_state = psci_idle_candidates[0];
	}
	nanoseconds_to_absolutetime((uint64_t)us * NSEC_PER_USEC, &psci_idle_min_abs);
	nanoseconds_to_absolutetime(300 * NSEC_PER_USEC, &psci_idle_margin_abs);      /* (woken this early: exit ~0.2 ms) */
	nanoseconds_to_absolutetime(1 * NSEC_PER_SEC, &psci_idle_max_abs);            /* (no deadline: wake within 1 s) */
	if (psci_idle_on) {
		/* PSCI_VERSION, and PSCI_FEATURES(CPU_SUSPEND): bit 1 the extended power-state format (power down = bit 30),
		 * bit 0 OS-initiated mode -- 11-12: 0x10000 (the device tree's) refused as an invalid parameter */
		int64_t version = psci_call(0x84000000u, 0, 0, 0);
		int64_t features = psci_call(0x8400000Au, PSCI_CPU_SUSPEND, 0, 0);

		printf("psci: cpuidle=%u: idle cores (mask 0x%x) power down (state 0x%x) when nothing is due for %u us -- once a "
		    "platform wake is given; PSCI %lld.%lld, CPU_SUSPEND features 0x%llx (%s power states%s)\n", psci_idle_on,
		    psci_idle_cpus, psci_idle_state, us, version >> 16, version & 0xffff, features, features < 0 ? "?" :
		    (features & 2) ? "extended" : "original", (features >= 0 && (features & 1)) ? ", OS-initiated mode offered" : "");
	}
}

void
psci_idle_set_wake(psci_idle_arm_t arm, psci_idle_disarm_t disarm)
{
	psci_idle_disarm = disarm;
	os_atomic_store(&psci_idle_arm, arm, release);
}

bool
psci_idle_wanted(void)
{
	return psci_idle_on != 0;
}

/*
 * From cpu_idle(), interrupts off, instead of its WFI: true when the wait was done here (the firmware came back without
 * powering down: an interrupt was pending, or it refused), false for a WFI. A core that did power down does not come
 * back here at all, but through psci_idle_resumed().
 */
bool
psci_idle_enter(cpu_data_t *cdp)
{
	psci_idle_arm_t arm = os_atomic_load(&psci_idle_arm, acquire);
	int cpu = cdp->cpu_number;
	uint64_t now, deadline = cdp->rtcPop, entry, context;
	int64_t status;

	if (arm == NULL || cpu >= MAX_CPUS || !(psci_idle_cpus & (1u << cpu)) || os_atomic_load(&psci_smp_stopping, relaxed)) {
		return false;
	}
	now = mach_absolute_time();
	if (psci_idle_refusals_in_a_row >= 1000) {
		if (psci_idle_ask_again_abs == 0) {
			psci_idle_ask_again_abs = now + psci_idle_max_abs;
		}
		if (now < psci_idle_ask_again_abs) {
			return false;
		}
		psci_idle_ask_again_abs = 0;
		psci_idle_refusals_in_a_row = 0;
	}
	if (deadline <= now || deadline - now < psci_idle_min_abs) {
		return false;
	}
	if (deadline - now > psci_idle_max_abs) {
		deadline = now + psci_idle_max_abs;
	}
	if (!arm(cpu, deadline - psci_idle_margin_abs)) {
		return false;
	}
	psci_idle_stat[cpu].tries++;
	entry = ml_static_vtop((vm_offset_t)&LowResetVectorBase);
	context = 0;
	if (psci_smp_el2_trampoline != 0) {
		context = entry;
		entry = psci_smp_el2_trampoline;
	}
	__builtin_arm_dsb(DSB_SY);
	if (psci_smp_el2_trampoline != 0 && cpu != master_cpu) {
		/*
		 * By HVC to the loader's EL2 stub, which makes the SMC from EL2 (xnuloader loader_el2_secondary_sync): the Pixel
		 * 8a's firmware refuses CPU_SUSPEND from EL1 (11-12: every state -2; on Android pKVM calls it, at EL2).
		 */
		register uint64_t x0 __asm__("x0") = PSCI_CPU_SUSPEND;
		register uint64_t x1 __asm__("x1") = psci_idle_state;
		register uint64_t x2 __asm__("x2") = entry;
		register uint64_t x3 __asm__("x3") = context;

		__asm__ volatile ("hvc #0" : "+r" (x0), "+r" (x1), "+r" (x2), "+r" (x3) : :
                    "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "memory");
		status = (int64_t)x0;
	} else {
		status = psci_call(PSCI_CPU_SUSPEND, psci_idle_state, entry, context);
	}
	if (psci_idle_disarm != NULL) {
		psci_idle_disarm(cpu);
	}
	if (status != 0) {
		psci_idle_stat[cpu].refused++;
		psci_idle_last_error = status;
		gic_v3_irq_cpu_sgi_ppi(&psci_idle_refusal[cpu].pending, &psci_idle_refusal[cpu].enabled);
		if (++psci_idle_refusals_in_a_row >= 50 && psci_idle_probe && psci_idle_taken == ~0u &&
		    psci_idle_candidate + 1 < sizeof(psci_idle_candidates) / sizeof(psci_idle_candidates[0])) {
			psci_idle_state = psci_idle_candidates[++psci_idle_candidate];
			psci_idle_refusals_in_a_row = 0;
		}
		return false;           /* (refused: the WFI after all -- never a loop of calls) */
	}
	if (psci_idle_taken == ~0u) {
		psci_idle_taken = psci_idle_state;
	}
	psci_idle_refusals_in_a_row = 0;
	psci_idle_stat[cpu].back++;
	return true;
}

/* A core that powered down is back, through the reset vector (arm_init_idle_cpu(), interrupts still off). */
void
psci_idle_resumed(void)
{
	cpu_data_t *cdp = getCpuDatap();
	int cpu = cdp->cpu_number;

	gic_v3_irq_cpu_resume();
	if (psci_idle_taken == ~0u) {
		psci_idle_taken = psci_idle_state;
	}
	if (psci_idle_disarm != NULL) {
		psci_idle_disarm(cpu);
	}
	if (cpu < MAX_CPUS) {
		psci_idle_stat[cpu].downs++;
	}
	setPop(cdp->rtcPop);                    /* (its timer's comparator was lost with the rest) */
}

/* the totals of every core: attempts, times it powered down, came back at once, refused */
int64_t
psci_idle_error(void)
{
	return psci_idle_last_error;
}

/* one core's counts, and its pending and enabled SGIs/PPIs as found after its last refusal */
void
psci_idle_core(int cpu, uint64_t *downs, uint64_t *refused, uint32_t *pending, uint32_t *enabled)
{
	*downs = psci_idle_stat[cpu].downs;
	*refused = psci_idle_stat[cpu].refused;
	*pending = psci_idle_refusal[cpu].pending;
	*enabled = psci_idle_refusal[cpu].enabled;
}

/* the state asked for now, and the one the firmware first took (~0: none) */
uint32_t
psci_idle_states(uint32_t *taken)
{
	*taken = psci_idle_taken;
	return psci_idle_state;
}

void
psci_idle_totals(uint64_t *tries, uint64_t *downs, uint64_t *back, uint64_t *refused)
{
	*tries = *downs = *back = *refused = 0;
	for (int i = 0; i < MAX_CPUS; i++) {
		*tries += psci_idle_stat[i].tries;
		*downs += psci_idle_stat[i].downs;
		*back += psci_idle_stat[i].back;
		*refused += psci_idle_stat[i].refused;
	}
}

#endif /* defined(GENERIC_PSCI_SMP) */
