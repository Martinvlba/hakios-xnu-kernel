/*
 * Copyright (c) 2016 Apple Inc. All rights reserved.
 */

/*
 * HakiOS: Apple's Raspberry Pi 3 board (xnu-8796 and older), kept as the generic HakiOS board:
 * a non-Apple ARMv8 SoC (Cortex-A cores) with a GICv3, PSCI and an architected timer, the kernel
 * at EL1, 4K pages, no PPL/SPTM/exclaves, no Apple PAC. xnu-12377 no longer has this board; this
 * is the 8796 HakiOS BCM2837.h, with 12377's spelling of the serial driver (PL011_UART) and the
 * defines 12377's VMAPPLE.h shows a non-Apple-family board needs. It deliberately does NOT include
 * apple_arm64_common.h (that would define APPLE_ARM64_ARCH_FAMILY and the Apple registers).
 */

#ifndef _PEXPERT_ARM_BCM2837_H
#define _PEXPERT_ARM_BCM2837_H

#define NO_MONITOR 1
#define NO_ECORE 1

#define BCM2837
#define BCM2837_BRINGUP
#define ARM_ARCH_TIMER
#define GENERIC_GIC_V3_IRQ                 1 /* interrupts via GICv3 Group 1, see osfmk/arm64/gic_v3_irq.c */
#define GENERIC_PSCI_SMP                  1 /* more cores are started through PSCI, see osfmk/arm64/psci_smp.c */

/*
 * The cores this kernel really runs on (Cortex-A55/A78, QEMU cortex-a76, Tensor G3's
 * Cortex-X3/A715/A510) are ARMv8.2 or later: they set PSTATE.PAN on every exception entry, so the
 * kernel must know to lift it around user memory accesses.
 */
#define __ARM_PAN_AVAILABLE__     1

/*
 * xnu-12377's sleh.c, pcb.c and machine_routines_common.c include apple_arm64_cpu.h directly, which
 * needs the MPIDR core-type fields that only apple_arm64_regs.h defines (not included here, see
 * above). Same text as there, so a later include of that header is a harmless identical redefinition.
 * Nothing on this board calls arm64_core_type().
 */
#define MPIDR_CORETYPE_SHIFT  (16)
#define MPIDR_CORETYPE_WIDTH  (3)
#define MPIDR_CORETYPE_MASK   ((1ULL << MPIDR_CORETYPE_WIDTH) - 1)
#define MPIDR_CORETYPE_ACC_E  (0ULL)
#define MPIDR_CORETYPE_ACC_P  (1ULL)
#define MPIDR_CORETYPE_ACC_M  (4ULL)

#define __ARM_ARCH__              8
#define __ARM_VMSA__              8
#define __ARM_VFP__               4
#define __ARM_COHERENT_CACHE__    1
#define __ARM_DEBUG__             7
#define __ARM64_PMAP_SUBPAGE_L1__ 1

#ifndef ASSEMBLER

#define PI3_UART                        /* (12377 has no Raspberry Pi UART driver any more; harmless) */
#define PL011_UART                      /* PL011, as found on QEMU virt (8796: VMAPPLE_UART) */

#define PI3_BREAK                               asm volatile("brk #0");

#define BCM2837_GPFSEL0_V               (pi3_gpio_base_vaddr + 0x0)
#define BCM2837_GPSET0_V                (pi3_gpio_base_vaddr + 0x1C)
#define BCM2837_GPCLR0_V                (pi3_gpio_base_vaddr + 0x28)
#define BCM2837_GPPUD_V                 (pi3_gpio_base_vaddr + 0x94)
#define BCM2837_GPPUDCLK0_V             (pi3_gpio_base_vaddr + 0x98)

#define BCM2837_FSEL_INPUT              0x0
#define BCM2837_FSEL_OUTPUT             0x1
#define BCM2837_FSEL_ALT0               0x4
#define BCM2837_FSEL_ALT1               0x5
#define BCM2837_FSEL_ALT2               0x6
#define BCM2837_FSEL_ALT3               0x7
#define BCM2837_FSEL_ALT4               0x3
#define BCM2837_FSEL_ALT5               0x2

#define BCM2837_FSEL_NFUNCS             54
#define BCM2837_FSEL_REG(func)          (BCM2837_GPFSEL0_V + (4 * ((func) / 10)))
#define BCM2837_FSEL_OFFS(func)         (((func) % 10) * 3)
#define BCM2837_FSEL_MASK(func)         (0x7 << BCM2837_FSEL_OFFS(func))

#define BCM2837_AUX_ENABLES_V           (pi3_aux_base_vaddr + 0x4)
#define BCM2837_AUX_MU_IO_REG_V         (pi3_aux_base_vaddr + 0x40)
#define BCM2837_AUX_MU_IER_REG_V        (pi3_aux_base_vaddr + 0x44)
#define BCM2837_AUX_MU_IIR_REG_V        (pi3_aux_base_vaddr + 0x48)
#define BCM2837_AUX_MU_LCR_REG_V        (pi3_aux_base_vaddr + 0x4C)
#define BCM2837_AUX_MU_MCR_REG_V        (pi3_aux_base_vaddr + 0x50)
#define BCM2837_AUX_MU_LSR_REG_V        (pi3_aux_base_vaddr + 0x54)
#define BCM2837_AUX_MU_MSR_REG_V        (pi3_aux_base_vaddr + 0x58)
#define BCM2837_AUX_MU_SCRATCH_V        (pi3_aux_base_vaddr + 0x5C)
#define BCM2837_AUX_MU_CNTL_REG_V       (pi3_aux_base_vaddr + 0x60)
#define BCM2837_AUX_MU_STAT_REG_V       (pi3_aux_base_vaddr + 0x64)
#define BCM2837_AUX_MU_BAUD_REG_V       (pi3_aux_base_vaddr + 0x68)
#define BCM2837_PUT32(addr, value) do { *((volatile uint32_t *) addr) = value; } while(0)
#define BCM2837_GET32(addr) *((volatile uint32_t *) addr)

/*
 * No fixed PLATFORM_PANIC_LOG_PADDR: the Raspberry Pi address (0x3c0fc000) is
 * not RAM on other boards. The panic log region comes from the device tree
 * instead (/pram and /chosen/embedded-panic-log-size), like on Apple devices.
 */
#endif /* ! ASSEMBLER */

#endif /* ! _PEXPERT_ARM_BCM2837_H */
