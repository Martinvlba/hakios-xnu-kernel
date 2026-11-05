/*
 * The Mali build's shader compiler (src/mali/compile_mali.c): a Metal compute function's AIR bitcode compiled for the
 * Pixel 8a's Mali-G715 -- hakios-kernel's AIR front end (tools/mesa/air) into NIR, Metal's buffers made addresses, Mesa
 * 26.2.5's panfrost compiler for arch 11. The chain of hakios-kernel's tools/mesa/pan/air_pan.c, as a library.
 *
 * What the code expects of the dispatch (src/mali/c_api_mali.cpp), its push constants (FAU, 32-bit words):
 *
 *   words 2n, 2n+1    [[buffer(n)]]'s address (offset included), n < MALI_KERNEL_BUFFERS
 *   words 32..34      the threadgroup's size (threads_per_threadgroup)
 *   words 35..37      the grid's size in threadgroups (threadgroups_per_grid)
 *   words 38..40      the grid's size in threads (a thread past it returns at once)
 *   word  41          zero
 *   words 42..        the compiler's own constants (consts[], from word MALI_KERNEL_RESERVED on), fau_words in all
 */
#ifndef HAKIOS_METAL_COMPILE_MALI_H
#define HAKIOS_METAL_COMPILE_MALI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MALI_KERNEL_BUFFERS         16
#define MALI_KERNEL_WORKGROUP_WORD  32
#define MALI_KERNEL_GROUPS_WORD     35
#define MALI_KERNEL_GRID_WORD       38
#define MALI_KERNEL_RESERVED        42
#define MALI_KERNEL_MAX_FAU         256
#define MALI_KERNEL_MAX_RESOURCES   64

struct mali_kernel_resource {
	unsigned kind;          /* enum hakios_metal_resource_kind (the front end's numbering, the same) */
	unsigned index;         /* [[buffer(n)]] ... */
	unsigned size;
	unsigned read, write;
};

struct mali_kernel_binary {
	void *code;             /* malloc'd */
	size_t size;
	unsigned work_regs;
	uint64_t preload;
	unsigned fau_words;     /* push constant words the code reads: MALI_KERNEL_RESERVED and the constants after */
	uint32_t consts[MALI_KERNEL_MAX_FAU];   /* words MALI_KERNEL_RESERVED .. fau_words - 1 */
	unsigned tls_size, wls_size;
	unsigned workgroup[3];  /* fixed by the kernel, or 0 0 0 */
	unsigned threadgroup_bytes;
	char name[128];
	unsigned num_resources;
	struct mali_kernel_resource resources[MALI_KERNEL_MAX_RESOURCES];
};

struct mali_constant {
	unsigned index;
	uint64_t bits;
};

/* 0 and *OUT filled, or -1 with the reason in ERROR */
int mali_kernel_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    struct mali_kernel_binary *out, char *error, size_t error_size);
void mali_kernel_binary_free(struct mali_kernel_binary *binary);

#ifdef __cplusplus
}
#endif

#endif
