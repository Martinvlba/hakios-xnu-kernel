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
#define MALI_KERNEL_TEXTURES        16      /* textures, images and samplers: indices 0 to 15 */
#define MALI_TABLE_SAMPLER          3       /* the resource table's tables (gallium panfrost's numbering) */
#define MALI_TABLE_TEXTURE          4
#define MALI_TABLE_IMAGE            5
#define MALI_TABLES                 6
#define MALI_KERNEL_WORKGROUP_WORD  32
#define MALI_KERNEL_GROUPS_WORD     35
#define MALI_KERNEL_GRID_WORD       38
#define MALI_KERNEL_RESERVED        42
/* a draw's push constants (both stages): words 2n, 2n+1 [[buffer(n)]]'s address as a kernel's; then the viewport
 * transform's scale (3 words) and offset (3), the first vertex and the first instance */
#define MALI_DRAW_VIEWPORT_SCALE    32
#define MALI_DRAW_VIEWPORT_OFFSET   35
#define MALI_DRAW_FIRST_VERTEX      38
#define MALI_DRAW_BASE_INSTANCE     39
#define MALI_KERNEL_MAX_FAU         256
#define MALI_KERNEL_MAX_RESOURCES   64

struct mali_kernel_resource {
	unsigned kind;          /* enum hakios_metal_resource_kind (the front end's numbering, the same) */
	unsigned index;         /* [[buffer(n)]] ... */
	unsigned size;
	unsigned read, write;
	unsigned inline_sampler;        /* a sampler the function declares (constexpr sampler): bound by the driver */
	uint64_t sampler_state;         /* ... its state word (AIR's __air_sampler_state) */
};

struct mali_kernel_binary {
	void *code;             /* malloc'd */
	size_t size;
	unsigned work_regs;
	uint64_t preload;
	unsigned fau_words;     /* push constant words the code reads: MALI_KERNEL_RESERVED and the constants after */
	uint32_t consts[MALI_KERNEL_MAX_FAU];   /* words MALI_KERNEL_RESERVED .. fau_words - 1 */
	unsigned tls_size, wls_size;    /* bytes a thread spills; bytes of static threadgroup arrays */
	unsigned workgroup[3];  /* fixed by the kernel, or 0 0 0 */
	unsigned threadgroup_bytes;
	unsigned threadgroup_arg_base;  /* where a [[threadgroup(0)]] argument starts; ~0u when there is none */
	char name[128];
	unsigned stage;                 /* 0 compute, 1 vertex, 2 fragment */
	/* a vertex function's: IDVS, the varying shader's offset in CODE (0: none), its registers and preload; the bytes
	 * of varyings a vertex has; the layout the fragment function is compiled against (Mesa's, opaque here) */
	unsigned idvs, secondary_offset, secondary_work_regs, varying_bytes;
	uint64_t secondary_preload;
	void *varyings;
	unsigned num_resources;
	struct mali_kernel_resource resources[MALI_KERNEL_MAX_RESOURCES];
	unsigned rt_written;             /* a fragment function's: bit n, [[color(n)]] written */
};

struct mali_constant {
	unsigned index;
	uint64_t bits;
};

/* 0 and *OUT filled, or -1 with the reason in ERROR */
int mali_kernel_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    struct mali_kernel_binary *out, char *error, size_t error_size);
/* Metal's vertex descriptor, its numbers straight through (MTLVertexFormat, MTLVertexStepFunction): the vertex
 * function's [[stage_in]] attributes fetched in the shader from [[buffer(n)]] (format 0: an attribute not described) */
#define MALI_VERTEX_SLOTS 31
struct mali_vertex_fetch {
	struct { unsigned format, offset, buffer; } attributes[MALI_VERTEX_SLOTS];
	struct { unsigned stride, step_function, step_rate; } layouts[MALI_VERTEX_SLOTS];
};
/* mali_stage_compile's vertex function, with its stage_in attributes fetched as FETCH says (NULL: as mali_stage_compile) */
int mali_vertex_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    const struct mali_vertex_fetch *fetch, struct mali_kernel_binary *out, char *error, size_t error_size);
/* a vertex function (FRAGMENT 0), or a fragment function (FRAGMENT 1) against VERTEX's varyings; as mali_kernel_compile */
int mali_stage_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count, int fragment,
    const struct mali_kernel_binary *vertex, struct mali_kernel_binary *out, char *error, size_t error_size);
/* the driver's preload draw (compile_mali.c): a triangle over the target, colour attachment n texture n's texel, for
 * TARGETS (1 to 4) attachments */
int mali_preload_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, unsigned targets, char *error,
    size_t error_size);
void mali_kernel_binary_free(struct mali_kernel_binary *binary);

#ifdef __cplusplus
}
#endif

#endif
