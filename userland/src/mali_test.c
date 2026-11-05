/*
 * mali_test: the Pixel 8a's GPU through HakiosGPU (devices/akita/drivers/gpu/hakios.akita_gpu.cpp), the service that
 * speaks Linux's panthor interface -- what Mesa's panfrost and panvk will use, here by hand: the device's description,
 * a VM, two buffers mapped into this process and bound into the VM, a group with one queue, and a command stream of
 * this program's own submitted N times with a sync object signalled by each, waited for, and its result read back.
 *
 * The job is the kernel's step 9 (mali=9), from user space: a shader of four Valhall instructions (assembled by
 * Mesa's valhall/asm.py) that stores 0x600DF00D where its push constant says, dispatched by RUN_COMPUTE after v11's
 * scoreboard setup (SET_STATE_IMM32, as panvk does), then a WAIT and a marker stored by the stream itself.
 *
 *   mali_test [N]       N submissions (3)
 *
 * Requests are the service's calls with DRM's request numbers as selectors (0x40 + panthor's, 0xbf.. sync objects);
 * a buffer's memory is the client memory whose type is its handle. Timeouts are relative nanoseconds.
 */
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mali_store_index.h"
#include "mali_air_fill_id.h"       /* AIR (hakios-metal's fill_id) -> tools/mesa/air -> Mesa's panfrost: tools/mesa/pan/air_pan.c */      /* Mesa 26.2.5's panfrost compiler for this GPU: tools/mesa/pan/pan_compile_test.c */

struct obj_array { uint32_t stride, count; uint64_t array; };
struct sync_op { uint32_t flags, handle; uint64_t timeline_value; };
struct dev_query { uint32_t type, size; uint64_t pointer; };
struct gpu_info {
	uint32_t gpu_id, gpu_rev, csf_id, l2_features, tiler_features, mem_features, mmu_features, thread_features,
	    max_threads, thread_max_workgroup_size, thread_max_barrier_size, coherency_features, texture_features[4],
	    as_present, selected_coherency;
	uint64_t shader_present, l2_present, tiler_present;
	uint32_t core_features, pad;
	uint64_t gpu_features;
};
struct csif_info { uint32_t csg_slot_count, cs_slot_count, cs_reg_count, scoreboard_slot_count, unpreserved, pad; };
struct vm_create { uint32_t flags, id; uint64_t user_va_range; };
struct bind_op { uint32_t flags, bo_handle; uint64_t bo_offset, va, size; struct obj_array syncs; };
struct vm_bind { uint32_t vm_id, flags; struct obj_array ops; };
struct bo_create { uint64_t size; uint32_t flags, exclusive_vm_id, handle, pad; };
struct queue_create { uint8_t priority, pad[3]; uint32_t ringbuf_size; };
struct group_create {
	struct obj_array queues;
	uint8_t max_compute_cores, max_fragment_cores, max_tiler_cores, priority;
	uint32_t pad;
	uint64_t compute_core_mask, fragment_core_mask, tiler_core_mask;
	uint32_t vm_id, group_handle;
};
struct queue_submit { uint32_t queue_index, stream_size; uint64_t stream_addr; uint32_t latest_flush, pad; struct obj_array syncs; };
struct group_submit { uint32_t group_handle, pad; struct obj_array queue_submits; };
struct syncobj_create { uint32_t handle, flags; };
struct syncobj_wait { uint64_t handles; int64_t timeout_nsec; uint32_t count_handles, flags, first_signaled, pad; uint64_t deadline; };
struct syncobj_array { uint64_t handles; uint32_t count_handles, pad; };

enum { DEV_QUERY = 0x40, VM_CREATE, VM_DESTROY, VM_BIND, VM_GET_STATE, BO_CREATE, BO_MMAP_OFFSET, GROUP_CREATE,
	GROUP_DESTROY, GROUP_SUBMIT, SYNCOBJ_CREATE = 0xbf, SYNCOBJ_WAIT = 0xc3, SYNCOBJ_RESET = 0xc4 };

static io_connect_t gpu;

static int
call(uint32_t selector, void *request, size_t size, const char *what)
{
	size_t out = size;
	kern_return_t kr = IOConnectCallStructMethod(gpu, selector, request, size, request, &out);

	if (kr != KERN_SUCCESS) {
		printf("mali_test: %s (0x%x): 0x%x\n", what, selector, kr);
		return -1;
	}
	return 0;
}

static void *
map_bo(uint32_t handle, uint64_t size)
{
	mach_vm_address_t address = 0;
	mach_vm_size_t mapped = size;

	if (IOConnectMapMemory64(gpu, handle, mach_task_self(), &address, &mapped, kIOMapAnywhere) != KERN_SUCCESS) {
		return NULL;
	}
	return (void *)(uintptr_t)address;
}

/* the shader: MOV.i32 r2, u0.w0; MOV.i32 r3, u0.w1; IADD_IMM.i32 r0, 0x0, #0x600DF00D; STORE.i32.slot0.end @r0, [r2^:r3^] */
static const uint8_t shader[32] = {
	0x80, 0x00, 0x00, 0x00, 0x00, 0xc2, 0x91, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0xc3, 0x91, 0x00,
	0xc0, 0x0d, 0xf0, 0x0d, 0x60, 0xc0, 0x10, 0x01, 0x42, 0x00, 0x00, 0x18, 0x02, 0x40, 0x61, 0x78,
};

#define CODE_VA 0x10000000ULL
#define DATA_VA 0x10100000ULL
#define SPD     0x000           /* in the data buffer: the shader program descriptor */
#define TSD     0x040           /* the local storage descriptor */
#define FAU     0x080           /* the push constants */
#define TARGET  0x100           /* the shader's word, at +0; the stream's marker at +0x10 */
#define STREAM  0x800

int
main(int argc, char **argv)
{
	int rounds = argc > 1 ? atoi(argv[1]) : 3, good = 0;
	io_service_t service = IOServiceGetMatchingService(0, IOServiceMatching("HakiosGPU"));
	struct gpu_info info;
	struct csif_info csif;
	struct dev_query q;
	struct vm_create vc = { 0 };
	struct bo_create code_bo = { .size = 0x4000 }, data_bo = { .size = 0x4000 };
	struct bind_op ops[2];
	struct vm_bind vb;
	struct queue_create qc = { .priority = 1, .ringbuf_size = 0x10000 };
	struct group_create gc;
	uint8_t *code;
	uint32_t *data;

	if (service == 0 || IOServiceOpen(service, mach_task_self(), 0, &gpu) != KERN_SUCCESS) {
		printf("mali_test: no HakiosGPU (boot-arg mali_keep=1)\n");
		return 1;
	}
	memset(&info, 0, sizeof(info));
	memset(&csif, 0, sizeof(csif));
	q = (struct dev_query){ .type = 0, .size = sizeof(info), .pointer = (uint64_t)(uintptr_t)&info };
	if (call(DEV_QUERY, &q, sizeof(q), "the GPU's description") != 0) {
		return 1;
	}
	q = (struct dev_query){ .type = 1, .size = sizeof(csif), .pointer = (uint64_t)(uintptr_t)&csif };
	if (call(DEV_QUERY, &q, sizeof(q), "the command stream interface") != 0) {
		return 1;
	}
	printf("mali_test: GPU 0x%08x (arch %u.%u), shaders 0x%llx, MMU 0x%x (VA %u bits), coherency features 0x%x; "
	    "%u groups of %u streams, %u registers, %u scoreboards\n", info.gpu_id, info.gpu_id >> 28, (info.gpu_id >> 24) & 15,
	    (unsigned long long)info.shader_present, info.mmu_features, info.mmu_features & 0xff, info.coherency_features,
	    csif.csg_slot_count, csif.cs_slot_count, csif.cs_reg_count, csif.scoreboard_slot_count);

	vc.user_va_range = 1ULL << 47;
	if (call(VM_CREATE, &vc, sizeof(vc), "a VM") != 0 || call(BO_CREATE, &code_bo, sizeof(code_bo), "a buffer") != 0 ||
	    call(BO_CREATE, &data_bo, sizeof(data_bo), "a buffer") != 0) {
		return 1;
	}
	code = map_bo(code_bo.handle, code_bo.size);
	data = map_bo(data_bo.handle, data_bo.size);
	if (code == NULL || data == NULL) {
		printf("mali_test: the buffers could not be mapped\n");
		return 1;
	}
	memset(ops, 0, sizeof(ops));
	ops[0] = (struct bind_op){ .flags = 1 /* read only */, .bo_handle = code_bo.handle, .va = CODE_VA, .size = code_bo.size };
	ops[1] = (struct bind_op){ .flags = 2 /* not executable */, .bo_handle = data_bo.handle, .va = DATA_VA, .size = data_bo.size };
	vb = (struct vm_bind){ .vm_id = vc.id, .ops = { sizeof(ops[0]), 2, (uint64_t)(uintptr_t)ops } };
	if (call(VM_BIND, &vb, sizeof(vb), "the binds") != 0) {
		return 1;
	}
	memset(&gc, 0, sizeof(gc));
	gc.queues = (struct obj_array){ sizeof(qc), 1, (uint64_t)(uintptr_t)&qc };
	gc.max_compute_cores = gc.max_fragment_cores = 7;
	gc.max_tiler_cores = 1;
	gc.priority = 1;
	gc.compute_core_mask = gc.fragment_core_mask = info.shader_present;
	gc.tiler_core_mask = info.tiler_present;
	gc.vm_id = vc.id;
	if (call(GROUP_CREATE, &gc, sizeof(gc), "a group") != 0) {
		return 1;
	}
	printf("mali_test: VM %u, buffers %u and %u (mapped at %p and %p), group %u\n", vc.id, code_bo.handle, data_bo.handle,
	    code, data, gc.group_handle);

	memcpy(code, shader, sizeof(shader));
	data[SPD / 4] = 8 | 1 << 4;                             /* shader program: type shader, stage compute */
	data[SPD / 4 + 2] = (uint32_t)CODE_VA;
	data[TSD / 4 + 1] = 31;                                 /* local storage: no workgroup memory */
	*(uint64_t *)&data[FAU / 4] = DATA_VA + TARGET;         /* push constant 0: where the shader stores */
	{
		uint64_t *s = (uint64_t *)&data[STREAM / 4];
		unsigned int n = 0;

		s[n++] = 1ULL << 56 | 0ULL << 48 | (DATA_VA + TARGET + 0x20);   /* first: a plain store (did the CALL run us?) */
		s[n++] = 2ULL << 56 | 4ULL << 48 | 0x5717a4edu;
		s[n++] = 21ULL << 56 | 4ULL << 48 | 0ULL << 40 | 0x1ULL << 16;
		s[n++] = 28ULL << 56 | 0ULL << 32 | 3;                  /* SET_STATE_IMM32: SB_SEL_ENDPOINT 3 */
		s[n++] = 28ULL << 56 | 9ULL << 32 | 0x8;                /* SB_MASK_WAIT 0x8 */
		s[n++] = 28ULL << 56 | 1ULL << 32 | 0;                  /* SB_SEL_OTHER 0 */
		s[n++] = 28ULL << 56 | 2ULL << 32 | 1;                  /* SB_SEL_DEFERRED 1 */
		s[n++] = 28ULL << 56 | 8ULL << 32 | 0xf0;               /* SB_MASK_STREAM 0xf0 */
		s[n++] = 1ULL << 56 | 16ULL << 48 | (DATA_VA + SPD);    /* MOVE48 r16 <- the shader program */
		s[n++] = 1ULL << 56 | 24ULL << 48 | (DATA_VA + TSD);    /* MOVE48 r24 <- the local storage */
		s[n++] = 1ULL << 56 | 8ULL << 48 | (DATA_VA + FAU);     /* MOVE48 r8 <- the push constants ... */
		s[n++] = 2ULL << 56 | 9ULL << 48 | 1u << 24;            /* ... one of them */
		s[n++] = 1ULL << 56 | 0ULL << 48;                       /* no resource table */
		for (int r = 32; r <= 36; r++) {
			s[n++] = 2ULL << 56 | (uint64_t)r << 48;        /* attribute offset, workgroup size 1, job offset 0 */
		}
		for (int r = 37; r <= 39; r++) {
			s[n++] = 2ULL << 56 | (uint64_t)r << 48 | 1;    /* job size 1x1x1 */
		}
		s[n++] = 4ULL << 56 | 1;                                /* RUN_COMPUTE */
		s[n++] = 3ULL << 56 | 0xffffULL << 16;                  /* WAIT */
		s[n++] = 1ULL << 56 | 0ULL << 48 | (DATA_VA + TARGET + 0x10);   /* the marker */
		s[n++] = 2ULL << 56 | 4ULL << 48 | 0x9a55ed00u;
		s[n++] = 21ULL << 56 | 4ULL << 48 | 0ULL << 40 | 0x1ULL << 16;

		for (int round = 1; round <= rounds; round++) {
			struct syncobj_create sc = { 0 };
			struct sync_op signal;
			struct queue_submit qs;
			struct group_submit gs;
			struct syncobj_wait w;
			volatile uint32_t *target = &data[TARGET / 4];

			target[0] = 0x33333333;
			target[4] = 0x44444444;
			target[8] = 0x55555555;
			target[12] = 0x66666666;
			/* the words written through the write-combining map made to reach memory before the GPU writes (else they
			 * could land after it, over its results) */
			__asm__ volatile ("dsb sy" ::: "memory");
			if (call(SYNCOBJ_CREATE, &sc, sizeof(sc), "a sync object") != 0) {
				return 1;
			}
			signal = (struct sync_op){ .flags = 1u << 31 /* signal */, .handle = sc.handle };
			qs = (struct queue_submit){ .queue_index = 0, .stream_size = n * 8, .stream_addr = DATA_VA + STREAM,
				.syncs = { sizeof(signal), 1, (uint64_t)(uintptr_t)&signal } };
			gs = (struct group_submit){ .group_handle = gc.group_handle,
				.queue_submits = { sizeof(qs), 1, (uint64_t)(uintptr_t)&qs } };
			if (call(GROUP_SUBMIT, &gs, sizeof(gs), "the submission") != 0) {
				return 1;
			}
			w = (struct syncobj_wait){ .handles = (uint64_t)(uintptr_t)&sc.handle, .timeout_nsec = 2000000000LL,
				.count_handles = 1, .flags = 1 };
			if (call(SYNCOBJ_WAIT, &w, sizeof(w), "the wait") != 0) {
				printf("mali_test: round %d: not done in 2 s; the shader's word 0x%08x, the marker 0x%08x\n", round,
				    target[0], target[4]);
				return 1;
			}
			printf("mali_test: round %d: done; the kernel's store 0x%08x, the stream's first store 0x%08x, the shader's "
			    "word 0x%08x, the marker 0x%08x: %s\n", round, target[12], target[8], target[0], target[4], target[0] == 0x600df00d && target[4] == 0x9a55ed00 ?
			    "RIGHT" : "WRONG");
			good += target[0] == 0x600df00d && target[4] == 0x9a55ed00;
			usleep(200000);
			printf("mali_test: round %d: 200 ms later: 0x%08x 0x%08x 0x%08x 0x%08x\n", round, target[12], target[8],
			    target[0], target[4]);
		}
	}
	/*
	 * Then a shader compiled by Mesa's own compiler (store_index: invocation i stores 3 * i + 7 into buffer[i]): its
	 * descriptor as the compiler's facts say (32 or 64 registers a thread, the preload of r48..r63 -- r60, the global
	 * invocation ID), 4 workgroups of 64, and the 256 words checked.
	 */
	{
		volatile uint32_t *out = &data[0x1000 / 4];
		uint64_t *s = (uint64_t *)&data[0x2000 / 4];
		unsigned int n = 0, right = 0;
		struct syncobj_create sc = { 0 };
		struct sync_op signal;
		struct queue_submit qs;
		struct group_submit gs;
		struct syncobj_wait w;

		memcpy(code + 0x100, store_index_shader, sizeof(store_index_shader));
		memset((void *)&data[0x200 / 4], 0, 0x100);
		data[0x200 / 4] = 8 | 1 << 4 | (STORE_INDEX_WORK_REGS <= 32 ? 2u : 0u) << 30;   /* shader program */
		data[0x200 / 4 + 1] = (uint32_t)(STORE_INDEX_PRELOAD >> 48);                    /* preload r48..r63 */
		data[0x200 / 4 + 2] = (uint32_t)(CODE_VA + 0x100);
		data[0x240 / 4 + 1] = 31;                                                       /* local storage */
		*(uint64_t *)&data[0x280 / 4] = DATA_VA + 0x1000;                               /* push constant 0 */
		for (int i = 0; i < 256; i++) {
			out[i] = 0xdeadbeef;
		}
		__asm__ volatile ("dsb sy" ::: "memory");
		s[n++] = 28ULL << 56 | 0ULL << 32 | 3;
		s[n++] = 28ULL << 56 | 9ULL << 32 | 0x8;
		s[n++] = 28ULL << 56 | 1ULL << 32 | 0;
		s[n++] = 28ULL << 56 | 2ULL << 32 | 1;
		s[n++] = 28ULL << 56 | 8ULL << 32 | 0xf0;
		s[n++] = 1ULL << 56 | 16ULL << 48 | (DATA_VA + 0x200);
		s[n++] = 1ULL << 56 | 24ULL << 48 | (DATA_VA + 0x240);
		s[n++] = 1ULL << 56 | 8ULL << 48 | (DATA_VA + 0x280);
		s[n++] = 2ULL << 56 | 9ULL << 48 | (uint64_t)STORE_INDEX_FAU_WORDS / 2 << 24;  /* FAU count in 64-bit words */
		s[n++] = 1ULL << 56 | 0ULL << 48;
		s[n++] = 2ULL << 56 | 32ULL << 48;
		s[n++] = 2ULL << 56 | 33ULL << 48 | 63;                                        /* workgroup 64x1x1 */
		for (int r = 34; r <= 36; r++) {
			s[n++] = 2ULL << 56 | (uint64_t)r << 48;
		}
		s[n++] = 2ULL << 56 | 37ULL << 48 | 4;                                         /* 4 workgroups along X */
		s[n++] = 2ULL << 56 | 38ULL << 48 | 1;
		s[n++] = 2ULL << 56 | 39ULL << 48 | 1;
		s[n++] = 4ULL << 56 | 1;                                                        /* RUN_COMPUTE */
		s[n++] = 3ULL << 56 | 0xffffULL << 16;                                          /* WAIT */
		if (call(SYNCOBJ_CREATE, &sc, sizeof(sc), "a sync object") == 0) {
			signal = (struct sync_op){ .flags = 1u << 31, .handle = sc.handle };
			qs = (struct queue_submit){ .stream_size = n * 8, .stream_addr = DATA_VA + 0x2000,
				.syncs = { sizeof(signal), 1, (uint64_t)(uintptr_t)&signal } };
			gs = (struct group_submit){ .group_handle = gc.group_handle, .queue_submits = { sizeof(qs), 1,
				(uint64_t)(uintptr_t)&qs } };
			w = (struct syncobj_wait){ .handles = (uint64_t)(uintptr_t)&sc.handle, .timeout_nsec = 2000000000LL,
				.count_handles = 1, .flags = 1 };
			if (call(GROUP_SUBMIT, &gs, sizeof(gs), "the compiled shader's submission") == 0 &&
			    call(SYNCOBJ_WAIT, &w, sizeof(w), "its wait") == 0) {
				for (int i = 0; i < 256; i++) {
					right += out[i] == (uint32_t)(3 * i + 7);
				}
			}
			printf("mali_test: Mesa's compiled shader: %u of 256 words right (out[0] %u, out[1] %u, out[255] %u; "
			    "3i+7 wanted)\n", right, out[0], out[1], out[255]);
		}
	}
	/*
	 * And an Apple-format shader: fill_id, AIR bitcode (out[gid] = gid * 0x9E3779B1 + c; out is [[buffer(0)]], c in
	 * [[buffer(1)]]) through our AIR front end and Mesa's compiler. Metal's buffers are addresses: buffer n's at push
	 * constant words 2n, 2n+1 (16 buffers' worth of push constants).
	 */
	{
		volatile uint32_t *out = &data[0x1400 / 4];
		volatile uint32_t *c = &data[0x1800 / 4];
		uint64_t *s = (uint64_t *)&data[0x2400 / 4];
		uint64_t *fau = (uint64_t *)&data[0x300 / 4];
		unsigned int n = 0, right = 0;
		struct syncobj_create sc = { 0 };
		struct sync_op signal;
		struct queue_submit qs;
		struct group_submit gs;
		struct syncobj_wait w;

		memcpy(code + 0x200, air_kernel_shader, sizeof(air_kernel_shader));
		memset((void *)&data[0x380 / 4], 0, 0x80);
		data[0x380 / 4] = 8 | 1 << 4 | (AIR_KERNEL_WORK_REGS <= 32 ? 2u : 0u) << 30;
		data[0x380 / 4 + 1] = (uint32_t)(AIR_KERNEL_PRELOAD >> 48);
		data[0x380 / 4 + 2] = (uint32_t)(CODE_VA + 0x200);
		memset(fau, 0, 128);
		fau[0] = DATA_VA + 0x1400;                       /* [[buffer(0)]]: out */
		fau[1] = DATA_VA + 0x1800;                       /* [[buffer(1)]]: c */
		*c = 0x1234;
		for (int i = 0; i < 256; i++) {
			out[i] = 0xdeadbeef;
		}
		__asm__ volatile ("dsb sy" ::: "memory");
		s[n++] = 28ULL << 56 | 0ULL << 32 | 3;
		s[n++] = 28ULL << 56 | 9ULL << 32 | 0x8;
		s[n++] = 28ULL << 56 | 1ULL << 32 | 0;
		s[n++] = 28ULL << 56 | 2ULL << 32 | 1;
		s[n++] = 28ULL << 56 | 8ULL << 32 | 0xf0;
		s[n++] = 1ULL << 56 | 16ULL << 48 | (DATA_VA + 0x380);
		s[n++] = 1ULL << 56 | 24ULL << 48 | (DATA_VA + 0x240);
		s[n++] = 1ULL << 56 | 8ULL << 48 | (DATA_VA + 0x300);
		s[n++] = 2ULL << 56 | 9ULL << 48 | (uint64_t)(AIR_KERNEL_FAU_WORDS / 2) << 24;
		s[n++] = 1ULL << 56 | 0ULL << 48;
		s[n++] = 2ULL << 56 | 32ULL << 48;
		s[n++] = 2ULL << 56 | 33ULL << 48 | 63;
		for (int r = 34; r <= 36; r++) {
			s[n++] = 2ULL << 56 | (uint64_t)r << 48;
		}
		s[n++] = 2ULL << 56 | 37ULL << 48 | 4;
		s[n++] = 2ULL << 56 | 38ULL << 48 | 1;
		s[n++] = 2ULL << 56 | 39ULL << 48 | 1;
		s[n++] = 4ULL << 56 | 1;
		s[n++] = 3ULL << 56 | 0xffffULL << 16;
		if (call(SYNCOBJ_CREATE, &sc, sizeof(sc), "a sync object") == 0) {
			signal = (struct sync_op){ .flags = 1u << 31, .handle = sc.handle };
			qs = (struct queue_submit){ .stream_size = n * 8, .stream_addr = DATA_VA + 0x2400,
				.syncs = { sizeof(signal), 1, (uint64_t)(uintptr_t)&signal } };
			gs = (struct group_submit){ .group_handle = gc.group_handle, .queue_submits = { sizeof(qs), 1,
				(uint64_t)(uintptr_t)&qs } };
			w = (struct syncobj_wait){ .handles = (uint64_t)(uintptr_t)&sc.handle, .timeout_nsec = 2000000000LL,
				.count_handles = 1, .flags = 1 };
			if (call(GROUP_SUBMIT, &gs, sizeof(gs), "the AIR kernel's submission") == 0 &&
			    call(SYNCOBJ_WAIT, &w, sizeof(w), "its wait") == 0) {
				for (int i = 0; i < 256; i++) {
					right += out[i] == (uint32_t)i * 0x9E3779B1u + 0x1234u;
				}
			}
			printf("mali_test: the AIR kernel fill_id: %u of 256 words right (out[0] 0x%x, out[1] 0x%x, out[255] 0x%x)\n",
			    right, out[0], out[1], out[255]);
		}
	}
	/*
	 * A frame with no draws, only a clear: the first fragment job. A 64 x 64 RGBA8 render target (linear) filled with junk,
	 * a tiler heap (panvk's sizes), its descriptor with a 64 KiB geometry buffer after it, a tiler context (empty polygon
	 * list), a framebuffer descriptor with one render target whose clear colour is 0xff3366cc (r 0xcc, g 0x66, b 0x33,
	 * a 0xff) and the sample positions (single-sampled); the stream: REQ_RESOURCE, HEAP_SET, RUN_FRAGMENT, WAIT. Layouts:
	 * Mesa 26.2.5's genxml v10.xml and panvk's CSF path (the port's log, "render pipelines").
	 */
	{
		struct bo_create desc_bo = { .size = 0x14000 }, rt_bo = { .size = 0x4000 };
		struct { uint32_t vm_id, initial, chunk_size, max_chunks, in_flight, handle; uint64_t ctx, first; } th;
		struct { uint32_t handle, pad; } td;
		const uint64_t DESC_VA = 0x10200000ULL, RT_VA = 0x10300000ULL;
		const unsigned W = 64, H = 64, levels = (info.tiler_features >> 8) & 0xf;
		static const uint32_t spread[8] = { 0, 0x80, 0x82, 0xa2, 0xaa, 0xea, 0xee, 0xfe };
		struct bind_op rops[2];
		struct vm_bind rvb;
		uint32_t *d, *rt;
		unsigned right = 0, n = 0;

		th = (typeof(th)){ .vm_id = vc.id, .initial = 5, .chunk_size = 2u << 20, .max_chunks = 64, .in_flight = 65535 };
		if (call(BO_CREATE, &desc_bo, sizeof(desc_bo), "the frame's descriptors") != 0 ||
		    call(BO_CREATE, &rt_bo, sizeof(rt_bo), "the render target") != 0 ||
		    (d = map_bo(desc_bo.handle, desc_bo.size)) == NULL || (rt = map_bo(rt_bo.handle, rt_bo.size)) == NULL ||
		    call(0x4b, &th, sizeof(th), "the frame's tiler heap") != 0) {
			printf("mali_test: the clear: no buffers or no tiler heap\n");
		} else {
			memset(rops, 0, sizeof(rops));
			rops[0] = (struct bind_op){ .flags = 2, .bo_handle = desc_bo.handle, .va = DESC_VA, .size = desc_bo.size };
			rops[1] = (struct bind_op){ .flags = 2, .bo_handle = rt_bo.handle, .va = RT_VA, .size = rt_bo.size };
			rvb = (struct vm_bind){ .vm_id = vc.id, .ops = { sizeof(rops[0]), 2, (uint64_t)(uintptr_t)rops } };
			call(VM_BIND, &rvb, sizeof(rvb), "the frame's binds");
			memset(d, 0, desc_bo.size);
			for (unsigned i = 0; i < W * H; i++) {
				rt[i] = 0xdeadbeef;
			}
			/* the tiler heap descriptor at 0: a buffer, of the tiler heap type; its geometry buffer at 0x1000 */
			d[0] = 9 | 2 << 4;
			d[1] = th.initial ? (2u << 20) : 0;
			*(uint64_t *)&d[2] = th.first;
			*(uint64_t *)&d[4] = th.first + 64;
			*(uint64_t *)&d[6] = th.first + (2u << 20);
			/* the tiler context at 0x11000 */
			uint32_t *tc = &d[0x11000 / 4];
			tc[2] = levels >= 8 ? 0xff : spread[levels];                    /* hierarchy mask; single-sampled */
			tc[3] = (W - 1) | (H - 1) << 16;
			tc[4] = 0;                                                      /* one layer */
			*(uint64_t *)&tc[6] = DESC_VA;                                  /* the heap descriptor */
			*(uint64_t *)&tc[8] = (DESC_VA + 0x1000) | (0x10000 >> 12);     /* the geometry buffer, 64 KiB */
			/* the framebuffer descriptor at 0x11100, its render target after its 128 bytes */
			uint32_t *fb = &d[0x11100 / 4], *rtd = &d[0x11180 / 4];
			*(uint64_t *)&fb[4] = DESC_VA + 0x11200;                        /* sample locations */
			fb[8] = (W - 1) | (H - 1) << 16;
			fb[9] = 0;                                                      /* bound min 0, 0 */
			fb[10] = (W - 1) | (H - 1) << 16;                               /* bound max */
			fb[11] = 0 /* 1 sample */ | 0 << 3 | 2 << 6 /* tie-break -180 in 0 out */ | 8 << 9 /* 16x16 tiles */ |
			    1u << 15 | 0 << 19 /* one render target */ | 1u << 24 /* colour buffer: 1 KiB */;
			fb[12] = 1u << 29;                                              /* empty tiles written too (no draws: all are) */
			*(uint64_t *)&fb[14] = DESC_VA + 0x11000;                       /* the tiler context */
			rtd[0] = 1u << 26;                                              /* internal format R8G8B8A8 */
			rtd[1] = 1 | 19 << 3 | 2 << 8 | (0 | 1 << 3 | 2 << 6 | 3 << 9) << 16 |  /* write, RGBA8, linear, identity, */
			    1u << 31;                                                   /* clean tiles written (the clear) */
			*(uint64_t *)&rtd[8] = RT_VA;
			rtd[10] = W * 4;                                                /* row stride */
			rtd[11] = W * H * 4;                                            /* surface stride */
			rtd[12] = rtd[13] = rtd[14] = rtd[15] = 0xff3366cc;             /* the clear colour */
			/* the sample positions at 0x11200: one sample, at the centre (and the origin, entry 32) */
			d[0x11200 / 4] = 128 | 128 << 16;
			d[0x11200 / 4 + 32] = 128 | 128 << 16;
			/* the stream at 0x11400 */
			uint64_t *st = (uint64_t *)&d[0x11400 / 4];
			st[n++] = 28ULL << 56 | 0ULL << 32 | 3;
			st[n++] = 28ULL << 56 | 9ULL << 32 | 0x8;
			st[n++] = 28ULL << 56 | 1ULL << 32 | 0;
			st[n++] = 28ULL << 56 | 2ULL << 32 | 1;
			st[n++] = 28ULL << 56 | 8ULL << 32 | 0xf0;
			st[n++] = 34ULL << 56 | 0xf;                                    /* REQ_RESOURCE: compute, fragment, tiler, IDVS */
			st[n++] = 1ULL << 56 | 0ULL << 48 | th.ctx;                     /* r0:r1 <- the heap context ... */
			st[n++] = 48ULL << 56 | 0ULL << 40;                             /* ... HEAP_SET */
			st[n++] = 1ULL << 56 | 40ULL << 48 | ((DESC_VA + 0x11100) | 1); /* r40 <- the FBD | (it is one), 1 RT */
			st[n++] = 2ULL << 56 | 42ULL << 48 | 0;                         /* bounding box min */
			st[n++] = 2ULL << 56 | 43ULL << 48 | ((W - 1) | (H - 1) << 16); /* max */
			st[n++] = 7ULL << 56;                                           /* RUN_FRAGMENT, Z order */
			st[n++] = 3ULL << 56 | 0xffffULL << 16;                         /* WAIT */
			{
				struct syncobj_create sc = { 0 };
				struct sync_op signal;
				struct queue_submit qs;
				struct group_submit gs;
				struct syncobj_wait w;

				uint64_t nothing = 0;

				__asm__ volatile ("dsb sy" ::: "memory");
				printf("mali_test: the clear: the GPU's state before it (kernel log)\n");
				call(0x7f, &nothing, sizeof(nothing), "the GPU's state before the clear");
				if (call(SYNCOBJ_CREATE, &sc, sizeof(sc), "a sync object") == 0) {
					signal = (struct sync_op){ .flags = 1u << 31, .handle = sc.handle };
					qs = (struct queue_submit){ .stream_size = n * 8, .stream_addr = DESC_VA + 0x11400,
						.syncs = { sizeof(signal), 1, (uint64_t)(uintptr_t)&signal } };
					gs = (struct group_submit){ .group_handle = gc.group_handle, .queue_submits = { sizeof(qs), 1,
						(uint64_t)(uintptr_t)&qs } };
					w = (struct syncobj_wait){ .handles = (uint64_t)(uintptr_t)&sc.handle, .timeout_nsec = 2000000000LL,
						.count_handles = 1, .flags = 1 };
					if (call(GROUP_SUBMIT, &gs, sizeof(gs), "the clear's submission") == 0 &&
					    call(SYNCOBJ_WAIT, &w, sizeof(w), "its wait") == 0) {
						for (unsigned i = 0; i < W * H; i++) {
							right += rt[i] == 0xff3366cc;
						}
					}
				}
			}
			{
				uint64_t nothing = 0;

				call(0x7f, &nothing, sizeof(nothing), "the GPU's state after the clear");
			}
			printf("mali_test: the clear (a fragment job, no draws; tiler levels %u): %u of %u pixels 0xff3366cc "
			    "(rt[0] 0x%x, rt[%u] 0x%x)\n", levels, right, W * H, rt[0], W * H - 1, rt[W * H - 1]);
			td = (typeof(td)){ .handle = th.handle };
			call(0x4c, &td, sizeof(td), "the frame's heap's destruction");
		}
	}
	{
		/* tiler heaps (panthor's TILER_HEAP_CREATE 0x4b, DESTROY 0x4c), with panvk's sizes: five 2 MiB chunks of 64 */
		struct { uint32_t vm_id, initial, chunk_size, max_chunks, in_flight, handle; uint64_t ctx, first; } th;
		struct { uint32_t handle, pad; } td;
		int heap_ok = 0;

		for (int round = 0; round < 2; round++) {
			th = (typeof(th)){ .vm_id = vc.id, .initial = 5, .chunk_size = 2u << 20, .max_chunks = 64, .in_flight = 65535 };
			if (call(0x4b, &th, sizeof(th), "a tiler heap") == 0 && th.handle != 0 && th.ctx >= 0x800000000000ULL &&
			    th.first >= 0x800000000000ULL && (th.first & ((2u << 20) - 1)) == 0) {
				printf("mali_test: tiler heap %u: context 0x%llx, first chunk 0x%llx\n", th.handle,
				    (unsigned long long)th.ctx, (unsigned long long)th.first);
				td = (typeof(td)){ .handle = th.handle };
				heap_ok += call(0x4c, &td, sizeof(td), "the heap's destruction") == 0;
			}
		}
		th = (typeof(th)){ .vm_id = vc.id, .initial = 1, .chunk_size = 3u << 20, .max_chunks = 1, .in_flight = 1 };
		printf("mali_test: tiler heaps: %d of 2 made and destroyed, a chunk size of 3 MiB %s\n", heap_ok,
		    call(0x4b, &th, sizeof(th), "a heap that must be refused") != 0 ? "refused (right)" : "ACCEPTED (wrong)");
	}
	{
		uint64_t nothing = 0;

		call(0x7f, &nothing, sizeof(nothing), "the queues' state into the kernel's log");
	}
	printf("mali_test: %d of %d submissions through HakiosGPU ran the shader\n", good, rounds);
	return good == rounds ? 0 : 1;
}
