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
	{
		uint64_t nothing = 0;

		call(0x7f, &nothing, sizeof(nothing), "the queues' state into the kernel's log");
	}
	printf("mali_test: %d of %d submissions through HakiosGPU ran the shader\n", good, rounds);
	return good == rounds ? 0 : 1;
}
