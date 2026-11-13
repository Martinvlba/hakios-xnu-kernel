// hakios_metal.h on the Pixel 8a's Mali-G715 (hakios-kernel devices/akita: HakiosGPU with Linux's panthor interface):
// what the Metal plug-in (plugin/HakiosMetal.m) asks of the GPU today -- the device, buffers, queues, command buffers
// with blit copies and fills -- done by the Mali. Built instead of src/c_api.cpp (`make ios-mali`); the Adreno's
// classes are not in it.
//
//   device          HakiosGPU opened (its "DRMDriver" property says panthor), a VM, one group with one queue
//   buffer          a BO of the service, write-combined for the processors, bound into the VM at an address of ours
//   command buffer  its copies and fills recorded; at commit one command stream of the GPU's (in a BO of its own)
//                   dispatches the blit shaders (src/mali/blit_shaders.h: Mesa 26.2.5's compiler for the G715, made by
//                   hakios-kernel's tools/mesa/pan/pan_blit_shaders.c), each after a WAIT for the one before, and is
//                   submitted with a sync object; the queue's thread waits for it, sets the status, runs the callback
//
//   kernel          a compute function's AIR compiled by src/mali/compile_mali.c (AIR -> NIR -> Mesa's panfrost); its
//                   code goes into a BO of the device the first time it is dispatched there
//   dispatch        recorded among the blits, in order; at commit a shader program descriptor, push constants (the
//                   buffers' addresses, the threadgroup's and the grid's sizes, the compiler's constants) and RUN_COMPUTE
//
// Kernels and buffers are counted references: a command buffer holds those it uses until the GPU is done with them,
// whatever their owners free meanwhile.
//
// Everything else of hakios_metal.h is not here yet: a plug-in that calls it does not link (on purpose).
#include "hakios-metal/hakios_metal.h"

#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <chrono>
#include <sys/stat.h>
#include <libkern/OSCacheControl.h>
#include <sys/time.h>
#include <dirent.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <mach-o/loader.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <map>
#include <vector>

#include "blit_shaders.h"
#include "compile_mali.h"

namespace {

// ---- panthor's interface, as much as is used (Linux's include/uapi/drm/panthor_drm.h, drm.h: MIT) ----
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
struct vm_create { uint32_t flags, id; uint64_t user_va_range; };
struct bind_op { uint32_t flags, bo_handle; uint64_t bo_offset, va, size; obj_array syncs; };
struct vm_bind { uint32_t vm_id, flags; obj_array ops; };
struct bo_create { uint64_t size; uint32_t flags, exclusive_vm_id, handle, pad; };
struct queue_create { uint8_t priority, pad[3]; uint32_t ringbuf_size; };
struct group_create {
    obj_array queues;
    uint8_t max_compute_cores, max_fragment_cores, max_tiler_cores, priority;
    uint32_t pad;
    uint64_t compute_core_mask, fragment_core_mask, tiler_core_mask;
    uint32_t vm_id, group_handle;
};
struct queue_submit { uint32_t queue_index, stream_size; uint64_t stream_addr; uint32_t latest_flush, pad; obj_array syncs; };
struct group_submit { uint32_t group_handle, pad; obj_array queue_submits; };
struct gem_close { uint32_t handle, pad; };
struct syncobj_create { uint32_t handle, flags; };
struct syncobj_destroy { uint32_t handle, pad; };
struct syncobj_wait { uint64_t handles; int64_t timeout_nsec; uint32_t count_handles, flags, first_signaled, pad; uint64_t deadline; };
enum : uint32_t {
    GEM_CLOSE = 0x09, DEV_QUERY = 0x40, VM_CREATE, VM_DESTROY, VM_BIND, VM_GET_STATE, BO_CREATE, BO_MMAP_OFFSET,
    GROUP_CREATE, GROUP_DESTROY, GROUP_SUBMIT, SYNCOBJ_CREATE = 0xbf, SYNCOBJ_DESTROY = 0xc0, SYNCOBJ_WAIT = 0xc3,
};
constexpr uint32_t BIND_READONLY = 1, BIND_NOEXEC = 2, BIND_UNMAP = 1u << 28;

thread_local std::string gError;

void
fail(const char *format, ...)
{
    char text[256];
    va_list ap;

    va_start(ap, format);
    vsnprintf(text, sizeof(text), format, ap);
    va_end(ap);
    gError = text;
}

// ---- the GPU's command stream (v11: v10's instructions with v12's scoreboard state; as hakios-kernel's mali_test) ----
inline uint64_t move48(unsigned r, uint64_t v) { return 1ULL << 56 | (uint64_t)r << 48 | (v & 0xffffffffffffULL); }
inline uint64_t move32(unsigned r, uint32_t v) { return 2ULL << 56 | (uint64_t)r << 48 | v; }
inline uint64_t set_state(unsigned state, uint32_t v) { return 28ULL << 56 | (uint64_t)state << 32 | v; }
constexpr uint64_t RUN_COMPUTE = 4ULL << 56 | 1, WAIT_ALL = 3ULL << 56 | 0xffffULL << 16;

}  // namespace

struct Bo {
    uint32_t handle = 0;
    uint64_t size = 0, va = 0;
    void *cpu = nullptr;
};

struct hakios_metal_device {
    io_connect_t connection = 0;
    uint64_t gpu_id = 0, shaders = 0, tiler = 0;
    uint32_t max_threads = 0, max_tasks = 0, core_range = 0;   // a core's threads and tasks; highest core id + 1
    Bo wls, tls;                                // threadgroup memory and spill space, grown as dispatches need
    std::vector<Bo> retired;                    // grown out of (a command buffer in flight may use one): freed at close
    uint32_t vm = 0, group = 0;
    uint64_t next_va = 0x100000000ULL;          // our addresses: the lower half is the program's
    std::mutex lock;                            // around the service's calls and next_va
    Bo descriptors;                             // the blit shaders' code, their program descriptors, the local storage
    uint64_t spd[5] = {};
    // draws: the queue's one tiler heap (HEAP_SET once a stream; a second heap on the same queue was not taken), its
    // descriptor and 64 KiB geometry buffer (heapdesc), the tiler's levels
    uint32_t heap_handle = 0, tiler_levels = 0;
    uint64_t heap_ctx = 0, heap_first = 0;
    std::atomic<bool> heap_set{false};          // the queue given its heap (HEAP_SET once a queue, as panvk and gallium)
    Bo heapdesc;
    bool heap_ok = false;                       // the heap made: draws can be served
    struct hakios_metal_program *preload_for[5][8] = {};   // the Load action's draw for 0 to 4 targets, with depth (1)
    bool preload_failed_for[5][8] = {};                    // and/or stencil (2) loaded too, multisample (4) (preloadProgram)
    struct hakios_metal_program *resolve_prog = nullptr;   // the driver's resolve draw (resolveProgram), under preload_lock
    bool resolve_failed = false;
    std::mutex preload_lock;
    struct hakios_metal_program *blit_prog = nullptr;  // the driver's blit draw (blitProgram), under preload_lock
    bool blit_failed = false;
    // where a colour output with no attachment goes (DrawJob::More::missing): the GPU writes such a target back even
    // with its write disabled -- at address 0 it faulted (AS1, a write at 0x80200) and the queue stopped
    Bo scratch_rt;
    uint64_t scratch_rt_size = 0;
    std::mutex scratch_lock;
    // WHAT STANDS FOR A RESOURCE A FUNCTION DECLARES AND NOTHING IS BOUND AT (11-12): Metal lets a draw or dispatch go
    // with such a hole as long as the function does not use it on the road it takes (MPS binds what a layer has);
    // refusing it stopped mediaserverd's networks. A buffer: 64 KiB of zeroes; a texture: 4x4 of zeroes.
    Bo nothing_bo;
    std::atomic<uint64_t> nothing_va{0};
    hakios_metal_texture *nothing_texture = nullptr;
    std::string preload_error;

    int call(uint32_t selector, void *request, size_t size)
    {
        size_t out = size;
        return IOConnectCallStructMethod(connection, selector, request, size, request, &out) == KERN_SUCCESS ? 0 : -1;
    }

    bool newBo(Bo &bo, uint64_t size, uint32_t bind_flags)
    {
        std::lock_guard<std::mutex> g(lock);
        bo_create c = {};
        c.size = (size + 0x3fff) & ~0x3fffULL;
        if (call(BO_CREATE, &c, sizeof(c)) != 0) {
            fail("no GPU memory of %llu bytes", (unsigned long long)size);
            return false;
        }
        return mapAndBind(bo, c.handle, c.size, bind_flags);
    }

    // a buffer the kernel holds (HANDLE, SIZE) mapped for the CPU and bound into the GPU's address space (lock held)
    bool mapAndBind(Bo &bo, uint32_t handle, uint64_t size, uint32_t bind_flags)
    {
        mach_vm_address_t address = 0;
        mach_vm_size_t mapped = size;
        if (IOConnectMapMemory64(connection, handle, mach_task_self(), &address, &mapped, kIOMapAnywhere) != KERN_SUCCESS) {
            fail("a buffer could not be mapped");
            return false;
        }
        bo.handle = handle;
        bo.size = size;
        bo.cpu = (void *)(uintptr_t)address;
        if (next_va >> 32 != (next_va + size - 1) >> 32) {
            next_va = (next_va + size - 1) & ~0xffffffffULL;    // (none across 4 GB: the local storage's must not)
        }
        bo.va = next_va;
        next_va += size + 0x4000;               // a page between them
        bind_op op = {};
        op.flags = bind_flags;
        op.bo_handle = handle;
        op.va = bo.va;
        op.size = size;
        vm_bind b = { vm, 0, { sizeof(op), 1, (uint64_t)(uintptr_t)&op } };
        if (call(VM_BIND, &b, sizeof(b)) != 0) {
            fail("a buffer could not be bound into the GPU's address space");
            return false;
        }
        return true;
    }

    // An IOSurface's memory as a buffer (HakiosGPU's 0x7e, as the Adreno service's), shared by every texture over it:
    // the kernel gives the same handle for the same surface, so it is closed with the last of them. false with the
    // reason; INFO (size, offset, row bytes, width, height, pixel format, planes) filled.
    struct SurfaceInfo { uint64_t size; uint32_t offset, row_bytes, width, height, pixel_format, planes; };
    // A SURFACE'S MIRROR (11-10): a texture of 64-byte rows standing for an IOSurface whose rows the Mali cannot sample
    // (mirrorOf), shared by every texture over that surface; copied from the surface on the GPU when made and when the
    // plug-in says the CPU changed the surface, copied back by each command buffer that draws into it (at its commit)
    struct Mirror { hakios_metal_texture *tex; int refs; uint64_t surface_va; uint32_t row_bytes, row_words, rows; };
    std::map<unsigned, Mirror> mirrors;
    std::mutex mirror_lock;
    hakios_metal_queue *mirror_queue = nullptr;
    struct Surface { Bo bo; SurfaceInfo info; int refs; };
    std::map<unsigned, Surface> surfaces;
    bool surfaceBo(unsigned id, Bo &bo, SurfaceInfo &info)
    {
        std::lock_guard<std::mutex> g(lock);
        auto it = surfaces.find(id);
        if (it != surfaces.end()) {
            it->second.refs++;
            bo = it->second.bo;
            info = it->second.info;
            return true;
        }
        struct { uint32_t surface, handle; uint64_t size; uint32_t offset, row_bytes, width, height, format, planes; } r = {};
        r.surface = id;
        int e = call(0x7e, &r, sizeof(r));
        if (e != 0) {
            fail("IOSurface %u could not be had as GPU memory (HakiosGPU: 0x%x)", id, e);
            return false;
        }
        Surface sf = {};
        sf.info = { r.size, r.offset, r.row_bytes, r.width, r.height, r.format, r.planes };
        if (!mapAndBind(sf.bo, r.handle, r.size, BIND_NOEXEC)) {
            gem_close gc = { r.handle, 0 };
            call(GEM_CLOSE, &gc, sizeof(gc));
            return false;
        }
        sf.refs = 1;
        surfaces[id] = sf;
        bo = sf.bo;
        info = sf.info;
        return true;
    }
    void surfaceRelease(unsigned id)
    {
        Bo bo;
        {
            std::lock_guard<std::mutex> g(lock);
            auto it = surfaces.find(id);
            if (it == surfaces.end() || --it->second.refs > 0) {
                return;
            }
            bo = it->second.bo;
            surfaces.erase(it);
        }
        freeBo(bo);
    }

    // COMMAND BUFFERS' STREAM BUFFERS, KEPT FOR THE NEXT: a buffer made, mapped and bound for each command buffer and
    // unbound after cost 100-400 us of the commit (metal_cbtime); sizes powers of two from 64 KiB, at most 16 kept
    std::vector<Bo> stream_pool;
    bool takeStream(Bo &bo, uint64_t size)
    {
        uint64_t want = 64 * 1024;
        while (want < size) want <<= 1;
        {
            std::lock_guard<std::mutex> g(lock);
            for (size_t i = 0; i < stream_pool.size(); i++) {
                if (stream_pool[i].size == want) {
                    bo = stream_pool[i];
                    stream_pool.erase(stream_pool.begin() + i);
                    return true;
                }
            }
        }
        return newBo(bo, want, BIND_NOEXEC);
    }
    void giveStream(Bo &bo)
    {
        {
            std::lock_guard<std::mutex> g(lock);
            if (bo.handle != 0 && stream_pool.size() < 16) {
                stream_pool.push_back(bo);
                bo = Bo();
                return;
            }
        }
        freeBo(bo);
    }

    void freeBo(Bo &bo)
    {
        std::lock_guard<std::mutex> g(lock);
        if (bo.handle == 0) {
            return;
        }
        bind_op op = {};
        op.flags = BIND_UNMAP;
        op.va = bo.va;
        op.size = bo.size;
        vm_bind b = { vm, 0, { sizeof(op), 1, (uint64_t)(uintptr_t)&op } };
        call(VM_BIND, &b, sizeof(b));
        IOConnectUnmapMemory64(connection, bo.handle, mach_task_self(), (mach_vm_address_t)(uintptr_t)bo.cpu);
        gem_close gc = { bo.handle, 0 };
        call(GEM_CLOSE, &gc, sizeof(gc));
        bo = Bo();
    }
};

// the zeroes that stand for a buffer or a texture a function declares and nothing is bound at (see the device's
// nothing_bo); 0 / nullptr if they cannot be made
static std::atomic<uint64_t> gNothingVa{0};
static uint64_t
nothingVa(hakios_metal_device *d)
{
    if (d->nothing_va.load() == 0) {
        std::lock_guard<std::mutex> g(d->scratch_lock);
        if (d->nothing_va.load() == 0 && d->newBo(d->nothing_bo, 0x10000, BIND_NOEXEC)) {
            memset(d->nothing_bo.cpu, 0, 0x10000);
            d->nothing_va = d->nothing_bo.va;
            gNothingVa = d->nothing_bo.va;
        }
    }
    return d->nothing_va.load();
}
static hakios_metal_texture *nothingTexture(hakios_metal_device *d);


namespace {
// DEVICE's buffer B at least SIZE bytes (the old one kept until the device closes); false with the reason
bool
grow(hakios_metal_device *d, Bo &b, uint64_t size)
{
    if (b.size >= size) {
        return true;
    }
    Bo bigger;
    if (!d->newBo(bigger, size, BIND_NOEXEC)) {
        return false;
    }
    std::lock_guard<std::mutex> g(d->lock);
    if (b.handle != 0) {
        d->retired.push_back(b);
    }
    b = bigger;
    return true;
}

inline uint32_t npot(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }
inline uint32_t log2u(uint32_t v) { uint32_t l = 0; while ((1u << l) < v) l++; return l; }
}  // namespace

/*
 * A BUFFER'S SLACK (11-11): a page past what was asked for, filled with a pattern and looked at when the buffer goes --
 * backboardd died writing 0x750 bytes into the page after one (CA::OGL::MetalBufferPool::write_buffer, TODO 8i): an
 * overrun lands in the slack now and is said (the length, how far past it), instead of taking the window server down.
 */
constexpr uint64_t kBufferSlack = 16384;
constexpr uint8_t kSlackByte = 0xa5;

struct hakios_metal_buffer {
    hakios_metal_device *device;
    Bo bo;
    uint64_t length;
    std::atomic<int> refs{1};       // its owner's, and each command buffer's that uses it

    void retain() { refs.fetch_add(1); }
    void checkSlack()
    {
        auto *p = (const uint8_t *)bo.cpu;
        if (p == nullptr || bo.size <= length) return;
        uint64_t last = 0;
        for (uint64_t i = length; i < bo.size; i++) {
            if (p[i] != kSlackByte) last = i + 1;
        }
        if (last != 0) {
            static std::atomic<unsigned> said{0};
            if (said++ < 20) {
                fprintf(stderr, "hakios-metal: A BUFFER WAS WRITTEN PAST ITS END: %llu bytes long, written to %llu bytes past "
                    "it (the slack caught it; TODO 8i)\n", (unsigned long long)length, (unsigned long long)(last - length));
            }
        }
    }
    void release()
    {
        if (refs.fetch_sub(1) == 1) {
            checkSlack();
            device->freeBo(bo);
            delete this;
        }
    }
};

struct hakios_metal_kernel {
    mali_kernel_binary binary = {};
    std::atomic<int> refs{1};
    std::mutex lock;                // around the code's upload
    hakios_metal_device *device = nullptr;
    Bo code;                        // the code, and its program descriptor at spd
    uint64_t spd = 0;

    void retain() { refs.fetch_add(1); }
    void release()
    {
        if (refs.fetch_sub(1) == 1) {
            if (device != nullptr) {
                device->freeBo(code);
            }
            mali_kernel_binary_free(&binary);
            delete this;
        }
    }

    unsigned maxThreads() const { return binary.work_regs <= 32 ? 1024 : 512; }

    // the code in the device's memory (once per kernel: one device in a process)
    bool upload(hakios_metal_device *d)
    {
        std::lock_guard<std::mutex> g(lock);
        if (device == d) {
            return true;
        }
        if (device != nullptr) {
            fail("%s: compiled for another device", binary.name);
            return false;
        }
        uint64_t at = (binary.size + 0x100 + 0x3f) & ~0x3fULL;     // (the shader core reads ahead of the code's end)
        if (!d->newBo(code, at + 0x40, BIND_READONLY)) {
            return false;
        }
        auto *bytes = (uint8_t *)code.cpu;
        memcpy(bytes, binary.code, binary.size);
        auto *w = (uint32_t *)(bytes + at);
        w[0] = 8 | 1 << 4 | (binary.work_regs <= 32 ? 2u : 0u) << 30;
        w[1] = (uint32_t)(binary.preload >> 48);
        w[2] = (uint32_t)code.va;
        w[3] = (uint32_t)(code.va >> 32);
        __asm__ volatile("dsb sy" ::: "memory");
        spd = code.va + at;
        device = d;
        return true;
    }
};

// A program: a vertex and a fragment function compiled for the Mali (compile_mali.c), linked by the vertex stage's
// varying layout; its code and three shader program descriptors (position, varying, fragment) in a BO of the device's
// the first time it is drawn with.
namespace {
bool bootArg(const char *word);         // (a word in the boot-args: defined below)
}  // namespace

struct hakios_metal_program {
    mali_kernel_binary vs = {}, fs = {};
    std::atomic<int> refs{1};
    std::mutex lock;
    hakios_metal_device *device = nullptr;
    Bo code;
    uint64_t pos_spd = 0, vary_spd = 0, frag_spd = 0;
    unsigned color_format = 0, samples = 1;
    bool blend_enabled = false, vertex_descriptor = false;
    // colour attachment 0's blend (hakios_metal_program_set_blend): the fixed-function equation word, whether it
    // reads the tile buffer, or why it cannot be done in fixed function
    uint32_t blend_equation = 0x132 | 0x132 << 12 | 0xfu << 28;    // replace, every channel written
    // the vertex descriptor (hakios_metal_program_set_vertex_*): the stages compiled again with the stage_in
    // attributes fetched in the shader (compile_mali.c's lower_vertex_fetch), once, before the first draw
    std::vector<uint8_t> vs_bitcode, fs_bitcode;
    std::vector<mali_constant> vs_constants, fs_constants;
    mali_vertex_fetch fetch = {};
    bool fetch_described = false, fetch_compiled = false;
    // the targets whose blend fixed function cannot do, blended in the fragment shader instead (compileFetch compiles it
    // again with mali_fragment_compile; their equation is a replace that reads the tile buffer)
    mali_shader_blend shader_blend = {};
    bool rt_blend_on[4] = {};                   // blending enabled at colour attachment n (hakios_metal_program_set_blend*)
    unsigned rt_format[4] = {};                 // colour attachment n's MTLPixelFormat (hakios_metal_program_set_attachment_format)
    std::string fetch_refusal;
    bool blend_reads_dest = false;
    std::string blend_refusal;
    // colour attachments 1 to 3's (hakios_metal_program_set_blend_at): the same, for the multiple-target draws
    uint32_t more_blend_equation[3] = { 0x132 | 0x132 << 12 | 0xfu << 28, 0x132 | 0x132 << 12 | 0xfu << 28,
                                        0x132 | 0x132 << 12 | 0xfu << 28 };
    bool more_blend_reads_dest[3] = {};
    std::string more_blend_refusal[3];
    char name[300] = "";
    char label[128] = "";       // the pipeline descriptor's (hakios_metal_program_set_label: diagnostics only)
    // a program specialised for a buffer's bytes (hakios_metal_program_specialize): its fragment stage compiled with
    // [[buffer(known_binding)]] read as KNOWN (compile_mali.c's mali_compile_known), every compile of it alike
    int known_binding = -1;
    std::vector<uint8_t> known;

    void retain() { refs.fetch_add(1); }
    void release()
    {
        if (refs.fetch_sub(1) == 1) {
            if (device != nullptr) {
                device->freeBo(code);
            }
            mali_kernel_binary_free(&vs);
            mali_kernel_binary_free(&fs);
            delete this;
        }
    }

    bool upload(hakios_metal_device *d)
    {
        std::lock_guard<std::mutex> g(lock);
        if (device == d) {
            return true;
        }
        if (device != nullptr) {
            fail("%s: made for another device", name);
            return false;
        }
        uint64_t fs_at = (vs.size + 0x100 + 0xff) & ~0xffULL, spd_at = (fs_at + fs.size + 0x100 + 0x3f) & ~0x3fULL;
        if (!d->newBo(code, spd_at + 0xc0, BIND_READONLY)) {
            return false;
        }
        auto *bytes = (uint8_t *)code.cpu;
        memcpy(bytes, vs.code, vs.size);
        memcpy(bytes + fs_at, fs.code, fs.size);
        auto spd = [&](unsigned slot, unsigned stage, unsigned regs, uint64_t preload, uint64_t va, bool helpers) {
            auto *w = (uint32_t *)(bytes + spd_at + 0x40 * slot);
            // (bit 28, "requires helper threads", fragment only -- 11-11, TODO 8h: SET FOR EVERY FRAGMENT SHADER. Mesa sets
            // it for barriers alone and lets the compiler end the helper lanes after their last use (VA_FLOW_DISCARD after
            // the last CLPER / implicit-LOD texture); with CoreAnimation's shaders through our AIR front end that left the
            // derivatives wrong wherever a triangle covers part of a 2x2 quad -- a light seam along every edge of its
            // meshes: Control Center's tiles' dashed inner outlines, the diagonals of each quad's split (the round
            // buttons', the tiles' corner pieces). Kept to the end, they are right; no cost seen (idle cube 109-111 fps
            // either way). Why the compiler's own end of helpers is too early is not known yet. metal_nohelpers: barriers
            // only, as Mesa)
            static const bool no_helpers = bootArg("metal_nohelpers");
            w[0] = 8 | stage << 4 | (regs <= 32 ? 2u : 0u) << 30 |
                   ((helpers || (stage == 2 && !no_helpers)) ? 1u << 28 : 0u);
            w[1] = (uint32_t)(preload >> 48);
            w[2] = (uint32_t)va;
            w[3] = (uint32_t)(va >> 32);
            return code.va + spd_at + 0x40 * slot;
        };
        pos_spd = spd(0, 3, vs.work_regs, vs.preload, code.va, false);
        vary_spd = vs.secondary_offset ? spd(1, 3, vs.secondary_work_regs, vs.secondary_preload, code.va + vs.secondary_offset, false) : 0;
        frag_spd = spd(2, 2, fs.work_regs, fs.preload, code.va + fs_at, fs.helpers != 0);
        __asm__ volatile("dsb sy" ::: "memory");
        device = d;
        return true;
    }
};

// A texture: linear rows in a BO of its own (Mali's "Generic Plane", clump ordering linear), described to the GPU per
// dispatch. Metal's pixel formats served, as Mesa's lib/pan_format.c encodes them for v9+: RGB component order |
// format << 12 | sRGB << 20.
struct hakios_metal_texture {
    hakios_metal_device *device;
    Bo bo;                                      // the memory (its own, an IOSurface's, a buffer's or a parent's)
    unsigned type, format, width, height, depth, usage;
    uint32_t mali_format, bytes, stride;        // per texel (a compressed format's: per block); per row (of blocks;
                                                //  64-byte aligned for its own)
    unsigned bw = 1, bh = 1;                    // a compressed format's block (ASTC), in texels
    uint64_t stencil_off = 0;                   // Depth32Float_Stencil8: the stencil plane (a byte a texel) at offset +
    uint32_t stencil_stride = 0;                //  stencil_off, rows stencil_stride apart (0: none)
    uint64_t stencil_level_off[16] = {};        // ... mipmapped: each level's stencil plane (from gpu()), after all the
    uint32_t stencil_level_stride[16] = {};     //  depth levels (0: level 0's is stencil_off alone)
    uint64_t offset = 0;                        // where its pixels start in BO
    unsigned origin = 0;                        // 0 its own memory, 1 an IOSurface's, 2 a buffer's, 4 a view's
    unsigned surface = 0;                       // the IOSurface's ID (origin 1)
    bool mirrored = false;                      // ... its memory the surface's mirror (Mirror), not the surface's own
    bool memoryless = false;                    // Metal's memoryless storage (hakios_metal_texture_set_memoryless): an
                                                //  extra colour attachment of it lives in the tile buffer alone
    std::atomic<unsigned> widest_rts{0};        // the most targets a draw with it as colour attachment 1 had (passWide)
    // (11-10, a diagnostic for metal_stats: CoreAnimation's second colour attachment -- is it ever read?)
    std::atomic<unsigned> as_extra_rt{0};       // draws it was colour attachment 1-3 of
    std::atomic<unsigned> reads{0};             // times it was sampled by a draw or dispatch, copied from, or read back
    unsigned levels = 1, samples = 1;
    // a mipmapped texture (levels > 1): level n's rows at offset + level_off[n], level_stride[n] apart, its size
    // levelWidth/levelHeight; the levels' plane descriptors (32 bytes each, level 0 first) at planes_off in BO, written
    // once when the texture is made (a view of levels n.. points at the parent's, 32 * n further on)
    uint64_t level_off[16] = {};
    uint32_t level_stride[16] = {};
    uint64_t planes_off = 0;                    // 0: none (one level, described per use)
    uint32_t swizzle = 0;                       // 0: the identity
    // A LAYERED TEXTURE (11-12: MTLTextureType 1DArray 1, 2DArray 3, Cube 5, CubeArray 6 -- a cube's faces are layers,
    // six a cube, in Metal's slice order, which is the Mali's): LAYERS of them, each the whole chain of its levels,
    // layer_stride bytes apart; the plane descriptors at planes_off layer by layer, each layer's levels in order
    // (Mesa's pan_emit_iview_texture_payload for v7 and later)
    unsigned layers = 1;
    uint64_t layer_stride = 0;
    hakios_metal_texture *parent = nullptr;     // a view's (held)
    hakios_metal_buffer *buffer = nullptr;      // a buffer texture's (held)
    std::atomic<int> refs{1};

    void retain() { refs.fetch_add(1); }
    void release();
    uint8_t *pixels() const { return (uint8_t *)bo.cpu + offset; }
    uint64_t gpu() const { return bo.va + offset; }
    unsigned levelWidth(unsigned l) const { return width >> l ? width >> l : 1; }
    unsigned levelHeight(unsigned l) const { return type == 0 || type == 1 ? 1 : height >> l ? height >> l : 1; }
    uint32_t levelStride(unsigned l) const { return l == 0 ? stride : level_stride[l]; }
    unsigned levelRows(unsigned l) const { return (levelHeight(l) + bh - 1) / bh; }    // rows of blocks
    uint64_t levelOffset(unsigned l) const { return l == 0 ? 0 : level_off[l]; }
    // the bytes from gpu() that its pixels (every level, every slice) span
    uint64_t span() const {
        if (layers > 1) return layer_stride * layers;
        if (levels > 1) return level_off[levels - 1] + (uint64_t)level_stride[levels - 1] * levelRows(levels - 1);
        return (uint64_t)stride * levelRows(0) * depth * samples;    // (a multisample texture: a plane a sample)
    }
};

struct hakios_metal_sampler {
    uint32_t words[8];                          // genxml v10 "Sampler"
};

static void mirrorRelease(hakios_metal_device *d, unsigned surface);

void
hakios_metal_texture::release()
{
    if (refs.fetch_sub(1) != 1) {
        return;
    }
    if (origin == 0) {
        device->freeBo(bo);
    } else if (origin == 1) {
        if (mirrored) {
            mirrorRelease(device, surface);
        }
        device->surfaceRelease(surface);
    }
    if (parent != nullptr) {
        parent->release();
    }
    if (buffer != nullptr) {
        buffer->release();
    }
    delete this;
}

namespace {
struct MaliFormat { unsigned metal; uint32_t mali; uint32_t bytes; unsigned bw = 1, bh = 1; };  // (bytes: a block's)
const MaliFormat kFormats[] = {
    { 1, 163u << 12 | 16, 1 },                  // A8Unorm: R8's, the texture's swizzle 000R (alphaSwizzle below)
    { 10, 163u << 12 | 16, 1 },                 // R8Unorm (R001 -> RGB1)
    { 23, 132u << 12 | 16, 2 },                 // R16Uint
    { 25, 199u << 12 | 16, 2 },                 // R16Float
    { 30, 171u << 12 | 16, 2 },                 // RG8Unorm
    { 53, 133u << 12 | 16, 4 },                 // R32Uint
    { 55, 167u << 12 | 16, 4 },                 // R32Float
    { 70, 187u << 12 | 0, 4 },                  // RGBA8Unorm
    { 71, 187u << 12 | 0 | 1u << 20, 4 },       // RGBA8Unorm_sRGB
    { 80, 187u << 12 | 4, 4 },                  // BGRA8Unorm
    { 81, 187u << 12 | 4 | 1u << 20, 4 },       // BGRA8Unorm_sRGB
    { 105, 175u << 12 | 16, 8 },                // RG32Float
    { 115, 223u << 12 | 0, 8 },                 // RGBA16Float
    { 123, 157u << 12 | 0, 16 },                // RGBA32Uint
    { 125, 191u << 12 | 0, 16 },                // RGBA32Float
    // packed 16-bit and 32-bit formats (Metal names the components from the low bits, as gallium does: Mesa's
    // pan_format.c entry for the gallium format of the same name)
    { 40, 64u << 12 | 20, 2 },                  // B5G6R5Unorm: RGB565, BGR1
    { 42, 186u << 12 | 12, 2 },                 // ABGR4Unorm: RGBA4_UNORM, ABGR
    { 43, 65u << 12 | 4, 2 },                   // BGR5A1Unorm: RGB5_A1_UNORM, BGRA
    { 20, 164u << 12 | 16, 2 },                 // R16Unorm
    { 60, 172u << 12 | 16, 4 },                 // RG16Unorm
    { 65, 207u << 12 | 16, 4 },                 // RG16Float
    { 72, 123u << 12 | 0, 4 },                  // RGBA8Snorm
    { 90, 67u << 12 | 0, 4 },                   // RGB10A2Unorm
    { 94, 67u << 12 | 4, 4 },                   // BGR10A2Unorm
    { 92, 89u << 12 | 16, 4 },                  // RG11B10Float
    { 93, 91u << 12 | 16, 4 },                  // RGB9E5Float
    { 110, 188u << 12 | 0, 8 },                 // RGBA16Unorm
    // integer formats (11-12: mediaserverd's video kernels read and write R16Sint; the rest of the family with it --
    // genxml v10's R8UI 131 .. RGBA32I 221)
    { 13, 131u << 12 | 16, 1 },                 // R8Uint
    { 14, 195u << 12 | 16, 1 },                 // R8Sint
    { 24, 196u << 12 | 16, 2 },                 // R16Sint
    { 33, 139u << 12 | 16, 2 },                 // RG8Uint
    { 34, 203u << 12 | 16, 2 },                 // RG8Sint
    { 54, 197u << 12 | 16, 4 },                 // R32Sint
    { 63, 140u << 12 | 16, 4 },                 // RG16Uint
    { 64, 204u << 12 | 16, 4 },                 // RG16Sint
    { 73, 155u << 12 | 0, 4 },                  // RGBA8Uint
    { 74, 219u << 12 | 0, 4 },                  // RGBA8Sint
    { 103, 141u << 12 | 16, 8 },                // RG32Uint
    { 104, 205u << 12 | 16, 8 },                // RG32Sint
    { 113, 156u << 12 | 0, 8 },                 // RGBA16Uint
    { 114, 220u << 12 | 0, 8 },                 // RGBA16Sint
    { 124, 221u << 12 | 0, 16 },                // RGBA32Sint
    // depth and stencil (render pass attachments; sampled as the plain format of their size): Depth16Unorm,
    // Depth32Float, Stencil8, Depth32Float_Stencil8 (its depth plane; the stencil plane after it, stencil_off)
    { 250, 164u << 12 | 16, 2 },
    { 252, 167u << 12 | 16, 4 },
    { 253, 163u << 12 | 16, 1 },
    { 260, 167u << 12 | 16, 4 },
    // ASTC, LDR (Metal 204..217) and sRGB (186..199), 16 bytes a block: on Valhall the format is the one it decodes to
    // (RGBA16F; sRGB RGBA8), the block size in the plane descriptor ("ASTC 2D Plane", describe())
    // (11-13: METAL'S NUMBERS HAVE A GAP after 6x6 -- 204..208 are 4x4..6x6, 209 is not a format, 210..218 are 8x5..12x12
    // (sRGB: 186..190, 192..200). Counted straight through, every block size from 8x5 up was the next one's: Weather's
    // radar tiles, 512x512 ASTC 8x8 (212), were taken for 10x5 -- 103 rows of blocks read from the 64 the application
    // gave, and it died in the copy, three times in the tester's hands)
#define ASTC(n, w, h) { 204 + n, 223u << 12 | 0, 16, w, h }, { 186 + n, 187u << 12 | 0 | 1u << 20, 16, w, h }
    ASTC(0, 4, 4), ASTC(1, 5, 4), ASTC(2, 5, 5), ASTC(3, 6, 5), ASTC(4, 6, 6), ASTC(6, 8, 5), ASTC(7, 8, 6), ASTC(8, 8, 8),
    ASTC(9, 10, 5), ASTC(10, 10, 6), ASTC(11, 10, 8), ASTC(12, 10, 10), ASTC(13, 12, 10), ASTC(14, 12, 12),
#undef ASTC
};
// A COLOUR ATTACHMENT'S FORMAT: the render target's word 1 (write-back format, linear block, sRGB, swizzle) for the
// formats the Mali build draws into; false for the rest. The tile buffer holds R8G8B8A8 for every one of them (the
// blend descriptors' and LD_TILE's conversion), converted at write-back. Metal names packed formats from the low bits:
// BGRA8, BGR5A1 and B5G6R5 all have blue first, hence the swizzle.
constexpr uint32_t kRtFloat = 1u << 30;       // (not the descriptor's: a float target, see below)
constexpr uint32_t kRtRaw = 1u << 29;         // (not the descriptor's: a raw 32-bit target, R32Float)
constexpr uint32_t kRtWide = 1u << 28;        // (not the descriptor's: 8 bytes a pixel in the tile buffer, RGBA16Float)
constexpr uint32_t kRtOurs = kRtFloat | kRtRaw | kRtWide;
bool
rtFormat(unsigned f, uint32_t &rt1)
{
    uint32_t wb;
    bool bgra, srgb = f == 71 || f == 81, flt = false, raw = false, wide = false;
    switch (f) {
    case 70: case 71: wb = 19; bgra = false; break;     // R8G8B8A8
    case 80: case 81: wb = 19; bgra = true; break;
    case 43: wb = 28; bgra = true; break;               // BGR5A1Unorm: R5G5B5A1
    case 40: wb = 21; bgra = true; break;               // B5G6R5Unorm: R5G6B5
    // (11-12, what widgets draw into once every process has the device: one and two 8-bit channels, the tile
    // buffer's RGBA8 written back as its first channel(s) -- panfrost's blendable R8_UNORM and R8G8_UNORM)
    case 10: wb = 16; bgra = false; break;              // R8Unorm: R8
    case 30: wb = 17; bgra = false; break;              // RG8Unorm: R8G8
    // HALF FLOATS (11-12: the widgets' accumulators, chronod): the tile buffer holds them as they are -- the internal
    // format R16F / RG16F (16, 17: four bytes a pixel like the rest, so the tile's kilobyte a target stands), the
    // write-back's "float colour format" of the same numbers, the blend descriptor's memory format the pixel
    // format's (rtBlend). kRtFloat marks the word for emitDraw, which takes it out again.
    case 25: wb = 16; bgra = false; flt = true; break;  // R16Float
    case 65: wb = 17; bgra = false; flt = true; break;  // RG16Float
    // R32Float (11-12 night: WebKit's GPU process draws its paths' coverage into one): no tile-buffer format of its
    // own -- raw, 32 bits as they are (internal RAW32, written back as RAW32; Mesa's get_rt_formats for a format that
    // is not blendable). Opaque writes only: the fixed-function blend does not know such a target.
    case 55: wb = 3; bgra = false; raw = true; break;
    // RGBA16Float (11-13: HDR and wide-colour content): the tile buffer's RGBA16F (internal format 18, written back as
    // the float colour format 19), EIGHT bytes a pixel -- two of the tile's kilobytes (rtKilobytes: the targets after it
    // start further on, the framebuffer's allocation counts it), its clear two words (rtClearHigh)
    case 115: wb = 19; bgra = false; flt = true; wide = true; break;
    // RG11B10Float (11-13; not run on the phone yet): the tile buffer's RG11F_B10F (internal format 20, the float
    // colour format 20 -- Mesa's BFMT_FLOAT(R11G11B10_FLOAT, RG11F_B10F) for this architecture), four bytes a pixel
    // like the rest; the blend's memory format the texture's (89), its clear the packed word (rtClear)
    case 92: wb = 20; bgra = false; flt = true; break;
    default: return false;
    }
    uint32_t swizzle = bgra ? (2 | 1 << 3 | 0 << 6 | 3 << 9) : (0 | 1 << 3 | 2 << 6 | 3 << 9);
    rt1 = wb << 3 | 2u << 8 | (srgb ? 1u << 14 : 0) | swizzle << 16 | (flt ? kRtFloat : 0u) | (raw ? kRtRaw : 0u) |
        (wide ? kRtWide : 0u);
    return true;
}
// the render target's internal format (word 0, bits 26..), the blend descriptor's memory format, and a clear colour
// as the tile buffer holds it, for a target whose word 1 is RT1
inline uint32_t rtInternal(uint32_t rt1) { return (rt1 & kRtRaw) ? 34 : (rt1 & kRtWide) ? 18 : (rt1 & kRtFloat) ? (rt1 >> 3) & 31 : 1; }
inline uint32_t rtKilobytes(uint32_t rt1) { return (rt1 & kRtWide) ? 2 : 1; }   // of the tile buffer, a 16x16 tile (or 8x8 of 4 samples)
inline uint32_t
rtBlend(uint32_t rt1)
{
    if (rt1 & kRtRaw) return 167u << 12 | 16;                   // R32F, as a texture of it is named
    if (rt1 & kRtWide) return 223u << 12;                       // RGBA16F
    if ((rt1 & kRtFloat) && ((rt1 >> 3) & 31) == 20) return 89u << 12;      // R11F_G11F_B10F
    // (11-13) AN sRGB TARGET's memory format says so (bit 20, as a texture of it does): without it the blend unit wrote
    // the shader's linear colour into the tile buffer as it was -- BGRA8Unorm_sRGB and RGBA8Unorm_sRGB targets came out
    // linear, some forty times too dark in the shadows: Maps' and Weather's maps in dark mode were all but black
    if (!(rt1 & kRtFloat)) return 237u << 12 | ((rt1 & 1u << 14) ? 1u << 20 : 0u);
    return ((rt1 >> 3) & 31) == 16 ? 199u << 12 : 207u << 12;
}
inline uint32_t
halfOf(double v)
{
    float f = (float)v;
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    if (e >= 31) return sign | 0x7c00 | (((x >> 23) & 0xff) == 0xff && mant ? 0x200 : 0);   // infinity, NaN
    if (e <= 0) return e < -10 ? sign : sign | ((mant | 0x800000) >> (14 - e));              // zero, denormal
    return sign | (uint32_t)e << 10 | mant >> 13;
}
uint8_t unorm8(double v);
inline uint32_t
rtClear(uint32_t rt1, const double *k)
{
    if (rt1 & kRtRaw) {
        float f = (float)k[0];
        uint32_t bits;
        memcpy(&bits, &f, 4);
        return bits;
    }
    if (rt1 & kRtWide) {
        return halfOf(k[0]) | halfOf(k[1]) << 16;               // (red, green; blue and alpha: rtClearHigh)
    }
    if ((rt1 & kRtFloat) && ((rt1 >> 3) & 31) == 20) {
        // RG11B10Float: red and green 5 bits of exponent and 6 of mantissa, blue 5 and 5, no sign (a half without its
        // sign and its low mantissa bits; a negative value is 0)
        uint32_t h[3];
        for (int i = 0; i < 3; i++) h[i] = k[i] > 0 ? halfOf(k[i]) & 0x7fff : 0;
        return h[0] >> 4 | (h[1] >> 4) << 11 | (h[2] >> 5) << 22;
    }
    if (rt1 & kRtFloat) {
        uint32_t r = halfOf(k[0]);
        return ((rt1 >> 3) & 31) == 16 ? r | r << 16 : r | halfOf(k[1]) << 16;
    }
    if (rt1 & 1u << 14) {
        // (11-13) an sRGB target's clear colour is linear, as a shader's output is: encoded here (alpha is not)
        auto enc = [](double v) { v = v < 0 ? 0 : v > 1 ? 1 : v; return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055; };
        return unorm8(enc(k[0])) | unorm8(enc(k[1])) << 8 | unorm8(enc(k[2])) << 16 | (uint32_t)unorm8(k[3]) << 24;
    }
    return unorm8(k[0]) | unorm8(k[1]) << 8 | unorm8(k[2]) << 16 | (uint32_t)unorm8(k[3]) << 24;
}
// the clear's second word (the render target's clear is four words: a pixel of 8 bytes twice, one of 4 or less four
// times)
inline uint32_t
rtClearHigh(uint32_t rt1, const double *k)
{
    return (rt1 & kRtWide) ? halfOf(k[2]) | halfOf(k[3]) << 16 : rtClear(rt1, k);
}
bool drawable(unsigned f) { uint32_t w; return rtFormat(f, w); }

// genxml's "ASTC 2D Dimension" for a block side
inline uint32_t astcDim(unsigned n) { return n == 4 ? 0 : n == 5 ? 1 : n == 6 ? 2 : n == 8 ? 4 : n == 10 ? 6 : 7; }
constexpr uint32_t kIdentitySwizzle = 0 | 1 << 3 | 2 << 6 | 3 << 9;
// A8Unorm's: zero (4) in red, green and blue, the R8 texel in alpha (as panfrost's A8_UNORM, 000R)
constexpr uint32_t kAlphaSwizzle = 4 | 4 << 3 | 4 << 6 | 0 << 9;

// a plane descriptor's first word: "Generic Plane" (clump RAW8..RAW128) or "ASTC 2D Plane" (block size; decoded wide,
// to RGBA16F, unless sRGB), linear
uint32_t
planeWord0(const hakios_metal_texture *t)
{
    if (t->bw > 1 || t->bh > 1) {
        bool srgb = (t->mali_format >> 20) & 1;
        return 10 | 4 << 4 | 2 << 8 | (srgb ? 0u : 1u << 25) | astcDim(t->bw) << 26 | astcDim(t->bh) << 29;
    }
    uint32_t clump = t->bytes == 1 ? 0 : t->bytes == 2 ? 1 : t->bytes == 4 ? 2 : t->bytes == 8 ? 3 : 4;
    return 10 | 1 << 4 | 2 << 8 | clump << 24;
}

// a texture's descriptor (genxml v10 "Texture", 32 bytes) and its plane's ("Generic Plane"), the plane at PLANE_VA
void
describe(const hakios_metal_texture *t, uint32_t *tex, uint32_t *plane, uint64_t plane_va)
{
    bool astc = t->bw > 1 || t->bh > 1;
    memset(tex, 0, 32);
    memset(plane, 0, 32);
    // a texture: 2D, 3D (MTLTextureType3D 7) or 1D (MTLTextureType1D 0, one row)
    // (a cube or cube array: dimension 0, its faces layers; a 1D array: 1D)
    const bool cube = t->type == 5 || t->type == 6;
    tex[0] = 2 | (t->type == 7 ? 3u : cube ? 0u : t->type == 0 || t->type == 1 ? 1u : 2u) << 4 | t->mali_format << 10;
    tex[1] = (t->width - 1) | (t->height - 1) << 16;
    tex[2] = t->swizzle ? t->swizzle : kIdentitySwizzle;                    // one level, from level 0
    if (astc) tex[2] |= 1u << 12;                                           // texel interleave (compressed, as Mesa)
    if ((t->levels > 1 || t->layers > 1) && t->planes_off != 0) {
        // every level: the plane descriptors the texture keeps (levels - 1, and the maximum LOD as 8.8 fixed point)
        tex[2] |= (t->levels - 1) << 16;
        tex[3] = ((t->levels - 1) * 256) << 16;
        plane_va = t->bo.va + t->planes_off;
    }
    if (t->layers > 1) {
        tex[6] = (cube ? t->layers / 6 : t->layers) - 1;                     // array size (cubes, for a cube array)
    }
    if (t->samples > 1) {
        // MULTISAMPLE: the sample count; each sample a plane (layered), the plane's slice stride apart (txf_ms)
        tex[3] |= (t->samples == 4 ? 2u : t->samples == 2 ? 1u : 0u) << 13;
    }
    tex[4] = (uint32_t)plane_va;
    tex[5] = (uint32_t)(plane_va >> 32);
    tex[7] = t->type == 7 ? t->depth - 1 : 0;
    plane[0] = planeWord0(t);                                               // a plane, generic (or ASTC), linear
    plane[1] = (uint32_t)(t->bo.size - t->offset);
    plane[2] = (uint32_t)t->gpu();
    plane[3] = (uint32_t)(t->gpu() >> 32);
    plane[4] = t->stride;
    plane[6] = t->stride * t->levelRows(0);                                 // (a slice: a 3D texture's depth)
    plane[7] = (t->width - 1) | (t->height - 1) << 16;
}
}  // namespace

namespace {
// one piece of work of a command buffer, in the order recorded
struct Job {
    unsigned shader;                // a blit: 0 copy_u32, 1 fill_u32, 2 copy_u8, 3 fill_u8; 4 a dispatch
    uint64_t dst, src;
    uint32_t value, count;
    uint32_t src_stride, dst_stride;    // COPY_ROWS: the rows' strides (value: words a row, count: words in all)
    hakios_metal_kernel *kernel;    // a dispatch's
    unsigned groups[3], threads[3], grid[3];
    uint64_t buffers[MALI_KERNEL_BUFFERS];
    uint32_t wls_size, wls_instances;   // a dispatch's threadgroup memory: bytes an instance (a power of two), instances
    hakios_metal_texture *textures[MALI_KERNEL_TEXTURES];  // bound at [[texture(n)]] (in the texture and image tables)
    uint32_t samplers[MALI_KERNEL_TEXTURES][8];
    unsigned texture_count, sampler_count;              // 0 0: no resource table
};
constexpr unsigned DISPATCH = 4, DRAW = 5, COPY_ROWS = 6;      // (a draw's Job: dst is its index in the command buffer's draws)

// One draw, everything resolved to what the GPU reads: one render pass of its own (begin, draw, end).
struct DrawJob {
    uint64_t pos_spd, vary_spd, frag_spd;
    unsigned vary_size;
    std::vector<uint32_t> vs_fau, fs_fau;               // the stages' push constants, whole
    uint64_t target_va;
    unsigned width, height, row_bytes;
    uint32_t rt1;                                       // the render target's word 1: write, format, block, swizzle
    uint32_t clear;                                     // (R8G8B8A8 in the tile buffer) and whether to write back
    uint32_t clear_high = 0;                            // (a wide target's second word: rtClearHigh)
    bool write;
    uint32_t blend[4];
    unsigned mode, count, instances, vertex_start;
    uint32_t dcd0, dcd1, dcd2, tiler_flags, scissor_min, scissor_max;
    hakios_metal_texture *textures[MALI_KERNEL_TEXTURES];
    uint32_t samplers[MALI_KERNEL_TEXTURES][8];
    unsigned texture_count, sampler_count;
    // (11-13) THE VERTEX STAGE's textures and samplers: a resource table of its own (register 0; the fragment's is 4)
    hakios_metal_texture *vs_textures[MALI_KERNEL_TEXTURES];
    uint32_t vs_samplers[MALI_KERNEL_TEXTURES][8];
    unsigned vs_texture_count, vs_sampler_count;
    uint64_t index_va = 0, index_size = 0;              // an indexed draw's indices (0: none)
    bool clear_only = false;                            // a pass with no geometry: the clear (hakios_metal_cb_clear)
    // DEPTH AND STENCIL (zs): the attachments' memory, formats, actions and clear values (the framebuffer descriptor's
    // word 12-13 and its ZS extension), and this draw's depth/stencil descriptor (zsd)
    unsigned samples = 1;                               // the attachments' (and the pipeline's): 1 or 4
    bool zs = false, has_z = false, has_s = false;
    uint64_t z_va = 0, s_va = 0;
    unsigned z_row = 0, s_row = 0;
    uint32_t z_format = 0, z_internal = 0;              // genxml "ZS Format" (D16 1, D32 14), "Z Internal Format"
    bool z_write = false, z_load = false, s_write = false, s_load = false;
    float z_clear = 0;
    uint32_t s_clear = 0;
    uint32_t zsd[8] = {};
    const char *name = "";                              // the program's (boot-arg metal_stats; held by the cb)
    const void *program = nullptr;                      // ... the program itself (boot-arg metal_glitchwatch)
    uint32_t tls_size = 0;                              // bytes a thread spills (the larger stage's): a local storage
                                                        // descriptor of its own at the area's end (drawBytes)
    // the Load action: the preload draw first, in the same Mali pass (compile_mali.c's mali_preload_compile)
    bool load = false;
    bool load0 = false;                                 // colour attachment 0 itself loads (else the preload leaves
                                                        // it to its clear: another target is what loads)
    uint64_t pre_pos_spd = 0, pre_frag_spd = 0;
    std::vector<uint32_t> pre_vs_fau, pre_fs_fau;
    hakios_metal_texture *pre_target = nullptr;
    // INCREMENTAL RENDERING (emitIr): the preload's fragment shader for every target as a pre-frame shader, its push
    // constants; 0: the pass has no tiler OOM handler (boot-arg metal_noir, or no preload program)
    uint64_t ir_frag_spd = 0;
    std::vector<uint32_t> ir_fs_fau;
    hakios_metal_texture *ir_target = nullptr;          // colour attachment 0 (held as the draw's target)
    hakios_metal_texture *target_tex = nullptr;         // ... always (boot-arg metal_glitchsay's scan)
    uint64_t zs_frag_spd = 0;                           // depth/stencil Load: the pre-frame shader that loads them
    std::vector<uint32_t> zs_fs_fau;
    // colour attachments 1 to 3 (rts > 1): each one's address, rows, RT word 1, clear, write, load, blend, texture
    unsigned rts = 1;
    struct More {
        uint64_t va;
        unsigned row_bytes;
        uint32_t rt1, clear, blend[4];
        uint32_t clear_high = 0;
        bool write, load;
        bool missing;                                   // written by the fragment function, absent from the pass
        bool untouched;                                 // this draw neither writes nor reads it (kept in, loaded and stored)
        bool memoryless;                                // a memoryless texture's: in the tile buffer only, never loaded/stored
        bool elide;                                     // no draw of its Mali pass touches it: not loaded, not written back
        hakios_metal_texture *texture;
    } more[3] = {};
};
struct IndexInfo { hakios_metal_buffer *buffer; uint64_t offset; unsigned type; int base_vertex; };

// the tiler's hierarchy mask for a W x H target of 16x16 tiles (Mesa's pan_select_tiler_hierarchy_mask, no budget)
uint32_t
hierarchyMask(unsigned w, unsigned h, unsigned levels)
{
    static const uint32_t spread[8] = { 0, 0x80, 0x82, 0xa2, 0xaa, 0xea, 0xee, 0xfe };
    uint32_t mask = levels >= 8 ? (1u << (levels > 13 ? 13 : levels)) - 1 : spread[levels];
    unsigned most = w > h ? w : h, last = 0;
    for (unsigned b = (most + 15) / 16; b != 0; b >>= 1) {
        last++;
    }
    unsigned max_levels = levels < 8 ? 8 : levels;
    if (last > max_levels) {
        mask <<= last - max_levels;
    }
    return mask & 0x1fff;
}

// One draw's descriptors at AREA (drawBytes(DJ) of them, 64-byte aligned) and its instructions into S from N on: the
// tiler context (fresh: an empty polygon list), the framebuffer descriptor and render target, the blend descriptor, the
// stages' push constants, the fragment stage's resource table; then RUN_IDVS, FINISH_TILING, RUN_FRAGMENT, each after a
// WAIT for the one before. Registers in v11's layout (upstream Mesa's genxml v11.xml: v10's plus TILER_FLAGS2 at r61);
// what each value is, hakios-kernel's the port's log ("render pipelines") and userland/src/mali_test.c.
uint64_t drawBytes(const DrawJob &dj);

// INCREMENTAL RENDERING (panvk's tiler OOM exception handler, csf/panvk_vX_exception_handler.c): a pass whose polygon
// lists outgrow the tiler heap's most chunks would wait for good. Each Mali pass has IR_BYTES of its own (ir): a
// handler the firmware runs on the queue when the kernel has no chunk left to give -- the lists so far drawn by a
// fragment job and stored, their chunks given back, the tiler context's lists emptied -- and three copies of the
// pass's framebuffer descriptor: the first flush (as the pass begins, every target stored), the later flushes (every
// target loaded by a pre-frame shader -- the preload's fragment shader -- and stored), and the pass's own last fragment
// job after a flush (loaded, then stored as the pass says). Word 0: the descriptor the next flush uses; word 8: the one
// the pass ends with (the stream loads r40 from it); 0x10: set when the handler is installed; 0x40: r40-r47 kept.
// (0xc00: the pre-frame DCDs of the later parts -- pre-frame 0, 1, post-frame, 128 bytes each; 0xd80: the pass's own,
// loading depth/stencil when it asks to, with their resource table at 0xf00, depth/stencil descriptors at 0x1040
// (the pass's) and 0x1060 (the later parts'), push constants at 0x1080)
constexpr uint64_t IR_BYTES = 0x1100, IR_HANDLER = 0x80, IR_HANDLER_BYTES = 0x180, IR_SRT = 0x200, IR_FBD = 0x600,
    IR_FBD_STRIDE = 0x200, IR_DCD = 0xc00, IR_ZS_DCD = 0xd80, IR_ZS_SRT = 0xf00, IR_ZSD = 0x1040, IR_IR_ZSD = 0x1060,
    IR_ZS_FAU = 0x1080;
inline uint64_t wait_on(unsigned mask) { return 3ULL << 56 | (uint64_t)mask << 16; }

// a texture descriptor and its plane for raw memory: 2D, one level, linear (the depth and stencil planes as the
// pre-frame shader reads them: R32F or R16 UNORM for depth, R8UI (131; 163 is R8 UNORM) for stencil)
void
describeRaw(uint32_t mali_format, unsigned bytes, uint64_t va, uint32_t stride, unsigned w, unsigned h, uint32_t *tex,
    uint32_t *plane, uint64_t plane_va, unsigned samples = 1)
{
    memset(tex, 0, 32);
    memset(plane, 0, 32);
    tex[0] = 2 | 2u << 4 | mali_format << 10;
    tex[1] = (w - 1) | (h - 1) << 16;
    tex[2] = kIdentitySwizzle;
    tex[3] = (samples == 4 ? 2u : 0u) << 13;                                    // (the planes a sample apart)
    tex[4] = (uint32_t)plane_va;
    tex[5] = (uint32_t)(plane_va >> 32);
    plane[0] = 10 | 1 << 4 | 2 << 8 | (bytes == 1 ? 0u : bytes == 2 ? 1u : 2u) << 24;
    plane[1] = stride * h;
    plane[2] = (uint32_t)va;
    plane[3] = (uint32_t)(va >> 32);
    plane[4] = stride;
    plane[6] = stride * h;
    plane[7] = (w - 1) | (h - 1) << 16;
}

// the depth and/or stencil (ZS bits 1, 2) of DJ's pass as textures FIRST.. of a resource table's texture table at
// TEX/TEX_VA (32 bytes each), their planes at PLANES/PLANES_VA
void
describeZs(const DrawJob &dj, unsigned zs, uint8_t *tex, uint8_t *planes, uint64_t planes_va)
{
    unsigned i = 0;
    if (zs & 1) {
        describeRaw(dj.z_format == 1 ? 164u << 12 | 16 : 167u << 12 | 16, dj.z_format == 1 ? 2 : 4, dj.z_va, dj.z_row,
            dj.width, dj.height, (uint32_t *)(tex + 32 * i), (uint32_t *)(planes + 32 * i), planes_va + 32 * i, dj.samples);
        i++;
    }
    if (zs & 2) {
        describeRaw(131u << 12 | 16, 1, dj.s_va, dj.s_row, dj.width, dj.height, (uint32_t *)(tex + 32 * i),
            (uint32_t *)(planes + 32 * i), planes_va + 32 * i, dj.samples);
    }
}

// a depth/stencil descriptor that writes what the pre-frame shader outputs (panvk's for its Z/S preload): depth
// always, written, from the shader; stencil always, replaced by the shader's
void
zsdFromShader(uint32_t *w, unsigned zs)
{
    memset(w, 0, 32);
    const uint32_t w4 = 1u << 21 | 1u << 22 | 1u << 23;
    w[0] = 7 | 7u << 4 | 7u << 16;
    w[4] = w4 | 2u << 25 | 7u << 29;
    if (zs & 1) {
        w[4] = w4 | 3u << 25 | 1u << 27 | 7u << 29;
    }
    if (zs & 2) {
        w[0] = 7 | 7u << 4 | 1u << 7 | 1u << 10 | 1u << 13 | 7u << 16 | 1u << 19 | 1u << 22 | 1u << 25 | 1u << 30 | 1u << 31;
        w[1] = 0xffffffffu;
    }
}

// the pass's own pre-frame DCD that loads its depth/stencil (dj.zs_frag_spd), at IR + IR_ZS_DCD
void
emitZsLoad(hakios_metal_device *d, const DrawJob &dj, uint8_t *ir, uint64_t ir_va)
{
    const unsigned zs = (dj.has_z && dj.z_load ? 1u : 0u) | (dj.has_s && dj.s_load ? 2u : 0u);
    const unsigned n = (zs & 1) + (zs >> 1);
    uint8_t *t = ir + IR_ZS_SRT;
    uint64_t t_va = ir_va + IR_ZS_SRT, tex_va = t_va + 0x80, samp_va = tex_va + 32 * n, plane_va = samp_va + 32;
    memset(t, 0, IR_ZSD - IR_ZS_SRT);
    auto entry = [&](unsigned table, uint64_t va, unsigned count) {
        auto *e = (uint32_t *)(t + 16 * table);
        e[0] = (uint32_t)va;
        e[1] = (uint32_t)(va >> 32) | 1u << 24;
        e[2] = count * 32;
    };
    entry(MALI_TABLE_TEXTURE, tex_va, n);
    entry(MALI_TABLE_SAMPLER, samp_va, 1);
    describeZs(dj, zs, ir + (tex_va - ir_va), ir + (plane_va - ir_va), plane_va);
    ((uint32_t *)(ir + (samp_va - ir_va)))[0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    zsdFromShader((uint32_t *)(ir + IR_ZSD), zs);
    memcpy(ir + IR_ZS_FAU, dj.zs_fs_fau.data(), dj.zs_fs_fau.size() * 4);
    auto *w = (uint32_t *)(ir + IR_ZS_DCD);
    memset(w, 0, 0x180);
    w[0] = 1u << 1 | 3u << 2 | 3u << 4 | 1u << 9;                             // killed, Z/S late, clean tiles written
    if (dj.samples > 1) w[0] |= 1u << 11 | 1u << 19;                           // (multisample: run a sample)
    w[1] = 0xffff;                                                              // no colour target
    w[5] = 1u << 24 | 1u << 25;
    w[7] = 0x3f800000;
    *(uint64_t *)&w[10] = ir_va + IR_ZSD;
    w[17] = (uint32_t)((dj.zs_fs_fau.size() + 1) / 2);
    *(uint64_t *)&w[24] = t_va | MALI_TABLES;
    *(uint64_t *)&w[26] = dj.zs_frag_spd;
    *(uint64_t *)&w[28] = d->descriptors.va + 0x1200;
    *(uint64_t *)&w[30] = ir_va + IR_ZS_FAU;
}

void
emitIr(hakios_metal_device *d, const DrawJob &dj, uint8_t *pass_area, uint64_t pass_va, uint8_t *ir, uint64_t ir_va,
    uint64_t rt_at)
{
    const unsigned nt = dj.rts;
    const uint64_t fbd_bytes = rt_at + 0x40 * nt - 0x80, tags = 1 | (dj.zs ? 2u : 0u) | (nt - 1) << 2;
    auto missing = [&](unsigned i) { return i > 0 && dj.more[i - 1].missing; };
    // the pre-frame shader's resources: each target as a texture (the preload's table), a sampler, a blend descriptor
    // a target: a replace, nothing written to an output with no attachment
    // (with depth/stencil: the depth attachment the next texture, the stencil plane the one after)
    const unsigned zs = (dj.has_z ? 1u : 0u) | (dj.has_s ? 2u : 0u), nzs = (zs & 1) + (zs >> 1), ntex = nt + nzs;
    uint8_t *t = ir + IR_SRT;
    uint64_t t_va = ir_va + IR_SRT, tex_va = t_va + 0x80, samp_va = tex_va + 32 * ntex, plane_va = samp_va + 32;
    uint64_t bl_at = (IR_SRT + 0x80 + (2 * ntex + 1) * 32 + 0x3f) & ~0x3fULL, fau_at = bl_at + 0x40;
    memset(t, 0, IR_FBD - IR_SRT);
    auto entry = [&](unsigned table, uint64_t va, unsigned count) {
        auto *e = (uint32_t *)(t + 16 * table);
        e[0] = (uint32_t)va;
        e[1] = (uint32_t)(va >> 32) | 1u << 24;
        e[2] = count * 32;
    };
    entry(MALI_TABLE_TEXTURE, tex_va, ntex);
    entry(MALI_TABLE_SAMPLER, samp_va, 1);
    describeZs(dj, zs, ir + (tex_va - ir_va) + 32 * nt, ir + (plane_va - ir_va) + 32 * nt, plane_va + 32 * nt);
    unsigned loaded = 0;
    for (unsigned i = 0; i < nt; i++) {
        uint32_t plane[8];
        describe(i == 0 ? dj.ir_target : dj.more[i - 1].texture, (uint32_t *)(ir + (tex_va - ir_va) + 32 * i), plane,
                 plane_va + 32 * i);
        memcpy(ir + (plane_va - ir_va) + 32 * i, plane, 32);
        bool loads = !missing(i);
        auto *b = (uint32_t *)(ir + bl_at) + 4 * i;
        b[0] = 1u << 9 | (loads ? 0u : 1u);
        b[1] = 0x132 | 0x132 << 12 | (loads ? 0xfu : 0u) << 28;
        b[2] = (loads ? 1u : 2u) | 3 << 3 | i << 16;
        b[3] = dj.blend[3];
        loaded |= loads ? 1u << i : 0;
    }
    ((uint32_t *)(ir + (samp_va - ir_va)))[0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    memcpy(ir + fau_at, dj.ir_fs_fau.data(), dj.ir_fs_fau.size() * 4);
    // the pre-frame draw descriptor (panvk's cmd_emit_dcd for a colour preload): Z/S forced early, may kill and be
    // killed, clean tiles written; every target; no depth/stencil test (the descriptor that lets all through)
    auto *w = (uint32_t *)(ir + IR_DCD);
    memset(w, 0, 0x180);
    w[0] = 1u << 0 | 1u << 1 | 1u << 9;
    if (zs != 0) {
        // (depth/stencil written by the shader: Z/S late, and it may not kill -- panvk's frame shaders)
        w[0] = 1u << 1 | 3u << 2 | 3u << 4 | 1u << 9;
        zsdFromShader((uint32_t *)(ir + IR_IR_ZSD), zs);
    }
    if (dj.samples > 1) w[0] |= 1u << 11 | 1u << 19;                           // (multisample: run a sample)
    w[1] = 0xffff | ((1u << nt) - 1) << 16;
    w[5] = dj.dcd2;
    w[7] = 0x3f800000;
    *(uint64_t *)&w[10] = zs != 0 ? ir_va + IR_IR_ZSD : d->descriptors.va + 0x1980;
    *(uint64_t *)&w[12] = (ir_va + bl_at) | nt;
    w[17] = (uint32_t)((dj.ir_fs_fau.size() + 1) / 2);
    *(uint64_t *)&w[24] = t_va | MALI_TABLES;
    *(uint64_t *)&w[26] = dj.ir_frag_spd;
    *(uint64_t *)&w[28] = d->descriptors.va + 0x1200;
    *(uint64_t *)&w[30] = ir_va + fau_at;
    // the three framebuffer descriptors, copies of the pass's as it now stands (its targets' write-back is the last
    // draw's): 0 the first flush, 1 the later ones, 2 the pass's end after a flush
    uint64_t fbd_va[3];
    for (unsigned k = 0; k < 3; k++) {
        uint8_t *v = ir + IR_FBD + k * IR_FBD_STRIDE;
        auto *f = (uint32_t *)v;
        fbd_va[k] = ir_va + IR_FBD + k * IR_FBD_STRIDE;
        memcpy(v, pass_area + 0x80, fbd_bytes);
        if (k != 2) {
            for (unsigned i = 0; i < nt; i++) {
                if (!missing(i)) ((uint32_t *)(v + rt_at - 0x80))[16 * i + 1] |= 1;
            }
            f[12] |= (dj.has_z ? 1u << 18 : 0) | (dj.has_s ? 1u << 8 : 0);
        }
        if (k != 0) {
            f[0] = (f[0] & ~7u) | 1;                                            // pre-frame 0: always (targets and
            *(uint64_t *)&f[6] = ir_va + IR_DCD;                                // depth/stencil loaded)
        }
    }
    *(uint64_t *)(ir + 0) = fbd_va[0] | tags;
    *(uint64_t *)(ir + 8) = (pass_va + 0x80) | tags;
    // the handler: registers r62-r79 are free between draws; r40-r47 (the fragment job's, also the next draw's) kept
    // and given back. Its own waits name their scoreboards (0: loads/stores and flushes; its fragment job on 4): the
    // IDVS job that ran out (slot 3) is still in flight
    auto *h = (uint64_t *)(ir + IR_HANDLER);
    unsigned k = 0;
    memset(h, 0, IR_HANDLER_BYTES);                                             // (NOPs after it)
    h[k++] = move48(76, ir_va);
    h[k++] = 21ULL << 56 | 40ULL << 48 | 76ULL << 40 | 0xffULL << 16 | 0x40;   // STORE_MULTIPLE r40-r47 -> [r76 + 0x40]
    h[k++] = 20ULL << 56 | 40ULL << 48 | 76ULL << 40 | 0x3ULL << 16 | 0;       // LOAD_MULTIPLE r40-r41 <- [r76]: this flush's
    h[k++] = move32(42, 0);
    h[k++] = move32(43, (dj.width - 1) | (dj.height - 1) << 16);
    h[k++] = move48(44, 0);
    h[k++] = move32(46, 0);
    h[k++] = wait_on(1);
    h[k++] = set_state(0, 4);                                                   // endpoint work on scoreboard 4
    h[k++] = 7ULL << 56;                                                        // RUN_FRAGMENT
    h[k++] = wait_on(1 << 4);
    h[k++] = set_state(0, 3);
    h[k++] = move48(78, pass_va);
    h[k++] = 20ULL << 56 | 62ULL << 48 | 78ULL << 40 | 0xfULL << 16 | 40;      // LOAD_MULTIPLE r62-r65 <- the completed chunks
    h[k++] = wait_on(1);
    h[k++] = 11ULL << 56 | 62ULL << 40 | 64ULL << 32;                          // FINISH_FRAGMENT, the pass not counted done
    h[k++] = wait_on(1);
    h[k++] = move48(66, 0);
    h[k++] = 21ULL << 56 | 66ULL << 48 | 78ULL << 40 | 0x3ULL << 16 | 0;       // the polygon list,
    h[k++] = 21ULL << 56 | 66ULL << 48 | 78ULL << 40 | 0x3ULL << 16 | 40;      // completed top and bottom: 0
    h[k++] = 21ULL << 56 | 66ULL << 48 | 78ULL << 40 | 0x3ULL << 16 | 48;
    h[k++] = move48(62, fbd_va[1] | tags);
    h[k++] = move48(64, fbd_va[2] | tags);
    h[k++] = 21ULL << 56 | 62ULL << 48 | 76ULL << 40 | 0xfULL << 16 | 0;       // the next flush's, the pass's end
    h[k++] = move32(68, 0);
    h[k++] = 36ULL << 56 | 68ULL << 40 | 2 << 8 | 1 << 4 | 1;                  // FLUSH_CACHE2: what was stored, seen by
    h[k++] = wait_on(1);                                                        // the pre-frame shader's texture reads
    h[k++] = 20ULL << 56 | 40ULL << 48 | 76ULL << 40 | 0xffULL << 16 | 0x40;   // r40-r47 back
    h[k++] = wait_on(1);
}

void
emitDraw(hakios_metal_device *d, const DrawJob &dj, uint8_t *area, uint64_t area_va, uint64_t *s, unsigned &n,
    bool first = true, bool last = true, uint8_t *pass_area = nullptr, uint64_t pass_va = 0, uint8_t *ir = nullptr,
    uint64_t ir_va = 0)
{
    // ONE MALI PASS FOR A RUN OF DRAWS (the stream loop: the draws of one Metal pass, one after the other into the same
    // attachments): the first opens it -- the tiler context, the framebuffer and its targets, the preload -- each draw
    // then only its own registers and RUN_IDVS into that tiler context, and the last closes it: tiling finished, one
    // fragment job, the targets written back as the last draw stores them. A pass a draw read and wrote the whole
    // target (the preload) for every draw before.
    if (first) {
        pass_area = area;
        pass_va = area_va;
    }
    // the tiler context, the framebuffer descriptor, a render target descriptor for each target (64 bytes each, after
    // the framebuffer's), their blend descriptors (16 bytes each, in 64), then the stages' push constants
    // (with depth or stencil the framebuffer's ZS extension comes first, 64 bytes, and the draw's depth/stencil
    // descriptor after the blend descriptors)
    const uint64_t rt_at = 0x100 + (dj.zs ? 0x40 : 0);
    uint64_t bl_at = rt_at + 0x40 * dj.rts;
    auto *tc = (uint32_t *)area, *fb = (uint32_t *)(area + 0x80), *rt = (uint32_t *)(area + rt_at);
    auto *bl = (uint32_t *)(area + bl_at);
    uint64_t zsd_at = bl_at + 0x40, vs_at = zsd_at + (dj.zs ? 0x40 : 0), fs_at = vs_at + ((dj.vs_fau.size() * 4 + 0x3f) & ~0x3fULL);
    uint64_t srt_at = fs_at + ((dj.fs_fau.size() * 4 + 0x3f) & ~0x3fULL), srt = 0;

    memset(area, 0, vs_at);
    // FIRST PROVOKING VERTEX (11-11): a flat varying is the first vertex's in Metal (and Vulkan), the last's in this
    // GPU's default -- CoreAnimation's quads showed a seam along their diagonals (Control Center's tiles, TODO 8h).
    // The tiler context's bit 18 and the framebuffer's word 11 bit 14 (Mesa's first_provoking_vertex); metal_lastpv:
    // the old way
    static const bool last_pv = bootArg("metal_lastpv");
    tc[2] = hierarchyMask(dj.width, dj.height, d->tiler_levels) | 1u << 17 | (last_pv ? 0u : 1u << 18);  // (sample test off)
    tc[3] = (dj.width - 1) | (dj.height - 1) << 16;
    *(uint64_t *)&tc[6] = d->heapdesc.va;
    *(uint64_t *)&tc[8] = (d->heapdesc.va + 0x1000) | (0x10000 >> 12);
    // sample positions: one, centred; four: the rotated 4x grid (descriptors + 0x2000)
    *(uint64_t *)&fb[4] = d->descriptors.va + (dj.samples > 1 ? 0x2000 : 0x1800);
    fb[8] = (dj.width - 1) | (dj.height - 1) << 16;
    fb[10] = (dj.width - 1) | (dj.height - 1) << 16;
    // 16x16 tiles, targets, 1 KiB a target a tile; four samples: 4 (log2 2), the rotated 4x grid (2), 8x8 tiles -- the
    // same kilobyte a target a tile (64 pixels of four samples)
    // (the colour buffer's allocation: each target's kilobytes of the tile -- one, or two for 8 bytes a pixel)
    uint32_t tile_kb = rtKilobytes(dj.rt1);
    for (unsigned i = 1; i < dj.rts; i++) tile_kb += rtKilobytes(dj.more[i - 1].rt1);
    fb[11] = (dj.samples > 1 ? 2u | 2u << 3 : 0u) | 2 << 6 | (dj.samples > 1 ? 6u : 8u) << 9 | (last_pv ? 0u : 1u << 14) |
        1u << 15 | (dj.rts - 1) << 19 | tile_kb << 24;
    fb[12] = 1u << 29;                                                          // empty tiles written (the clear)
    *(uint64_t *)&fb[14] = area_va;
    if (dj.zs) {
        auto *ext = (uint32_t *)(area + 0x100);
        fb[12] |= 1u << 21;                                                     // the ZS extension
        ext[0] = 1u << 31;                                                      // clean tiles written
        if (dj.has_z) {
            fb[12] |= dj.z_internal << 16 | (dj.z_write ? 1u << 18 : 0);
            memcpy(&fb[13], &dj.z_clear, 4);
            ext[0] |= dj.z_format | 2u << 4 | (dj.samples > 1 ? 3u << 8 : 0u);  // linear (layered samples)
            *(uint64_t *)&ext[8] = dj.z_va;
            ext[10] = dj.z_row;
            ext[11] = dj.z_row * dj.height;
        }
        if (dj.has_s) {
            fb[12] |= (dj.s_clear & 0xff) | (dj.s_write ? 1u << 8 : 0);
            ext[0] |= 1u << 16 | 2u << 20 | (dj.samples > 1 ? 3u << 24 : 0u);   // S8, linear (layered samples)
            *(uint64_t *)&ext[12] = dj.s_va;
            ext[14] = dj.s_row;
            ext[15] = dj.s_row * dj.height;
        }
        memcpy(area + zsd_at, dj.zsd, 32);
        if (first && dj.zs_frag_spd != 0 && ir != nullptr) {
            // DEPTH/STENCIL LOADED by the pre-frame shader (the descriptor's Z/S preload bits, 19 and 9, load nothing on
            // the Mali: what a pass after another found was its clear values)
            emitZsLoad(d, dj, ir, ir_va);
            fb[0] = (fb[0] & ~7u) | 1;                                          // pre-frame 0: always
            *(uint64_t *)&fb[6] = ir_va + IR_ZS_DCD;
        }
    }
    rt[0] = rtInternal(dj.rt1) << 26;                                          // internal format (R8G8B8A8, or a float one)
    rt[1] = (dj.rt1 & ~kRtOurs) | (dj.write ? 1u : 0u) | 1u << 31 | (dj.samples > 1 ? 3u << 12 : 0u);   // clean tiles written too
    // (multisample: written back layered, a plane a sample, rt[11] apart)
    *(uint64_t *)&rt[8] = dj.target_va;
    rt[10] = dj.row_bytes;
    rt[11] = dj.row_bytes * dj.height;
    rt[12] = rt[14] = dj.clear;
    rt[13] = rt[15] = (dj.rt1 & kRtWide) ? dj.clear_high : dj.clear;
    memcpy(bl, dj.blend, 16);
    // (11-13) the blend descriptor's sRGB bit (10) for an sRGB target: the fixed-function blender decodes what it reads
    // of the tile buffer and encodes what it writes -- without it a blended or masked draw wrote linear values there
    // (opaque mode converts by the memory format alone: rtBlend)
    if (dj.rt1 & 1u << 14) bl[0] |= 1u << 10;
    uint32_t tile_at = rtKilobytes(dj.rt1) * 1024;      // where the next target's part of the tile buffer starts
    for (unsigned i = 1; i < dj.rts; i++) {
        // target i: its kilobyte of the tile buffer (internal buffer offset, 16-byte units), its own clear and memory
        const DrawJob::More &m = dj.more[i - 1];
        auto *r = rt + 16 * i;
        r[0] = rtInternal(m.rt1) << 26 | (tile_at / 16) << 4;
        tile_at += rtKilobytes(m.rt1) * 1024;
        r[1] = (m.rt1 & ~kRtOurs) | (m.write && !m.elide ? 1u : 0u) | 1u << 31 | (dj.samples > 1 ? 3u << 12 : 0u);
        *(uint64_t *)&r[8] = m.va;
        r[10] = m.row_bytes;
        r[11] = m.row_bytes * dj.height;
        r[12] = r[14] = m.clear;
        r[13] = r[15] = (m.rt1 & kRtWide) ? m.clear_high : m.clear;
        memcpy(bl + 4 * i, m.blend, 16);
        if (m.rt1 & 1u << 14) bl[4 * i] |= 1u << 10;
    }
    memcpy(area + vs_at, dj.vs_fau.data(), dj.vs_fau.size() * 4);
    memcpy(area + fs_at, dj.fs_fau.data(), dj.fs_fau.size() * 4);
    if (dj.texture_count || dj.sampler_count) {
        // the fragment stage's resource table, as a dispatch's (texture table 4, sampler 3, image 5)
        uint8_t *t = area + srt_at;
        uint64_t t_va = area_va + srt_at, tex_va = t_va + 0x80, samp_va = tex_va + dj.texture_count * 32;
        uint64_t img_va = samp_va + dj.sampler_count * 32, plane_va = img_va + dj.texture_count * 32;
        memset(t, 0, 0x80);
        auto entry = [&](unsigned table, uint64_t va, unsigned count) {
            auto *e = (uint32_t *)(t + 16 * table);
            e[0] = (uint32_t)va;
            e[1] = (uint32_t)(va >> 32) | 1u << 24;
            e[2] = count * 32;
        };
        entry(MALI_TABLE_TEXTURE, tex_va, dj.texture_count);
        entry(MALI_TABLE_SAMPLER, samp_va, dj.sampler_count);
        entry(MALI_TABLE_IMAGE, img_va, dj.texture_count);
        for (unsigned i = 0; i < dj.texture_count; i++) {
            auto *tex = (uint32_t *)(area + (tex_va - area_va) + 32 * i), *img = (uint32_t *)(area + (img_va - area_va) + 32 * i);
            uint32_t plane[8];
            if (dj.textures[i] == nullptr) {
                memset(tex, 0, 32);
                memset(img, 0, 32);
                continue;
            }
            describe(dj.textures[i], tex, plane, plane_va);
            memcpy(area + (plane_va - area_va), plane, 32);
            memcpy(img, tex, 32);
            plane_va += 32;
        }
        for (unsigned i = 0; i < dj.sampler_count; i++) {
            memcpy(area + (samp_va - area_va) + 32 * i, dj.samplers[i], 32);
        }
        srt = t_va | MALI_TABLES;
    }
    // the vertex stage's resource table, after the fragment's (64-byte aligned): the same three tables
    const uint64_t fs_srt_bytes = dj.texture_count || dj.sampler_count ? 0x80 + (dj.texture_count * 3 + dj.sampler_count) * 32 : 0;
    const uint64_t vsrt_at = (srt_at + fs_srt_bytes + 0x3f) & ~0x3fULL;
    const uint64_t vs_srt_bytes = dj.vs_texture_count || dj.vs_sampler_count ? 0x80 + (dj.vs_texture_count * 3 + dj.vs_sampler_count) * 32 : 0;
    uint64_t vsrt = 0;
    if (vs_srt_bytes != 0) {
        uint8_t *t = area + vsrt_at;
        uint64_t t_va = area_va + vsrt_at, tex_va = t_va + 0x80, samp_va = tex_va + dj.vs_texture_count * 32;
        uint64_t img_va = samp_va + dj.vs_sampler_count * 32, plane_va = img_va + dj.vs_texture_count * 32;
        memset(t, 0, 0x80);
        auto entry = [&](unsigned table, uint64_t va, unsigned count) {
            auto *e = (uint32_t *)(t + 16 * table);
            e[0] = (uint32_t)va;
            e[1] = (uint32_t)(va >> 32) | 1u << 24;
            e[2] = count * 32;
        };
        entry(MALI_TABLE_TEXTURE, tex_va, dj.vs_texture_count);
        entry(MALI_TABLE_SAMPLER, samp_va, dj.vs_sampler_count);
        entry(MALI_TABLE_IMAGE, img_va, dj.vs_texture_count);
        for (unsigned i = 0; i < dj.vs_texture_count; i++) {
            auto *tex = (uint32_t *)(area + (tex_va - area_va) + 32 * i), *img = (uint32_t *)(area + (img_va - area_va) + 32 * i);
            uint32_t plane[8];
            if (dj.vs_textures[i] == nullptr) {
                memset(tex, 0, 32);
                memset(img, 0, 32);
                continue;
            }
            describe(dj.vs_textures[i], tex, plane, plane_va);
            memcpy(area + (plane_va - area_va), plane, 32);
            memcpy(img, tex, 32);
            plane_va += 32;
        }
        for (unsigned i = 0; i < dj.vs_sampler_count; i++) {
            memcpy(area + (samp_va - area_va) + 32 * i, dj.vs_samplers[i], 32);
        }
        vsrt = t_va | MALI_TABLES;
    }
    uint64_t vs_fau_va = area_va + vs_at, fs_fau_va = area_va + fs_at;
    // A PASS OPENED (11-10): the texture caches dropped first (and the load/store caches cleaned) -- what an earlier pass
    // of this command buffer wrote back is in L2, but a texture cache may still hold that memory as it was before:
    // CoreAnimation's blur samples the passes before it (the tester's slow-motion clip: frames partly unblurred and
    // streaked mid-swipe). Passes were only ordered (WAIT_ALL). This Mali (v11) does not invalidate them at RUN_FRAGMENT
    // (v13+ does, implicitly; Arm's own driver forces it before every fragment job -- Mesa's panvk, IMPLICIT_OTHERS_INV).
    // A pass that loads its targets has the same flush before its preload, below. metal_passinv=0: not done (a test).
    static const bool pass_inv = [] {
        char args[1024] = {0};
        size_t size = sizeof(args) - 1;
        return !(sysctlbyname("kern.bootargs", args, &size, nullptr, 0) == 0 && strstr(args, "metal_passinv=0") != nullptr);
    }();
    // (with every target that asked for a load elided -- passElide -- there is nothing to preload)
    bool preload = dj.load;
    if (preload && !dj.load0) {
        preload = false;
        for (unsigned i = 1; i < dj.rts; i++) {
            preload = preload || (dj.more[i - 1].load && !dj.more[i - 1].missing && !dj.more[i - 1].elide);
        }
    }
    if (first && pass_inv && !preload && !dj.clear_only) {
        s[n++] = move32(74, 0);
        s[n++] = 36ULL << 56 | 74ULL << 40 | 2 << 8 | 1 << 4;                  // FLUSH_CACHE2: LSC clean, others invalidated
        s[n++] = WAIT_ALL;
    }
    if (first && ir != nullptr) {
        *(uint64_t *)(ir + 0x10) = 0;
        if (!dj.clear_only && dj.ir_frag_spd != 0) {
            *(uint64_t *)(ir + 0x10) = 1;                                       // (emitIr when the pass ends)
            s[n++] = move48(76, ir_va + IR_HANDLER);
            s[n++] = move32(78, IR_HANDLER_BYTES);
            s[n++] = 25ULL << 56 | 76ULL << 40 | 78ULL << 32 | 2;              // SET_EXCEPTION_HANDLER: tiler OOM
        }
    }
    if (preload && first) {
        // THE LOAD ACTION: the target's own pixels drawn first, one triangle over it whose fragment is the target's
        // texel (texture 0, txf), into the same tiler context as the draw -- the tile buffer preloaded through the
        // ordinary draw path. Caches first, as at a command buffer's start: what an earlier pass wrote back reaches
        // L2, and the texture cache drops what an earlier preload read of the same target.
        uint64_t end_at = vs_srt_bytes != 0 ? vsrt_at + vs_srt_bytes : srt_at + fs_srt_bytes;
        uint64_t pvs_at = (end_at + 0x3f) & ~0x3fULL, pfs_at = pvs_at + ((dj.pre_vs_fau.size() * 4 + 0x3f) & ~0x3fULL);
        uint64_t psrt_at = pfs_at + ((dj.pre_fs_fau.size() * 4 + 0x3f) & ~0x3fULL);
        const unsigned nt = dj.rts;
        memcpy(area + pvs_at, dj.pre_vs_fau.data(), dj.pre_vs_fau.size() * 4);
        memcpy(area + pfs_at, dj.pre_fs_fau.data(), dj.pre_fs_fau.size() * 4);
        uint8_t *t = area + psrt_at;
        uint64_t t_va = area_va + psrt_at, tex_va = t_va + 0x80, samp_va = tex_va + 32 * nt, plane_va = samp_va + 32;
        memset(t, 0, 0x80 + (2 * nt + 1) * 32 + 0x40 + 16 * nt);
        // the preload's own blend descriptors: a replace for a target that loads, nothing written to one that does not
        // (the draw's may blend, the preload must not)
        uint64_t pbl_at = (psrt_at + 0x80 + (2 * nt + 1) * 32 + 0x3f) & ~0x3fULL;
        auto *pbl = (uint32_t *)(area + pbl_at);
        for (unsigned i = 0; i < nt; i++) {
            bool loads = i == 0 ? dj.load0 : (dj.more[i - 1].load && !dj.more[i - 1].missing && !dj.more[i - 1].elide);
            auto *b = pbl + 4 * i;
            b[0] = 1u << 9 | (loads ? 0u : 1u);
            b[1] = 0x132 | 0x132 << 12 | (loads ? 0xfu : 0u) << 28;
            b[2] = (loads ? 1u : 2u) | 3 << 3 | i << 16;
            b[3] = dj.blend[3];
        }
        auto entry = [&](unsigned table, uint64_t va, unsigned count) {
            auto *e = (uint32_t *)(t + 16 * table);
            e[0] = (uint32_t)va;
            e[1] = (uint32_t)(va >> 32) | 1u << 24;
            e[2] = count * 32;
        };
        entry(MALI_TABLE_TEXTURE, tex_va, nt);
        entry(MALI_TABLE_SAMPLER, samp_va, 1);
        for (unsigned i = 0; i < nt; i++) {
            uint32_t plane[8];
            describe(i == 0 ? dj.pre_target : dj.more[i - 1].texture, (uint32_t *)(area + (tex_va - area_va) + 32 * i), plane,
                     plane_va + 32 * i);
            memcpy(area + (plane_va - area_va) + 32 * i, plane, 32);
        }
        ((uint32_t *)(area + (samp_va - area_va)))[0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
        uint64_t pvs_va = area_va + pvs_at, pfs_va = area_va + pfs_at;
        s[n++] = move32(74, 0);
        s[n++] = 36ULL << 56 | 74ULL << 40 | 2 << 8 | 1 << 4 | 1;              // FLUSH_CACHE2: L2/LSC clean, others invalidated
        s[n++] = WAIT_ALL;
        s[n++] = move48(0, 0);
        s[n++] = move48(4, t_va | MALI_TABLES);
        s[n++] = move48(8, pvs_va);
        s[n++] = move32(9, (uint32_t)(pvs_va >> 32) | (uint32_t)((dj.pre_vs_fau.size() + 1) / 2) << 24);
        s[n++] = move48(12, pfs_va);
        s[n++] = move32(13, (uint32_t)(pfs_va >> 32) | (uint32_t)((dj.pre_fs_fau.size() + 1) / 2) << 24);
        s[n++] = move48(16, dj.pre_pos_spd);
        s[n++] = move48(18, 0);                                                 // no varying shader
        s[n++] = move48(20, dj.pre_frag_spd);
        s[n++] = move48(24, d->descriptors.va + 0x1200);
        s[n++] = move32(32, 0);
        s[n++] = move32(33, 3);                                                 // one triangle
        s[n++] = move32(34, 1);
        s[n++] = move32(35, 0);
        s[n++] = move32(36, 0);
        s[n++] = move32(37, 0);
        s[n++] = move32(38, dj.dcd2);
        s[n++] = move32(39, 0);
        s[n++] = move48(40, area_va);
        s[n++] = move32(42, 0);                                                 // the whole target
        s[n++] = move32(43, (dj.width - 1) | (dj.height - 1) << 16);
        s[n++] = move32(44, 0);
        s[n++] = move32(45, 0x3f800000);
        s[n++] = move48(46, 0);
        s[n++] = move32(48, 0);
        s[n++] = move48(50, (area_va + pbl_at) | nt);                          // the preload's
        s[n++] = move48(52, d->descriptors.va + 0x1980);
        s[n++] = move48(54, 0);
        s[n++] = move32(56, 8 | 1u << 22);                                      // triangles, basic FIFO
        s[n++] = move32(57, 1u << 23 | 1u << 26 | (dj.samples > 1 ? 1u << 11 | 1u << 19 : 0u));   // no face culled
        // (multisample: run a sample, each its own texel)
        s[n++] = move32(58, dj.dcd1);
        s[n++] = move32(60, 0x3f800000);
        s[n++] = move32(61, 0);
        s[n++] = 49ULL << 56 | 0ULL << 32;                                      // HEAP_OPERATION: vertex/tiler started
        s[n++] = 6ULL << 56 | 1ULL << 33 | 1ULL << 38;                          // RUN_IDVS
    }
    s[n++] = move48(0, vsrt);                                                   // the vertex stage's SRT (0: none)
    s[n++] = move48(4, srt);                                                    // fragment SRT
    s[n++] = move48(8, vs_fau_va);
    s[n++] = move32(9, (uint32_t)(vs_fau_va >> 32) | (uint32_t)((dj.vs_fau.size() + 1) / 2) << 24);
    s[n++] = move48(12, fs_fau_va);
    s[n++] = move32(13, (uint32_t)(fs_fau_va >> 32) | (uint32_t)((dj.fs_fau.size() + 1) / 2) << 24);
    s[n++] = move48(16, dj.pos_spd);
    s[n++] = move48(18, dj.vary_spd);
    s[n++] = move48(20, dj.frag_spd);
    if (dj.tls_size != 0) {
        // SPILLED REGISTERS: a shader too big for its registers (CoreAnimation's fixed_frag_lph_cpf) spills to thread
        // local storage, which a draw had none of -- its stores went to addresses near 0 and the GPU faulted (AS1,
        // writes at 0x600000, 0x8000000, ...). The descriptor as a dispatch's: per thread 16 << word 0 bytes, no
        // threadgroup memory (word 1 = 31), the device's spill buffer (sized in drawImpl)
        uint64_t ls_at = (drawBytes(dj) - 0x80 + 0x3f) & ~0x3fULL;
        auto *ls = (uint32_t *)(area + ls_at);
        memset(ls, 0, 32);
        ls[0] = log2u((dj.tls_size + 15) / 16);
        ls[1] = 31;
        ls[2] = (uint32_t)(d->tls.va >> 8);
        ls[3] = (uint32_t)(d->tls.va >> 40) | 1u << 28;
        s[n++] = move48(24, area_va + ls_at);
    } else {
        s[n++] = move48(24, d->descriptors.va + 0x1200);                       // local storage: none
    }
    s[n++] = move32(32, 0);
    s[n++] = move32(33, dj.count);
    s[n++] = move32(34, dj.instances);
    s[n++] = move32(35, 0);
    s[n++] = move32(36, dj.vertex_start);
    s[n++] = move32(37, 0);
    s[n++] = move32(38, dj.dcd2);
    s[n++] = move32(39, (uint32_t)dj.index_size);                               // the index buffer's bytes
    s[n++] = move48(40, pass_va);                                               // the pass's tiler context
    s[n++] = move32(42, dj.scissor_min);
    s[n++] = move32(43, dj.scissor_max);
    s[n++] = move32(44, 0);                                                     // depth clamps 0 .. 1
    s[n++] = move32(45, 0x3f800000);
    s[n++] = move48(46, 0);                                                     // no occlusion query
    s[n++] = move32(48, dj.vary_size);
    s[n++] = move48(50, (area_va + bl_at) | dj.rts);                           // a blend descriptor a target
    s[n++] = move48(52, dj.zs ? area_va + zsd_at : d->descriptors.va + 0x1980);   // depth/stencil (none: all pass)
    s[n++] = move48(54, dj.index_va);                                           // the index buffer (0: none)
    s[n++] = move32(56, dj.tiler_flags);
    s[n++] = move32(57, dj.dcd0);
    s[n++] = move32(58, dj.dcd1);
    s[n++] = move32(60, 0x3f800000);                                            // primitive size 1.0
    s[n++] = move32(61, 0);                                                     // TILER_FLAGS2: VRS 1x1
    if (!dj.clear_only) {                                                       // (a clear: the fragment job alone)
        if (first && !dj.load) {                                                // (started already by the preload)
            s[n++] = 49ULL << 56 | 0ULL << 32;                                  // HEAP_OPERATION: vertex/tiler started
        }
        s[n++] = 6ULL << 56 | 1ULL << 33 | 1ULL << 38;                          // RUN_IDVS: malloc, the fragment's SRT
        if (!last) {
            return;                                                             // (the pass goes on with the next draw)
        }
        s[n++] = WAIT_ALL;
        s[n++] = 9ULL << 56;                                                    // FINISH_TILING
        s[n++] = WAIT_ALL;
        s[n++] = 49ULL << 56 | 1ULL << 32;                                      // HEAP_OPERATION: vertex/tiler completed
        s[n++] = WAIT_ALL;
    }
    const bool ir_on = ir != nullptr && *(uint64_t *)(ir + 0x10) != 0;
    if (ir_on) {
        s[n++] = move48(76, 0);                                                 // the handler off: tiling is done
        s[n++] = move32(78, 0);
        s[n++] = 25ULL << 56 | 76ULL << 40 | 78ULL << 32 | 2;
    }
    if (!first) {
        // the targets written back as the last draw stores them (the pass's descriptors are the first draw's)
        auto *prt = (uint32_t *)(pass_area + rt_at);
        prt[1] = (prt[1] & ~1u) | (dj.write ? 1u : 0u);
        for (unsigned i = 1; i < dj.rts; i++) {
            prt[16 * i + 1] = (prt[16 * i + 1] & ~1u) | (dj.more[i - 1].write && !dj.more[i - 1].elide ? 1u : 0u);
        }
        if (dj.zs) {
            auto *pfb = (uint32_t *)(pass_area + 0x80);
            pfb[12] = (pfb[12] & ~(1u << 18 | 1u << 8)) | (dj.has_z && dj.z_write ? 1u << 18 : 0) |
                (dj.has_s && dj.s_write ? 1u << 8 : 0);
        }
    }
    if (ir_on) {
        // the framebuffer descriptor: the pass's own, or after a flush its copy that loads what the flushes stored
        emitIr(d, dj, pass_area, pass_va, ir, ir_va, rt_at);
        s[n++] = move48(76, ir_va);
        s[n++] = 20ULL << 56 | 40ULL << 48 | 76ULL << 40 | 0x3ULL << 16 | 8;   // LOAD_MULTIPLE r40-r41 <- [r76 + 8]
        s[n++] = wait_on(1);
    } else {
        s[n++] = move48(40, (pass_va + 0x80) | 1 | (dj.zs ? 2u : 0u) | (dj.rts - 1) << 2);   // the framebuffer descriptor, its targets
    }
    s[n++] = move32(42, 0);
    s[n++] = move32(43, (dj.width - 1) | (dj.height - 1) << 16);
    s[n++] = move48(44, 0);                                                     // no tile enable map (r44 held the
    s[n++] = move32(46, 0);                                                     // depth clamps), its row stride
    s[n++] = 7ULL << 56;                                                        // RUN_FRAGMENT
    s[n++] = WAIT_ALL;
    // the pass finished, as panvk ends one: the tiler context's completed top and bottom (byte 40, four words) into
    // r70-r73, and FINISH_FRAGMENT gives that heap memory back and counts the pass done. Without it a queue drew one
    // pass and then binned nothing (the heap's in-flight passes never went down).
    // (A CLEAR -- no tiling, no heap pass started -- does not: counted done, the firmware's count of the heap's passes
    // in flight went one below the truth, and a later pass that ran out of heap had "none in flight": its tiler OOM
    // handler never ran and the queue waited for good)
    if (dj.clear_only) {
        return;
    }
    s[n++] = move48(74, pass_va);
    s[n++] = 20ULL << 56 | 70ULL << 48 | 74ULL << 40 | 0xfULL << 16 | 40;      // LOAD_MULTIPLE r70-73 <- [r74 + 40]
    s[n++] = WAIT_ALL;
    s[n++] = 11ULL << 56 | 70ULL << 40 | 72ULL << 32 | 1;                       // FINISH_FRAGMENT: chunks r70..r72, count it
    s[n++] = WAIT_ALL;
}
constexpr unsigned DRAW_WORDS = 62 + 45 + 10 + 3; // (instructions emitDraw writes, at most: the preload's 45, a pass's flush, the
                                              // tiler OOM handler's on/off included)

// whether draw B goes on in the Mali pass draw A is in: the same attachments, and B keeps what is in them (Load)
bool
samePass(const DrawJob &a, const DrawJob &b)
{
    if (a.clear_only || b.clear_only || !b.load || !b.load0 || a.target_va != b.target_va || a.rts != b.rts || a.width != b.width ||
        a.height != b.height || a.row_bytes != b.row_bytes || a.zs != b.zs || a.z_va != b.z_va || a.s_va != b.s_va ||
        (b.zs && ((b.has_z && !b.z_load) || (b.has_s && !b.s_load)))) {
        return false;
    }
    for (unsigned i = 1; i < a.rts; i++) {
        if (a.more[i - 1].va != b.more[i - 1].va || a.more[i - 1].missing != b.more[i - 1].missing ||
            (!b.more[i - 1].missing && !b.more[i - 1].load && !b.more[i - 1].memoryless)) {
            return false;
        }
    }
    // a draw that samples one of the pass's attachments needs what the draws before it left there IN MEMORY: in the
    // same Mali pass that is still in the tile buffer (the home screen's search label went missing so)
    for (unsigned t = 0; t < b.texture_count; t++) {
        const hakios_metal_texture *x = b.textures[t];
        if (x == nullptr) continue;
        uint64_t va = x->gpu(), end = va + x->span();  // (every level of a mipmapped one: a pass into one level)
        if (a.target_va >= va && a.target_va < end) return false;
        for (unsigned i = 1; i < a.rts; i++) {
            if (a.more[i - 1].va >= va && a.more[i - 1].va < end) return false;
        }
    }
    return true;
}

// the bytes a draw needs in its command buffer's stream BO, past the instructions
uint64_t
drawBytes(const DrawJob &dj)
{
    uint64_t bytes = 0x100 + 0x40 * dj.rts + 0x40 + (dj.zs ? 0x80 : 0) + ((dj.vs_fau.size() * 4 + 0x3f) & ~0x3fULL) +
        ((dj.fs_fau.size() * 4 + 0x3f) & ~0x3fULL);
    if (dj.texture_count || dj.sampler_count) {
        bytes += 0x80 + (dj.texture_count * 3 + dj.sampler_count) * 32 + 0x40;
    }
    if (dj.vs_texture_count || dj.vs_sampler_count) {
        bytes += 0x80 + (dj.vs_texture_count * 3 + dj.vs_sampler_count) * 32 + 0x80;
    }
    if (dj.tls_size != 0) {
        bytes += 0x80;                                  // (the local storage descriptor, 64-byte aligned, last)
    }
    if (dj.load) {
        bytes += 0x40 + ((dj.pre_vs_fau.size() * 4 + 0x3f) & ~0x3fULL) + ((dj.pre_fs_fau.size() * 4 + 0x3f) & ~0x3fULL) + 0x80 +
            (2 * dj.rts + 1) * 32 + 0x80 + 16 * dj.rts;
    }
    return bytes;
}
}  // namespace

namespace {
void passStats(hakios_metal_cb *cb);
void glitchScan(hakios_metal_cb *cb, bool before);
void glitchWatch(const char *event, hakios_metal_cb *cb);
void glitchWatchCommit(hakios_metal_cb *cb);
void glitchWatchDone();
bool bootArg(const char *word);
}  // namespace

struct hakios_metal_cb {
    hakios_metal_queue *queue;
    void (*scheduled)(hakios_metal_cb *, void *) = nullptr;  // (hakios_metal_cb_on_scheduled: after the submission)
    void *scheduled_context = nullptr;
    std::vector<unsigned> busy;                         // the IOSurfaces it draws into, marked busy (HakiosGPU 0x7d)
    std::vector<unsigned> mirror_out;                   // ... of them, those drawn through a mirror: copied back at commit
    hakios_metal_device *busy_device = nullptr;         // ... the device the marks were made through
    std::chrono::steady_clock::time_point committed_at; // (boot-arg metal_stats)
    Bo timestamps;                                      // ... the GPU's timestamp at each Mali pass's start and end
    std::chrono::steady_clock::time_point bo_at, submitted_at, waited_at;   // (boot-arg metal_cbtime)
    unsigned timestamp_count = 0;
    std::vector<Job> blits;
    // the vertex stage's [[texture(n)]] and [[sampler(m)]] for the draws that follow (hakios_metal_cb_set_vertex_textures)
    hakios_metal_texture *vs_textures[MALI_KERNEL_TEXTURES] = {};
    uint32_t vs_samplers[MALI_KERNEL_TEXTURES][8] = {};
    bool vs_sampler_set[MALI_KERNEL_TEXTURES] = {};
    unsigned vs_texture_count = 0, vs_sampler_count = 0;
    std::vector<hakios_metal_buffer *> held;    // retained until the GPU is done
    std::vector<hakios_metal_kernel *> kernels;
    std::vector<hakios_metal_texture *> held_textures;
    std::vector<DrawJob> draws;                 // a DRAW Job's (its dst the index)
    std::vector<uint64_t> draw_areas;
                                                // (where each draw's descriptors are in the stream BO: HAKIOS_MALI_TRACE)
    std::vector<hakios_metal_program *> programs;
    unsigned threadgroup_length = 0;            // setThreadgroupMemoryLength:atIndex: 0, for the dispatches after
    std::string error;
    enum hakios_metal_status status = HAKIOS_METAL_NOT_ENQUEUED;
    bool skip_refused = false;                  // a refused piece left out, not the whole command buffer failed
    unsigned skipped = 0;
    std::string skip_reason;
    unsigned seq = 0;                           // (boot-arg metal_glitchwatch: its number, in commit order)
    Bo stream;
    uint32_t sync = 0;
    void (*callback)(hakios_metal_cb *, void *) = nullptr;
    void *context = nullptr;
    bool freed = false;             // freed by its owner while in flight: the queue's thread frees it after
    std::mutex lock;
    std::condition_variable done;

    void hold(hakios_metal_buffer *b)
    {
        b->retain();
        held.push_back(b);
    }
    void letGo()
    {
        // the surfaces it drew into: not busy any more (the display waits for these marks to go: markBusy)
        for (unsigned sid : busy) {
            struct { uint32_t surface; int32_t delta; } r = { sid, -1 };
            std::lock_guard<std::mutex> g(busy_device->lock);
            busy_device->call(0x7d, &r, sizeof(r));
        }
        busy.clear();
        for (auto *b : held) {
            b->release();
        }
        for (auto *k : kernels) {
            k->release();
        }
        for (auto *t : held_textures) {
            t->release();
        }
        for (auto *p : programs) {
            p->release();
        }
        programs.clear();
        draws.clear();
        held.clear();
        kernels.clear();
        held_textures.clear();
    }
};

struct hakios_metal_queue {
    hakios_metal_device *device;
    std::thread thread;
    std::atomic<bool> self_free{false};         // freed from its own thread (a completion handler): run()'s end deletes it
    std::mutex lock;
    std::condition_variable work;
    std::deque<hakios_metal_cb *> pending;
    bool stopping = false;

    void run()
    {
        for (;;) {
            hakios_metal_cb *cb;
            {
                std::unique_lock<std::mutex> l(lock);
                work.wait(l, [&] { return stopping || !pending.empty(); });
                if (pending.empty()) {
                    return;
                }
                cb = pending.front();
                pending.pop_front();
            }
            if (cb->status != HAKIOS_METAL_ERROR) {
                syncobj_wait w = {};
                w.handles = (uint64_t)(uintptr_t)&cb->sync;
                w.timeout_nsec = 10000000000LL;         // relative (HakiosGPU's convention)
                w.count_handles = 1;
                w.flags = 1;
                // (WITHOUT the device's lock, 11-10: it guards the device's buffers and addresses, not the call -- held
                // here, every other call waited for the GPU to finish this command buffer: the next one's submission in
                // commit, and HakiosGPU's surface marks at a command buffer's first draw, so CoreAnimation recorded its
                // frames one GPU completion at a time; the kernel drops its own lock while it waits)
                int r = device->call(SYNCOBJ_WAIT, &w, sizeof(w));
                cb->waited_at = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> g(cb->lock);
                if (r != 0) {
                    cb->error = "the GPU did not finish the command buffer in 10 s";
                    cb->status = HAKIOS_METAL_ERROR;
                } else if (w.pad != 0) {
                    // (HakiosGPU's: the queue was restarted after a GPU fault with this submission not done)
                    cb->error = "a GPU fault: the command buffer's work is lost (the queue was restarted)";
                    cb->status = HAKIOS_METAL_ERROR;
                } else {
                    cb->status = HAKIOS_METAL_COMPLETED;
                }
            }
            {
                std::lock_guard<std::mutex> g(device->lock);
                if (cb->sync) {
                    syncobj_destroy d = { cb->sync, 0 };
                    device->call(SYNCOBJ_DESTROY, &d, sizeof(d));
                    cb->sync = 0;
                }
            }
            if (getenv("HAKIOS_MALI_TRACE") != nullptr && cb->stream.cpu != nullptr) {
                // what the tiler did with each draw: the polygon list (its top bits set when it binned something), and
                // the heap descriptor's bottom (primitive data, up) and top (bins, down)
                auto *h = (uint32_t *)device->heapdesc.cpu;
                for (uint64_t at : cb->draw_areas) {
                    auto *tc = (uint32_t *)((uint8_t *)cb->stream.cpu + at);
                    fprintf(stderr, "hakios-metal: draw: at 0x%llx: ", (unsigned long long)(cb->stream.va + at));
                    fprintf(stderr, "tiler context %08x %08x %08x %08x; completed %08x%08x %08x%08x; "
                        "private %08x %08x %08x %08x %08x %08x %08x %08x; heap bottom %08x%08x top %08x%08x\n",
                        tc[0], tc[1], tc[2], tc[3], tc[11], tc[10], tc[13], tc[12], tc[24], tc[25], tc[26], tc[27], tc[28],
                        tc[29], tc[30], tc[31], h[5], h[4], h[7], h[6]);
                }
                uint64_t nothing = 0;
                device->call(0x7f, &nothing, sizeof(nothing));        // (HakiosGPU: the queues and heaps into its log)
            }
            static const bool stats = bootArg("metal_stats") || bootArg("mdsay="), cbtime = bootArg("metal_cbtime");
            if (stats) {
                passStats(cb);
            }
            if (cbtime) {
                // commit -> stream BO made -> built and submitted -> the GPU's wait returned -> here (freed, called back)
                static unsigned said;
                auto us = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
                    return (long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
                };
                if (said++ < 400) {
                    auto now = std::chrono::steady_clock::now();
                    fprintf(stderr, "hakios-metal: cbtime: bo %ld, build+submit %ld, wait %ld, after %ld us (%zu jobs)\n",
                        us(cb->committed_at, cb->bo_at), us(cb->bo_at, cb->submitted_at),
                        us(cb->submitted_at, cb->waited_at), us(cb->waited_at, now), cb->blits.size());
                }
            }
            glitchScan(cb, false);
            glitchWatch("completion", cb);
            glitchWatchDone();
            device->giveStream(cb->stream);
            if (cb->timestamps.size != 0) {
                device->freeBo(cb->timestamps);
                cb->timestamps.size = 0;
            }
            cb->letGo();
            cb->done.notify_all();
            if (cb->callback) {
                cb->callback(cb, cb->context);  // (it may free the command buffer)
            } else if (cb->freed) {
                delete cb;
            }
        }
    }
};

namespace {
// what draws need of the device: the queue's tiler heap (panvk's sizes) and its descriptor with a 64 KiB geometry buffer
// after it (heapdesc + 0x1000); the sample positions (descriptors + 0x1800), a depth/stencil descriptor that lets all
// through (+ 0x1980). (The GPU's pages are outer shareable -- hakios-kernel's PTE_SH_OS: without that the tiler read
// what the shader cores had written as memory held it before, and every pass looked like the one before it.)

bool
openDraws(hakios_metal_device *d)
{
    // (HAKIOS_MALI_HEAP_CHUNKS=N: the heap's most chunks, 5 to 64 -- a test of incremental rendering, mali_draw_test)
    const char *most = getenv("HAKIOS_MALI_HEAP_CHUNKS");
    uint32_t max_chunks = most != nullptr && atoi(most) >= 5 && atoi(most) <= 64 ? (uint32_t)atoi(most) : 64;
    // (11-13) THE HEAP'S FIRST CHUNKS: five (10 MB) for the window server, which draws the screen from its first frame;
    // ONE for every other process -- with Metal for every process each client held 10 MB of the kernel's memory from its
    // start, drawn or not (141 MB of them in an application sweep), and the heap grows by itself when a pass fills it
    // (HakiosGPU's tiler out-of-memory answer, a chunk a time up to max_chunks). HAKIOS_MALI_HEAP_FIRST=N: N.
    const char *prog = getprogname(), *first_env = getenv("HAKIOS_MALI_HEAP_FIRST");
    uint32_t first_chunks = first_env != nullptr && atoi(first_env) >= 1 && atoi(first_env) <= 5 ? (uint32_t)atoi(first_env) :
        prog != nullptr && strcmp(prog, "backboardd") == 0 ? 5 : 1;
    struct { uint32_t vm_id, initial, chunk_size, max_chunks, in_flight, handle; uint64_t ctx, first; } th =
        { d->vm, first_chunks, 2u << 20, max_chunks, 65535, 0, 0, 0 };
    if (d->call(0x4b /* TILER_HEAP_CREATE */, &th, sizeof(th)) != 0) {
        fail("HakiosGPU: no tiler heap (hakios-kernel with heaps, 11-05 or later)");
        return false;
    }
    d->heap_handle = th.handle;
    d->heap_ctx = th.ctx;
    d->heap_first = th.first;
    if (!d->newBo(d->heapdesc, 0x18000, BIND_NOEXEC)) {
        return false;
    }
    auto *h = (uint32_t *)d->heapdesc.cpu;
    memset(h, 0, d->heapdesc.size);
    h[0] = 9 | 2 << 4;                                  // a buffer: a tiler heap
    h[1] = 2u << 20;
    *(uint64_t *)&h[2] = th.first;
    *(uint64_t *)&h[4] = th.first + 64;                 // (past the chunk's header)
    *(uint64_t *)&h[6] = th.first + (2u << 20);
    auto *desc = (uint8_t *)d->descriptors.cpu;
    auto *pos = (uint32_t *)(desc + 0x1800), *zs = (uint32_t *)(desc + 0x1980);
    memset(pos, 0, 256);
    pos[0] = pos[32] = 128 | 128 << 16;                 // one sample, at the pixel's centre (and the origin)
    // four samples (descriptors + 0x2000): Mesa's rotated 4x grid (pan_samples.c), 1/256 pixel
    auto *pos4 = (uint32_t *)(desc + 0x2000);
    memset(pos4, 0, 256);
    pos4[0] = 96 | 32u << 16;
    pos4[1] = 224 | 96u << 16;
    pos4[2] = 32 | 160u << 16;
    pos4[3] = 160 | 224u << 16;
    pos4[32] = 128 | 128u << 16;
    memset(zs, 0, 32);
    zs[0] = 7 | 7 << 4 | 7 << 16;                       // stencil faces: always
    zs[4] = 1u << 21 | 1u << 22 | 1u << 23 | 2u << 25 | 7u << 29;   // depth: always, nothing written (zsOf's word 4)
    d->heap_ok = true;
    return true;
}
}  // namespace

extern "C" {

const char *
hakios_metal_error(void)
{
    return gError.c_str();
}

/*
 * The kernel starts the Mali mali_delay (8) s after its own start, once the firmware's file can be read: on the Pixel 8a
 * HakiosGPU appears about 20 s into the boot, after backboardd has asked for its device (15 s) -- and a window server
 * without one composites in software for good. With the GPU kept (mali_keep, mali= among the boot-args) the service is
 * waited for, up to 15 s; without it the answer is at once, as before.
 */
static bool
gpu_expected(void)
{
    char args[1024] = {};
    size_t n = sizeof(args) - 1;
    if (sysctlbyname("kern.bootargs", args, &n, nullptr, 0) != 0) {
        return false;
    }
    for (const char *p = args; *p != 0;) {
        while (*p == ' ') p++;
        if ((strncmp(p, "mali_keep=", 10) == 0 && p[10] != '0') || (strncmp(p, "mali=", 5) == 0 && p[5] != '0')) {
            return true;
        }
        while (*p != 0 && *p != ' ') p++;
    }
    return false;
}

static io_service_t
find_gpu_service(void)
{
    io_service_t service = IOServiceGetMatchingService(0, IOServiceMatching("HakiosGPU"));
    if (service != 0 || !gpu_expected()) {
        return service;
    }
    for (int i = 0; i < 150 && service == 0; i++) {     // 15 s in steps of 100 ms
        usleep(100000);
        service = IOServiceGetMatchingService(0, IOServiceMatching("HakiosGPU"));
    }
    if (service != 0) {
        fprintf(stderr, "hakios-metal: HakiosGPU appeared while it was waited for\n");
    }
    return service;
}

hakios_metal_device *
hakios_metal_open(void)
{
    io_service_t service = find_gpu_service();
    if (service == 0) {
        fail("no HakiosGPU service (hakios-kernel with mali_keep=1)");
        return nullptr;
    }
    CFTypeRef kind = IORegistryEntryCreateCFProperty(service, CFSTR("DRMDriver"), kCFAllocatorDefault, 0);
    bool panthor = kind != nullptr && CFGetTypeID(kind) == CFStringGetTypeID() &&
        CFStringCompare((CFStringRef)kind, CFSTR("panthor"), 0) == kCFCompareEqualTo;
    if (kind != nullptr) {
        CFRelease(kind);
    }
    if (!panthor) {
        fail("HakiosGPU is not a Mali (panthor) here");
        IOObjectRelease(service);
        return nullptr;
    }
    auto *d = new hakios_metal_device;
    if (IOServiceOpen(service, mach_task_self(), 0, &d->connection) != KERN_SUCCESS) {
        IOObjectRelease(service);
        delete d;
        fail("HakiosGPU could not be opened");
        return nullptr;
    }
    IOObjectRelease(service);
    gpu_info info = {};
    dev_query q = { 0, sizeof(info), (uint64_t)(uintptr_t)&info };
    vm_create vc = {};
    vc.user_va_range = 1ULL << 47;
    if (d->call(DEV_QUERY, &q, sizeof(q)) != 0 || d->call(VM_CREATE, &vc, sizeof(vc)) != 0) {
        fail("HakiosGPU: no description or no VM");
        hakios_metal_close(d);
        return nullptr;
    }
    d->gpu_id = info.gpu_id;
    d->shaders = info.shader_present;
    d->tiler = info.tiler_present;
    d->max_threads = info.max_threads;
    d->max_tasks = info.thread_features >> 24;
    for (d->core_range = 64; d->core_range > 0 && !(info.shader_present >> (d->core_range - 1) & 1); d->core_range--) {
    }
    d->vm = vc.id;
    // the blit shaders (code read only and executable), their program descriptors and the local storage
    if (!d->newBo(d->descriptors, 0x4000, BIND_READONLY)) {
        hakios_metal_close(d);
        return nullptr;
    }
    const mali_blit_shader *shaders[4] = { &mali_copy_u32, &mali_fill_u32, &mali_copy_u8, &mali_fill_u8 };
    auto *bytes = (uint8_t *)d->descriptors.cpu;
    for (unsigned i = 0; i < 4; i++) {
        memcpy(bytes + 0x200 * i, shaders[i]->code, shaders[i]->size);     // code at 0x000, 0x200, 0x400, 0x600
        auto *spd = (uint32_t *)(bytes + 0x1000 + 0x40 * i);                // program descriptors from 0x1000
        spd[0] = 8 | 1 << 4 | (shaders[i]->work_regs <= 32 ? 2u : 0u) << 30;
        spd[1] = (uint32_t)(shaders[i]->preload >> 48);
        spd[2] = (uint32_t)(d->descriptors.va + 0x200 * i);
        spd[3] = (uint32_t)((d->descriptors.va + 0x200 * i) >> 32);
        d->spd[i] = d->descriptors.va + 0x1000 + 0x40 * i;
    }
    {
        // copy_rows_u32 (COPY_ROWS): its code at 0x800 (up to 2 KiB), its program descriptor at 0x1100
        const mali_blit_shader *r = &mali_copy_rows_u32;
        memcpy(bytes + 0x800, r->code, r->size);
        auto *spd = (uint32_t *)(bytes + 0x1100);
        spd[0] = 8 | 1 << 4 | (r->work_regs <= 32 ? 2u : 0u) << 30;
        spd[1] = (uint32_t)(r->preload >> 48);
        spd[2] = (uint32_t)(d->descriptors.va + 0x800);
        spd[3] = (uint32_t)((d->descriptors.va + 0x800) >> 32);
        d->spd[4] = d->descriptors.va + 0x1100;
    }
    ((uint32_t *)(bytes + 0x1200))[1] = 31;                                 // local storage: no workgroup memory
    __asm__ volatile("dsb sy" ::: "memory");
    // one group with one queue
    queue_create qc = {};
    qc.priority = 1;
    qc.ringbuf_size = 0x10000;
    group_create gc = {};
    gc.queues = { sizeof(qc), 1, (uint64_t)(uintptr_t)&qc };
    gc.max_compute_cores = gc.max_fragment_cores = 64;
    gc.max_tiler_cores = 1;
    gc.priority = 1;
    gc.compute_core_mask = gc.fragment_core_mask = d->shaders;
    gc.tiler_core_mask = d->tiler;
    gc.vm_id = d->vm;
    d->tiler_levels = (info.tiler_features >> 8) & 0xf;
    openDraws(d);                                       // (without it, draws are refused by name)
    if (d->call(GROUP_CREATE, &gc, sizeof(gc)) != 0) {
        fail("HakiosGPU: no group");
        hakios_metal_close(d);
        return nullptr;
    }
    d->group = gc.group_handle;
    return d;
}

hakios_metal_device *
hakios_metal_open_fake(void)
{
    fail("no fake GPU in the Mali build");
    return nullptr;
}

void
hakios_metal_close(hakios_metal_device *d)
{
    if (d == nullptr) {
        return;
    }
    for (Bo &b : d->stream_pool) {
        d->freeBo(b);
    }
    d->stream_pool.clear();
    d->freeBo(d->descriptors);
    d->freeBo(d->heapdesc);
    if (d->heap_handle != 0) {
        struct { uint32_t handle, pad; } td = { d->heap_handle, 0 };
        d->call(0x4c /* TILER_HEAP_DESTROY */, &td, sizeof(td));
    }
    d->freeBo(d->wls);
    d->freeBo(d->tls);
    for (Bo &b : d->retired) {
        d->freeBo(b);
    }
    if (d->connection) {
        IOServiceClose(d->connection);
    }
    delete d;
}

uint64_t
hakios_metal_gpu_id(hakios_metal_device *d)
{
    return d ? d->gpu_id : 0;
}

int
hakios_metal_has_compiler(void)
{
    return 1;
}

hakios_metal_buffer *
hakios_metal_buffer_new(hakios_metal_device *d, uint64_t length)
{
    auto *b = new hakios_metal_buffer;
    b->device = d;
    b->length = length;
    if (!d->newBo(b->bo, (length ? length : 1) + kBufferSlack, BIND_NOEXEC)) {
        delete b;
        return nullptr;
    }
    if (b->bo.cpu != nullptr && b->bo.size > length) {
        memset((uint8_t *)b->bo.cpu + length, kSlackByte, (size_t)(b->bo.size - length));
    }
    return b;
}

void
hakios_metal_buffer_free(hakios_metal_buffer *b)
{
    if (b != nullptr) {
        b->release();       // (a command buffer that uses it keeps it until the GPU is done)
    }
}

void *
hakios_metal_buffer_contents(hakios_metal_buffer *b)
{
    return b ? b->bo.cpu : nullptr;
}

uint64_t
hakios_metal_buffer_length(hakios_metal_buffer *b)
{
    return b ? b->length : 0;
}

uint64_t
hakios_metal_buffer_gpu_address(hakios_metal_buffer *b)
{
    return b ? b->bo.va : 0;
}

hakios_metal_queue *
hakios_metal_queue_new(hakios_metal_device *d)
{
    auto *q = new hakios_metal_queue;
    q->device = d;
    q->thread = std::thread([q] {
        q->run();
        if (q->self_free.load()) {
            delete q;
        }
    });
    return q;
}

void
hakios_metal_queue_free(hakios_metal_queue *q)
{
    if (q == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> l(q->lock);
        q->stopping = true;
    }
    q->work.notify_all();
    if (std::this_thread::get_id() == q->thread.get_id()) {
        // FREED ON ITS OWN THREAD (11-12): a command buffer's completion handler let the last reference to the queue
        // go -- joining oneself is libc++'s "thread::join failed: Resource deadlock avoided", uncaught: mediaserverd
        // aborted. The thread ends by itself once the handler returns (stopping is set) and deletes the queue.
        q->self_free = true;
        q->thread.detach();
        return;
    }
    q->thread.join();
    delete q;
}

hakios_metal_cb *
hakios_metal_cb_new(hakios_metal_queue *q)
{
    auto *cb = new hakios_metal_cb;
    cb->queue = q;
    return cb;
}

void
hakios_metal_cb_free(hakios_metal_cb *cb)
{
    if (cb == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> l(cb->lock);
    if (cb->status == HAKIOS_METAL_COMMITTED || cb->status == HAKIOS_METAL_SCHEDULED) {
        if (cb->callback == nullptr) {
            cb->freed = true;   // the queue's thread deletes it after
            return;
        }
        cb->done.wait(l, [&] { return cb->status == HAKIOS_METAL_COMPLETED || cb->status == HAKIOS_METAL_ERROR; });
    }
    l.unlock();
    cb->letGo();            // (never committed: what it held)
    delete cb;
}

int
hakios_metal_cb_copy(hakios_metal_cb *cb, hakios_metal_buffer *from, uint64_t from_offset, hakios_metal_buffer *to,
    uint64_t to_offset, uint64_t length)
{
    if (from_offset + length > from->bo.size || to_offset + length > to->bo.size || length >= (1ULL << 32)) {
        fail("copy: out of the buffers");
        return -1;
    }
    bool words = ((from_offset | to_offset | length) & 3) == 0;
    Job j = {};
    j.shader = words ? 0u : 2u;
    j.dst = to->bo.va + to_offset;
    j.src = from->bo.va + from_offset;
    j.count = (uint32_t)(words ? length / 4 : length);
    cb->blits.push_back(j);
    cb->hold(from);
    cb->hold(to);
    return 0;
}

int
hakios_metal_cb_fill(hakios_metal_cb *cb, hakios_metal_buffer *buffer, uint64_t offset, uint64_t length, uint8_t value)
{
    if (offset + length > buffer->bo.size || length >= (1ULL << 32)) {
        fail("fill: out of the buffer");
        return -1;
    }
    bool words = ((offset | length) & 3) == 0;
    Job j = {};
    j.shader = words ? 1u : 3u;
    j.dst = buffer->bo.va + offset;
    if (getenv("HAKIOS_MALI_FAULT_TEST") != nullptr) {
        j.dst = 0x1000;                                 // (a test of the kernel's recovery: an address nothing maps)
    }
    j.value = words ? value * 0x01010101u : value;
    j.count = (uint32_t)(words ? length / 4 : length);
    cb->blits.push_back(j);
    cb->hold(buffer);
    return 0;
}

void
hakios_metal_cb_fail(hakios_metal_cb *cb, const char *why)
{
    if (cb->error.empty()) {
        cb->error = why;
    }
}

static void mirrorsBack(hakios_metal_cb *cb, hakios_metal_device *d);

/*
 * AN ATTACHMENT NO DRAW OF A MALI PASS TOUCHES (11-10): CoreAnimation's passes carry a second colour attachment that
 * most of their draws neither write nor read; each such draw keeps it loaded and stored back unchanged, so a full-screen
 * Mali pass read and wrote ~10 MB of it for nothing -- Spotlight's opening drew ~8 full-screen passes a frame at 2.1-2.8
 * ms each, 17-25 ms of GPU time a frame (metal_stats_ms). Over the passes as the encoder will group them (samePass,
 * which reads none of this), an attachment every draw of the pass leaves untouched is elided: not preloaded, not
 * written back -- its memory keeps what it had, as a load and a store of it unchanged would leave it. A pass with
 * nothing else to load has no preload at all. Boot-arg metal_noelide: not done.
 */
static void
passElide(hakios_metal_cb *cb)
{
    static const bool off = bootArg("metal_noelide");
    if (off) return;
    for (size_t i = 0; i < cb->blits.size(); i++) {
        if (cb->blits[i].shader != DRAW) continue;
        size_t k = i;
        while (k + 1 < cb->blits.size() && cb->blits[k + 1].shader == DRAW &&
               samePass(cb->draws[cb->blits[k].dst], cb->draws[cb->blits[k + 1].dst])) {
            k++;
        }
        const DrawJob &head = cb->draws[cb->blits[i].dst];
        for (unsigned r = 1; r < head.rts; r++) {
            bool all = true;
            for (size_t j = i; j <= k && all; j++) {
                const DrawJob::More &m = cb->draws[cb->blits[j].dst].more[r - 1];
                all = m.untouched && !m.missing;
            }
            for (size_t j = i; j <= k; j++) {
                cb->draws[cb->blits[j].dst].more[r - 1].elide = all;
            }
        }
        i = k;
    }
}

void
hakios_metal_cb_commit(hakios_metal_cb *cb, void (*callback)(hakios_metal_cb *cb, void *context), void *context)
{
    hakios_metal_device *d = cb->queue->device;

    cb->callback = callback;
    cb->context = context;
    cb->committed_at = std::chrono::steady_clock::now();
    mirrorsBack(cb, d);
    passElide(cb);
    // NOT INTO THE SURFACE ON THE PANEL (11-10): the GPU service holds this until the panel has left each surface it draws
    // into (HakiosGPU 0x80, at most 25 ms) -- CoreAnimation records its next frame into the one still shown, and the GPU
    // overwrote what the panel was reading. Here, not at the first draw: the recording itself goes on meanwhile.
    for (unsigned sid : cb->busy) {
        struct { uint32_t surface; int32_t delta; } r = { sid, 0 };
        d->call(0x80, &r, sizeof(r));
    }
    glitchWatchCommit(cb);
    glitchScan(cb, true);
    cb->status = HAKIOS_METAL_COMMITTED;
    if (!cb->error.empty()) {
        cb->status = HAKIOS_METAL_ERROR;
    } else if (!cb->blits.empty()) {
        // the stream (from 0), then each job's push constants: 32 bytes for a blit, a kernel's fau_words for a dispatch
        static const bool stats = bootArg("metal_stats") || bootArg("mdsay=");
        size_t words = 18 + cb->blits.size() * 24 + cb->draws.size() * (DRAW_WORDS + (stats ? 4 : 0));   // (24: a compute job's 15, its flush's 3)
        if (stats && !cb->draws.empty() && cb->timestamps.size == 0) {
            if (!d->newBo(cb->timestamps, ((cb->draws.size() * 16) + 0xfff) & ~0xfffULL, BIND_NOEXEC)) {
                cb->timestamps.size = 0;
            }
        }
        cb->timestamp_count = 0;
        uint64_t fau_area = (words * 8 + 0xff) & ~0xffULL, fau_size = 0;
        std::vector<uint64_t> fau_at(cb->blits.size()), ir_at(cb->blits.size(), 0);
        for (size_t i = 0; i < cb->blits.size(); i++) {
            fau_at[i] = fau_area + fau_size;
            if (cb->blits[i].shader == DRAW) {
                fau_size += (drawBytes(cb->draws[cb->blits[i].dst]) + 0x3f) & ~0x3fULL;
                // a Mali pass's first draw: the pass's incremental rendering (emitIr) after its area
                if (i == 0 || cb->blits[i - 1].shader != DRAW ||
                    !samePass(cb->draws[cb->blits[i - 1].dst], cb->draws[cb->blits[i].dst])) {
                    ir_at[i] = fau_area + fau_size;
                    fau_size += IR_BYTES;
                }
                continue;
            }
            // (a dispatch: its push constants, then its local storage descriptor, 64-byte aligned)
            fau_size += cb->blits[i].shader == DISPATCH ? ((cb->blits[i].kernel->binary.fau_words * 4 + 0x3f) & ~0x3fULL) + 0x40
                                                        : 0x40;
            // (and its resource table: 6 entries, then the texture, sampler and image tables, and the planes)
            if (cb->blits[i].shader == DISPATCH && (cb->blits[i].texture_count || cb->blits[i].sampler_count)) {
                fau_size += 0x80 + (cb->blits[i].texture_count * 3 + cb->blits[i].sampler_count) * 32 + 0x40;
            }
        }
        bool stream_ok = d->takeStream(cb->stream, fau_area + fau_size);
        cb->bo_at = std::chrono::steady_clock::now();
        if (!stream_ok) {
            cb->error = gError;
            cb->status = HAKIOS_METAL_ERROR;
        } else {
            auto *s = (uint64_t *)cb->stream.cpu;
            unsigned n = 0;
            s[n++] = set_state(0, 3);           // v11's scoreboards, as panvk sets them up
            s[n++] = set_state(9, 0x8);
            s[n++] = set_state(1, 0);
            s[n++] = set_state(2, 1);
            s[n++] = set_state(8, 0xf0);
            // the GPU's caches made to see what the CPU wrote, and (at the end) to show the CPU what the GPU wrote, as
            // gallium's panfrost ends a batch: L2 and load/store cleaned, the others (the tiler's, descriptors')
            // invalidated. Without it a queue drew its first render pass only: later passes found the tiler heap and
            // contexts as the GPU's caches held them.
            s[n++] = move32(74, 0);
            s[n++] = 36ULL << 56 | 74ULL << 40 | 2 << 8 | 1 << 4 | 1;          // FLUSH_CACHE2
            s[n++] = WAIT_ALL;
            if (!cb->draws.empty()) {
                s[n++] = 34ULL << 56 | 0xf;                                     // REQ_RESOURCE: compute, fragment, tiler, IDVS
                if (!d->heap_set.exchange(true)) {                              // the queue's tiler heap, once
                    s[n++] = move48(0, d->heap_ctx);
                    s[n++] = 48ULL << 56;                                       // HEAP_SET
                }
            }
            uint8_t *pass_area = nullptr;           // the open Mali pass's first draw (emitDraw), or none
            uint64_t pass_va = 0;
            uint8_t *pass_ir = nullptr;             // ... and its incremental rendering (emitIr)
            uint64_t pass_ir_va = 0;
            for (size_t i = 0; i < cb->blits.size(); i++) {
                const Job &b = cb->blits[i];
                auto *pc = (uint8_t *)cb->stream.cpu + fau_at[i];
                if (b.shader == DRAW) {
                    const DrawJob &dj = cb->draws[b.dst];
                    bool first = pass_area == nullptr;
                    bool last = i + 1 >= cb->blits.size() || cb->blits[i + 1].shader != DRAW ||
                        !samePass(dj, cb->draws[cb->blits[i + 1].dst]);
                    if (first) {
                        pass_area = pc;
                        pass_va = cb->stream.va + fau_at[i];
                        pass_ir = ir_at[i] != 0 ? (uint8_t *)cb->stream.cpu + ir_at[i] : nullptr;
                        pass_ir_va = ir_at[i] != 0 ? cb->stream.va + ir_at[i] : 0;
                    }
                    // (metal_stats: STORE_STATE timestamp, opcode 40, into the command buffer's timestamps, r80:r81
                    // the address; after everything before it is done)
                    auto stamp = [&] {
                        if (cb->timestamps.size != 0 && (cb->timestamp_count + 1) * 8 <= cb->timestamps.size) {
                            s[n++] = WAIT_ALL;
                            s[n++] = move48(80, cb->timestamps.va + 8ULL * cb->timestamp_count++);
                            s[n++] = 40ULL << 56 | 80ULL << 40 | 0ULL << 32;
                        }
                    };
                    if (first) stamp();
                    emitDraw(d, dj, pc, cb->stream.va + fau_at[i], s, n, first, last, pass_area, pass_va, pass_ir, pass_ir_va);
                    if (last) stamp();
                    cb->draw_areas.push_back(fau_at[i]);
                    if (last) {
                        pass_area = nullptr;
                    }
                    continue;
                }
                uint64_t fau_va = cb->stream.va + fau_at[i], spd;
                unsigned fau_entries, size[3], groups[3];
                uint64_t local_storage = d->descriptors.va + 0x1200;            // none
                uint64_t resources = 0;
                if (b.shader == DISPATCH) {
                    const mali_kernel_binary &k = b.kernel->binary;
                    auto *w = (uint32_t *)pc;
                    memset(pc, 0, k.fau_words * 4);
                    memcpy(pc, b.buffers, sizeof(b.buffers));                       // words 2n, 2n+1
                    for (int c = 0; c < 3; c++) {
                        w[MALI_KERNEL_WORKGROUP_WORD + c] = b.threads[c];
                        w[MALI_KERNEL_GROUPS_WORD + c] = b.groups[c];
                        w[MALI_KERNEL_GRID_WORD + c] = b.grid[c];
                    }
                    memcpy(w + MALI_KERNEL_RESERVED, k.consts, (k.fau_words - MALI_KERNEL_RESERVED) * 4);
                    if (k.tls_size != 0 || b.wls_size != 0) {
                        // the local storage descriptor (Mesa's genxml v10 "Local Storage"): spill space, packed, per thread
                        // 16 << word 0 bytes; threadgroup memory 2^(word 1 & 31) instances of 2^((word 1 >> 8) - 1) bytes
                        uint64_t at = (k.fau_words * 4 + 0x3f) & ~0x3fULL;
                        auto *ls = (uint32_t *)(pc + at);
                        memset(ls, 0, 32);
                        if (k.tls_size != 0) {
                            ls[0] = log2u((k.tls_size + 15) / 16);
                            ls[2] = (uint32_t)(d->tls.va >> 8);
                            ls[3] = (uint32_t)(d->tls.va >> 40) | 1u << 28;
                        }
                        if (b.wls_size != 0) {
                            ls[1] = log2u(b.wls_instances) | (log2u(b.wls_size) + 1) << 8;
                            ls[4] = (uint32_t)d->wls.va;
                            ls[5] = (uint32_t)(d->wls.va >> 32);
                        } else {
                            ls[1] = 31;                                         // no threadgroup memory
                        }
                        local_storage = fau_va + at;
                    }
                    if (b.texture_count || b.sampler_count) {
                        // the resource table (r0): 16-byte entries, table t's descriptors at entry t (gallium's
                        // numbering: sampler 3, texture 4, image 5); each texture once in the texture table, once in the
                        // image table, its plane after them
                        uint64_t at = ((k.fau_words * 4 + 0x3f) & ~0x3fULL) + 0x40;
                        uint8_t *rt = pc + at;
                        uint64_t rt_va = fau_va + at, tex_va = rt_va + 0x80;
                        uint64_t samp_va = tex_va + b.texture_count * 32, img_va = samp_va + b.sampler_count * 32;
                        uint64_t plane_va = img_va + b.texture_count * 32;
                        memset(rt, 0, 0x80);
                        auto entry = [&](unsigned table, uint64_t va, unsigned count) {
                            auto *e = (uint32_t *)(rt + 16 * table);
                            e[0] = (uint32_t)va;
                            e[1] = (uint32_t)(va >> 32) | 1u << 24;                 // contains descriptors
                            e[2] = count * 32;
                        };
                        entry(MALI_TABLE_TEXTURE, tex_va, b.texture_count);
                        entry(MALI_TABLE_SAMPLER, samp_va, b.sampler_count);
                        entry(MALI_TABLE_IMAGE, img_va, b.texture_count);
                        for (unsigned t = 0; t < b.texture_count; t++) {
                            auto *tex = (uint32_t *)(pc + (tex_va - fau_va) + 32 * t);
                            auto *img = (uint32_t *)(pc + (img_va - fau_va) + 32 * t);
                            uint32_t plane[8];
                            if (b.textures[t] == nullptr) {
                                memset(tex, 0, 32);
                                memset(img, 0, 32);
                                continue;
                            }
                            // (one plane, shared by the two descriptors; 32-byte aligned as everything here)
                            uint64_t pva = plane_va;
                            plane_va = pva + 32;
                            describe(b.textures[t], tex, plane, pva);
                            memcpy(pc + (pva - fau_va), plane, 32);
                            memcpy(img, tex, 32);
                        }
                        for (unsigned t = 0; t < b.sampler_count; t++) {
                            memcpy(pc + (samp_va - fau_va) + 32 * t, b.samplers[t], 32);
                        }
                        resources = rt_va | MALI_TABLES;
                    }
                    spd = b.kernel->spd;
                    fau_entries = (k.fau_words + 1) / 2;
                    memcpy(size, b.threads, sizeof(size));
                    memcpy(groups, b.groups, sizeof(groups));
                } else {
                    memset(pc, 0, 32);
                    memcpy(pc + 0, &b.dst, 8);
                    memcpy(pc + 8, &b.src, 8);
                    memcpy(pc + 16, &b.value, 4);
                    memcpy(pc + 20, &b.count, 4);
                    if (b.shader == COPY_ROWS) {
                        memcpy(pc + 24, &b.src_stride, 4);
                        memcpy(pc + 28, &b.dst_stride, 4);
                        spd = d->spd[4];
                    } else {
                        spd = d->spd[b.shader];
                    }
                    fau_entries = 4;                                                // 8 words: 4 of 64 bits
                    size[0] = 64, size[1] = size[2] = 1;                            // 64 invocations a workgroup
                    groups[0] = (b.count + 63) / 64, groups[1] = groups[2] = 1;
                }
                // A COMPUTE JOB AFTER OTHER WORK OF THIS COMMAND BUFFER (11-10): the load/store caches cleaned and the
                // texture caches dropped first, as a render pass now opens (pass_inv) -- a blit or kernel reading what a
                // pass or another job just wrote (CoreAnimation's backdrop blurs copy what is drawn so far) could read a
                // core's stale texture lines: the widgets page's and the App Library's blurred backgrounds came out
                // empty or with noise for a frame (the tester's clips). Jobs were only ordered (WAIT_ALL).
                if (i > 0) {
                    s[n++] = move32(74, 0);
                    s[n++] = 36ULL << 56 | 74ULL << 40 | 2 << 8 | 1 << 4;      // FLUSH_CACHE2: LSC clean, others invalidated
                    s[n++] = WAIT_ALL;
                }
                s[n++] = move48(16, spd);
                s[n++] = move48(24, local_storage);
                s[n++] = move48(8, fau_va);
                s[n++] = move32(9, (uint32_t)(fau_va >> 32) | fau_entries << 24);
                s[n++] = move48(0, resources);                                      // the resource table, or none
                s[n++] = move32(32, 0);
                s[n++] = move32(33, (size[0] - 1) | (size[1] - 1) << 10 | (size[2] - 1) << 20);
                s[n++] = move32(34, 0);
                s[n++] = move32(35, 0);
                s[n++] = move32(36, 0);
                s[n++] = move32(37, groups[0]);
                s[n++] = move32(38, groups[1]);
                s[n++] = move32(39, groups[2]);
                s[n++] = RUN_COMPUTE;
                s[n++] = WAIT_ALL;                                                  // in order, as Metal's encoders are
            }
            s[n++] = move32(74, 0);
            s[n++] = 36ULL << 56 | 74ULL << 40 | 2 << 8 | 1 << 4 | 1;          // FLUSH_CACHE2, as at the start
            s[n++] = WAIT_ALL;
            __asm__ volatile("dsb sy" ::: "memory");
            syncobj_create sc = {};
            sync_op signal = {};
            queue_submit qs = {};
            group_submit gs = {};
            std::lock_guard<std::mutex> g(d->lock);
            if (d->call(SYNCOBJ_CREATE, &sc, sizeof(sc)) != 0) {
                cb->error = "no sync object";
                cb->status = HAKIOS_METAL_ERROR;
            } else {
                cb->sync = sc.handle;
                signal = { 1u << 31, sc.handle, 0 };
                qs.stream_size = n * 8;
                qs.stream_addr = cb->stream.va;
                qs.syncs = { sizeof(signal), 1, (uint64_t)(uintptr_t)&signal };
                gs.group_handle = d->group;
                gs.queue_submits = { sizeof(qs), 1, (uint64_t)(uintptr_t)&qs };
                int submitted = d->call(GROUP_SUBMIT, &gs, sizeof(gs));
                cb->submitted_at = std::chrono::steady_clock::now();
                if (submitted != 0) {
                    cb->error = "HakiosGPU did not take the submission";
                    cb->status = HAKIOS_METAL_ERROR;
                } else {
                    cb->status = HAKIOS_METAL_SCHEDULED;
                }
            }
        }
    } else {
        // nothing recorded: done at once, in its turn
        syncobj_create sc = {};
        sc.flags = 1;           // signalled
        std::lock_guard<std::mutex> g(d->lock);
        if (d->call(SYNCOBJ_CREATE, &sc, sizeof(sc)) == 0) {
            cb->sync = sc.handle;
            cb->status = HAKIOS_METAL_SCHEDULED;
        } else {
            cb->status = HAKIOS_METAL_ERROR;
        }
    }
    // THE SCHEDULED HANDLERS NOW (11-10), the work submitted and not yet the queue thread's (completed comes after):
    // CoreAnimation presents its frame from one, and the framebuffer waits for what was submitted before the
    // presentation (MiniIOSurfaceDrawn). Run before the submission, as they were, the frame's own work was not among it:
    // the panel showed surfaces the Mali had not drawn yet -- icons missing, an older page through the new one, bands
    // of half-written tiles (the tester's slow-motion clips of 11-10, the App Library and sb_cube=1's cube)
    if (cb->scheduled != nullptr) {
        auto scheduled = cb->scheduled;
        cb->scheduled = nullptr;
        scheduled(cb, cb->scheduled_context);
    }
    {
        std::lock_guard<std::mutex> l(cb->queue->lock);
        cb->queue->pending.push_back(cb);
    }
    cb->queue->work.notify_all();
}

void
hakios_metal_cb_on_scheduled(hakios_metal_cb *cb, void (*scheduled)(hakios_metal_cb *, void *), void *context)
{
    cb->scheduled = scheduled;
    cb->scheduled_context = context;
}

void
hakios_metal_cb_wait(hakios_metal_cb *cb)
{
    std::unique_lock<std::mutex> l(cb->lock);
    cb->done.wait(l, [&] { return cb->status == HAKIOS_METAL_COMPLETED || cb->status == HAKIOS_METAL_ERROR ||
                                  cb->status == HAKIOS_METAL_NOT_ENQUEUED; });
}

enum hakios_metal_status
hakios_metal_cb_status(hakios_metal_cb *cb)
{
    return cb->status;
}

const char *
hakios_metal_cb_error(hakios_metal_cb *cb)
{
    return cb->error.c_str();
}

int64_t
hakios_metal_cb_gpu_nanoseconds(hakios_metal_cb *cb)
{
    (void)cb;
    return 0;
}

// ---- kernels (src/mali/compile_mali.c) and their dispatch ----

unsigned
hakios_metal_bitcode_constant_count(const void *bitcode, size_t size)
{
    return bitcode != nullptr && size != 0 ? mali_bitcode_constant_count(bitcode, size) : 0;
}

hakios_metal_kernel *
hakios_metal_kernel_compile_bitcode_constants(const void *bitcode, size_t size, const hakios_metal_constant *constants,
    unsigned count)
{
    std::vector<mali_constant> values(count);
    char why[512];

    for (unsigned i = 0; i < count; i++) {
        values[i] = { constants[i].index, constants[i].bits };
    }
    auto *k = new hakios_metal_kernel;
    if (mali_kernel_compile(bitcode, size, values.data(), count, &k->binary, why, sizeof(why)) != 0) {
        fail("%s", why);
        delete k;
        return nullptr;
    }
    return k;
}

hakios_metal_kernel *
hakios_metal_kernel_compile_bitcode(const void *bitcode, size_t size)
{
    return hakios_metal_kernel_compile_bitcode_constants(bitcode, size, nullptr, 0);
}

void
hakios_metal_kernel_free(hakios_metal_kernel *k)
{
    if (k != nullptr) {
        k->release();       // (a command buffer that dispatches it keeps it until the GPU is done)
    }
}

const char *
hakios_metal_kernel_name(hakios_metal_kernel *k)
{
    return k ? k->binary.name : "";
}

unsigned
hakios_metal_kernel_resource_count(hakios_metal_kernel *k)
{
    return k ? k->binary.num_resources : 0;
}

int
hakios_metal_kernel_resource(hakios_metal_kernel *k, unsigned i, unsigned *kind, unsigned *index, unsigned *size_bytes)
{
    if (k == nullptr || i >= k->binary.num_resources) {
        return -1;
    }
    const mali_kernel_resource &r = k->binary.resources[i];
    if (kind) {
        *kind = r.kind;
    }
    if (index) {
        *index = r.index;
    }
    if (size_bytes) {
        *size_bytes = r.size;
    }
    return 0;
}

void
hakios_metal_kernel_threadgroup_size(hakios_metal_kernel *k, unsigned size[3])
{
    for (int i = 0; i < 3; i++) {
        size[i] = k ? k->binary.workgroup[i] : 0;
    }
}

unsigned
hakios_metal_kernel_max_threads(hakios_metal_kernel *k)
{
    return k ? k->maxThreads() : 0;
}

unsigned
hakios_metal_kernel_execution_width(hakios_metal_kernel *k)
{
    (void)k;
    return 16;              // Valhall's warp
}

unsigned
hakios_metal_kernel_threadgroup_memory(hakios_metal_kernel *k)
{
    return k ? k->binary.threadgroup_bytes : 0;
}

namespace {
static void markBusy(hakios_metal_cb *cb, hakios_metal_device *d, hakios_metal_texture *t);

int
dispatchGrid(hakios_metal_cb *cb, hakios_metal_kernel *k, const unsigned groups[3], const unsigned threads[3],
    const unsigned grid[3], hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count,
    hakios_metal_texture *const *textures = nullptr, unsigned texture_count = 0,
    hakios_metal_sampler *const *samplers = nullptr, unsigned sampler_count = 0)
{
    Job j = {};
    unsigned total = 1;

    if (k == nullptr) {
        fail("dispatch: no kernel");
        return -1;
    }
    for (int i = 0; i < 3; i++) {
        if (threads[i] == 0 || threads[i] > 1024) {
            fail("dispatch of %s: a threadgroup of %u x %u x %u", k->binary.name, threads[0], threads[1], threads[2]);
            return -1;
        }
        if (k->binary.workgroup[i] != 0 && k->binary.workgroup[i] != threads[i]) {
            fail("dispatch of %s: the kernel's threadgroup is %u x %u x %u", k->binary.name, k->binary.workgroup[0],
                k->binary.workgroup[1], k->binary.workgroup[2]);
            return -1;
        }
        total *= threads[i];
    }
    if (total > k->maxThreads()) {
        fail("dispatch of %s: %u threads a threadgroup, %u at most", k->binary.name, total, k->maxThreads());
        return -1;
    }
    if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return 0;           // nothing to run
    }
    for (unsigned r = 0; r < k->binary.num_resources; r++) {
        const mali_kernel_resource &res = k->binary.resources[r];
        if (res.kind != HAKIOS_METAL_RESOURCE_CONSTANT_BUFFER && res.kind != HAKIOS_METAL_RESOURCE_DEVICE_BUFFER) {
            continue;
        }
        if (res.index >= count || buffers[res.index] == nullptr) {
            // (nothing bound: the device's zeroes stand for it)
            uint64_t nothing = nothingVa(cb->queue->device);
            if (nothing == 0) {
                fail("dispatch of %s: nothing bound at [[buffer(%u)]]", k->binary.name, res.index);
                return -1;
            }
            j.buffers[res.slot] = nothing;
            continue;
        }
        hakios_metal_buffer *b = buffers[res.index];
        uint64_t offset = offsets ? offsets[res.index] : 0;
        if (offset > b->bo.size || (res.size != 0 && offset + res.size > b->bo.size)) {
            fail("dispatch of %s: [[buffer(%u)]] at offset %llu is past its buffer", k->binary.name, res.index,
                (unsigned long long)offset);
            return -1;
        }
        j.buffers[res.slot] = b->bo.va + offset;        // (its slot in the push constants: the index, under 16)
    }
    // textures and samplers the kernel declares: bound, and of a kind it can use
    uint32_t nothing_at = 0;
    for (unsigned r = 0; r < k->binary.num_resources; r++) {
        const mali_kernel_resource &res = k->binary.resources[r];
        if (res.kind == HAKIOS_METAL_RESOURCE_TEXTURE || res.kind == HAKIOS_METAL_RESOURCE_IMAGE) {
            if (res.index >= texture_count || textures[res.index] == nullptr) {
                // (nothing bound: the device's 4x4 of zeroes stands for it)
                if (res.index >= MALI_KERNEL_TEXTURES || nothingTexture(cb->queue->device) == nullptr) {
                    fail("dispatch of %s: nothing bound at [[texture(%u)]]", k->binary.name, res.index);
                    return -1;
                }
                nothing_at |= 1u << res.index;
                j.texture_count = res.index + 1 > j.texture_count ? res.index + 1 : j.texture_count;
                continue;
            }
            if (res.kind == HAKIOS_METAL_RESOURCE_IMAGE) {
                markBusy(cb, cb->queue->device, textures[res.index]);   // (a kernel writing a surface: marked as a draw is)
            }
            if (res.kind == HAKIOS_METAL_RESOURCE_IMAGE && !(textures[res.index]->usage & 2)) {
                fail("dispatch of %s: [[texture(%u)]] is written and was made without MTLTextureUsageShaderWrite", k->binary.name,
                    res.index);
                return -1;
            }
            j.texture_count = res.index + 1 > j.texture_count ? res.index + 1 : j.texture_count;
        } else if (res.kind == HAKIOS_METAL_RESOURCE_SAMPLER) {
            if (res.index >= MALI_KERNEL_TEXTURES) {
                fail("dispatch of %s: nothing bound at [[sampler(%u)]]", k->binary.name, res.index);
                return -1;
            }
            if (res.index >= sampler_count || samplers[res.index] == nullptr) {
                // (nothing bound, 11-12: a nearest, clamped sampler stands for it -- as for a texel fetch below)
                j.samplers[res.index][0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
            }
            j.sampler_count = res.index + 1 > j.sampler_count ? res.index + 1 : j.sampler_count;
        }
    }
    if (j.texture_count != 0 && j.sampler_count == 0) {
        j.sampler_count = 1;                // (a texel fetch wants a valid sampler 0: a nearest, clamped one)
        j.samplers[0][0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    }
    for (unsigned t = 0; t < j.texture_count; t++) {
        j.textures[t] = t < texture_count && textures[t] != nullptr ? textures[t] :
            (nothing_at & 1u << t) ? nothingTexture(cb->queue->device) : nullptr;
        { hakios_metal_texture *rt_ = (j.textures[t]); while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent; if (rt_ != nullptr) rt_->reads++; }
    }
    for (unsigned t = 0; t < j.sampler_count && t < sampler_count; t++) {
        if (samplers[t] != nullptr) {
            memcpy(j.samplers[t], samplers[t]->words, 32);
        }
    }
    hakios_metal_device *d = cb->queue->device;
    if (!k->upload(d)) {
        return -1;
    }
    // threadgroup memory: the static arrays and the encoder's length, an instance per workgroup the cores can hold at once
    // (Mesa's pan_calc_wls_instances); spill space: every thread a core can run (pan_get_total_stack_size)
    uint32_t wls = k->binary.wls_size;
    if (k->binary.threadgroup_arg_base != ~0u && k->binary.threadgroup_arg_base + cb->threadgroup_length > wls) {
        wls = k->binary.threadgroup_arg_base + cb->threadgroup_length;     // the argument, past the static arrays
    }
    if (wls != 0) {
        uint32_t per_task = d->max_threads / (d->max_tasks ? d->max_tasks : 1);
        uint32_t wg_per_task = (per_task + total - 1) / total;
        uint32_t most = npot(wg_per_task * (d->max_tasks ? d->max_tasks : 1));
        uint32_t wanted = npot(groups[0]) * npot(groups[1]) * npot(groups[2]);
        j.wls_size = npot(wls < 128 ? 128 : wls);
        j.wls_instances = wanted < most ? wanted : most;
        if (!grow(d, d->wls, (uint64_t)j.wls_size * j.wls_instances * d->core_range)) {
            return -1;
        }
    }
    if (k->binary.tls_size != 0 &&
        !grow(d, d->tls, (uint64_t)npot((k->binary.tls_size + 15) & ~15u) * d->max_threads * d->core_range)) {
        return -1;
    }
    j.shader = DISPATCH;
    j.kernel = k;
    memcpy(j.groups, groups, sizeof(j.groups));
    memcpy(j.threads, threads, sizeof(j.threads));
    memcpy(j.grid, grid, sizeof(j.grid));
    cb->blits.push_back(j);
    k->retain();
    cb->kernels.push_back(k);
    for (unsigned r = 0; r < k->binary.num_resources; r++) {
        const mali_kernel_resource &res = k->binary.resources[r];
        if ((res.kind == HAKIOS_METAL_RESOURCE_CONSTANT_BUFFER || res.kind == HAKIOS_METAL_RESOURCE_DEVICE_BUFFER) &&
            res.index < count && buffers[res.index] != nullptr) {
            cb->hold(buffers[res.index]);
        }
    }
    for (unsigned t = 0; t < j.texture_count; t++) {
        if (j.textures[t] != nullptr) {
            j.textures[t]->retain();
            cb->held_textures.push_back(j.textures[t]);
        }
    }
    return 0;
}
}  // namespace

void
hakios_metal_cb_set_threadgroup_length(hakios_metal_cb *cb, unsigned length)
{
    cb->threadgroup_length = length;
}

int
hakios_metal_cb_dispatch(hakios_metal_cb *cb, hakios_metal_kernel *k, const unsigned groups[3], const unsigned threads[3],
    hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count)
{
    unsigned grid[3];

    for (int i = 0; i < 3; i++) {
        grid[i] = groups[i] * threads[i];
    }
    return dispatchGrid(cb, k, groups, threads, grid, buffers, offsets, count);
}

int
hakios_metal_cb_dispatch_threads(hakios_metal_cb *cb, hakios_metal_kernel *k, const unsigned grid[3], const unsigned threads[3],
    hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count)
{
    unsigned groups[3];

    for (int i = 0; i < 3; i++) {
        groups[i] = threads[i] ? (grid[i] + threads[i] - 1) / threads[i] : 0;
    }
    return dispatchGrid(cb, k, groups, threads, grid, buffers, offsets, count);
}

// ---- textures and samplers ----

namespace {
// BOOT-ARG metal_stats: a command buffer that took 40 ms or more from its commit to its completion, said pass by pass
// (each Mali pass: target size, draws, whether it preloads, its first program; the copies and dispatches between)
// (boot-arg metal_glitchsay: every target a finished command buffer drew into, sampled a row in 16: rows mostly pure
// blue or cyan -- the tester's video of a launch, 11-09 -- said with the command buffer's draws, once each up to 30)
void
glitchScan(hakios_metal_cb *cb, bool before)
{
    static const bool on = bootArg("metal_glitchsay");
    static std::atomic<unsigned> said{0}, mapped{0};
    if (!on || said >= 30) return;
    std::vector<hakios_metal_texture *> targets;
    for (const DrawJob &dj : cb->draws) {
        auto add = [&](hakios_metal_texture *t) {
            if (t != nullptr && t->bytes == 4 && t->samples == 1 && std::find(targets.begin(), targets.end(), t) == targets.end())
                targets.push_back(t);
        };
        add(dj.target_tex);
        for (unsigned i = 1; i < dj.rts; i++) if (!dj.more[i - 1].missing) add(dj.more[i - 1].texture);
    }
    std::string bad;
    std::vector<uint32_t> row;
    for (hakios_metal_texture *t : targets) {
        bool bgra = t->format == 80 || t->format == 81;
        unsigned rows_bad = 0, first = 0;
        row.resize(t->width);
        static const bool flush = bootArg("metal_glitchflush");    // (a test: the CPU's cached lines of it dropped first)
        for (unsigned y = 0; y < t->height; y += 16) {
            if (flush) sys_dcache_flush(t->pixels() + (uint64_t)y * t->stride, t->width * 4);
            if (hakios_metal_texture_read(t, 0, y, t->width, 1, row.data(), t->width * 4) != 0) break;
            unsigned n = 0;
            for (unsigned x = 0; x < t->width; x++) {
                uint32_t v = row[x];
                unsigned r = bgra ? (v >> 16) & 0xff : v & 0xff, g = (v >> 8) & 0xff, b = bgra ? v & 0xff : (v >> 16) & 0xff;
                n += (b == 255 && g < 8 && r < 8) || (b == 255 && g == 255 && r < 100);
            }
            if (n * 10 > t->width * 6) { if (rows_bad++ == 0) first = y; }
        }
        if (rows_bad != 0) {
            char line[160];
            snprintf(line, sizeof(line), " [target %p %ux%u format %u: %u rows blue/cyan from y %u]", (void *)t, t->width,
                t->height, t->format, rows_bad, first);
            bad += line;
            if (mapped++ < 2) {
                // a coarse map of it: k black, w white, b blue, c cyan, r red, g green, y yellow, m magenta, . other
                for (unsigned gy = 0; gy < 30; gy++) {
                    char cells[64];
                    unsigned y = gy * t->height / 30, n = 0;
                    if (hakios_metal_texture_read(t, 0, y, t->width, 1, row.data(), t->width * 4) != 0) break;
                    for (unsigned gx = 0; gx < 27; gx++) {
                        uint32_t v = row[gx * t->width / 27];
                        unsigned r = bgra ? (v >> 16) & 0xff : v & 0xff, g = (v >> 8) & 0xff, b = bgra ? v & 0xff : (v >> 16) & 0xff;
                        auto hi = [](unsigned c) { return c > 200; };
                        auto lo = [](unsigned c) { return c < 60; };
                        char ch = lo(r) && lo(g) && lo(b) ? 'k' : hi(r) && hi(g) && hi(b) ? 'w' : lo(r) && lo(g) && hi(b) ? 'b' :
                            lo(r) && hi(g) && hi(b) ? 'c' : hi(r) && lo(g) && lo(b) ? 'r' : lo(r) && hi(g) && lo(b) ? 'g' :
                            hi(r) && hi(g) && lo(b) ? 'y' : hi(r) && lo(g) && hi(b) ? 'm' : '.';
                        cells[n++] = ch;
                    }
                    cells[n] = 0;
                    fprintf(stderr, "hakios-metal: glitch map %p %s %2u %s\n", (void *)t, before ? "before" : "after", gy, cells);
                }
            }
        }
    }
    if (bad.empty()) return;
    said++;
    fprintf(stderr, "hakios-metal: glitch: a command buffer %s blue/cyan rows in%s; its draws:\n",
        before ? "FOUND, before it ran," : "left", bad.c_str());
    for (size_t i = 0; i < cb->draws.size(); i++) {
        const DrawJob &dj = cb->draws[i];
        std::string tex;
        for (unsigned k = 0; k < dj.texture_count; k++) {
            char one[64];
            hakios_metal_texture *x = dj.textures[k];
            snprintf(one, sizeof(one), " %ux%u/%u", x ? x->width : 0, x ? x->height : 0, x ? x->format : 0);
            tex += one;
        }
        fprintf(stderr, "hakios-metal: glitch:  %zu %s into %p %ux%u (rts %u, load %d, clear_only %d, zs %d), textures%s\n", i,
            dj.name, (void *)dj.target_tex, dj.width, dj.height, dj.rts, dj.load, dj.clear_only, dj.zs, tex.c_str());
    }
}

// BOOT-ARG metal_glitchwatch: every 1080x2340-class screen target (4 bytes a texel, drawn into by 20 command buffers)
// kept watched, scanned (a row in 64) at every commit and every completion of any command buffer, and at CPU uploads:
// the moment one turns blue/cyan is said with the event, the command buffer and its jobs (copies/fills with whether
// they land in it, draws with their attachments). metal_glitchsync too: a commit first waits (up to 200 ms) for every
// command buffer in flight, so "turned at the commit of N" means between N-1's completion and N's commit: no GPU work
// of ours -- a CPU write (ours, CoreAnimation's, another process's) or the kernel.
std::mutex gwLock;
std::vector<std::pair<hakios_metal_texture *, bool>> gwTargets;  // (retained) and whether it was blue at the last scan
std::atomic<unsigned> gwSeq{0}, gwInFlight{0};

unsigned
gwBlueRows(hakios_metal_texture *t, unsigned step, unsigned *first)
{
    bool bgra = t->format == 80 || t->format == 81;
    std::vector<uint32_t> row(t->width);
    unsigned bad = 0;
    for (unsigned y = 0; y < t->height; y += step) {
        if (hakios_metal_texture_read(t, 0, y, t->width, 1, row.data(), t->width * 4) != 0) break;
        unsigned n = 0;
        for (unsigned x = 0; x < t->width; x++) {
            uint32_t v = row[x];
            unsigned r = bgra ? (v >> 16) & 0xff : v & 0xff, g = (v >> 8) & 0xff, b = bgra ? v & 0xff : (v >> 16) & 0xff;
            n += (b == 255 && g < 8 && r < 8) || (b == 255 && g == 255 && r < 100);
        }
        if (n * 10 > t->width * 6 && bad++ == 0 && first) *first = y;
    }
    return bad;
}

// a coarse colour map of T, 27x30 cells: k black, w white, b blue, c cyan, r red, g green, y yellow, m magenta, . other
void
gwMap(hakios_metal_texture *t, const char *what)
{
    bool bgra = t->format == 80 || t->format == 81;
    std::vector<uint32_t> row(t->width);
    for (unsigned gy = 0; gy < 30; gy++) {
        char cells[64];
        unsigned y = gy * t->height / 30, n = 0;
        if (hakios_metal_texture_read(t, 0, y, t->width, 1, row.data(), t->width * 4) != 0) break;
        for (unsigned gx = 0; gx < 27; gx++) {
            uint32_t v = row[gx * t->width / 27];
            unsigned r = bgra ? (v >> 16) & 0xff : v & 0xff, g = (v >> 8) & 0xff, b = bgra ? v & 0xff : (v >> 16) & 0xff;
            auto hi = [](unsigned c) { return c > 200; };
            auto lo = [](unsigned c) { return c < 60; };
            cells[n++] = lo(r) && lo(g) && lo(b) ? 'k' : hi(r) && hi(g) && hi(b) ? 'w' : lo(r) && lo(g) && hi(b) ? 'b' :
                lo(r) && hi(g) && hi(b) ? 'c' : hi(r) && lo(g) && lo(b) ? 'r' : lo(r) && hi(g) && lo(b) ? 'g' :
                hi(r) && hi(g) && lo(b) ? 'y' : hi(r) && lo(g) && hi(b) ? 'm' : '.';
        }
        cells[n] = 0;
        uint32_t mid = row[t->width / 2];
        fprintf(stderr, "hakios-metal: glitchwatch: map %s %p %2u %s  (middle %08x)\n", what, (void *)t, gy, cells, mid);
    }
}

// a command buffer's jobs as lines (consecutive draws alike said once, with how many), marking what lands in T
std::vector<std::string>
gwDescribe(hakios_metal_cb *cb, hakios_metal_texture *t)
{
    std::vector<std::string> out;
    if (cb == nullptr) return out;
    uint64_t lo = t->gpu(), hi = lo + t->span();
    static const char *kinds[] = { "copy_u32", "fill_u32", "copy_u8", "fill_u8", "dispatch", "draw", "copy_rows" };
    std::string last;
    size_t last_at = 0, alike = 0;
    char text[700];
    auto flush = [&] {
        if (!last.empty()) {
            snprintf(text, sizeof(text), "job %zu%s draw %s", last_at,
                alike > 1 ? (" (x" + std::to_string(alike) + ")").c_str() : "", last.c_str());
            out.push_back(text);
        }
        last.clear();
        alike = 0;
    };
    for (size_t i = 0; i < cb->blits.size(); i++) {
        const Job &b = cb->blits[i];
        if (b.shader == DRAW) {
            if (b.dst >= cb->draws.size()) continue;
            const DrawJob &dj = cb->draws[b.dst];
            std::string more, tex;
            for (unsigned k = 1; k < dj.rts; k++) {
                const auto &m = dj.more[k - 1];
                char one[128];
                snprintf(one, sizeof(one), " rt%u %ux%u/%u%s%s%s blend %08x", k, m.texture ? m.texture->width : 0,
                    m.texture ? m.texture->height : 0, m.texture ? m.texture->format : 0, m.missing ? " missing" : "",
                    m.write ? " w" : "", m.va >= lo && m.va < hi ? " IN-IT" : "", m.blend[0]);
                more += one;
            }
            for (unsigned k = 0; k < dj.texture_count; k++) {
                char one[48];
                hakios_metal_texture *x = dj.textures[k];
                if (x == nullptr) continue;
                snprintf(one, sizeof(one), " %u:%ux%u/%u%s", k, x->width, x->height, x->format, x == t ? "(IT)" : "");
                tex += one;
            }
            char dtext[700];
            snprintf(dtext, sizeof(dtext), "%s (%p) into %ux%u%s load %d write %d blend %08x %08x count %u inst %u scissor %08x-%08x%s "
                "textures%s", dj.name, dj.program, dj.width, dj.height, dj.target_va >= lo && dj.target_va < hi ? " (IT)" : "", dj.load,
                dj.write, dj.blend[0], dj.blend[1], dj.count, dj.instances, dj.scissor_min, dj.scissor_max, more.c_str(),
                tex.c_str());
            if (last == dtext) { alike++; continue; }
            flush();
            last = dtext;
            last_at = i;
            alike = 1;
            continue;
        }
        flush();
        if (b.shader == DISPATCH) {
            bool in = false;
            for (unsigned k = 0; k < MALI_KERNEL_BUFFERS; k++) in |= b.buffers[k] >= lo && b.buffers[k] < hi;
            for (unsigned k = 0; k < b.texture_count; k++) in |= b.textures[k] == t;
            snprintf(text, sizeof(text), "job %zu dispatch %ux%ux%u%s", i, b.groups[0], b.groups[1], b.groups[2],
                in ? " (USES IT)" : "");
        } else {
            unsigned unit = b.shader <= 1 ? 4 : 1;
            bool in = b.dst < hi && b.dst + (uint64_t)b.count * unit > lo;
            snprintf(text, sizeof(text), "job %zu %s dst 0x%llx count %u value %08x src 0x%llx%s", i,
                kinds[b.shader < 7 ? b.shader : 5], (unsigned long long)b.dst, b.count, b.value,
                (unsigned long long)b.src, in ? " (INTO IT)" : "");
        }
        out.push_back(text);
    }
    flush();
    return out;
}

void
glitchWatch(const char *event, hakios_metal_cb *cb)
{
    static const bool on = bootArg("metal_glitchwatch");
    static std::atomic<unsigned> said{0};
    if (!on) return;
    std::lock_guard<std::mutex> g(gwLock);
    if (cb != nullptr) {
        // screen targets: drawn into by 20 command buffers or more (the screen's drawables, not the launch snapshots);
        // the six most recently drawn kept
        static std::map<hakios_metal_texture *, unsigned> seen;
        std::vector<hakios_metal_texture *> mine;
        for (const DrawJob &dj : cb->draws) {
            hakios_metal_texture *t = dj.target_tex;
            if (t == nullptr || t->bytes != 4 || t->samples != 1 || t->width < 1000 || t->height < 2000) continue;
            if (std::find(mine.begin(), mine.end(), t) == mine.end()) mine.push_back(t);
        }
        for (hakios_metal_texture *t : mine) {
            auto at = std::find_if(gwTargets.begin(), gwTargets.end(), [&](auto &w) { return w.first == t; });
            if (at != gwTargets.end()) {
                auto keep = *at;                    // most recently drawn last
                gwTargets.erase(at);
                gwTargets.push_back(keep);
                continue;
            }
            if (++seen[t] < 20) continue;
            if (seen.size() > 256) seen.clear();
            t->retain();
            gwTargets.push_back({ t, false });
            fprintf(stderr, "hakios-metal: glitchwatch: watching target %p %ux%u format %u (command buffer #%u)\n", (void *)t,
                t->width, t->height, t->format, cb->seq);
            if (gwTargets.size() > 6) {
                gwTargets.front().first->release();
                gwTargets.erase(gwTargets.begin());
            }
        }
    }
    // this event in a line: the command buffer, the targets it draws into (with its draw count into each), and each
    // watched target's sampled blue rows now; the last 16 lines said when a target changes, and the 8 after it
    static std::deque<std::string> history;
    static unsigned after = 0;
    std::string line;
    char one[160];
    snprintf(one, sizeof(one), "%s #%u (in flight %u, %ld us):", event, cb ? cb->seq : 0, gwInFlight.load(),
        (long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() % 100000000);
    line = one;
    if (cb != nullptr) {
        std::vector<std::pair<hakios_metal_texture *, unsigned>> into;
        for (const DrawJob &dj : cb->draws) {
            auto at = std::find_if(into.begin(), into.end(), [&](auto &x) { return x.first == dj.target_tex; });
            if (at == into.end()) into.push_back({ dj.target_tex, 1 }); else at->second++;
        }
        for (auto &x : into) {
            snprintf(one, sizeof(one), " %p%s*%u", (void *)x.first,
                x.first && x.first->width >= 1000 ? "(screen)" : "", x.second);
            line += one;
        }
        if (cb->blits.size() != cb->draws.size()) {
            snprintf(one, sizeof(one), " +%zu copies/dispatches", cb->blits.size() - cb->draws.size());
            line += one;
        }
    }
    line += " | watched:";
    bool changed = false;
    std::vector<std::pair<hakios_metal_texture *, unsigned>> turned;
    static std::map<hakios_metal_texture *, std::vector<std::string>> before;  // the last clean frame's jobs
    for (auto &w : gwTargets) {
        unsigned first = 0, bad = gwBlueRows(w.first, 64, &first);
        bool blue = bad * 64 >= 512;     // 8 sampled rows (about a fifth of the screen) or more
        snprintf(one, sizeof(one), " %p=%u", (void *)w.first, bad);
        line += one;
        bool draws_it = false;
        if (cb != nullptr && !strcmp(event, "completion")) {
            for (const DrawJob &dj : cb->draws) draws_it |= dj.target_tex == w.first;
        }
        if (blue == w.second) {
            if (draws_it && !blue && said < 40) before[w.first] = gwDescribe(cb, w.first);
            continue;
        }
        w.second = blue;
        changed = true;
        if (blue) turned.push_back({ w.first, first });
    }
    history.push_back(line);
    if (history.size() > 16) history.pop_front();
    if (said >= 40) return;
    if (changed) {
        said++;
        fprintf(stderr, "hakios-metal: glitchwatch: a watched target changed (sampled blue/cyan rows of 37 per target); "
            "the events up to it:\n");
        for (auto &h : history) fprintf(stderr, "hakios-metal: glitchwatch:   %s\n", h.c_str());
        for (auto &x : turned) {
            fprintf(stderr, "hakios-metal: glitchwatch: target %p TURNED BLUE/CYAN (from y %u) at the %s of command buffer #%u; "
                "its jobs:\n", (void *)x.first, x.second, event, cb ? cb->seq : 0);
            for (auto &l : gwDescribe(cb, x.first)) fprintf(stderr, "hakios-metal: glitchwatch:  now %s\n", l.c_str());
            for (auto &l : before[x.first]) fprintf(stderr, "hakios-metal: glitchwatch:  last clean frame %s\n", l.c_str());
            // the large textures its draws sample, as the CPU reads them now: blue there too (the data), or not (the
            // GPU's reading of it)
            std::vector<hakios_metal_texture *> sampled;
            if (cb != nullptr) {
                for (const DrawJob &dj : cb->draws) {
                    for (unsigned k = 0; k < dj.texture_count; k++) {
                        hakios_metal_texture *s2 = dj.textures[k];
                        if (s2 && s2->bytes == 4 && s2->width >= 500 && s2->height >= 500 && s2->levels == 1 &&
                            std::find(sampled.begin(), sampled.end(), s2) == sampled.end()) sampled.push_back(s2);
                    }
                }
            }
            for (hakios_metal_texture *s2 : sampled) {
                unsigned f2 = 0, b2 = gwBlueRows(s2, 64, &f2);
                fprintf(stderr, "hakios-metal: glitchwatch: sampled texture %p %ux%u format %u origin %u surface %u: %u sampled rows "
                    "blue/cyan\n", (void *)s2, s2->width, s2->height, s2->format, s2->origin, s2->surface, b2);
                if (said <= 3) gwMap(s2, "sampled");
            }
            if (said <= 3) gwMap(x.first, "target");
        }
        after = 8;
    } else if (after != 0) {
        after--;
        fprintf(stderr, "hakios-metal: glitchwatch:   then %s\n", line.c_str());
    }
}

void
glitchWatchCommit(hakios_metal_cb *cb)
{
    static const bool on = bootArg("metal_glitchwatch"), sync = bootArg("metal_glitchsync");
    if (!on) return;
    cb->seq = ++gwSeq;
    if (sync) {
        auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        while (gwInFlight.load() != 0 && std::chrono::steady_clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    glitchWatch("commit", cb);
    gwInFlight++;
}

void
glitchWatchDone()
{
    static const bool on = bootArg("metal_glitchwatch");
    if (on) gwInFlight--;
}

void
passStats(hakios_metal_cb *cb)
{
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cb->committed_at).count();
    // metal_stats_ms=N (11-10): the threshold, 40 ms by default -- Spotlight's opening has buffers of 23-28 ms; at most
    // 300 said
    static double limit = -1;
    static std::atomic<unsigned> said{0};
    if (limit < 0) {
        char args[1024] = {0};
        size_t size = sizeof(args) - 1;
        const char *at;
        limit = 40;
        if (sysctlbyname("kern.bootargs", args, &size, nullptr, 0) == 0 && (at = strstr(args, "metal_stats_ms=")) != nullptr) {
            limit = atof(at + 15);
        } else if ((at = strstr(args, "mdsay=")) != nullptr) {
            limit = atof(at + 6);                       // (mdsay=N: the threshold, and each slow pass's draws -- short:
        }                                               //  the command line holds 490 characters)
    }
    if (ms < limit || said >= 300) {
        return;
    }
    said++;
    double gpu_sum = 0;
    unsigned passes = 0, draws = 0, loads = 0, other = 0;
    uint64_t pixels = 0;
    std::string list;
    for (size_t i = 0; i < cb->blits.size(); i++) {
        const Job &b = cb->blits[i];
        if (b.shader != DRAW) {
            other++;
            continue;
        }
        const DrawJob &dj = cb->draws[b.dst];
        size_t k = i;
        unsigned n = 1;
        while (k + 1 < cb->blits.size() && cb->blits[k + 1].shader == DRAW && samePass(cb->draws[cb->blits[k].dst],
                                                                                         cb->draws[cb->blits[k + 1].dst])) {
            k++;
            n++;
        }
        // why the next draw (into the same target) opened a pass of its own
        const char *why = "";
        if (k + 1 < cb->blits.size() && cb->blits[k + 1].shader == DRAW) {
            const DrawJob &a = cb->draws[cb->blits[k].dst], &c = cb->draws[cb->blits[k + 1].dst];
            if (a.target_va == c.target_va) {
                why = c.clear_only || a.clear_only ? "clear" : !c.load ? "no load" : a.rts != c.rts ? "targets" :
                    a.zs != c.zs ? "zs" : "samples a target";
                for (unsigned m = 1; m < a.rts && m < c.rts; m++) {
                    if (a.more[m - 1].va != c.more[m - 1].va) why = "other target n";
                    else if (a.more[m - 1].missing != c.more[m - 1].missing) why = "missing n";
                    else if (!c.more[m - 1].missing && !c.more[m - 1].load) why = "target n not loaded";
                }
            }
        }
        unsigned touch1 = 0;                            // (draws of the pass that write or read attachment 1)
        for (size_t j = i; j <= k; j++) {
            const DrawJob &x = cb->draws[cb->blits[j].dst];
            touch1 += x.rts > 1 && !x.more[0].missing && !x.more[0].untouched;
        }
        double gpu_us = -1;
        if (cb->timestamps.size != 0 && 2 * passes + 1 < cb->timestamp_count) {
            auto *ts0 = (volatile uint64_t *)cb->timestamps.cpu;
            double us = (double)(ts0[2 * passes + 1] - ts0[2 * passes]) / 24.576;
            static const bool draws_say = bootArg("mdsay=");
            if (draws_say && us >= limit * 1000 * 0.5) {
                // (boot-arg mdsay=MS, 11-11: a slow pass's draws -- each one's area (its scissor), vertices,
                // program, the textures it samples (size, format, stride; an IOSurface's ID) and its flags)
                std::string d;
                for (size_t j = i; j <= k && d.size() < 3000; j++) {
                    const DrawJob &x = cb->draws[cb->blits[j].dst];
                    unsigned w = (x.scissor_max & 0xffff) - (x.scissor_min & 0xffff) + 1;
                    unsigned h = (x.scissor_max >> 16) - (x.scissor_min >> 16) + 1;
                    char one[400];
                    const hakios_metal_program *xp = (const hakios_metal_program *)x.program;
                    const char *lab = xp ? xp->label : "";
                    if (strncmp(lab, "com.apple.coreanimation.draw.", 29) == 0) lab += 29;
                    int at = snprintf(one, sizeof(one), " {%ux%u@%u,%u %uv %s %s fs %u dcd0 %x blend %x", w, h,
                        x.scissor_min & 0xffff, x.scissor_min >> 16, x.count * (x.instances ? x.instances : 1),
                        x.name ? x.name + (strlen(x.name) > 12 ? strlen(x.name) - 12 : 0) : "", lab, xp ? xp->fs.size : 0,
                        x.dcd0, x.blend[0]);
                    for (unsigned t = 0; t < x.texture_count && at < (int)sizeof(one) - 60; t++) {
                        const hakios_metal_texture *tx = x.textures[t];
                        if (tx) at += snprintf(one + at, sizeof(one) - at, " T%ux%u f%u s%u%s", tx->width, tx->height, tx->format,
                            tx->stride, tx->surface ? " ios" : "");
                    }
                    snprintf(one + at, sizeof(one) - at, "}");
                    d += one;
                }
                const DrawJob &h0 = cb->draws[cb->blits[i].dst];
                const hakios_metal_texture *tt = h0.target_tex;
                fprintf(stderr, "hakios-metal: slow pass %.0f us, %u draws into %ux%u f%u s%u origin %u surface %u%s%s%s at %llx:%s\n",
                    us, n, tt ? tt->width : 0, tt ? tt->height : 0, tt ? tt->format : 0, tt ? tt->stride : 0, tt ? tt->origin : 0,
                    tt ? tt->surface : 0, tt && tt->mirrored ? " mirrored" : "", h0.load ? " L" : "", h0.write ? " W" : "",
                    (unsigned long long)h0.target_va, d.c_str());
            }
        }
        if (cb->timestamps.size != 0 && 2 * passes + 1 < cb->timestamp_count) {
            auto *ts = (volatile uint64_t *)cb->timestamps.cpu;
            gpu_us = (double)(ts[2 * passes + 1] - ts[2 * passes]) / 24.576;    // (the system counter: 24.576 MHz)
            gpu_sum += gpu_us;
        }
        passes++;
        draws += n;
        loads += dj.load;
        pixels += (uint64_t)dj.width * dj.height;
        if (list.size() < 2500) {
            char one[400];
            char a1[160] = "";
            if (dj.rts > 1 && dj.more[0].texture != nullptr && !dj.more[0].missing) {
                const hakios_metal_texture *x = dj.more[0].texture;
                int at = snprintf(a1, sizeof(a1), " a1 fmt %u %ux%u: attachment %u times, READ %u @%llx", x->format, x->width,
                    x->height, x->as_extra_rt.load(), x->reads.load(), (unsigned long long)(dj.more[0].va & 0xffffffffff));
                if (dj.rts > 2 && dj.more[1].texture != nullptr && !dj.more[1].missing) {
                    const hakios_metal_texture *y = dj.more[1].texture;
                    snprintf(a1 + at, sizeof(a1) - at, " a2 %c fmt %u %ux%u @%llx", dj.more[1].memoryless ? 't' : dj.more[1].load ?
                        'l' : dj.more[1].write ? 'w' : '-', y->format, y->width, y->height,
                        (unsigned long long)(dj.more[1].va & 0xffffffffff));
                }
            }
            snprintf(one, sizeof(one), " [%.0fus %ux%u %u%s%s %s rts %u%s t%u%s%s%s]", gpu_us, dj.width, dj.height, n, dj.load ? " L" : "",
                dj.clear_only ? " C" : "", dj.name ? dj.name : "", dj.rts,
                dj.rts > 1 ? (dj.more[0].missing ? "m" : dj.more[0].memoryless ? "t" : dj.more[0].elide ? "e" : dj.more[0].load ? "l" :
                              dj.more[0].write ? "w" : "-") : "",
                touch1, a1, why[0] ? " | " : "", why);
            list += one;
        }
        i = k;
    }
    fprintf(stderr, "hakios-metal: stats: %.1f ms (the passes' GPU time %.1f ms): %u Mali passes (%u preload), %u draws, %u "
        "copies/dispatches, %.1f MP of targets:%s\n", ms, gpu_sum / 1e3, passes, loads, draws, other, pixels / 1e6, list.c_str());
}

// a word in the boot-args (testing switches: metal_nozs)
bool
bootArg(const char *word)
{
    char args[1024] = {0};
    size_t size = sizeof(args) - 1;
    return sysctlbyname("kern.bootargs", args, &size, nullptr, 0) == 0 && strstr(args, word) != nullptr;
}
}  // namespace

hakios_metal_texture *
hakios_metal_texture_new(hakios_metal_device *d, unsigned type, unsigned pixel_format, unsigned width, unsigned height,
    unsigned usage)
{
    static const bool no_zs = bootArg("metal_nozs");
    if (no_zs && pixel_format >= 250 && pixel_format <= 261) {
        fail("a depth or stencil texture of MTLPixelFormat %u (boot-arg metal_nozs)", pixel_format);
        return nullptr;
    }
    const MaliFormat *f = nullptr;
    for (const MaliFormat &m : kFormats) {
        if (m.metal == pixel_format) {
            f = &m;
        }
    }
    if (f == nullptr) {
        fail("a texture of MTLPixelFormat %u (not served on the Mali yet)", pixel_format);
        return nullptr;
    }
    if ((type != 2 && !(type == 0 && height == 1)) || width == 0 || height == 0 || width > 16384 || height > 16384) {
        fail("a texture of type %u, %u x %u (the Mali build makes 2D and 1D ones up to 16384)", type, width, height);
        return nullptr;
    }
    auto *t = new hakios_metal_texture;
    t->device = d;
    t->type = type;
    t->format = pixel_format;
    t->width = width;
    t->height = height;
    t->depth = 1;
    t->usage = usage;
    t->mali_format = f->mali;
    if (f->metal == 1) t->swizzle = kAlphaSwizzle;
    t->bytes = f->bytes;
    t->bw = f->bw;
    t->bh = f->bh;
    t->stride = ((width + f->bw - 1) / f->bw * f->bytes + 63) & ~63u;
    uint64_t size = (uint64_t)t->stride * t->levelRows(0);
    if (pixel_format == 260) {
        t->stencil_off = (size + 63) & ~63ULL;
        t->stencil_stride = (width + 63) & ~63u;
        size = t->stencil_off + (uint64_t)t->stencil_stride * height;
    }
    if (!d->newBo(t->bo, size, BIND_NOEXEC)) {
        delete t;
        return nullptr;
    }
    return t;
}

void
hakios_metal_texture_free(hakios_metal_texture *t)
{
    if (t != nullptr) {
        t->release();       // (a command buffer that uses it keeps it until the GPU is done)
    }
}

void *hakios_metal_texture_contents(hakios_metal_texture *t) { return t ? t->pixels() : nullptr; }
uint64_t hakios_metal_texture_bytes_per_row(hakios_metal_texture *t) { return t ? t->stride : 0; }
uint64_t hakios_metal_texture_gpu_address(hakios_metal_texture *t) { return t ? t->gpu() : 0; }
unsigned hakios_metal_texture_width(hakios_metal_texture *t) { return t ? t->width : 0; }
unsigned hakios_metal_texture_height(hakios_metal_texture *t) { return t ? t->height : 0; }
unsigned hakios_metal_texture_format(hakios_metal_texture *t) { return t ? t->format : 0; }
unsigned hakios_metal_texture_type(hakios_metal_texture *t) { return t ? t->type : 0; }
unsigned hakios_metal_texture_usage(hakios_metal_texture *t) { return t ? t->usage : 0; }

namespace {
int
copyRows(hakios_metal_texture *t, unsigned x, unsigned y, unsigned w, unsigned h, uint8_t *bytes, uint64_t bytes_per_row,
    bool in, unsigned level = 0)
{
    if (t == nullptr || level >= t->levels || x + w > t->levelWidth(level) || y + h > t->levelHeight(level) ||
        bytes_per_row < (uint64_t)(w + t->bw - 1) / t->bw * t->bytes) {
        fail("a texture region outside the texture (level %u of %u), or rows shorter than its width", level,
            t ? t->levels : 0);
        return -1;
    }
    uint8_t *base = t->pixels() + t->levelOffset(level);
    uint32_t stride = t->levelStride(level);
    {
        // (11-13) THE TEXTURE'S OWN MEMORY: a copy that would run past it is refused with its numbers -- Weather died in
        // this memcpy as its radar map came up, at the first address after a mapping
        const uint64_t rows = (uint64_t)(y + h + t->bh - 1) / t->bh, cols = (uint64_t)(x + w + t->bw - 1) / t->bw;
        const uint64_t end = t->offset + t->levelOffset(level) + (rows ? rows - 1 : 0) * stride + cols * t->bytes;
        if (h != 0 && w != 0 && end > t->bo.size) {
            fail("a texture copy past the texture's memory: %ux%u at (%u, %u) of level %u, MTLPixelFormat %u, rows %u bytes "
                "apart from %llu + %llu: it would end at %llu of %llu bytes", w, h, x, y, level, t->format, stride,
                (unsigned long long)t->offset, (unsigned long long)t->levelOffset(level), (unsigned long long)end,
                (unsigned long long)t->bo.size);
            return -1;
        }
    }
    if (t->bw > 1 || t->bh > 1) {
        // a compressed format: the region in whole blocks (its edge blocks may be partly past the level), rows of blocks
        if (x % t->bw || y % t->bh) {
            fail("a compressed texture region not on its %ux%u blocks", t->bw, t->bh);
            return -1;
        }
        x /= t->bw;
        y /= t->bh;
        w = (w + t->bw - 1) / t->bw;
        h = (h + t->bh - 1) / t->bh;
    }
    for (unsigned row = 0; row < h; row++) {
        uint8_t *gpu = base + (uint64_t)(y + row) * stride + (uint64_t)x * t->bytes;
        uint8_t *cpu = bytes + row * bytes_per_row;
        if (in) {
            memcpy(gpu, cpu, (size_t)w * t->bytes);
        } else {
            memcpy(cpu, gpu, (size_t)w * t->bytes);
        }
    }
    return 0;
}
}  // namespace

int
hakios_metal_texture_replace(hakios_metal_texture *t, unsigned x, unsigned y, unsigned w, unsigned h, const void *bytes,
    uint64_t bytes_per_row)
{
    int r = copyRows(t, x, y, w, h, (uint8_t *)bytes, bytes_per_row, true);
    glitchWatch("CPU upload (texture replace)", nullptr);
    // (boot-arg metal_glitchsay: what a CPU upload of 64 rows or more brings that is mostly pure blue or cyan, with the
    // caller -- the screen surface was found blue BEFORE any command buffer drew it, 11-09)
    static const bool say = bootArg("metal_glitchsay");
    static std::atomic<unsigned> said{0};
    if (say && r == 0 && t != nullptr && t->bytes == 4 && h >= 64 && w >= 64 && said < 20) {
        bool bgra = t->format == 80 || t->format == 81;
        unsigned bad = 0, rows = 0;
        for (unsigned j = 0; j < h; j += 16, rows++) {
            const uint32_t *row = (const uint32_t *)((const uint8_t *)bytes + (uint64_t)j * bytes_per_row);
            unsigned n = 0;
            for (unsigned i = 0; i < w; i++) {
                uint32_t v = row[i];
                unsigned rr = bgra ? (v >> 16) & 0xff : v & 0xff, g = (v >> 8) & 0xff, b = bgra ? v & 0xff : (v >> 16) & 0xff;
                n += (b == 255 && g < 8 && rr < 8) || (b == 255 && g == 255 && rr < 100);
            }
            bad += n * 10 > w * 6;
        }
        if (bad * 2 > rows) {
            said++;
            Dl_info info = {};
            void *caller = __builtin_return_address(0);
            dladdr(caller, &info);
            fprintf(stderr, "hakios-metal: glitch: a CPU upload of %ux%u at (%u,%u) into texture %p %ux%u format %u is blue/cyan "
                "(%u of %u rows), from %s+%#lx\n", w, h, x, y, (void *)t, t->width, t->height, t->format, bad, rows,
                info.dli_sname ? info.dli_sname : "?", (unsigned long)((uintptr_t)caller - (uintptr_t)info.dli_saddr));
        }
    }
    return r;
}

int
hakios_metal_texture_read(hakios_metal_texture *t, unsigned x, unsigned y, unsigned w, unsigned h, void *bytes,
    uint64_t bytes_per_row)
{
    { hakios_metal_texture *rt_ = (t); while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent; if (rt_ != nullptr) rt_->reads++; }
    return copyRows(t, x, y, w, h, (uint8_t *)bytes, bytes_per_row, false);
}

// MTLSamplerDescriptor as Mali's sampler (genxml v10): wraps, filters, normalized coordinates
hakios_metal_sampler *
hakios_metal_sampler_new(hakios_metal_device *d, const struct hakios_metal_sampler_desc *desc)
{
    static const uint32_t wraps[6] = { 9, 13, 8, 12, 11, 11 };     // Metal's address modes -> Mali's wrap modes
    (void)d;
    if (desc->s_address > 5 || desc->t_address > 5 || desc->r_address > 5) {
        fail("a sampler with an address mode past ClampToBorderColor");
        return nullptr;
    }
    auto *s = new hakios_metal_sampler;
    memset(s->words, 0, sizeof(s->words));
    s->words[0] = 1 | wraps[desc->r_address] << 8 | wraps[desc->t_address] << 12 | wraps[desc->s_address] << 16 | 1u << 21 |
        1u << 23 | (desc->normalized_coordinates ? 1u << 25 : 0) | 1u << 26 | (desc->min_filter == 0 ? 1u << 27 : 0) |
        (desc->mag_filter == 0 ? 1u << 28 : 0);
    // the levels: Metal's mip filter as the mipmap mode (NotMipmapped: none, level min_lod alone; Nearest; Linear:
    // trilinear) and its LOD clamps, 8.8 fixed point (13 bits: up to 31.99)
    auto ulod = [](float v) { return (uint32_t)(v <= 0 ? 0 : v >= 31.99f ? 0x1fff : v * 256); };
    uint32_t lo = ulod(desc->lod_min_clamp), hi = desc->mip_filter == 0 ? lo : ulod(desc->lod_max_clamp);
    s->words[0] |= (desc->mip_filter == 0 ? 1u : desc->mip_filter == 1 ? 0u : 3u) << 30;
    s->words[1] = lo | (hi < lo ? lo : hi) << 16;
    if (desc->s_address == 4 || desc->t_address == 4 || desc->r_address == 4 || desc->border_color == 0) {
        // ClampToZero, and the transparent black border: zeros (the border colour's words stay 0)
    } else if (desc->border_color == 1) {
        s->words[7] = 0x3f800000;                               // opaque black: alpha 1.0
    } else {
        s->words[4] = s->words[5] = s->words[6] = s->words[7] = 0x3f800000;    // opaque white
    }
    return s;
}

void hakios_metal_sampler_free(hakios_metal_sampler *s) { delete s; }
const uint32_t *hakios_metal_sampler_words(hakios_metal_sampler *s) { return s ? s->words : nullptr; }

int
hakios_metal_cb_dispatch2(hakios_metal_cb *cb, hakios_metal_kernel *k, const unsigned groups[3], const unsigned threads[3],
    hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count, hakios_metal_texture *const *textures,
    unsigned texture_count, hakios_metal_sampler *const *samplers, unsigned sampler_count)
{
    unsigned grid[3];

    for (int i = 0; i < 3; i++) {
        grid[i] = groups[i] * threads[i];
    }
    return dispatchGrid(cb, k, groups, threads, grid, buffers, offsets, count, textures, texture_count, samplers, sampler_count);
}

int
hakios_metal_cb_dispatch_threads2(hakios_metal_cb *cb, hakios_metal_kernel *k, const unsigned grid[3], const unsigned threads[3],
    hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count, hakios_metal_texture *const *textures,
    unsigned texture_count, hakios_metal_sampler *const *samplers, unsigned sampler_count)
{
    unsigned groups[3];

    for (int i = 0; i < 3; i++) {
        groups[i] = threads[i] ? (grid[i] + threads[i] - 1) / threads[i] : 0;
    }
    return dispatchGrid(cb, k, groups, threads, grid, buffers, offsets, count, textures, texture_count, samplers, sampler_count);
}

// ---- programs and draws ----

// ---- the compiled programs kept on disk ----
//
// A program compiled once is not compiled again at the next start: CoreAnimation builds its pipelines at every start,
// and its uber shaders take 14 to 27 s of the phone's processor each (fixed_frag_lph_cpf, for every pixel format) --
// minutes of a busy, warming phone at every boot. Both stages are kept in a file of their own under
// /private/var/mobile/Library/Caches/com.hakios.metal-mali, named for a hash of what decides them: both stages'
// bitcode and constants, the colour format, the vertex descriptor's fetch (when it is compiled with one) and this
// library's own image UUID -- a rebuilt compiler or front end is another image, and another name. Written to a
// temporary file and renamed, so a file is whole or absent. Boot-arg metal_cache=0: not kept, not read.
namespace {
constexpr char kCacheMagic[8] = { 'H', 'K', 'M', 'A', 'L', 'I', '0', '1' };

bool
cacheOn()
{
    static int on = -1;
    if (on < 0) {
        char args[1024] = {};
        size_t n = sizeof(args) - 1;
        on = !(sysctlbyname("kern.bootargs", args, &n, nullptr, 0) == 0 && strstr(args, "metal_cache=0") != nullptr);
    }
    return on != 0;
}

// THE CACHE'S SIZE (TODO 17): a key holds this library's build (fnvImage), so every new build leaves the old entries
// behind for good -- on a test phone each build. Over CACHE_MOST bytes, the least recently used files (their mtime:
// cacheLoad touches what it reads) go until CACHE_LEAST remain. Once a process, in the background.
constexpr uint64_t CACHE_MOST = 64ull << 20, CACHE_LEAST = 48ull << 20;

void
cacheTrim(const std::string &dir)
{
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) return;
    struct Entry { std::string path; time_t mtime; uint64_t size; };
    std::vector<Entry> files;
    uint64_t total = 0;
    while (struct dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string p = dir + "/" + e->d_name;
        struct stat st;
        if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.push_back({ p, st.st_mtime, (uint64_t)st.st_size });
            total += (uint64_t)st.st_size;
        }
    }
    closedir(d);
    if (total <= CACHE_MOST) return;
    std::sort(files.begin(), files.end(), [](const Entry &a, const Entry &b) { return a.mtime < b.mtime; });
    unsigned removed = 0;
    uint64_t was = total;
    for (const Entry &f : files) {
        if (total <= CACHE_LEAST) break;
        if (unlink(f.path.c_str()) == 0) {
            total -= f.size;
            removed++;
        }
    }
    fprintf(stderr, "hakios-metal: the shader cache was %llu KB: %u of its oldest files removed, %llu KB left\n",
        (unsigned long long)(was >> 10), removed, (unsigned long long)(total >> 10));
}

const std::string &
cacheDir()
{
    static std::string dir;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *tries[] = { "/private/var/mobile/Library/Caches/com.hakios.metal-mali", "/private/var/tmp/com.hakios.metal-mali" };
        for (const char *t : tries) {
            mkdir(t, 0755);
            struct stat st;
            if (stat(t, &st) == 0 && S_ISDIR(st.st_mode) && access(t, W_OK) == 0) {
                dir = t;
                break;
            }
        }
        if (!dir.empty()) {
            std::string d = dir;
            std::thread([d] { cacheTrim(d); }).detach();
        }
    });
    return dir;
}

void
fnv(uint64_t &h, const void *data, size_t n)
{
    auto *b = (const uint8_t *)data;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
}

// this library's image UUID (LC_UUID): another build, another key
void
fnvImage(uint64_t &h)
{
    static uint8_t uuid[16];
    static bool have = false;
    static std::once_flag once;
    std::call_once(once, [] {
        Dl_info info;
        if (dladdr((const void *)&fnvImage, &info) == 0 || info.dli_fbase == nullptr) {
            return;
        }
        auto *mh = (const struct mach_header_64 *)info.dli_fbase;
        auto *lc = (const struct load_command *)(mh + 1);
        for (uint32_t i = 0; i < mh->ncmds; i++, lc = (const struct load_command *)((const uint8_t *)lc + lc->cmdsize)) {
            if (lc->cmd == LC_UUID) {
                memcpy(uuid, ((const struct uuid_command *)lc)->uuid, 16);
                have = true;
                break;
            }
        }
    });
    if (have) {
        fnv(h, uuid, 16);
    }
}

const uint32_t sizeof_binary = sizeof(mali_kernel_binary);

uint64_t
cacheKey(const void *vs, size_t vs_size, const std::vector<mali_constant> &vc, const void *fs, size_t fs_size,
    const std::vector<mali_constant> &fc, unsigned format, const mali_vertex_fetch *fetch,
    const mali_shader_blend *blend = nullptr)
{
    uint64_t h = 14695981039346656037ULL;
    fnv(h, kCacheMagic, sizeof(kCacheMagic));
#ifdef HAKIOS_MALI_CACHE_BUILD
    // the compiler's own make-up (the Makefile's MALI_CACHE_BUILD), not this library's build: see there
    fnv(h, HAKIOS_MALI_CACHE_BUILD, sizeof(HAKIOS_MALI_CACHE_BUILD) - 1);
    fnv(h, &sizeof_binary, sizeof(sizeof_binary));
    {
        // (and the compile options the boot-args choose: compile_mali.c's mali_mediump=0)
        static const uint32_t no_mediump = bootArg("mali_mediump=0") ? 1 : 0;
        fnv(h, &no_mediump, sizeof(no_mediump));
    }
#else
    fnvImage(h);
#endif
    fnv(h, &vs_size, sizeof(vs_size));
    fnv(h, vs, vs_size);
    for (const mali_constant &k : vc) fnv(h, &k, sizeof(k));
    fnv(h, &fs_size, sizeof(fs_size));
    fnv(h, fs, fs_size);
    for (const mali_constant &k : fc) fnv(h, &k, sizeof(k));
    fnv(h, &format, sizeof(format));
    if (fetch != nullptr) fnv(h, fetch, sizeof(*fetch));
    if (blend != nullptr) {
        static const char tag[] = "shader blend";
        fnv(h, tag, sizeof(tag));
        fnv(h, blend, sizeof(*blend));
    }
    return h;
}

std::string
cachePath(uint64_t key)
{
    char name[32];
    snprintf(name, sizeof(name), "/%016llx.prog", (unsigned long long)key);
    return cacheDir() + name;
}

bool
writeAll(int fd, const void *p, size_t n)
{
    auto *b = (const uint8_t *)p;
    while (n != 0) {
        ssize_t w = write(fd, b, n);
        if (w <= 0) return false;
        b += w;
        n -= (size_t)w;
    }
    return true;
}

bool
readAll(int fd, void *p, size_t n)
{
    auto *b = (uint8_t *)p;
    while (n != 0) {
        ssize_t r = read(fd, b, n);
        if (r <= 0) return false;
        b += r;
        n -= (size_t)r;
    }
    return true;
}

// (the cache's use, said once a process after 40 looks: 11-10, seven pipelines compiled again at every boot)
std::atomic<unsigned> cache_hits{0}, cache_misses{0}, cache_bad{0}, cache_unwritten{0}, cache_seeded{0};
void
cacheSay()
{
    unsigned n = cache_hits + cache_misses + cache_bad;
    if (n == 40) {
        fprintf(stderr, "hakios-metal: shader cache %s: %u found (%u of them in the image's seed), %u not there, %u not whole, "
            "%u not written (of the first 40 looks)\n", cacheDir().empty() ? "(none: no directory)" : cacheDir().c_str(),
            (unsigned)cache_hits, (unsigned)cache_seeded, (unsigned)cache_misses, (unsigned)cache_bad, (unsigned)cache_unwritten);
    }
}

// THE IMAGE'S SEED (TODO 21, 11-11): programs compiled on an earlier boot and put into the image's RAM disk
// (devices/akita/tools/usbmux-pull-cache.py -> userland/hakios-metal-cache-mali -> userland/build.sh), read-only, looked
// at before the phone's own cache: at boot a 160 KB file of the latter took 1.1-2.5 s to read (behind the boot's ~2 GB
// of storage reads: CoreAnimation's seven generic pipelines ~12 s, their draws skipped meanwhile), and a new image's
// first boot compiled for minutes. Names as the cache's own (cachePath); boot-arg metal_seed=0: not looked at.
const std::string &
seedDir()
{
    static std::string dir;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *d = "/redwood/share/hakios-metal-cache-mali";
        struct stat st;
        if (!bootArg("metal_seed=0") && stat(d, &st) == 0 && S_ISDIR(st.st_mode)) {
            dir = d;
        }
    });
    return dir;
}

// both stages from the file KEY names; false (and nothing left allocated) when there is none or it is not whole
bool
cacheLoad(uint64_t key, mali_kernel_binary *vs, mali_kernel_binary *fs)
{
    if (!cacheOn()) { cache_misses++; cacheSay(); return false; }
    int fd = -1;
    bool seeded = false;
    if (!seedDir().empty()) {
        char name[32];
        snprintf(name, sizeof(name), "/%016llx.prog", (unsigned long long)key);
        fd = open((seedDir() + name).c_str(), O_RDONLY);
        seeded = fd >= 0;
    }
    if (fd < 0) {
        if (cacheDir().empty()) { cache_misses++; cacheSay(); return false; }
        std::string path = cachePath(key);
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { cache_misses++; cacheSay(); return false; }
        utimes(path.c_str(), nullptr);          // (used now: the trim keeps it)
    }
    char magic[8];
    uint64_t k = 0;
    uint32_t struct_size = 0;
    bool ok = readAll(fd, magic, 8) && memcmp(magic, kCacheMagic, 8) == 0 && readAll(fd, &k, 8) && k == key &&
        readAll(fd, &struct_size, 4) && struct_size == sizeof(mali_kernel_binary);
    mali_kernel_binary *b[2] = { vs, fs };
    for (int i = 0; i < 2 && ok; i++) {
        ok = readAll(fd, b[i], sizeof(mali_kernel_binary));
        b[i]->code = nullptr;
        b[i]->varyings = nullptr;               // (the layout a fragment stage is compiled against: not kept)
        if (ok && (b[i]->size == 0 || b[i]->size > (16u << 20))) ok = false;
        if (ok) {
            b[i]->code = malloc(b[i]->size);
            ok = b[i]->code != nullptr && readAll(fd, b[i]->code, b[i]->size);
        }
    }
    close(fd);
    (ok ? cache_hits : cache_bad)++;
    if (ok && seeded) cache_seeded++;
    cacheSay();
    if (!ok) {
        free(vs->code);
        free(fs->code);
        memset(vs, 0, sizeof(*vs));
        memset(fs, 0, sizeof(*fs));
    }
    return ok;
}

void
cacheStore(uint64_t key, const mali_kernel_binary &vs, const mali_kernel_binary &fs)
{
    if (!cacheOn() || cacheDir().empty()) return;
    std::string path = cachePath(key), tmp = path + "." + std::to_string(getpid()) + ".tmp";
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { cache_unwritten++; return; }
    uint32_t struct_size = sizeof(mali_kernel_binary);
    bool ok = writeAll(fd, kCacheMagic, 8) && writeAll(fd, &key, 8) && writeAll(fd, &struct_size, 4);
    const mali_kernel_binary *b[2] = { &vs, &fs };
    for (int i = 0; i < 2 && ok; i++) {
        ok = writeAll(fd, b[i], sizeof(mali_kernel_binary)) && writeAll(fd, b[i]->code, b[i]->size);
    }
    close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        cache_unwritten++;
    }
}
}  // namespace

hakios_metal_program *
hakios_metal_program_new_bitcode_constants(const void *vs_bitcode, size_t vs_size, const hakios_metal_constant *vs_constants,
    unsigned vs_count, const void *fs_bitcode, size_t fs_size, const hakios_metal_constant *fs_constants, unsigned fs_count,
    unsigned color_pixel_format)
{
    std::vector<mali_constant> vc(vs_count), fc(fs_count);
    char why[512];

    if (fs_bitcode == nullptr || fs_size == 0) {
        fail("a pipeline with no fragment function (depth-only: not on the Mali yet)");
        return nullptr;
    }
    for (unsigned i = 0; i < vs_count; i++) {
        vc[i] = { vs_constants[i].index, vs_constants[i].bits };
    }
    for (unsigned i = 0; i < fs_count; i++) {
        fc[i] = { fs_constants[i].index, fs_constants[i].bits };
    }
    auto *p = new hakios_metal_program;
    auto t0 = std::chrono::steady_clock::now();
    uint64_t key = cacheKey(vs_bitcode, vs_size, vc, fs_bitcode, fs_size, fc, color_pixel_format, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    bool found = cacheLoad(key, &p->vs, &p->fs);
    auto t2 = std::chrono::steady_clock::now();
    {
        // (a slow program's time by step, 11-10: CoreAnimation's uber pipelines took 1-3 s with their cache file found)
        auto ms = [](std::chrono::steady_clock::duration d) {
            return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
        };
        if (ms(t2 - t0) >= 200) {
            fprintf(stderr, "hakios-metal: program %016llx: key %lld ms (bitcode %zu + %zu bytes), cache %s %lld ms (%u + %u bytes of code)\n",
                (unsigned long long)key, ms(t1 - t0), vs_size, fs_size, found ? "found in" : "not found in", ms(t2 - t1),
                found ? p->vs.size : 0, found ? p->fs.size : 0);
        }
    }
    if (!found) {
        if (mali_stage_compile(vs_bitcode, vs_size, vc.data(), vs_count, 0, nullptr, &p->vs, why, sizeof(why)) != 0 ||
            mali_stage_compile(fs_bitcode, fs_size, fc.data(), fs_count, 1, &p->vs, &p->fs, why, sizeof(why)) != 0) {
            fail("%s", why);
            p->release();
            return nullptr;
        }
        cacheStore(key, p->vs, p->fs);
    }
    p->color_format = color_pixel_format;
    snprintf(p->name, sizeof(p->name), "%s+%s", p->vs.name, p->fs.name);
    {
        // (boot-arg metal_shaderstats, 11-11: each program's compiled size, registers and spilling -- CoreAnimation's
        // full-screen cph passes took 7 ms each with a few draws: how big is the shader)
        static const bool stats = bootArg("metal_shaderstats");
        if (stats) {
            fprintf(stderr, "hakios-metal: shader %s: vertex %u bytes %u regs tls %u; fragment %u bytes %u regs tls %u, "
                "%u constants, %u resources%s\n", p->name, p->vs.size, p->vs.work_regs, p->vs.tls_size, p->fs.size,
                p->fs.work_regs, p->fs.tls_size, fs_count, p->fs.num_resources, found ? " (cached)" : "");
        }
    }
    p->vs_bitcode.assign((const uint8_t *)vs_bitcode, (const uint8_t *)vs_bitcode + vs_size);
    p->fs_bitcode.assign((const uint8_t *)fs_bitcode, (const uint8_t *)fs_bitcode + fs_size);
    p->vs_constants = vc;
    p->fs_constants = fc;
    return p;
}

hakios_metal_program *
hakios_metal_program_new_bitcode(const void *vs_bitcode, size_t vs_size, const void *fs_bitcode, size_t fs_size,
    unsigned color_pixel_format)
{
    return hakios_metal_program_new_bitcode_constants(vs_bitcode, vs_size, nullptr, 0, fs_bitcode, fs_size, nullptr, 0,
        color_pixel_format);
}

void hakios_metal_program_free(hakios_metal_program *p) { if (p) p->release(); }
const char *hakios_metal_program_name(hakios_metal_program *p) { return p ? p->name : ""; }
void hakios_metal_program_set_label(hakios_metal_program *p, const char *label)
{
    if (p != nullptr && label != nullptr) snprintf(p->label, sizeof(p->label), "%s", label);
}

unsigned
hakios_metal_program_stage_words(hakios_metal_program *p, unsigned stage)
{
    return p == nullptr ? 0 : (unsigned)((stage == 0 ? p->vs.size : p->fs.size) / 8);
}

unsigned
hakios_metal_program_resource_count(hakios_metal_program *p, unsigned stage)
{
    return p == nullptr ? 0 : stage == 0 ? p->vs.num_resources : p->fs.num_resources;
}

int
hakios_metal_program_resource(hakios_metal_program *p, unsigned stage, unsigned i, unsigned *kind, unsigned *index,
    unsigned *size_bytes)
{
    const mali_kernel_binary *b = p == nullptr ? nullptr : stage == 0 ? &p->vs : &p->fs;
    if (b == nullptr || i >= b->num_resources) {
        return -1;
    }
    if (kind) *kind = b->resources[i].kind;
    if (index) *index = b->resources[i].index;
    if (size_bytes) *size_bytes = b->resources[i].size;
    return 0;
}

// (recorded; a draw refuses what the Mali build does not serve yet, by name)
int
hakios_metal_program_set_vertex_attribute(hakios_metal_program *p, unsigned index, unsigned format, unsigned offset,
    unsigned buffer_index)
{
    if (p == nullptr || index >= MALI_VERTEX_SLOTS) {
        fail("a vertex attribute past 30");
        return -1;
    }
    std::lock_guard<std::mutex> g(p->lock);
    p->fetch.attributes[index].format = format;
    p->fetch.attributes[index].offset = offset;
    p->fetch.attributes[index].buffer = buffer_index;
    if (format != 0) {
        p->fetch_described = true;              // (every slot is set, described or not: only a format counts)
    }
    p->fetch_compiled = false;
    return 0;
}

int
hakios_metal_program_set_vertex_layout(hakios_metal_program *p, unsigned buffer_index, unsigned stride,
    unsigned step_function, unsigned step_rate)
{
    if (p == nullptr || buffer_index >= MALI_VERTEX_SLOTS) {
        fail("a vertex layout past 30");
        return -1;
    }
    std::lock_guard<std::mutex> g(p->lock);
    p->fetch.layouts[buffer_index].stride = stride;
    p->fetch.layouts[buffer_index].step_function = step_function;
    p->fetch.layouts[buffer_index].step_rate = step_rate;
    p->fetch_compiled = false;
    return 0;
}

namespace {
// the fragment constants as the cache's key holds them: a specialised program's buffer bytes after them (pseudo-
// constants: what the stage is compiled for, not what the front end is given)
std::vector<mali_constant>
keyedConstants(const hakios_metal_program *p, const std::vector<mali_constant> &fc)
{
    if (p->known_binding < 0) return fc;
    std::vector<mali_constant> k(fc.size() + 1 + (p->known.size() + 7) / 8);   // (value-initialised: no stray padding)
    std::copy(fc.begin(), fc.end(), k.begin());
    size_t at = fc.size();
    k[at].index = 0x40000000u | (unsigned)p->known_binding;
    k[at++].bits = p->known.size();
    for (size_t i = 0; i < p->known.size(); i += 8, at++) {
        uint64_t w = 0;
        memcpy(&w, p->known.data() + i, std::min<size_t>(8, p->known.size() - i));
        k[at].index = 0x40010000u + (unsigned)(i / 8);
        k[at].bits = w;
    }
    return k;
}

// this thread's fragment compiles fold P's known buffer while it lives
struct KnownNow {
    mali_known_buffer kb = {};
    explicit KnownNow(const hakios_metal_program *p)
    {
        if (p->known_binding >= 0) {
            kb = { (unsigned)p->known_binding, p->known.data(), p->known.size() };
            mali_compile_known(&kb);
        }
    }
    ~KnownNow() { mali_compile_known(nullptr); }
};
}  // namespace

namespace {
// the stages again with the vertex descriptor's attributes fetched in the shader, before the program's first upload;
// true when there is nothing to do or it was done, false with p->fetch_refusal said
bool
compileFetch(hakios_metal_program *p)
{
    std::lock_guard<std::mutex> g(p->lock);
    bool blended = p->shader_blend.mask != 0 || p->shader_blend.format[0] || p->shader_blend.format[1] ||
        p->shader_blend.format[2] || p->shader_blend.format[3];   // (a float target's format: the fetch's conversion)
    if ((!p->fetch_described && !blended) || p->fetch_compiled) {
        return p->fetch_refusal.empty();
    }
    p->fetch_compiled = true;
    if (p->device != nullptr) {
        p->fetch_refusal = "a vertex descriptor or blending set after the program's first draw";
        return false;
    }
    if (p->vs_bitcode.empty() || p->fs_bitcode.empty()) {
        p->fetch_refusal = "blending in the shader for a program without its bitcode";
        return false;
    }
    mali_kernel_binary vs = {}, fs = {};
    char why[512] = "";
    uint64_t key = cacheKey(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants, p->fs_bitcode.data(),
                            p->fs_bitcode.size(), keyedConstants(p, p->fs_constants), p->color_format,
                            p->fetch_described ? &p->fetch : nullptr, blended ? &p->shader_blend : nullptr);
    if (!cacheLoad(key, &vs, &fs)) {
        KnownNow known(p);
        int r = p->fetch_described ?
            mali_vertex_compile(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants.data(), p->vs_constants.size(),
                                &p->fetch, &vs, why, sizeof(why)) :
            mali_stage_compile(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants.data(), p->vs_constants.size(), 0,
                               nullptr, &vs, why, sizeof(why));
        if (r != 0 ||
            mali_fragment_compile(p->fs_bitcode.data(), p->fs_bitcode.size(), p->fs_constants.data(), p->fs_constants.size(),
                                  &vs, blended ? &p->shader_blend : nullptr, &fs, why, sizeof(why)) != 0) {
            mali_kernel_binary_free(&vs);
            mali_kernel_binary_free(&fs);
            p->fetch_refusal = std::string(p->fetch_described ? "the vertex descriptor: " : "blending in the shader: ") + why;
            return false;
        }
        cacheStore(key, vs, fs);
    }
    mali_kernel_binary_free(&p->vs);
    mali_kernel_binary_free(&p->fs);
    p->vs = vs;
    p->fs = fs;
    p->fetch_refusal.clear();
    return true;
}
}  // namespace

namespace {
// Metal's blend factor as Mali's fixed-function operand: the base (C operand, with ONE as its own) and whether it is
// inverted; MTLBlendFactorZero is ONE inverted, as gallium's. false for what the hardware has no operand for.
struct MaliFactor { unsigned base; bool inv; };
enum { MF_ONE = 1, MF_SRC = 2, MF_DEST = 3, MF_SRC_ALPHA = 5, MF_DEST_ALPHA = 6 };    // (the C operand's values)
bool
maliFactor(unsigned metal, MaliFactor &f)
{
    static const MaliFactor table[10] = {
        { MF_ONE, true }, { MF_ONE, false }, { MF_SRC, false }, { MF_SRC, true }, { MF_SRC_ALPHA, false },
        { MF_SRC_ALPHA, true }, { MF_DEST, false }, { MF_DEST, true }, { MF_DEST_ALPHA, false }, { MF_DEST_ALPHA, true },
    };
    if (metal >= 10) {
        return false;           // alpha saturated, the blend constant, the second source: not yet
    }
    f = table[metal];
    return true;
}

// One channel group's A + B * C (pan_blend.c's to_mali_function, Mesa 26.2.5), for Metal's operation (0 add, 1
// subtract, 2 reverse subtract); 0 and WHY for what fixed function cannot do (min, max; factors that neither match nor
// are zero or one)
uint32_t
maliBlendFunction(unsigned op, unsigned src_metal, unsigned dst_metal, bool alpha, const char *&why)
{
    MaliFactor src, dst;
    if (op > 2) {
        why = "a min or max blend operation (fixed function adds and subtracts only)";
        return 0;
    }
    if (!maliFactor(src_metal, src) || !maliFactor(dst_metal, dst)) {
        why = "a blend factor with no fixed-function operand (alpha saturated, the blend colour, the second source)";
        return 0;
    }
    enum { A_ZERO = 1, A_SRC = 2, A_DEST = 3 };
    enum { B_MINUS = 0, B_PLUS = 1, B_SRC = 2, B_DEST = 3 };
    unsigned a = 0, b = 0, c = 0;
    bool neg_a = false, neg_b = false, inv_c = false;
    // the hardware's C has zero and inverts it to one, gallium's ONE is uninverted: hence the extra invert for ONE
    auto inverted = [](MaliFactor f) { return f.inv ^ (f.base == MF_ONE); };
    auto c_of = [](MaliFactor f) { return f.base == MF_ONE ? 1u /* zero */ : f.base; };
    bool src_zero = src.base == MF_ONE && src.inv, src_one = src.base == MF_ONE && !src.inv;
    bool dst_zero = dst.base == MF_ONE && dst.inv, dst_one = dst.base == MF_ONE && !dst.inv;
    bool two_src_dest = op == 0 && (src.base == MF_DEST || (alpha && src.base == MF_DEST_ALPHA)) && !src.inv &&
        (dst.base == MF_SRC || (alpha && dst.base == MF_SRC_ALPHA)) && !dst.inv;
    if (src_zero) {
        a = A_ZERO; b = B_DEST; neg_b = op == 1; inv_c = inverted(dst); c = c_of(dst);
    } else if (src_one) {
        a = A_SRC; b = B_DEST; neg_b = op == 1; neg_a = op == 2; inv_c = inverted(dst); c = c_of(dst);
    } else if (dst_zero) {
        a = A_ZERO; b = B_SRC; neg_b = op == 2; inv_c = inverted(src); c = c_of(src);
    } else if (dst_one) {
        a = A_DEST; b = B_SRC; neg_a = op == 1; neg_b = op == 2; inv_c = inverted(src); c = c_of(src);
    } else if (src.base == dst.base && src.inv == dst.inv) {
        a = A_ZERO; inv_c = inverted(src); c = c_of(src);
        b = op == 0 ? B_PLUS : B_MINUS;
        neg_b = op == 2;
    } else if (two_src_dest) {
        a = A_ZERO; b = B_DEST; c = 4;                  // dest * (2 * src)
    } else if (src.base == dst.base) {                  // the same factor, one of them inverted
        a = A_DEST; inv_c = inverted(src); c = c_of(src);
        if (op == 0) { b = B_MINUS; }
        else if (op == 2) { b = B_PLUS; neg_b = true; }
        else { b = B_PLUS; neg_a = true; }
    } else {
        why = "blend factors that neither match nor are zero or one (fixed function has one multiplier)";
        return 0;
    }
    return a | (neg_a ? 1u << 3 : 0) | b << 4 | (neg_b ? 1u << 7 : 0) | c << 8 | (inv_c ? 1u << 11 : 0);
}

}  // namespace

namespace {
// one colour attachment's blend state as the fixed-function equation word, whether it reads the tile buffer, or why not
void
encodeBlend(int enabled, unsigned rgb_op, unsigned alpha_op, unsigned src_rgb, unsigned dst_rgb, unsigned src_alpha,
    unsigned dst_alpha, unsigned write_mask, uint32_t &equation, bool &reads_dest, std::string &refusal,
    mali_blend_rt *shader = nullptr, bool *in_shader = nullptr)
{
    if (in_shader) *in_shader = false;
    // Metal's write mask: alpha 1, blue 2, green 4, red 8; Mali's colour mask: red bit 0 .. alpha bit 3
    uint32_t mask = (write_mask & 8 ? 1u : 0) | (write_mask & 4 ? 2u : 0) | (write_mask & 2 ? 4u : 0) | (write_mask & 1 ? 8u : 0);
    refusal.clear();
    reads_dest = mask != 0xf;                   // a channel kept is read from the tile buffer
    if (!enabled) {
        equation = 0x132 | 0x132 << 12 | mask << 28;
        return;
    }
    const char *why = nullptr;
    uint32_t rgb = maliBlendFunction(rgb_op, src_rgb, dst_rgb, false, why);
    uint32_t alpha = why == nullptr ? maliBlendFunction(alpha_op, src_alpha, dst_alpha, true, why) : 0;
    // in the shader instead: any operation and factor -- the blend colour from the draw's push constants
    // (MALI_DRAW_BLEND_COLOR), the second source the fragment function's [[color(n), index(1)]] (nir_lower_blend)
    auto shaderable = [](unsigned f) { return f <= 18; };
    const bool can_shade = shader != nullptr && rgb_op <= 4 && alpha_op <= 4 && shaderable(src_rgb) && shaderable(dst_rgb) &&
        shaderable(src_alpha) && shaderable(dst_alpha);
    if (can_shade) {
        // (kept whichever way it is blended: a raw target's blend is the shader's even when fixed function could --
        // rawBlend)
        *shader = { rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask };
    }
    if (why != nullptr && can_shade) {
        *in_shader = true;
        equation = 0x132 | 0x132 << 12 | 0xfu << 28;    // the shader's result as it is (it applied the write mask)
        reads_dest = true;
        return;
    }
    if (why != nullptr) {
        char factors[96];
        snprintf(factors, sizeof(factors), " (operations %u/%u, factors %u %u / %u %u)", rgb_op, alpha_op, src_rgb, dst_rgb,
            src_alpha, dst_alpha);
        refusal = std::string("blending: ") + why + factors;
        return;
    }
    equation = rgb | alpha << 12 | mask << 28;
    reads_dest = true;
}
}  // namespace

namespace {
// A RAW COLOUR ATTACHMENT (R32Float: the tile buffer holds its 32 bits as they are, and the fixed-function blend does
// not know such a target) with blending on: the blend computed in the fragment shader (nir_lower_blend over the
// target's own format), the hardware writing the result as it is -- as when fixed function has no such equation
void
rawBlend(hakios_metal_program *p, unsigned i)
{
    if (i > 3) {
        return;
    }
    // WHAT THE COMPILE IS TOLD OF THE ATTACHMENT'S FORMAT (11-13): R32Float always (the fetch's conversion); a half-float
    // one only when its blend is computed in the shader (nir_lower_blend then reads and writes it as the floats it holds
    // -- it took every target for RGBA8, clamped to 0..1) -- told otherwise, the program takes the shader-blend compile
    // for nothing
    const unsigned f = p->rt_format[i];
    static const bool told_always = bootArg("metal_rtfmt=1");   // (a test: every float attachment's format told)
    p->shader_blend.format[i] = f == 55 || ((told_always || (p->shader_blend.mask & 1u << i)) && (f == 25 || f == 65 || f == 115 || f == 92)) ||
        f == 71 || f == 81 ? f : 0;       // (an sRGB target's always: a framebuffer fetch from it must decode, 11-13)
    if (f != 55 || !p->rt_blend_on[i]) {
        return;
    }
    p->shader_blend.mask |= 1u << i;
    if (i == 0) {
        p->blend_equation = 0x132 | 0x132 << 12 | 0xfu << 28;
        p->blend_reads_dest = true;
        p->blend_refusal.clear();
    } else {
        p->more_blend_equation[i - 1] = 0x132 | 0x132 << 12 | 0xfu << 28;
        p->more_blend_reads_dest[i - 1] = true;
        p->more_blend_refusal[i - 1].clear();
    }
}
}  // namespace

// colour attachment INDEX's MTLPixelFormat, from the pipeline's descriptor (11-12 night): the ones that are not 8 bits a
// channel are what the stage's compile and the blend go by
void
hakios_metal_program_set_attachment_format(hakios_metal_program *p, unsigned index, unsigned pixel_format)
{
    if (p == nullptr || index > 3) {
        return;
    }
    p->rt_format[index] = pixel_format;
    rawBlend(p, index);
}

void
hakios_metal_program_set_blend(hakios_metal_program *p, int enabled, unsigned rgb_op, unsigned alpha_op, unsigned src_rgb,
    unsigned dst_rgb, unsigned src_alpha, unsigned dst_alpha, unsigned write_mask)
{
    if (p == nullptr) {
        return;
    }
    p->blend_enabled = enabled != 0;
    p->rt_blend_on[0] = enabled != 0;
    bool in_shader = false;
    encodeBlend(enabled, rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask, p->blend_equation,
                p->blend_reads_dest, p->blend_refusal, &p->shader_blend.rt[0], &in_shader);
    p->shader_blend.mask = (p->shader_blend.mask & ~1u) | (in_shader ? 1u : 0u);
    rawBlend(p, 0);
}

int
hakios_metal_program_set_blend_at(hakios_metal_program *p, unsigned index, int enabled, unsigned rgb_op, unsigned alpha_op,
    unsigned src_rgb, unsigned dst_rgb, unsigned src_alpha, unsigned dst_alpha, unsigned write_mask)
{
    if (index > 7) {
        fail("a colour attachment past 7");
        return -1;
    }
    if (index == 0) {
        hakios_metal_program_set_blend(p, enabled, rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask);
    } else if (p != nullptr && index <= 3) {
        bool in_shader = false;
        encodeBlend(enabled, rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask,
                    p->more_blend_equation[index - 1], p->more_blend_reads_dest[index - 1], p->more_blend_refusal[index - 1],
                    &p->shader_blend.rt[index], &in_shader);
        p->shader_blend.mask = (p->shader_blend.mask & ~(1u << index)) | (in_shader ? 1u << index : 0u);
        p->rt_blend_on[index] = enabled != 0;
        rawBlend(p, index);
    }
    return 0;
}

int
hakios_metal_program_set_samples(hakios_metal_program *p, unsigned samples)
{
    if (p == nullptr || (samples != 1 && samples != 2 && samples != 4)) {
        fail("a sample count of %u", samples);
        return -1;
    }
    p->samples = samples;
    return 0;
}

namespace {
// the stage's push constants for a draw: [[buffer(n)]]'s addresses, the viewport transform, the first vertex/instance
/*
 * A SURFACE THIS COMMAND BUFFER DRAWS INTO (11-10), marked busy with HakiosGPU (0x7d) from its first draw into it until
 * the command buffer is done or dropped (letGo): the display waits for the marks to go before it shows the surface
 * (MiniIOSurfaceDrawn, hakios.akita_gpu.cpp), as Apple's waits on a surface's own GPU writes. CoreAnimation draws some
 * frames with several command buffers and presents from the first one's scheduled handler, before the others are even
 * recorded: waiting only for what had been submitted, the panel showed such frames half drawn.
 */
static void
markBusy(hakios_metal_cb *cb, hakios_metal_device *d, hakios_metal_texture *t)
{
    while (t != nullptr && t->origin == 4) {
        t = t->parent;
    }
    if (t == nullptr || t->origin != 1 || t->surface == 0) {
        return;
    }
    if (t->mirrored && std::find(cb->mirror_out.begin(), cb->mirror_out.end(), t->surface) == cb->mirror_out.end()) {
        cb->mirror_out.push_back(t->surface);
    }
    for (unsigned sid : cb->busy) {
        if (sid == t->surface) {
            return;
        }
    }
    struct { uint32_t surface; int32_t delta; } r = { t->surface, 1 };
    {
        std::lock_guard<std::mutex> g(d->lock);
        if (d->call(0x7d, &r, sizeof(r)) != 0) {
            return;
        }
    }
    cb->busy.push_back(t->surface);
    cb->busy_device = d;
}

bool
stagePushConstants(const mali_kernel_binary &b, hakios_metal_buffer *const *buffers, const uint64_t *offsets, unsigned count,
    const float scale[3], const float offset[3], unsigned first_vertex, unsigned base_instance, std::vector<uint32_t> &fau,
    std::vector<hakios_metal_buffer *> &used)
{
    fau.assign(b.fau_words, 0);
    for (unsigned r = 0; r < b.num_resources; r++) {
        const mali_kernel_resource &res = b.resources[r];
        if (res.kind != HAKIOS_METAL_RESOURCE_CONSTANT_BUFFER && res.kind != HAKIOS_METAL_RESOURCE_DEVICE_BUFFER) {
            continue;
        }
        if (res.index >= count || buffers[res.index] == nullptr) {
            // (nothing bound: the device's zeroes stand for it -- gNothingVa, set by the draw before this)
            if (gNothingVa.load() == 0) {
                fail("draw with %s: nothing bound at [[buffer(%u)]]", b.name, res.index);
                return false;
            }
            fau[2 * res.slot] = (uint32_t)gNothingVa.load();
            fau[2 * res.slot + 1] = (uint32_t)(gNothingVa.load() >> 32);
            continue;
        }
        uint64_t va = buffers[res.index]->bo.va + (offsets ? offsets[res.index] : 0);
        fau[2 * res.slot] = (uint32_t)va;               // (its slot in the push constants: the index, under 16)
        fau[2 * res.slot + 1] = (uint32_t)(va >> 32);
        used.push_back(buffers[res.index]);
    }
    for (int i = 0; i < 3; i++) {
        memcpy(&fau[MALI_DRAW_VIEWPORT_SCALE + i], &scale[i], 4);
        memcpy(&fau[MALI_DRAW_VIEWPORT_OFFSET + i], &offset[i], 4);
    }
    fau[MALI_DRAW_FIRST_VERTEX] = first_vertex;
    fau[MALI_DRAW_BASE_INSTANCE] = base_instance;
    if (b.fau_words > MALI_KERNEL_RESERVED) {
        memcpy(&fau[MALI_KERNEL_RESERVED], b.consts, (b.fau_words - MALI_KERNEL_RESERVED) * 4);
    }
    return true;
}

uint8_t
unorm8(double v)
{
    return (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255.0 + 0.5);
}

// A function's constexpr sampler (AIR's __air_sampler_state word; upstream tools/mesa/air's air_decode_sampler_state:
// address s/t/r bits 0/3/6, mag 9, min 11, mip 13, coord::pixel 15, compare 16, anisotropy 20, lod 24/40, border 56)
// as the API's samplers are made: AIR's clamp_to_zero 0 / clamp_to_edge 1 / repeat 2 / mirrored_repeat 3 /
// clamp_to_border 4 are Metal's 4 / 0 / 2 / 3 / 5. A comparison (compare_func other than none 0 and never 8) is refused.
bool
inlineSampler(uint64_t w, uint32_t words[8])
{
    static const unsigned address[5] = { 4, 0, 2, 3, 5 };
    auto field = [&](unsigned lo, unsigned n) { return (unsigned)((w >> lo) & ((1ull << n) - 1)); };
    hakios_metal_sampler_desc desc = {};
    for (unsigned a = 0; a < 3; a++) {
        if (field(3 * a, 3) > 4) {
            fail("an address mode AIR does not have (%u)", field(3 * a, 3));
            return false;
        }
    }
    unsigned compare = field(16, 4);
    if (compare != 0 && compare != 8) {
        fail("a comparison sampler (compare_func %u, MSL's numbering): not on the Mali yet", compare);
        return false;
    }
    desc.s_address = address[field(0, 3)];
    desc.t_address = address[field(3, 3)];
    desc.r_address = address[field(6, 3)];
    desc.mag_filter = field(9, 2) ? 1 : 0;                  // (bicubic as linear)
    desc.min_filter = field(11, 2) ? 1 : 0;
    desc.mip_filter = field(13, 2);
    desc.normalized_coordinates = field(15, 1) == 0;
    desc.max_anisotropy = field(20, 4) + 1;
    desc.border_color = field(56, 2);
    desc.lod_min_clamp = 0;                                  // (MSL's lod_clamp default: every level)
    desc.lod_max_clamp = 1e30f;
    hakios_metal_sampler *s = hakios_metal_sampler_new(nullptr, &desc);
    if (s == nullptr) {
        return false;
    }
    memcpy(words, s->words, 32);
    delete s;
    return true;
}

// the Load action's program (compile_mali.c's mali_preload_compile), compiled and uploaded on first use; false with
// d->preload_error said when it cannot be
hakios_metal_texture *attachmentOf(hakios_metal_texture *t, unsigned level, unsigned slice);

// THE PASS'S DEPTH AND STENCIL into DJ (its attachments, actions, clear values; with DRAW, the draw's depth/stencil
// state as a "Depth/stencil" descriptor), the attachments the size of WIDTH x HEIGHT; false with WHY
bool
zsOf(const struct hakios_metal_render_pass *pass, DrawJob &dj, unsigned width, unsigned height, bool draw, const char *&why)
{
    hakios_metal_texture *z = pass->depth, *st = pass->stencil;
    if (z == nullptr && st == nullptr) {
        if (draw && (pass->depth_state_set || pass->stencil_state_set)) {
            // (a depth/stencil state with no attachment: Metal tests against nothing, every fragment passes)
        }
        return true;
    }
    if (z != nullptr && z->format != 252 && z->format != 250 && z->format != 260) {
        why = "a depth attachment not Depth16Unorm, Depth32Float or Depth32Float_Stencil8 (not on the Mali yet)";
        return false;
    }
    if (st != nullptr && st->format != 253 && st->format != 260) {
        why = "a stencil attachment not Stencil8 or Depth32Float_Stencil8 (not on the Mali yet)";
        return false;
    }
    if ((z && z->samples != dj.samples) || (st && st->samples != dj.samples)) {
        why = "depth or stencil attachments of another sample count than the colour attachments";
        return false;
    }
    if ((z && (z->width != width || z->height != height)) || (st && (st->width != width || st->height != height))) {
        why = "depth or stencil attachments of another size than the colour attachments";
        return false;
    }
    dj.zs = true;
    {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "hakios-metal: the first pass with depth/stencil: depth MTLPixelFormat %u, stencil %u, %ux%u, "
                "depth state %d (compare %u write %d), stencil state %d\n", z ? z->format : 0, st ? st->format : 0, width,
                height, pass->depth_state_set, pass->depth_compare, pass->depth_write, pass->stencil_state_set);
        }
    }
    if (z != nullptr) {
        dj.has_z = true;
        dj.z_va = z->gpu();
        dj.z_row = z->stride;
        dj.z_format = z->format == 250 ? 1 : 14;
        dj.z_internal = z->format == 250 ? 0 : 2;
        dj.z_write = pass->depth_store_action == 1;
        dj.z_load = pass->depth_load_action == 1;
        dj.z_clear = (float)pass->clear_depth;
    }
    if (st != nullptr) {
        dj.has_s = true;
        dj.s_va = st->format == 260 ? st->gpu() + st->stencil_off : st->gpu();
        dj.s_row = st->format == 260 ? st->stencil_stride : st->stride;
        dj.s_write = pass->stencil_store_action == 1;
        dj.s_load = pass->stencil_load_action == 1;
        dj.s_clear = pass->clear_stencil & 0xff;
    }
    auto *w = dj.zsd;
    memset(w, 0, 32);
    w[0] = 7 | 7u << 4 | 7u << 16;                     // stencil faces: always (no test)
    // (word 4: v11's separated dependency tracking 21, depth cull 22, clamp mode [0, 1] (1) at 23, depth from fixed
    // function (2) at 25 -- the two genxml defaults that are not zero, without which the depth test did nothing)
    const uint32_t w4 = 1u << 21 | 1u << 22 | 1u << 23 | 2u << 25;
    w[4] = w4 | 7u << 29;                               // depth: always, nothing written
    if (!draw) {
        return true;
    }
    if (dj.has_z && pass->depth_state_set) {
        // MTLCompareFunction is genxml's "Func" (Never 0 .. Always 7)
        w[4] = w4 | (pass->depth_write ? 1u << 27 : 0) | (pass->depth_compare & 7) << 29;
    }
    if (dj.has_s && pass->stencil_state_set) {
        // MTLStencilOperation (Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert, IncrementWrap,
        // DecrementWrap) as genxml's "Stencil Op"
        static const uint32_t op[8] = { 0, 2, 1, 6, 7, 3, 4, 5 };
        const auto &f = pass->stencil_front, &b = pass->stencil_back;
        w[0] = 7 | (f.compare & 7) << 4 | op[f.stencil_fail & 7] << 7 | op[f.depth_fail & 7] << 10 | op[f.pass & 7] << 13 |
            (b.compare & 7) << 16 | op[b.stencil_fail & 7] << 19 | op[b.depth_fail & 7] << 22 | op[b.pass & 7] << 25 |
            1u << 31;
        w[1] = (f.write_mask & 0xff) | (b.write_mask & 0xff) << 8 | (f.read_mask & 0xff) << 16 | (b.read_mask & 0xff) << 24;
        w[2] = (pass->stencil_front_reference & 0xff) | (pass->stencil_back_reference & 0xff) << 8;
    }
    return true;
}

// A PASS INTO A MIP LEVEL OR SLICE: the pass as one whose attachments are those levels as textures of their own
// (attachmentOf), held in VIEWS until the caller releases them; false (and nothing held) with the reason
bool
passAtLevels(const struct hakios_metal_render_pass *pass, struct hakios_metal_render_pass &q, hakios_metal_texture *views[10])
{
    q = *pass;
    for (unsigned i = 0; i < 10; i++) views[i] = nullptr;
    auto release = [&] { for (unsigned i = 0; i < 10; i++) if (views[i]) { views[i]->release(); views[i] = nullptr; } };
    if (q.color0 != nullptr && (q.color0_level || q.color0_slice)) {
        if ((views[0] = attachmentOf(q.color0, q.color0_level, q.color0_slice)) == nullptr) return false;
        q.color0 = views[0];
        q.color0_level = q.color0_slice = 0;
    }
    for (unsigned i = 0; i < 7; i++) {
        auto &m = q.more[i];
        if (m.texture != nullptr && (m.level || m.slice)) {
            if ((views[i + 1] = attachmentOf(m.texture, m.level, m.slice)) == nullptr) { release(); return false; }
            m.texture = views[i + 1];
            m.level = m.slice = 0;
        }
    }
    if (q.depth != nullptr && (q.depth_level || q.depth_slice)) {
        // the depth attachment's level (with its own stencil plane when it is Depth32Float_Stencil8 and the stencil
        // attachment too: Metal's stencil descriptor names the same level)
        if ((views[8] = attachmentOf(q.depth, q.depth_level, q.depth_slice)) == nullptr) { release(); return false; }
        if (q.stencil == q.depth) q.stencil = views[8];
        q.depth = views[8];
        q.depth_level = q.depth_slice = 0;
    }
    return true;
}

bool
atLevels(const struct hakios_metal_render_pass *pass)
{
    if (pass == nullptr) return false;
    if (pass->color0 != nullptr && (pass->color0_level || pass->color0_slice)) return true;
    if (pass->depth != nullptr && (pass->depth_level || pass->depth_slice)) return true;
    for (unsigned i = 0; i < 7; i++) {
        if (pass->more[i].texture != nullptr && (pass->more[i].level || pass->more[i].slice)) return true;
    }
    return false;
}

hakios_metal_program *
preloadProgram(hakios_metal_device *d, unsigned targets, unsigned zs)
{
    std::lock_guard<std::mutex> g(d->preload_lock);
    unsigned k = targets > 4 ? 4 : targets;
    zs &= 7;
    if (d->preload_for[k][zs] != nullptr || d->preload_failed_for[k][zs]) {
        return d->preload_for[k][zs];
    }
    auto *p = new hakios_metal_program;
    char why[256] = "";
    if (mali_preload_zs_compile(&p->vs, &p->fs, targets, zs, why, sizeof(why)) != 0 || !p->upload(d)) {
        d->preload_error = std::string("load action Load: the preload: ") + (why[0] ? why : gError.c_str());
        d->preload_failed_for[k][zs] = true;
        p->release();
        return nullptr;
    }
    snprintf(p->name, sizeof(p->name), "hakios_preload%u_%u", targets, zs);
    d->preload_for[k][zs] = p;
    return p;
}

int
drawImpl(hakios_metal_cb *cb, hakios_metal_program *p, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned vertex_start, unsigned vertex_count, unsigned instance_count, unsigned base_instance,
    hakios_metal_buffer *const *vs_buffers, const uint64_t *vs_offsets, unsigned vs_count,
    hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets, unsigned fs_count,
    hakios_metal_texture *const *fs_textures, unsigned fs_texture_count, hakios_metal_sampler *const *fs_samplers,
    unsigned fs_sampler_count, const IndexInfo *ix = nullptr, const uint32_t *fs_words = nullptr, unsigned fs_word_count = 0)
{
    if (atLevels(pass)) {
        struct hakios_metal_render_pass q;
        hakios_metal_texture *views[10];
        if (!passAtLevels(pass, q, views)) {
            std::string why = gError;
            fail("%s%s%s", p ? p->name : "", p ? ": " : "", why.c_str());
            return -1;
        }
        int r = drawImpl(cb, p, &q, viewport, scissor, primitive, vertex_start, vertex_count, instance_count, base_instance,
            vs_buffers, vs_offsets, vs_count, fs_buffers, fs_offsets, fs_count, fs_textures, fs_texture_count, fs_samplers,
            fs_sampler_count, ix, fs_words, fs_word_count);
        for (auto *v : views) if (v) v->release();     // (the draw holds what it uses)
        return r;
    }
    hakios_metal_device *d = cb->queue->device;
    const char *refused = nullptr;
    hakios_metal_texture *t = pass ? pass->color0 : nullptr;

    // what the Mali build draws so far; the rest by name
    if (!d->heap_ok) refused = "draws (no tiler heap on this device: hakios_metal_open said why)";
    else if (p == nullptr || t == nullptr) refused = "a draw without a program or a colour attachment";
    else if (!drawable(t->format)) {
        static thread_local char why[120];      // (the format named: which ones applications ask for)
        snprintf(why, sizeof(why), "a colour attachment of MTLPixelFormat %u, not RGBA8/BGRA8/BGR5A1/B5G6R5 (not on the Mali yet)", t->format);
        refused = why;
    }
    else if (!(t->usage & HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET)) refused = "a colour attachment made without render-target usage";
    else if (pass->store_action > 1) refused = "a store action other than DontCare and Store";
    else if (pass->visibility && pass->visibility_mode) refused = "visibility results (not on the Mali yet)";
    else if (!p->blend_refusal.empty()) refused = p->blend_refusal.c_str();
    else if (!compileFetch(p)) refused = p->fetch_refusal.c_str();
    else if (p->samples == 2) refused = "2 samples a pixel (the Mali-G715 has no 2x sample pattern -- Mesa serves it from v12; 4 is served)";
    else if (t->samples != p->samples) refused = "a colour attachment of another sample count than the pipeline's rasterSampleCount";
    else if (primitive > 4) refused = "a primitive type past MTLPrimitiveTypeTriangleStrip";
    else if (instance_count == 0) refused = "no instances";
    else if (ix && (ix->buffer == nullptr || ix->type > 1 || ix->base_vertex < 0 ||
                    ix->offset + (uint64_t)vertex_count * (ix->type ? 4 : 2) > ix->buffer->length))
        refused = "an indexed draw past its index buffer, of an unknown index type or with a negative base vertex";
    // colour attachments past 0: the ones the fragment function writes are drawn into (up to 3 more: RGBA8/BGRA8, the
    // size of the first, level and slice 0); one it does not write is left as it is -- CoreAnimation attaches more
    // targets than most of its draws write -- and a clear of such a one is not done (said once).
    unsigned rts = 1;
    for (unsigned w = p ? p->fs.rt_written >> 1 : 0, i = 1; refused == nullptr && w != 0; w >>= 1, i++) {
        if (!(w & 1)) continue;
        const hakios_metal_render_pass::hakios_metal_color_attachment &m = pass->more[i - 1];
        if (i > 3) refused = "a fragment function writing a colour attachment past 3 (not on the Mali yet)";
        else if (m.texture == nullptr) { rts = i + 1; continue; }   // (Metal discards the output: see dj.more below)
        else if (!drawable(m.texture->format)) {
            static thread_local char why[120];
            snprintf(why, sizeof(why), "colour attachment %u of MTLPixelFormat %u, not RGBA8/BGRA8/BGR5A1/B5G6R5 (not on the Mali yet)", i, m.texture->format);
            refused = why;
        }
        else if (m.texture->width != t->width || m.texture->height != t->height)
            refused = "colour attachments of different sizes";
        else if (m.texture->samples != t->samples) refused = "colour attachments of different sample counts";
        else if (!(m.texture->usage & HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET)) refused = "a colour attachment made without render-target usage";
        else if (m.store_action > 1) refused = "a store action other than DontCare and Store";
        else if (!p->more_blend_refusal[i - 1].empty()) refused = p->more_blend_refusal[i - 1].c_str();
        rts = i + 1;
    }
    // EVERY ATTACHMENT OF THE PASS IN EVERY DRAW (up to 4 targets): one the fragment function does not write is still
    // in the Mali pass -- loaded, stored, cleared as the pass says -- its blend writing nothing. Without this a pass
    // whose draws wrote attachment 1 only now and then (CoreAnimation's in Settings) split into a Mali pass at every
    // change, each preloading and writing back the whole screen: 20-36 passes a frame, 100-240 ms.
    for (unsigned i = rts; refused == nullptr && i <= 3; i++) {
        const hakios_metal_render_pass::hakios_metal_color_attachment &m = pass->more[i - 1];
        if (m.texture == nullptr) continue;
        if (!drawable(m.texture->format) || m.texture->width != t->width || m.texture->height != t->height ||
            m.store_action > 1 || m.texture->samples != t->samples) {
            break;                                      // (left out, as before: a later one too)
        }
        rts = i + 1;
    }
    // ONE MALI PASS FOR CROSS-FADED DRAWS (11-11): CoreAnimation's fixed_frag_lph_cpf (the generic pipeline) writes a
    // third colour output its passes have no attachment for -- a target of the scratch's (dj.more below) -- and so it had
    // a Mali pass of three targets amid the passes of two of the same encoder: the home screen's frames split into up to
    // 16 full-screen passes, each preloading and writing back the screen (~1.1 ms each, 18 ms a frame; metal_stats_ms).
    // So each target set (keyed on colour attachment 1, CoreAnimation's memoryless one, or else attachment 0) remembers
    // the most targets a draw of it had, and a draw with fewer takes the missing ones too -- outputs it never writes, a
    // scratch target never loaded nor written back -- and samePass keeps it in the Mali pass. Boot-arg mwide=0: not done.
    static const bool no_wide = bootArg("mwide=0");
    if (refused == nullptr && !no_wide && t != nullptr) {
        hakios_metal_texture *key = rts > 1 && pass->more[0].texture != nullptr ? pass->more[0].texture : t;
        unsigned seen = key->widest_rts.load();
        if (rts > seen) {
            key->widest_rts.store(rts);
        } else if (seen > rts) {
            bool none = true;                           // (only past every attachment the pass has)
            for (unsigned i = rts; i < seen && none; i++) {
                none = pass->more[i - 1].texture == nullptr;
            }
            if (none) {
                rts = seen;
            }
        }
    }
    for (unsigned i = rts; refused == nullptr && i < 8; i++) {
        if (pass->more[i - 1].texture != nullptr && pass->more[i - 1].load_action == 2) {
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) {
                fprintf(stderr, "hakios-metal: colour attachment %u's clear is not done: no draw writes it\n", i);
            }
        }
    }
    bool loads = refused == nullptr && pass->load_action == 1;
    for (unsigned i = 1; refused == nullptr && i < rts; i++) {
        static const bool no_ml = bootArg("metal_nomemoryless"), mlread_old = bootArg("mlread=1");
        const bool ml = pass->more[i - 1].texture != nullptr && pass->more[i - 1].texture->memoryless && !no_ml &&
            (!(p->fs.rt_read & 1u << i) || !mlread_old);    // (a memoryless attachment is not loaded: below)
        loads = loads || (pass->more[i - 1].texture != nullptr && !ml &&
                          (pass->more[i - 1].load_action == 1 || !(p->fs.rt_written & 1u << i)));
    }
    const unsigned ms4 = t != nullptr && t->samples > 1 ? 4u : 0u;     // (preloads of multisample attachments: a sample a run)
    hakios_metal_program *pre = loads ? preloadProgram(d, rts, ms4) : nullptr;
    if (loads && pre == nullptr) refused = d->preload_error.c_str();
    if (refused != nullptr) {
        fail("%s%s%s", p ? p->name : "", p ? ": " : "", refused);
        return -1;
    }
    if (vertex_count == 0) {
        return 0;
    }
    {
        // (boot-arg metal_skipprog=A,B, 11-11 for TODO 8h: the draws of programs whose names or pipeline labels hold A
        // or B are not done -- a screenshot without them says which draw makes a mark)
        static const std::vector<std::string> skip = [] {
            std::vector<std::string> v;
            char args[1024] = {};
            size_t size = sizeof(args) - 1;
            const char *at;
            if (sysctlbyname("kern.bootargs", args, &size, nullptr, 0) == 0 && (at = strstr(args, "metal_skipprog=")) != nullptr) {
                std::string w;
                for (at += strlen("metal_skipprog="); ; at++) {
                    if (*at == ',' || *at == ' ' || *at == 0) {
                        if (!w.empty()) v.push_back(w);
                        w.clear();
                        if (*at != ',') break;
                    } else {
                        w += *at;
                    }
                }
            }
            for (const auto &w : v) fprintf(stderr, "hakios-metal: draws of programs named *%s* skipped (metal_skipprog)\n", w.c_str());
            return v;
        }();
        for (const auto &w : skip) {
            if (strstr(p->name, w.c_str()) != nullptr || strstr(p->label, w.c_str()) != nullptr) return 0;
        }
    }
    if (!p->upload(d)) {
        return -1;
    }
    {
        // (boot-arg metal_progsay: each program's first draw, with the time since boot as the kernel log's)
        static const bool say = bootArg("metal_progsay");
        if (say) {
            static std::mutex m;
            static std::vector<const void *> seen;
            std::lock_guard<std::mutex> g(m);
            if (std::find(seen.begin(), seen.end(), (const void *)p) == seen.end()) {
                seen.push_back(p);
                mach_timebase_info_data_t tb;
                mach_timebase_info(&tb);
                double t = (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
                fprintf(stderr, "hakios-metal: first draw of %s [%s] at %.3f s: fs push words %u, textures %u, samplers %u, blend "
                    "in shader 0x%x, written 0x%x, read 0x%x, target %ux%u format %u, samples %u, %u vertices x %u\n",
                    p->name, p->label, t, p->fs.fau_words, fs_texture_count, fs_sampler_count, p->shader_blend.mask, p->fs.rt_written,
                    p->fs.rt_read, pass && pass->color0 ? pass->color0->width : 0, pass && pass->color0 ? pass->color0->height : 0, pass && pass->color0 ? pass->color0->format : 0, p->samples,
                    vertex_count, instance_count);
                for (unsigned i = 0; i < 4; i++) {
                    if (p->shader_blend.mask & 1u << i) {
                        const mali_blend_rt &b = p->shader_blend.rt[i];
                        fprintf(stderr, "hakios-metal:   %s: target %u blended in the shader: ops %u/%u, factors %u %u / %u %u, "
                            "write mask 0x%x; blend colour %g %g %g %g\n", p->name, i, b.rgb_op, b.alpha_op, b.src_rgb, b.dst_rgb,
                            b.src_alpha, b.dst_alpha, b.write_mask, pass->blend_color[0], pass->blend_color[1],
                            pass->blend_color[2], pass->blend_color[3]);
                    }
                }
            }
        }
    }
    DrawJob dj = {};
    dj.samples = t->samples;
    dj.target_tex = t;
    markBusy(cb, d, t);
    {
        // marksay=1 (a test, 11-10): every big render target said once -- what it is over, and whether it is marked
        static const bool marksay = bootArg("marksay");
        static std::mutex said_lock;
        static std::vector<hakios_metal_texture *> said;
        if (marksay && t->width >= 1000 && t->height >= 1000) {
            std::lock_guard<std::mutex> g(said_lock);
            if (std::find(said.begin(), said.end(), t) == said.end() && said.size() < 64) {
                said.push_back(t);
                hakios_metal_texture *r = t;
                unsigned depth = 0;
                while (r != nullptr && r->origin == 4) { r = r->parent; depth++; }
                fprintf(stderr, "hakios-metal: marksay: target %p %ux%u fmt %u origin %u (after %u views: origin %u surface %u)%s\n",
                    (void *)t, t->width, t->height, t->format, t->origin, depth, r ? r->origin : 99, r ? r->surface : 0,
                    r && r->origin == 1 ? " MARKED" : " NOT MARKED");
            }
        }
    }
    for (unsigned i = 0; i < 3; i++) {
        markBusy(cb, d, pass->more[i].texture);
    }
    std::vector<hakios_metal_buffer *> used;
    struct hakios_metal_viewport vp = viewport ? *viewport :
        hakios_metal_viewport{ 0, 0, (double)t->width, (double)t->height, 0, 1 };
    // Metal's viewport: origin top left, NDC y up, z in [0, 1]
    float scale[3] = { (float)(vp.width / 2), (float)(-vp.height / 2), (float)(vp.zfar - vp.znear) };
    float offset[3] = { (float)(vp.x + vp.width / 2), (float)(vp.y + vp.height / 2), (float)vp.znear };
    nothingVa(d);       // (for a buffer a stage declares and nothing is bound at)
    if (!stagePushConstants(p->vs, vs_buffers, vs_offsets, vs_count, scale, offset, vertex_start, base_instance, dj.vs_fau, used) ||
        !stagePushConstants(p->fs, fs_buffers, fs_offsets, fs_count, scale, offset, vertex_start, base_instance, dj.fs_fau, used)) {
        return -1;
    }
    for (unsigned i = 0; i < fs_word_count && i < dj.fs_fau.size(); i++) {
        dj.fs_fau[i] = fs_words[i];                     // (the driver's own programs: their words from 0, no buffers)
    }
    if (pass != nullptr && dj.fs_fau.size() > MALI_DRAW_BLEND_COLOR + 1) {
        // the blend colour, for blending in the shader (BlendColor/BlendAlpha factors): four halves
        uint16_t half[4];
        for (int c = 0; c < 4; c++) {
            __fp16 h = (__fp16)pass->blend_color[c];
            memcpy(&half[c], &h, 2);
        }
        dj.fs_fau[MALI_DRAW_BLEND_COLOR] = half[0] | (uint32_t)half[1] << 16;
        dj.fs_fau[MALI_DRAW_BLEND_COLOR + 1] = half[2] | (uint32_t)half[3] << 16;
    }
    // the fragment stage's textures and samplers, as a dispatch's
    uint32_t fs_nothing_at = 0;
    for (unsigned r = 0; r < p->fs.num_resources; r++) {
        const mali_kernel_resource &res = p->fs.resources[r];
        if (res.kind == HAKIOS_METAL_RESOURCE_TEXTURE) {
            if (res.index >= fs_texture_count || fs_textures[res.index] == nullptr) {
                // (nothing bound: the device's 4x4 of zeroes stands for it)
                if (res.index >= MALI_KERNEL_TEXTURES || nothingTexture(d) == nullptr) {
                    fail("draw with %s: nothing bound at [[texture(%u)]]", p->name, res.index);
                    return -1;
                }
                fs_nothing_at |= 1u << res.index;
            }
            dj.texture_count = res.index + 1 > dj.texture_count ? res.index + 1 : dj.texture_count;
        } else if (res.kind == HAKIOS_METAL_RESOURCE_SAMPLER) {
            if (res.inline_sampler) {
                // the function's own (constexpr sampler): made from its state word below, nothing bound
            } else if (res.index >= MALI_KERNEL_TEXTURES) {
                fail("draw with %s: nothing bound at [[sampler(%u)]]", p->name, res.index);
                return -1;
            } else if (res.index >= fs_sampler_count || fs_samplers[res.index] == nullptr) {
                // (nothing bound, 11-12: a nearest, clamped sampler stands for it)
                dj.samplers[res.index][0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
            }
            dj.sampler_count = res.index + 1 > dj.sampler_count ? res.index + 1 : dj.sampler_count;
        } else if (res.kind == HAKIOS_METAL_RESOURCE_IMAGE) {
            fail("draw with %s: a texture the fragment function writes (not on the Mali yet)", p->name);
            return -1;
        }
    }
    if (dj.texture_count != 0 && dj.sampler_count == 0) {
        dj.sampler_count = 1;
        dj.samplers[0][0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    }
    for (unsigned i = 0; i < dj.texture_count; i++) {
        dj.textures[i] = i < fs_texture_count && fs_textures[i] != nullptr ? fs_textures[i] :
            (fs_nothing_at & 1u << i) ? nothingTexture(d) : nullptr;
        { hakios_metal_texture *rt_ = (dj.textures[i]); while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent; if (rt_ != nullptr) rt_->reads++; }
    }
    for (unsigned i = 0; i < dj.sampler_count && i < fs_sampler_count; i++) {
        if (fs_samplers[i] != nullptr) {
            memcpy(dj.samplers[i], fs_samplers[i]->words, 32);
        }
    }
    for (unsigned r = 0; r < p->fs.num_resources; r++) {
        const mali_kernel_resource &res = p->fs.resources[r];
        if (res.kind != HAKIOS_METAL_RESOURCE_SAMPLER || !res.inline_sampler || res.index >= MALI_KERNEL_TEXTURES) {
            continue;
        }
        if (!inlineSampler(res.sampler_state, dj.samplers[res.index])) {
            fail("draw with %s: its constexpr sampler at %u: %s", p->name, res.index, gError.c_str());
            return -1;
        }
    }
    // the vertex stage's, the same way from what the command buffer was given (hakios_metal_cb_set_vertex_textures)
    {
        const uint32_t nearest = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
        uint32_t vs_nothing_at = 0;
        for (unsigned r = 0; r < p->vs.num_resources; r++) {
            const mali_kernel_resource &res = p->vs.resources[r];
            if (res.kind == HAKIOS_METAL_RESOURCE_TEXTURE) {
                if (res.index >= MALI_KERNEL_TEXTURES || (cb->vs_textures[res.index] == nullptr && nothingTexture(d) == nullptr)) {
                    fail("draw with %s: nothing bound at the vertex function's [[texture(%u)]]", p->name, res.index);
                    return -1;
                }
                if (cb->vs_textures[res.index] == nullptr) vs_nothing_at |= 1u << res.index;
                dj.vs_texture_count = res.index + 1 > dj.vs_texture_count ? res.index + 1 : dj.vs_texture_count;
            } else if (res.kind == HAKIOS_METAL_RESOURCE_SAMPLER) {
                if (res.index >= MALI_KERNEL_TEXTURES) {
                    fail("draw with %s: nothing bound at the vertex function's [[sampler(%u)]]", p->name, res.index);
                    return -1;
                }
                if (res.inline_sampler) {
                    if (!inlineSampler(res.sampler_state, dj.vs_samplers[res.index])) {
                        fail("draw with %s: its vertex function's constexpr sampler at %u: %s", p->name, res.index, gError.c_str());
                        return -1;
                    }
                } else if (cb->vs_sampler_set[res.index]) {
                    memcpy(dj.vs_samplers[res.index], cb->vs_samplers[res.index], 32);
                } else {
                    dj.vs_samplers[res.index][0] = nearest;
                }
                dj.vs_sampler_count = res.index + 1 > dj.vs_sampler_count ? res.index + 1 : dj.vs_sampler_count;
            } else if (res.kind == HAKIOS_METAL_RESOURCE_IMAGE) {
                fail("draw with %s: a texture the vertex function writes (not on the Mali yet)", p->name);
                return -1;
            }
        }
        if (dj.vs_texture_count != 0 && dj.vs_sampler_count == 0) {
            dj.vs_sampler_count = 1;
            dj.vs_samplers[0][0] = nearest;
        }
        for (unsigned i = 0; i < dj.vs_texture_count; i++) {
            hakios_metal_texture *t = cb->vs_textures[i] != nullptr ? cb->vs_textures[i] : (vs_nothing_at & 1u << i) ? nothingTexture(d) : nullptr;
            dj.vs_textures[i] = t;
            if (t != nullptr) {
                hakios_metal_texture *rt_ = t;
                while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent;
                if (rt_ != nullptr) rt_->reads++;
                t->retain();
                cb->held_textures.push_back(t);
            }
        }
    }
    dj.name = p->name;
    dj.program = p;
    dj.pos_spd = p->pos_spd;
    dj.vary_spd = p->vary_spd;
    dj.frag_spd = p->frag_spd;
    {
        uint32_t tls = p->vs.tls_size > p->fs.tls_size ? p->vs.tls_size : p->fs.tls_size;
        if (tls != 0) {
            if (!grow(d, d->tls, (uint64_t)npot((tls + 15) & ~15u) * d->max_threads * d->core_range)) {
                fail("draw with %s: no memory for %u bytes of spilled registers a thread", p->name, tls);
                return -1;
            }
            dj.tls_size = tls;
        }
    }
    {
        const char *why = nullptr;
        if (!zsOf(pass, dj, t->width, t->height, true, why)) {
            fail("%s: %s", p->name, why);
            return -1;
        }
    }
    dj.vary_size = p->vs.varying_bytes;
    dj.target_va = t->gpu();
    dj.width = t->width;
    dj.height = t->height;
    dj.row_bytes = t->stride;
    rtFormat(t->format, dj.rt1);
    dj.write = pass->store_action == 1;
    if (pass->load_action == 2) {
        dj.clear = rtClear(dj.rt1, pass->clear_color);
        dj.clear_high = rtClearHigh(dj.rt1, pass->clear_color);
    }
    if (loads) {
        // Load: the preload draw first (emitDraw), target n its texture n, over the whole target
        float full_scale[3] = { (float)t->width / 2, -(float)t->height / 2, 1 }, full_offset[3] = { (float)t->width / 2, (float)t->height / 2, 0 };
        if (!stagePushConstants(pre->vs, nullptr, nullptr, 0, full_scale, full_offset, 0, 0, dj.pre_vs_fau, used) ||
            !stagePushConstants(pre->fs, nullptr, nullptr, 0, full_scale, full_offset, 0, 0, dj.pre_fs_fau, used)) {
            return -1;
        }
        dj.load = true;
        dj.load0 = pass->load_action == 1;
        dj.pre_pos_spd = pre->pos_spd;
        dj.pre_frag_spd = pre->frag_spd;
        dj.pre_target = t;                              // (held with the draw's target below)
    }
    {
        static const bool no_ir = bootArg("metal_noir");
        const unsigned irzs = (dj.has_z ? 1u : 0u) | (dj.has_s ? 2u : 0u);
        hakios_metal_program *irp = no_ir ? nullptr : loads && irzs == 0 ? pre : preloadProgram(d, rts, irzs | ms4);
        float full_scale[3] = { (float)t->width / 2, -(float)t->height / 2, 1 }, full_offset[3] = { (float)t->width / 2, (float)t->height / 2, 0 };
        if (irp != nullptr &&
            stagePushConstants(irp->fs, nullptr, nullptr, 0, full_scale, full_offset, 0, 0, dj.ir_fs_fau, used)) {
            dj.ir_frag_spd = irp->frag_spd;
            dj.ir_target = t;
        }
        // DEPTH/STENCIL LOADED: by a pre-frame shader (the framebuffer descriptor's Z/S preload bits load nothing)
        const unsigned zsl = (dj.has_z && dj.z_load ? 1u : 0u) | (dj.has_s && dj.s_load ? 2u : 0u);
        if (zsl != 0) {
            hakios_metal_program *zp = preloadProgram(d, 0, zsl | ms4);
            if (zp == nullptr || !stagePushConstants(zp->fs, nullptr, nullptr, 0, full_scale, full_offset, 0, 0, dj.zs_fs_fau, used)) {
                fail("%s: depth/stencil Load: %s", p->name, d->preload_error.c_str());
                return -1;
            }
            dj.zs_frag_spd = zp->frag_spd;
        }
    }
    // the blend descriptor: enabled; the tile buffer loaded when the equation or the mask reads it; opaque (mode 1)
    // for a plain replace of every channel, fixed function (2) otherwise
    // (a target the fragment function reads -- framebuffer fetch, LD_TILE -- is loaded too, as panfrost's blend for a
    // shader that reads the tile buffer: in opaque mode the pixel before is not there to read)
    bool reads0 = p->blend_reads_dest || (p->fs.rt_read & 1);
    dj.blend[0] = 1u << 9 | (reads0 ? 1u : 0);
    dj.blend[1] = p->blend_equation;
    // (a raw target, R32Float: opaque always -- the fixed-function blender does not know it; what the shader read of it
    // came through LD_TILE, the tile loaded by blend[0]'s bit all the same)
    dj.blend[2] = (reads0 && !(dj.rt1 & kRtRaw) ? 2u : 1u) | 3 << 3;   // mode, 4 components, RT 0
    dj.blend[3] = rtBlend(dj.rt1);                           // the tile buffer's RGBA8
    dj.rts = rts;
    for (unsigned i = 1; i < rts; i++) {
        const hakios_metal_render_pass::hakios_metal_color_attachment &a = pass->more[i - 1];
        DrawJob::More &m = dj.more[i - 1];
        if (a.texture == nullptr) {
            // AN OUTPUT WITH NO ATTACHMENT, which Metal discards: a target of its own in the tile buffer, never written
            // back, whose blend writes no channel (CoreAnimation's fixed_frag_lph_cpf writes two in passes of one: its
            // draws were refused, pieces missing while the home screen moved)
            m.missing = true;
            m.row_bytes = (t->width * 4 + 63) & ~63u;
            {
                std::lock_guard<std::mutex> g(d->scratch_lock);
                uint64_t need = (uint64_t)m.row_bytes * t->height * t->samples;
                if (d->scratch_rt_size < need) {
                    if (d->scratch_rt_size != 0) {
                        d->freeBo(d->scratch_rt);       // (a draw in flight may still write the old one: rare, and only
                    }                                   //  its own scratch pixels, which nobody reads)
                    d->scratch_rt_size = d->newBo(d->scratch_rt, need, BIND_NOEXEC) ? need : 0;
                }
                if (d->scratch_rt_size == 0) {
                    fail("no scratch memory for a colour output with no attachment");
                    return -1;
                }
                m.va = d->scratch_rt.va;
            }
            m.rt1 = 19 << 3 | 2 << 8;
            m.write = false;
            m.load = false;
            m.clear = 0;
            m.texture = t;                              // (only for the preload's table: its blend writes nothing here)
            m.blend[0] = 1u << 9 | 1u;
            m.blend[1] = 0x132 | 0x132 << 12;           // colour mask 0
            m.blend[2] = 2u | 3 << 3 | i << 16;
            m.blend[3] = rtBlend(m.rt1);
            continue;
        }
        m.va = a.texture->gpu();
        m.row_bytes = a.texture->stride;
        a.texture->as_extra_rt++;
        /*
         * MEMORYLESS (11-10): CoreAnimation's second colour attachment is a memoryless texture -- an Apple GPU keeps it in
         * tile memory, its contents undefined past the pass; it is never sampled, copied or read (128682 draws, 0 reads).
         * The Mali has a tile buffer too: there it stays, never preloaded nor written back. Loaded and stored as any
         * other, a full-screen pass moved ~20 MB for it -- Spotlight's opening, ~8 such passes a frame at 2.1-2.8 ms
         * each. A draw whose fragment function reads it (framebuffer fetch) keeps the load and store, as before.
         * Boot-arg metal_nomemoryless: as before everywhere.
         */
        // (11-11: read by the fragment function too -- framebuffer fetch reads the tile buffer, and Metal never loads a
        // memoryless attachment from memory: CoreAnimation's cross-fade reads it, and its Mali pass loaded and wrote back
        // ~20 MB of stale memory for it. Boot-arg mlread=1: loaded and stored when read, as before)
        static const bool no_memoryless = bootArg("metal_nomemoryless"), mlread_old = bootArg("mlread=1");
        m.memoryless = a.texture->memoryless && !no_memoryless && (!(p->fs.rt_read & 1u << i) || !mlread_old);
        rtFormat(a.texture->format, m.rt1);
        m.write = a.store_action == 1;
        m.load = a.load_action == 1;
        m.clear = a.load_action == 2 ? rtClear(m.rt1, a.clear_color) : 0;
        m.clear_high = a.load_action == 2 ? rtClearHigh(m.rt1, a.clear_color) : 0;
        m.texture = a.texture;
        if (!(p->fs.rt_written & 1u << i)) {
            // not written by this draw: in the pass only to keep the pass whole, exactly as it is -- loaded and stored
            // back unchanged, never cleared here (as when it was left out: CoreAnimation's Settings showed another
            // cell's label when this draw applied the attachment's Clear/DontCare). When no draw of the Mali pass
            // touches it, it is not loaded nor stored at all (passElide)
            m.load = !m.memoryless;
            m.write = !m.memoryless;
            m.untouched = !(p->fs.rt_read & 1u << i) && !p->more_blend_reads_dest[i - 1];
            m.clear = 0;
            m.blend[0] = 1u << 9 | 1u;
            m.blend[1] = 0x132 | 0x132 << 12;           // colour mask 0
            m.blend[2] = 2u | 3 << 3 | i << 16;
            m.blend[3] = rtBlend(m.rt1);
            continue;
        }
        if (m.memoryless) {
            m.load = false;                             // (its clear, if any, still starts the tile buffer's copy)
            m.write = false;
        }
        bool reads = p->more_blend_reads_dest[i - 1] || (p->fs.rt_read & 1u << i);
        m.blend[0] = 1u << 9 | (reads ? 1u : 0);
        m.blend[1] = p->more_blend_equation[i - 1];
        m.blend[2] = (reads && !(m.rt1 & kRtRaw) ? 2u : 1u) | 3 << 3 | i << 16;
        m.blend[3] = rtBlend(m.rt1);
    }
    // Metal's point, line, line strip, triangle, triangle strip as Mali's draw modes (genxml v10 "Draw Mode")
    static const unsigned kDrawModes[5] = { 1, 2, 4, 8, 10 };
    dj.mode = kDrawModes[primitive];
    dj.count = vertex_count;
    dj.instances = instance_count;
    dj.vertex_start = ix ? (unsigned)ix->base_vertex : vertex_start;
    // DCD0: scissor to the bounding box, cull zero area (v11); the face culled and the front's winding.
    // THE WINDING IS FLIPPED (11-10): Metal names it in its normalized coordinates, y up; the Mali judges it where
    // the vertex shader's viewport transform puts the triangle, in the target's pixels, y down (scale -height/2,
    // above) -- the y flip turns Metal's counter-clockwise into the Mali's clockwise (panvk sets "front face CCW"
    // from Vulkan's framebuffer-space winding). Taken as it was, culling kept the faces turned away and dropped the
    // ones facing the viewer: sb_cube=1's cube showed its far faces, their numbers mirrored. metal_winding=metal:
    // as before (a test).
    static const bool winding_as_is = [] {
        char args[1024] = {0};
        size_t size = sizeof(args) - 1;
        return sysctlbyname("kern.bootargs", args, &size, nullptr, 0) == 0 && strstr(args, "metal_winding=metal") != nullptr;
    }();
    bool front_ccw = winding_as_is ? pass->winding == 1 : pass->winding == 0;
    dj.dcd0 = 1u << 23 | 1u << 26 | (front_ccw ? 1u << 16 : 0) | (pass->cull_mode == 1 ? 1u << 17 : 0) |
        (pass->cull_mode == 2 ? 1u << 18 : 0) | (dj.samples > 1 ? 1u << 19 : 0);  // (19: multisample)
    if (dj.zs && ((dj.zsd[4] >> 29) != 7 || (dj.zsd[4] & 1u << 27) || (dj.zsd[0] >> 31))) {
        // a depth or stencil test or write: pixels killed weakly early, Z/S updated late (a fragment the shader
        // discards must not write them; whether a shader discards is not tracked here)
        dj.dcd0 |= 2u << 2 | 3u << 4;
    }
    dj.dcd1 = 0xffff | 1u << 16;
    // (11-12 night) THE READ MASK: the colour targets the fragment function reads from the tile buffer (framebuffer fetch,
    // or the blend computed in the shader) -- Mesa's cfg.read_mask = rt_read. Without it two draws over the same pixels
    // in one pass each read the tile as it was before either: WebKit's path coverage, added into an R32Float target, came
    // out as noise (mali_draw_test "blending into R32Float": two draws added once)
    dj.dcd2 = 1u << 12 | 1u << 24 | 1u << 25 | (p->fs.rt_read & 0xffu);
    dj.tiler_flags = dj.mode | 1u << 22 | (dj.vary_spd ? 1u << 18 : 0);   // basic FIFO; the varying shader
    if (ix) {
        // the indices: 16 or 32 bits (Mali's index type 2, 3), primitive restart at the type's all-ones as Metal has it
        dj.index_va = ix->buffer->bo.va + ix->offset;
        dj.index_size = ix->buffer->length - ix->offset;
        dj.tiler_flags |= (ix->type ? 3u : 2u) << 8 | 1u << 19;
        used.push_back(ix->buffer);
    }
    unsigned x0 = 0, y0 = 0, x1 = t->width, y1 = t->height;
    if (scissor != nullptr) {
        x0 = scissor->x < t->width ? scissor->x : t->width;
        y0 = scissor->y < t->height ? scissor->y : t->height;
        x1 = scissor->x + scissor->width < t->width ? scissor->x + scissor->width : t->width;
        y1 = scissor->y + scissor->height < t->height ? scissor->y + scissor->height : t->height;
        if (x1 <= x0 || y1 <= y0) {
            return 0;                                   // nothing inside the scissor
        }
    }
    dj.scissor_min = x0 | y0 << 16;
    dj.scissor_max = (x1 - 1) | (y1 - 1) << 16;
    // recorded: the GPU's state kept alive until it is done
    Job j = {};
    j.shader = DRAW;
    j.dst = cb->draws.size();
    cb->draws.push_back(dj);
    cb->blits.push_back(j);
    p->retain();
    cb->programs.push_back(p);
    t->retain();
    cb->held_textures.push_back(t);
    for (hakios_metal_texture *x : { pass->depth, pass->stencil }) {
        if (x != nullptr && dj.zs) {
            x->retain();
            cb->held_textures.push_back(x);
        }
    }
    for (unsigned i = 1; i < dj.rts; i++) {
        dj.more[i - 1].texture->retain();
        cb->held_textures.push_back(dj.more[i - 1].texture);
    }
    for (unsigned i = 0; i < dj.texture_count; i++) {
        if (dj.textures[i] != nullptr) {
            dj.textures[i]->retain();
            cb->held_textures.push_back(dj.textures[i]);
        }
    }
    for (auto *b : used) {
        cb->hold(b);
    }
    return 0;
}
}  // namespace

int
hakios_metal_cb_draw(hakios_metal_cb *cb, hakios_metal_program *program, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned vertex_start, unsigned vertex_count, unsigned instance_count, hakios_metal_buffer *const *vs_buffers,
    const uint64_t *vs_offsets, unsigned vs_count, hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets,
    unsigned fs_count)
{
    return drawImpl(cb, program, pass, viewport, scissor, primitive, vertex_start, vertex_count, instance_count, 0,
        vs_buffers, vs_offsets, vs_count, fs_buffers, fs_offsets, fs_count, nullptr, 0, nullptr, 0);
}

int
hakios_metal_cb_draw2(hakios_metal_cb *cb, hakios_metal_program *program, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned vertex_start, unsigned vertex_count, unsigned instance_count, hakios_metal_buffer *const *vs_buffers,
    const uint64_t *vs_offsets, unsigned vs_count, hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets,
    unsigned fs_count, hakios_metal_texture *const *fs_textures, unsigned fs_texture_count,
    hakios_metal_sampler *const *fs_samplers, unsigned fs_sampler_count)
{
    return drawImpl(cb, program, pass, viewport, scissor, primitive, vertex_start, vertex_count, instance_count, 0,
        vs_buffers, vs_offsets, vs_count, fs_buffers, fs_offsets, fs_count, fs_textures, fs_texture_count, fs_samplers,
        fs_sampler_count);
}

int
hakios_metal_cb_draw3(hakios_metal_cb *cb, hakios_metal_program *program, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned vertex_start, unsigned vertex_count, unsigned instance_count, unsigned base_instance,
    hakios_metal_buffer *const *vs_buffers, const uint64_t *vs_offsets, unsigned vs_count,
    hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets, unsigned fs_count,
    hakios_metal_texture *const *fs_textures, unsigned fs_texture_count, hakios_metal_sampler *const *fs_samplers,
    unsigned fs_sampler_count)
{
    return drawImpl(cb, program, pass, viewport, scissor, primitive, vertex_start, vertex_count, instance_count,
        base_instance, vs_buffers, vs_offsets, vs_count, fs_buffers, fs_offsets, fs_count, fs_textures, fs_texture_count,
        fs_samplers, fs_sampler_count);
}

// ---- what upstream's CoreAnimation plug-in (userland/src/HakiosMetalCA.m) asks besides: served, or refused by name ----

namespace {
const MaliFormat *
findFormat(unsigned pixel_format)
{
    for (const MaliFormat &m : kFormats) {
        if (m.metal == pixel_format) {
            return &m;
        }
    }
    return nullptr;
}

// a texture over memory not its own: OWNER's BO at OFFSET, STRIDE apart
hakios_metal_texture *
textureOver(hakios_metal_device *d, const Bo &bo, uint64_t offset, uint32_t stride, unsigned type, const MaliFormat *f,
    unsigned pixel_format, unsigned width, unsigned height, unsigned usage, unsigned origin)
{
    auto *t = new hakios_metal_texture;
    t->device = d;
    t->bo = bo;
    t->offset = offset;
    t->type = type;
    t->format = pixel_format;
    t->width = width;
    t->height = height;
    t->depth = 1;
    t->usage = usage;
    t->mali_format = f->mali;
    if (f->metal == 1) t->swizzle = kAlphaSwizzle;
    t->bytes = f->bytes;
    t->bw = f->bw;
    t->bh = f->bh;
    t->stride = stride;
    t->origin = origin;
    return t;
}
}  // namespace

/*
 * MIRRORS (11-10). The Mali samples a linear texture only from rows a multiple of 64 bytes apart: an IOSurface of other
 * rows (the App Library's background, a material's snapshot of 4320-byte rows) read so came out as nothing -- black. The
 * plug-in's way round it, a copy on the CPU (HakiosMetalCA.m, -hakiosShadow:), cost ~160 ms as the App Library came: 10
 * MB taken, written back after the draw into it, taken again. Here a texture over such a surface is over its MIRROR: a
 * texture of 64-byte rows, one for all the textures over the surface; filled from the surface on the GPU (copy_rows_u32)
 * when made and when the plug-in says the CPU changed the surface (hakios_metal_texture_surface_changed), and copied
 * back into the surface by every command buffer that draws into it, as its last job (mirrorsBack, from markBusy's list).
 * boot-arg metal_nomirror: such textures refused (the plug-in's CPU copy).
 */
static Job
mirrorJob(const hakios_metal_device::Mirror &m, bool in)
{
    Job j = {};
    j.shader = COPY_ROWS;
    j.dst = in ? m.tex->gpu() : m.surface_va;
    j.src = in ? m.surface_va : m.tex->gpu();
    j.src_stride = in ? m.row_bytes : m.tex->stride;
    j.dst_stride = in ? m.tex->stride : m.row_bytes;
    j.value = m.row_words;
    j.count = m.row_words * m.rows;
    return j;
}

// the surface into its mirror, now (the mirror lock held): a command buffer of its own, waited for
static bool
mirrorFill(hakios_metal_device *d, const hakios_metal_device::Mirror &m)
{
    if (d->mirror_queue == nullptr && (d->mirror_queue = hakios_metal_queue_new(d)) == nullptr) {
        return false;
    }
    hakios_metal_cb *cb = hakios_metal_cb_new(d->mirror_queue);
    if (cb == nullptr) {
        return false;
    }
    cb->blits.push_back(mirrorJob(m, true));
    hakios_metal_cb_commit(cb, nullptr, nullptr);
    hakios_metal_cb_wait(cb);
    bool ok = cb->status == HAKIOS_METAL_COMPLETED;
    hakios_metal_cb_free(cb);
    return ok;
}

static hakios_metal_texture *
mirrorOf(hakios_metal_device *d, unsigned id, const Bo &bo, const hakios_metal_device::SurfaceInfo &info, const MaliFormat *f,
    unsigned pixel_format)
{
    std::lock_guard<std::mutex> g(d->mirror_lock);
    auto it = d->mirrors.find(id);
    if (it != d->mirrors.end()) {
        it->second.refs++;
        return it->second.tex;
    }
    hakios_metal_device::Mirror m = {};
    m.tex = hakios_metal_texture_new_levels(d, 2, pixel_format, info.width, info.height, 1, 1 | 2 | 4);
    if (m.tex == nullptr) {
        return nullptr;
    }
    m.refs = 1;
    m.surface_va = bo.va + info.offset;
    m.row_bytes = info.row_bytes;
    m.row_words = info.width * f->bytes / 4;
    m.rows = info.height;
    if (!mirrorFill(d, m)) {
        m.tex->release();
        fail("IOSurface %u's mirror could not be filled", id);
        return nullptr;
    }
    static unsigned told;
    if (told++ < 8) {
        fprintf(stderr, "hakios-metal: IOSurface %u (%ux%u, rows of %u bytes) sampled through a mirror of rows of %u, "
            "copied on the GPU\n", id, info.width, info.height, info.row_bytes, m.tex->stride);
    }
    d->mirrors[id] = m;
    return m.tex;
}

static void
mirrorRelease(hakios_metal_device *d, unsigned surface)
{
    hakios_metal_texture *gone = nullptr;
    {
        std::lock_guard<std::mutex> g(d->mirror_lock);
        auto it = d->mirrors.find(surface);
        if (it == d->mirrors.end() || --it->second.refs > 0) {
            return;
        }
        gone = it->second.tex;
        d->mirrors.erase(it);
    }
    gone->release();
}

static void
mirrorsBack(hakios_metal_cb *cb, hakios_metal_device *d)
{
    if (cb->mirror_out.empty()) {
        return;
    }
    std::lock_guard<std::mutex> g(d->mirror_lock);
    for (unsigned id : cb->mirror_out) {
        auto it = d->mirrors.find(id);
        if (it != d->mirrors.end()) {
            cb->blits.push_back(mirrorJob(it->second, false));
            it->second.tex->retain();
            cb->held_textures.push_back(it->second.tex);
        }
    }
    cb->mirror_out.clear();
}

void
hakios_metal_texture_surface_changed(hakios_metal_texture *t)
{
    while (t != nullptr && t->origin == 4) {
        t = t->parent;
    }
    if (t == nullptr || !t->mirrored) {
        return;
    }
    hakios_metal_device *d = t->device;
    std::lock_guard<std::mutex> g(d->mirror_lock);
    auto it = d->mirrors.find(t->surface);
    if (it != d->mirrors.end()) {
        mirrorFill(d, it->second);
    }
}

void
hakios_metal_texture_set_memoryless(hakios_metal_texture *t, int on)
{
    if (t != nullptr) {
        t->memoryless = on != 0;
    }
}

int
hakios_metal_texture_mirrored(hakios_metal_texture *t)
{
    while (t != nullptr && t->origin == 4) {
        t = t->parent;
    }
    return t != nullptr && t->mirrored;
}

hakios_metal_texture *
hakios_metal_texture_new_iosurface_plane(hakios_metal_device *d, unsigned surface_id, uint64_t plane_delta, unsigned row_bytes,
    unsigned pixel_format, unsigned width, unsigned height, unsigned usage)
{
    // a later plane of a planar surface, as a texture of the format and size asked (SpringBoard asks RGBA8 of a '420f'
    // surface's CbCr plane: its bytes four at a time): the same memory, further on
    const MaliFormat *f = findFormat(pixel_format);
    if (f == nullptr || f->bw != 1 || f->bh != 1 || width == 0 || height == 0) {
        fail("a texture over a plane of IOSurface %u of MTLPixelFormat %u (not served on the Mali yet)", surface_id, pixel_format);
        return nullptr;
    }
    Bo bo;
    hakios_metal_device::SurfaceInfo info;
    if (!d->surfaceBo(surface_id, bo, info)) {
        return nullptr;
    }
    const uint64_t offset = info.offset + plane_delta;
    if ((uint64_t)width * f->bytes > row_bytes || offset + (uint64_t)row_bytes * (height - 1) + (uint64_t)width * f->bytes > info.size ||
        (row_bytes & 63) || (offset & 63)) {
        d->surfaceRelease(surface_id);
        fail("a texture of %u x %u, MTLPixelFormat %u over a plane of IOSurface %u at %llu, rows of %u bytes, in %llu bytes (rows "
            "and start of multiples of 64 bytes, inside the surface)", width, height, pixel_format, surface_id,
            (unsigned long long)offset, row_bytes, (unsigned long long)info.size);
        return nullptr;
    }
    hakios_metal_texture *t = textureOver(d, bo, offset, row_bytes, 2, f, pixel_format, width, height, usage, 1);
    t->surface = surface_id;
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_iosurface(hakios_metal_device *d, unsigned surface_id, unsigned plane, unsigned pixel_format,
    unsigned width, unsigned height, unsigned usage)
{
    const MaliFormat *f = findFormat(pixel_format);
    if (f == nullptr || plane != 0) {
        fail("a texture over IOSurface %u of MTLPixelFormat %u, plane %u (not served on the Mali yet)", surface_id,
            pixel_format, plane);
        return nullptr;
    }
    Bo bo;
    hakios_metal_device::SurfaceInfo info;
    if (!d->surfaceBo(surface_id, bo, info)) {
        return nullptr;
    }
    if (width > info.width || height > info.height || (uint64_t)width * f->bytes > info.row_bytes ||
        info.offset + (uint64_t)info.row_bytes * (height - 1) + (uint64_t)width * f->bytes > info.size) {
        d->surfaceRelease(surface_id);
        fail("a texture of %u x %u over IOSurface %u of %u x %u (rows of %u bytes)", width, height, surface_id, info.width,
            info.height, info.row_bytes);
        return nullptr;
    }
    static const bool no_mirror = bootArg("metal_nomirror");
    if (((info.row_bytes & 63) || (info.offset & 63)) && !no_mirror && f->bw == 1 && f->bh == 1 &&
        ((info.row_bytes | info.offset | info.width * f->bytes) & 3) == 0) {
        hakios_metal_texture *m = mirrorOf(d, surface_id, bo, info, f, pixel_format);
        if (m == nullptr) {
            d->surfaceRelease(surface_id);
            return nullptr;
        }
        hakios_metal_texture *t = textureOver(d, m->bo, m->offset, m->stride, 2, f, pixel_format, width, height, usage, 1);
        t->surface = surface_id;
        t->mirrored = true;
        return t;
    }
    if ((info.row_bytes & 63) || (info.offset & 63)) {
        // rows not a multiple of 64 bytes, as buffer textures are held to (hakios_metal_texture_new_buffer): sampled so,
        // the App Library's background -- a material's snapshot SpringBoard asks for with 4320-byte rows, drawn by
        // backboardd into the surface right (its pixels checked) -- came out black on the panel (11-10). Refused, the
        // plug-in reads such a surface through a copy of its rows and writes what is drawn into it back, as it did on
        // the Adreno (HakiosMetalCA.m, -hakiosShadow:)
        d->surfaceRelease(surface_id);
        fail("a texture over IOSurface %u of rows of %u bytes at offset %u: the Mali takes rows of multiples of 64 bytes",
            surface_id, info.row_bytes, (unsigned)info.offset);
        return nullptr;
    }
    hakios_metal_texture *t = textureOver(d, bo, info.offset, info.row_bytes, 2, f, pixel_format, width, height, usage, 1);
    t->surface = surface_id;
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_buffer(hakios_metal_buffer *b, unsigned type, unsigned pixel_format, unsigned width,
    unsigned height, unsigned usage, uint64_t offset, uint64_t bytes_per_row)
{
    const MaliFormat *f = findFormat(pixel_format);
    if (b == nullptr || f == nullptr || type != 2 || (offset & 63) || (bytes_per_row & 63) ||
        bytes_per_row < (uint64_t)width * f->bytes || offset + bytes_per_row * height > b->length) {
        fail("a texture over a buffer: of MTLPixelFormat %u, type %u, offset %llu, rows of %llu bytes (64-byte aligned, "
            "2D, inside the buffer)", pixel_format, type, (unsigned long long)offset, (unsigned long long)bytes_per_row);
        return nullptr;
    }
    hakios_metal_texture *t = textureOver(b->device, b->bo, offset, (uint32_t)bytes_per_row, type, f, pixel_format, width,
        height, usage, 2);
    b->retain();
    t->buffer = b;
    return t;
}

namespace {
// levels BASE .. BASE + COUNT - 1 of P as a texture of their own over P's memory (P held): level BASE its level 0; a
// view of one level is described as any one-level texture, of more through P's plane descriptors
hakios_metal_texture *
levelsOf(hakios_metal_texture *p, const MaliFormat *f, unsigned pixel_format, unsigned type, unsigned base, unsigned count,
    unsigned layer = 0)
{
    // (LAYER of a layered texture: that layer's levels, a texture of one layer)
    hakios_metal_texture *t = textureOver(p->device, p->bo, p->offset + p->layer_stride * layer + p->levelOffset(base),
        p->levelStride(base), type, f, pixel_format, p->levelWidth(base), p->levelHeight(base), p->usage, 4);
    t->depth = p->depth;
    t->levels = count;
    for (unsigned l = 0; l < count; l++) {
        t->level_off[l] = p->levelOffset(base + l) - p->levelOffset(base);
        t->level_stride[l] = p->levelStride(base + l);
    }
    t->planes_off = count > 1 && p->planes_off != 0 ? p->planes_off + 32 * ((uint64_t)layer * p->levels + base) : 0;
    if (p->stencil_level_stride[base] != 0) {
        const uint64_t lo = p->levelOffset(base);
        t->stencil_off = p->stencil_level_off[base] - lo;
        t->stencil_stride = p->stencil_level_stride[base];
        for (unsigned l = 0; l < count; l++) {
            t->stencil_level_off[l] = p->stencil_level_off[base + l] - lo;
            t->stencil_level_stride[l] = p->stencil_level_stride[base + l];
        }
    } else if (base == 0) {
        t->stencil_off = p->stencil_off;
        t->stencil_stride = p->stencil_stride;
    }
    p->retain();
    t->parent = p;
    t->surface = p->surface;
    return t;
}

// a render pass's attachment: level LEVEL (and a 3D texture's depth plane SLICE) of T as a one-level 2D texture, or
// nullptr with the reason
hakios_metal_texture *
attachmentOf(hakios_metal_texture *t, unsigned level, unsigned slice)
{
    if (level >= t->levels || slice >= (t->type == 7 ? t->depth : t->layers)) {
        fail("an attachment at level %u, slice %u of a texture of %u level(s) and %u slice(s)", level, slice, t->levels,
            t->type == 7 ? t->depth : t->layers);
        return nullptr;
    }
    if (t->layers > 1) {
        // a 2D array's layer or a cube's face: that layer's level, a 2D texture
        hakios_metal_texture *v = levelsOf(t, findFormat(t->format), t->format, t->type == 1 ? 0 : 2, level, 1, slice);
        v->depth = 1;
        return v;
    }
    hakios_metal_texture *v = levelsOf(t, findFormat(t->format), t->format, t->type == 7 ? 2 : t->type, level, 1);
    if (slice != 0) {
        v->offset += (uint64_t)v->stride * v->height * slice;
    }
    v->depth = 1;
    return v;
}
}  // namespace

hakios_metal_texture *
hakios_metal_texture_new_view(hakios_metal_texture *p, unsigned pixel_format, unsigned type, unsigned base_level,
    unsigned levels, unsigned base_slice, unsigned slices)
{
    const MaliFormat *f = findFormat(pixel_format);
    if (p != nullptr && f != nullptr && f->bytes == p->bytes && p->layers > 1 && levels != 0 &&
        base_level + levels <= p->levels && slices != 0 && base_slice + slices <= p->layers) {
        // SLICES OF A LAYERED TEXTURE: one layer (any levels of it) as a 2D or 1D texture; or a run of whole layers
        // (every level) as an array or a cube -- the layers' plane descriptors are one after the other
        const bool one = slices == 1 && (type == 2 || type == 0 || type == 3 || type == 1);
        const bool run = (base_level == 0 && levels == p->levels) &&
                         (type == 3 || type == 1 || ((type == 5 || type == 6) && slices % 6 == 0));
        if (one || run) {
            hakios_metal_texture *v = levelsOf(p, f, pixel_format, one ? (type == 3 ? 2 : type == 1 ? 0 : type) : type,
                base_level, levels, base_slice);
            if (!one || slices > 1) {
                v->layers = slices;
                v->layer_stride = p->layer_stride;
                v->planes_off = p->planes_off + 32 * (uint64_t)base_slice * p->levels;
            }
            return v;
        }
    }
    if (p == nullptr || f == nullptr || f->bytes != p->bytes || type != p->type || levels == 0 ||
        base_level + levels > p->levels || base_slice != 0 || slices != 1) {
        fail("a texture view of MTLPixelFormat %u, type %u, levels %u+%u, slices %u+%u of one with %u level(s) (the Mali "
            "build views levels of a texture, as a format of the same size, all its slices)", pixel_format, type,
            base_level, levels, base_slice, slices, p ? p->levels : 0);
        return nullptr;
    }
    return levelsOf(p, f, pixel_format, type, base_level, levels);
}

hakios_metal_texture *
hakios_metal_texture_new_swizzled(hakios_metal_texture *p, unsigned red, unsigned green, unsigned blue, unsigned alpha)
{
    // MTLTextureSwizzle (0 zero, 1 one, 2 red .. 5 alpha) as Mali's component selects (0 R .. 3 A, 4 zero, 5 one), over
    // what P already reads
    static const uint32_t to_mali[6] = { 4, 5, 0, 1, 2, 3 };
    if (p == nullptr || red > 5 || green > 5 || blue > 5 || alpha > 5) {
        fail("a texture swizzle past MTLTextureSwizzleAlpha");
        return nullptr;
    }
    hakios_metal_texture *t = textureOver(p->device, p->bo, p->offset, p->stride, p->type, findFormat(p->format), p->format,
        p->width, p->height, p->usage, 4);
    uint32_t base = p->swizzle ? p->swizzle : kIdentitySwizzle;
    auto pick = [&](unsigned m) -> uint32_t {
        uint32_t sel = to_mali[m];
        return sel < 4 ? (base >> (3 * sel)) & 7 : sel;
    };
    t->swizzle = pick(red) | pick(green) << 3 | pick(blue) << 6 | pick(alpha) << 9;
    p->retain();
    t->parent = p;
    t->surface = p->surface;
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_levels(hakios_metal_device *d, unsigned type, unsigned pixel_format, unsigned width,
    unsigned height, unsigned levels, unsigned usage)
{
    hakios_metal_texture *t = hakios_metal_texture_new(d, type, pixel_format, width, height, usage);
    if (t == nullptr || levels <= 1) {
        return t;
    }
    // MIPMAPPED: every level in one BO, each level's rows 64-byte aligned after the one before, then the levels' plane
    // descriptors ("Generic Plane", level 0 first: the texture descriptor's Surfaces, Levels and Maximum LOD)
    unsigned most = 1;
    while (most < 16 && ((width >> most) || (type != 0 && (height >> most)))) most++;
    if (levels > most) levels = most;
    t->levels = levels;
    uint64_t at = 0;
    for (unsigned l = 0; l < levels; l++) {
        t->level_off[l] = at;
        t->level_stride[l] = ((t->levelWidth(l) + t->bw - 1) / t->bw * t->bytes + 63) & ~63u;
        at += ((uint64_t)t->level_stride[l] * t->levelRows(l) + 63) & ~63ULL;
    }
    if (pixel_format == 260) {
        // Depth32Float_Stencil8: a stencil plane a level, after the depth levels (the one level's plane hakios_metal_
        // texture_new made had no room in the bigger BO: stencil written to a level past 0 landed on the planes)
        for (unsigned l = 0; l < levels; l++) {
            t->stencil_level_off[l] = at;
            t->stencil_level_stride[l] = (t->levelWidth(l) + 63) & ~63u;
            at += ((uint64_t)t->stencil_level_stride[l] * t->levelHeight(l) + 63) & ~63ULL;
        }
        t->stencil_off = t->stencil_level_off[0];
        t->stencil_stride = t->stencil_level_stride[0];
    }
    Bo bigger;
    if (!d->newBo(bigger, at + 32 * levels, BIND_NOEXEC)) {
        t->release();
        return nullptr;
    }
    d->freeBo(t->bo);
    t->bo = bigger;
    t->planes_off = at;
    for (unsigned l = 0; l < levels; l++) {
        auto *plane = (uint32_t *)((uint8_t *)t->bo.cpu + at + 32 * l);
        uint64_t va = t->gpu() + t->level_off[l];
        uint32_t w = t->levelWidth(l), h = t->levelHeight(l);
        memset(plane, 0, 32);
        plane[0] = planeWord0(t);
        plane[1] = t->level_stride[l] * t->levelRows(l);
        plane[2] = (uint32_t)va;
        plane[3] = (uint32_t)(va >> 32);
        plane[4] = t->level_stride[l];
        plane[6] = t->level_stride[l] * t->levelRows(l);
        plane[7] = (w - 1) | (h - 1) << 16;
    }
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_multisample(hakios_metal_device *d, unsigned pixel_format, unsigned width, unsigned height,
    unsigned samples, unsigned usage)
{
    if (samples <= 1) {
        return hakios_metal_texture_new(d, 2, pixel_format, width, height, usage);
    }
    if (samples != 4) {
        fail("a multisample texture of %u samples (the Mali-G715 has no 2x sample pattern -- Mesa serves it from v12; 4 is "
            "served)", samples);
        return nullptr;
    }
    // FOUR SAMPLES A PIXEL, LAYERED: each sample a plane of the texture's rows, one after the other (the render target's
    // and depth/stencil's "Layered" write-back, the texture descriptor's sample count with the plane as its slice
    // stride); Depth32Float_Stencil8's stencil planes after the depth planes
    hakios_metal_texture *t = hakios_metal_texture_new(d, 2, pixel_format, width, height, usage);
    if (t == nullptr) {
        return nullptr;
    }
    uint64_t plane = ((uint64_t)t->stride * t->levelRows(0) + 63) & ~63ULL;
    if (plane != (uint64_t)t->stride * t->levelRows(0)) {
        // (the planes are stride * rows apart: a texture whose plane is not a multiple of 64 bytes is not served)
        t->release();
        fail("a multisample texture of %ux%u, MTLPixelFormat %u: its sample planes would not be 64-byte aligned", width,
            height, pixel_format);
        return nullptr;
    }
    uint64_t size = plane * samples;
    if (pixel_format == 260) {
        t->stencil_off = size;
        size += (uint64_t)t->stencil_stride * height * samples;
    }
    Bo bigger;
    if (!d->newBo(bigger, size, BIND_NOEXEC)) {
        t->release();
        return nullptr;
    }
    d->freeBo(t->bo);
    t->bo = bigger;
    t->samples = samples;
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_layered(hakios_metal_device *d, unsigned type, unsigned pixel_format, unsigned width,
    unsigned height, unsigned layers, unsigned levels, unsigned usage)
{
    if (layers <= 1 && (type == 2 || type == 3)) {
        return hakios_metal_texture_new_levels(d, 2, pixel_format, width, height, levels, usage);
    }
    const bool cube = type == 5 || type == 6;
    if ((type != 1 && type != 3 && !cube) || layers == 0 || layers > 2048 || (cube && (layers % 6 != 0 || width != height)) ||
        (type == 1 && height != 1) || pixel_format == 253 || pixel_format == 260) {
        fail("a layered texture: type %u, %u layers, %u x %u, MTLPixelFormat %u (the Mali build makes 1D and 2D arrays, cubes "
            "and cube arrays up to 2048 layers; no stencil)", type, layers, width, height, pixel_format);
        return nullptr;
    }
    hakios_metal_texture *t = hakios_metal_texture_new(d, type == 1 ? 0 : 2, pixel_format, width, height, usage);
    if (t == nullptr) {
        return nullptr;
    }
    t->type = type;
    unsigned most = 1;
    while (most < 16 && ((width >> most) || (type != 1 && (height >> most)))) most++;
    if (levels == 0) levels = 1;
    if (levels > most) levels = most;
    t->levels = levels;
    uint64_t at = 0;
    for (unsigned l = 0; l < levels; l++) {
        t->level_off[l] = at;
        t->level_stride[l] = ((t->levelWidth(l) + t->bw - 1) / t->bw * t->bytes + 63) & ~63u;
        at += ((uint64_t)t->level_stride[l] * t->levelRows(l) + 63) & ~63ULL;
    }
    Bo bigger;
    if (!d->newBo(bigger, at * layers + 32ULL * levels * layers, BIND_NOEXEC)) {
        t->release();
        return nullptr;
    }
    d->freeBo(t->bo);
    t->bo = bigger;
    t->layers = layers;
    t->layer_stride = at;
    t->planes_off = at * layers;
    for (unsigned y = 0; y < layers; y++) {
        for (unsigned l = 0; l < levels; l++) {
            auto *plane = (uint32_t *)((uint8_t *)t->bo.cpu + t->planes_off + 32 * ((uint64_t)y * levels + l));
            uint64_t va = t->gpu() + at * y + t->level_off[l];
            uint32_t w = t->levelWidth(l), h = t->levelHeight(l);
            memset(plane, 0, 32);
            plane[0] = planeWord0(t);
            plane[1] = t->level_stride[l] * t->levelRows(l);
            plane[2] = (uint32_t)va;
            plane[3] = (uint32_t)(va >> 32);
            plane[4] = t->level_stride[l];
            plane[6] = t->level_stride[l] * t->levelRows(l);
            plane[7] = (w - 1) | (h - 1) << 16;
        }
    }
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_3d(hakios_metal_device *d, unsigned pixel_format, unsigned width, unsigned height,
    unsigned depth, unsigned usage)
{
    // a 2D texture's rows, DEPTH planes of them one after the other (describe(): a 3D texture, the plane's slice size)
    if (depth == 0 || depth > 2048 || width > 2048 || height > 2048) {
        fail("a 3D texture of %u x %u x %u (the Mali build makes them up to 2048 each way)", width, height, depth);
        return nullptr;
    }
    hakios_metal_texture *t = hakios_metal_texture_new(d, 2, pixel_format, width, height, usage);
    if (t == nullptr || depth == 1) {
        if (t != nullptr) t->type = 7;
        return t;
    }
    Bo bigger;
    if (!d->newBo(bigger, (uint64_t)t->stride * height * depth, BIND_NOEXEC)) {
        t->release();
        return nullptr;
    }
    d->freeBo(t->bo);
    t->bo = bigger;
    t->type = 7;
    t->depth = depth;
    return t;
}

hakios_metal_texture *
hakios_metal_texture_new_compressed(hakios_metal_device *d, unsigned pixel_format, unsigned width, unsigned height,
    unsigned usage)
{
    // (the plug-in's UBWC on the Adreno; on the Mali, AFBC is not done yet: a linear texture, read the same way)
    return hakios_metal_texture_new(d, 2, pixel_format, width, height, usage);
}

unsigned hakios_metal_texture_iosurface(hakios_metal_texture *t) { return t ? t->surface : 0; }
unsigned hakios_metal_texture_levels(hakios_metal_texture *t) { return t ? t->levels : 0; }
unsigned hakios_metal_texture_samples(hakios_metal_texture *t) { return t ? t->samples : 0; }
unsigned hakios_metal_texture_depth(hakios_metal_texture *t) { return t ? (t->layers > 1 ? t->layers : t->depth) : 0; }
unsigned hakios_metal_texture_bytes_per_pixel(hakios_metal_texture *t) { return t ? t->bytes : 0; }
unsigned hakios_metal_texture_origin(hakios_metal_texture *t) { return t ? (t->origin == 4 ? (t->parent ? hakios_metal_texture_origin(t->parent) : 0) : t->origin) : 0; }
uint64_t hakios_metal_texture_allocated_bytes(hakios_metal_texture *t) { return t ? t->bo.size : 0; }
uint64_t hakios_metal_texture_resource_id(hakios_metal_texture *t) { return (uint64_t)(uintptr_t)t; }
uint64_t hakios_metal_sampler_resource_id(hakios_metal_sampler *s) { return (uint64_t)(uintptr_t)s; }
uint64_t hakios_metal_device_allocated_bytes(hakios_metal_device *d) { (void)d; return 0; }

int
hakios_metal_format_block(unsigned pixel_format, unsigned *block_w, unsigned *block_h, unsigned *bytes)
{
    const MaliFormat *f = findFormat(pixel_format);
    if (f == nullptr) {
        fail("MTLPixelFormat %u is not served on the Mali yet", pixel_format);
        return 0;
    }
    *block_w = f->bw;
    *block_h = f->bh;
    *bytes = f->bytes;
    return 1;
}

int
hakios_metal_texture_plan_bytes(unsigned type, unsigned pixel_format, unsigned width, unsigned height, unsigned depth,
    unsigned levels, unsigned samples, unsigned usage, uint64_t *bytes)
{
    (void)usage;
    const MaliFormat *f = findFormat(pixel_format);
    // (11-12: the layered types too -- DEPTH their layers, a cube's faces among them)
    const bool layered = type == 1 || type == 3 || type == 5 || type == 6;
    unsigned layers = layered ? (depth ? depth : 1) : 1;
    if (f == nullptr || (type != 2 && type != 4 && !layered) || (!layered && depth > 1) || (samples != 1 && samples != 4) ||
        (samples > 1 && (levels > 1 || layered)) || width == 0 || height == 0) {
        fail("a texture of type %u, MTLPixelFormat %u, %u levels, %u samples (not on the Mali yet)", type, pixel_format,
            levels, samples);
        return -1;
    }
    // as hakios_metal_texture_new(_levels) lays it out: each level's rows (of blocks) 64-byte aligned, the levels' plane
    // descriptors after them
    uint64_t at = 0;
    unsigned l = 0;
    for (; l < (levels ? levels : 1) && l < 16 && (l == 0 || (width >> l) || (height >> l)); l++) {
        unsigned w = width >> l ? width >> l : 1, h = height >> l ? height >> l : 1;
        at += ((uint64_t)(((w + f->bw - 1) / f->bw * f->bytes + 63) & ~63u) * ((h + f->bh - 1) / f->bh) + 63) & ~63ULL;
    }
    if (layers > 1) at = at * layers + 32ULL * l * layers;
    else if (l > 1) at += 32 * l;
    if (samples > 1) at = at * samples + (pixel_format == 260 ? (uint64_t)((width + 63) & ~63u) * height * samples : 0);
    *bytes = (at + 0x3fff) & ~0x3fffULL;
    return 0;
}

int
hakios_metal_texture_read_level(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned w, unsigned h,
    void *bytes, uint64_t bytes_per_row)
{
    return copyRows(t, x, y, w, h, (uint8_t *)bytes, bytes_per_row, false, level);
}

int
hakios_metal_texture_replace_level(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned w,
    unsigned h, const void *bytes, uint64_t bytes_per_row)
{
    return copyRows(t, x, y, w, h, (uint8_t *)bytes, bytes_per_row, true, level);
}

int
hakios_metal_texture_read_3d(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned z, unsigned w,
    unsigned h, unsigned d, void *bytes, uint64_t bytes_per_row, uint64_t bytes_per_image)
{
    if (t != nullptr && t->layers > 1 && d == 1 && z < t->layers && level < t->levels) {
        // a layer of an array, a face of a cube (Z): that layer's level
        hakios_metal_texture *v = levelsOf(t, findFormat(t->format), t->format, t->type == 1 ? 0 : 2, level, 1, z);
        int r = copyRows(v, x, y, w, h, (uint8_t *)bytes, bytes_per_row, false);
        v->release();
        return r;
    }
    if (t != nullptr && t->type == 7 && level == 0 && d >= 1 && z + d <= t->depth && (z != 0 || d != 1)) {
        // a 3D texture's depth planes Z .. Z + D - 1, each image BYTES_PER_IMAGE further on in the caller's memory
        // (SpringBoard's 64 x 64 x 64 colour table)
        const uint64_t image = bytes_per_image ? bytes_per_image : bytes_per_row * h;
        for (unsigned k = 0; k < d; k++) {
            hakios_metal_texture *v = attachmentOf(t, 0, z + k);
            if (v == nullptr) return -1;
            int r = copyRows(v, x, y, w, h, (uint8_t *)bytes + image * k, bytes_per_row, false);
            v->release();
            if (r != 0) return r;
        }
        return 0;
    }
    if (z != 0 || d != 1) {
        fail("a 3D region of a texture (not on the Mali yet)");
        return -1;
    }
    return hakios_metal_texture_read_level(t, level, x, y, w, h, bytes, bytes_per_row);
}

int
hakios_metal_texture_replace_3d(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned z, unsigned w,
    unsigned h, unsigned d, const void *bytes, uint64_t bytes_per_row, uint64_t bytes_per_image)
{
    if (t != nullptr && t->layers > 1 && d == 1 && z < t->layers && level < t->levels) {
        // a layer of an array, a face of a cube (Z): that layer's level
        hakios_metal_texture *v = levelsOf(t, findFormat(t->format), t->format, t->type == 1 ? 0 : 2, level, 1, z);
        int r = copyRows(v, x, y, w, h, (uint8_t *)bytes, bytes_per_row, true);
        v->release();
        return r;
    }
    if (t != nullptr && t->type == 7 && level == 0 && d >= 1 && z + d <= t->depth && (z != 0 || d != 1)) {
        // a 3D texture's depth planes Z .. Z + D - 1, each image BYTES_PER_IMAGE further on in the caller's memory
        // (SpringBoard's 64 x 64 x 64 colour table)
        const uint64_t image = bytes_per_image ? bytes_per_image : bytes_per_row * h;
        for (unsigned k = 0; k < d; k++) {
            hakios_metal_texture *v = attachmentOf(t, 0, z + k);
            if (v == nullptr) return -1;
            int r = copyRows(v, x, y, w, h, (uint8_t *)bytes + image * k, bytes_per_row, true);
            v->release();
            if (r != 0) return r;
        }
        return 0;
    }
    if (z != 0 || d != 1) {
        fail("a 3D region of a texture (not on the Mali yet)");
        return -1;
    }
    return hakios_metal_texture_replace_level(t, level, x, y, w, h, bytes, bytes_per_row);
}

// the clear alone: a pass of no geometry (the fragment job writes the clear into every tile, as mali_test's clear did)
int
hakios_metal_cb_clear(hakios_metal_cb *cb, const struct hakios_metal_render_pass *pass)
{
    if (cb != nullptr && atLevels(pass)) {
        struct hakios_metal_render_pass q;
        hakios_metal_texture *views[10];
        if (!passAtLevels(pass, q, views)) return -1;
        int r = hakios_metal_cb_clear(cb, &q);
        for (auto *v : views) if (v) v->release();
        return r;
    }
    // (the surfaces it writes marked busy, as a draw's: the display waits for them -- markBusy)
    if (cb != nullptr && pass != nullptr) {
        markBusy(cb, cb->queue->device, pass->color0);
        for (unsigned i = 0; i < 7; i++) {
            markBusy(cb, cb->queue->device, pass->more[i].texture);
        }
    }
    // THE ATTACHMENTS CLEARED (Clear load action), any of the eight: four a Mali pass (the tile buffer's four kilobytes
    // a tile), each its own render target -- one not cleared is not in the pass at all
    struct Cleared { hakios_metal_texture *t; const double *color; unsigned store; };
    Cleared c[8];
    unsigned nc = 0;
    if (cb == nullptr || pass == nullptr || pass->color0 == nullptr) {
        fail("a clear without a colour attachment");
        return -1;
    }
    if (pass->load_action == 2) c[nc++] = { pass->color0, pass->clear_color, pass->store_action };
    for (unsigned i = 0; i < 7; i++) {
        const auto &m = pass->more[i];
        if (m.texture != nullptr && m.load_action == 2) c[nc++] = { m.texture, m.clear_color, m.store_action };
    }
    bool zs_clear = (pass->depth && pass->depth_load_action == 2) || (pass->stencil && pass->stencil_load_action == 2);
    static const double kNone[4] = { 0, 0, 0, 0 };
    if (nc == 0 && !zs_clear) {
        return 0;                                       // nothing cleared, nothing drawn: nothing to do
    }
    if (nc == 0) {
        c[nc++] = { pass->color0, kNone, 0 };           // (depth or stencil alone: colour 0 in the pass, not written)
    }
    hakios_metal_texture *t0 = c[0].t;
    for (unsigned i = 0; i < nc; i++) {
        hakios_metal_texture *t = c[i].t;
        if (!cb->queue->device->heap_ok || !drawable(t->format) || t->width != t0->width || t->height != t0->height) {
            fail("a clear of MTLPixelFormat %u, %ux%u (RGBA8/BGRA8/BGR5A1/B5G6R5 attachments of one size on the Mali yet)",
                t->format, t->width, t->height);
            return -1;
        }
    }
    auto rt1Of = [](const hakios_metal_texture *t) { uint32_t w = 0; rtFormat(t->format, w); return w; };
    for (unsigned at = 0; at < nc; at += 4) {
        DrawJob dj = {};
        hakios_metal_texture *t = c[at].t;
        dj.clear_only = true;
        dj.samples = t->samples;
        dj.target_va = t->gpu();
        dj.width = t->width;
        dj.height = t->height;
        dj.row_bytes = t->stride;
        dj.rt1 = rt1Of(t);
        dj.write = c[at].store == 1;
        dj.clear = rtClear(dj.rt1, c[at].color);
        dj.clear_high = rtClearHigh(dj.rt1, c[at].color);
        dj.blend[0] = 1u << 9;
        dj.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;
        dj.blend[2] = 1 | 3 << 3;
        dj.blend[3] = rtBlend(dj.rt1);
        dj.rts = nc - at < 4 ? nc - at : 4;
        for (unsigned i = 1; i < dj.rts; i++) {
            DrawJob::More &m = dj.more[i - 1];
            hakios_metal_texture *u = c[at + i].t;
            if (u->samples != t->samples) {
                fail("a clear of colour attachments of different sample counts");
                return -1;
            }
            m.va = u->gpu();
            m.row_bytes = u->stride;
            m.rt1 = rt1Of(u);
            m.clear = rtClear(m.rt1, c[at + i].color);
            m.clear_high = rtClearHigh(m.rt1, c[at + i].color);
            m.write = c[at + i].store == 1;
            m.texture = u;
            m.blend[0] = 1u << 9;
            m.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;
            m.blend[2] = 1 | 3 << 3 | i << 16;
            m.blend[3] = rtBlend(m.rt1);
        }
        if (at == 0 && zs_clear) {
            const char *why = nullptr;
            if (!zsOf(pass, dj, t->width, t->height, false, why)) {
                fail("a clear: %s", why);
                return -1;
            }
            for (hakios_metal_texture *x : { pass->depth, pass->stencil }) {
                if (x != nullptr) {
                    x->retain();
                    cb->held_textures.push_back(x);
                }
            }
        }
        dj.mode = 8;
        dj.instances = 1;
        dj.scissor_max = (t->width - 1) | (t->height - 1) << 16;
        Job j = {};
        j.shader = DRAW;
        j.dst = cb->draws.size();
        cb->draws.push_back(dj);
        cb->blits.push_back(j);
        for (unsigned i = 0; i < dj.rts; i++) {
            c[at + i].t->retain();
            cb->held_textures.push_back(c[at + i].t);
        }
    }
    return 0;
}

int
hakios_metal_cb_draw_indexed(hakios_metal_cb *cb, hakios_metal_program *program, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned index_count, unsigned index_type, hakios_metal_buffer *index_buffer, uint64_t index_offset,
    unsigned instance_count, int base_vertex, unsigned base_instance,
    hakios_metal_buffer *const *vs_buffers, const uint64_t *vs_offsets, unsigned vs_count,
    hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets, unsigned fs_count,
    hakios_metal_texture *const *fs_textures, unsigned fs_texture_count,
    hakios_metal_sampler *const *fs_samplers, unsigned fs_sampler_count)
{
    IndexInfo ix = { index_buffer, index_offset, index_type, base_vertex };
    return drawImpl(cb, program, pass, viewport, scissor, primitive, 0, index_count, instance_count, base_instance,
        vs_buffers, vs_offsets, vs_count, fs_buffers, fs_offsets, fs_count, fs_textures, fs_texture_count, fs_samplers,
        fs_sampler_count, &ix);
}

int
hakios_metal_cb_refuse(hakios_metal_cb *cb, const char *why)
{
    if (cb != nullptr) {
        if (cb->skip_refused) {
            cb->skipped++;
            cb->skip_reason = why ? why : "";
        } else {
            hakios_metal_cb_fail(cb, why ? why : "refused");
        }
    }
    return -1;
}

void hakios_metal_cb_skip_refused(hakios_metal_cb *cb, int on) { if (cb) cb->skip_refused = on != 0; }

// TEXTURE COPIES AND SCALED BLITS. A copy of the same size between formats of one texel size is a byte copy, row by
// row, on the compute blit shaders (any format); a scale, a rotation, or a big copy into an RGBA8/BGRA8 texture is a draw
// of the driver's blit program (mali_blit_compile): the source level sampled at each destination pixel's centre.
namespace {
hakios_metal_program *
blitProgram(hakios_metal_device *d)
{
    std::lock_guard<std::mutex> g(d->preload_lock);
    if (d->blit_prog != nullptr || d->blit_failed) {
        if (d->blit_prog == nullptr) fail("the driver's blit draw could not be built");
        return d->blit_prog;
    }
    auto *p = new hakios_metal_program;
    char why[256] = "";
    if (mali_blit_compile(&p->vs, &p->fs, why, sizeof(why)) != 0 || !p->upload(d)) {
        fail("the driver's blit draw: %s", why[0] ? why : gError.c_str());
        d->blit_failed = true;
        p->release();
        return nullptr;
    }
    snprintf(p->name, sizeof(p->name), "hakios_blit");
    d->blit_prog = p;
    return p;
}

unsigned linearOf(unsigned f) { return f == 71 ? 70 : f == 81 ? 80 : f; }

hakios_metal_program *
resolveProgram(hakios_metal_device *d)
{
    std::lock_guard<std::mutex> g(d->preload_lock);
    if (d->resolve_prog != nullptr || d->resolve_failed) {
        if (d->resolve_prog == nullptr) fail("the driver's resolve draw could not be built");
        return d->resolve_prog;
    }
    auto *p = new hakios_metal_program;
    char why[256] = "";
    if (mali_resolve_compile(&p->vs, &p->fs, 4, why, sizeof(why)) != 0 || !p->upload(d)) {
        fail("the driver's resolve draw: %s", why[0] ? why : gError.c_str());
        d->resolve_failed = true;
        p->release();
        return nullptr;
    }
    snprintf(p->name, sizeof(p->name), "hakios_resolve4");
    d->resolve_prog = p;
    return p;
}

// A MULTISAMPLE RESOLVE (MTLStoreActionMultisampleResolve, which the Metal side turns into this at -endEncoding): the
// SW x SH pixels at (SX, SY) of FROM (4 samples) averaged into the same size at (DX, DY) of TO's level TL, drawn
int
resolveDraw(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned sx, unsigned sy, hakios_metal_texture *to,
    unsigned tl, unsigned dx, unsigned dy, unsigned w, unsigned h)
{
    hakios_metal_device *d = cb->queue->device;
    hakios_metal_program *rp = resolveProgram(d);
    if (rp == nullptr) return -1;
    hakios_metal_texture *dst = levelsOf(to, findFormat(to->format), to->format, 2, tl, 1);
    dst->depth = 1;
    dst->usage |= HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
    hakios_metal_sampler smp;
    memset(smp.words, 0, sizeof(smp.words));
    smp.words[0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 23 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    struct hakios_metal_render_pass pass;
    memset(&pass, 0, sizeof(pass));
    pass.color0 = dst;
    pass.load_action = dx == 0 && dy == 0 && w == dst->width && h == dst->height ? 0 : 1;
    pass.store_action = 1;
    struct hakios_metal_viewport vp = { (double)dx, (double)dy, (double)w, (double)h, 0, 1 };
    struct hakios_metal_scissor sc = { dx, dy, w, h };
    uint32_t words[2] = { (uint32_t)((int)sx - (int)dx), (uint32_t)((int)sy - (int)dy) };
    hakios_metal_texture *texs[1] = { from };
    hakios_metal_sampler *smps[1] = { &smp };
    int r = drawImpl(cb, rp, &pass, &vp, &sc, 3, 0, 3, 1, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, texs, 1, smps, 1,
        nullptr, words, 2);
    dst->release();
    return r;
}

// a draw of FROM's level FL into the DW x DH rectangle at (DX, DY) of TO's level TL, the destination pixel (x, y) (its
// centre) sampled at normalized (x m0 + y m1 + m2, x m3 + y m4 + m5) of the source level
int
blitDraw(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned fl, hakios_metal_texture *to, unsigned tl, unsigned dx,
    unsigned dy, unsigned dw, unsigned dh, const float m[6], bool linear, bool raw)
{
    hakios_metal_device *d = cb->queue->device;
    hakios_metal_program *bp = blitProgram(d);
    if (bp == nullptr) return -1;
    // the two levels as textures of their own; a byte copy reads and writes them without sRGB conversion
    unsigned ff = raw ? linearOf(from->format) : from->format, tf = raw ? linearOf(to->format) : to->format;
    hakios_metal_texture *src = levelsOf(from, findFormat(ff), ff, from->type == 7 ? 2 : from->type, fl, 1);
    hakios_metal_texture *dst = levelsOf(to, findFormat(tf), tf, 2, tl, 1);
    src->depth = dst->depth = 1;
    dst->usage |= HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;   // (the Mali draws into any RGBA8 texture)
    hakios_metal_sampler smp;
    memset(smp.words, 0, sizeof(smp.words));
    smp.words[0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 23 | 1u << 25 | 1u << 26 |
        (linear ? 0u : 1u << 27 | 1u << 28);
    struct hakios_metal_render_pass pass;
    memset(&pass, 0, sizeof(pass));
    pass.color0 = dst;
    pass.load_action = dx == 0 && dy == 0 && dw == dst->width && dh == dst->height ? 0 : 1;
    pass.store_action = 1;
    struct hakios_metal_viewport vp = { (double)dx, (double)dy, (double)dw, (double)dh, 0, 1 };
    struct hakios_metal_scissor sc = { dx, dy, dw, dh };
    uint32_t words[6];
    memcpy(words, m, sizeof(words));
    hakios_metal_texture *texs[1] = { src };
    hakios_metal_sampler *smps[1] = { &smp };
    int r = drawImpl(cb, bp, &pass, &vp, &sc, 3, 0, 3, 1, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, texs, 1, smps, 1,
        nullptr, words, 6);
    src->release();
    dst->release();
    return r;
}
}  // namespace

// (11-13) A copy between LAYERS -- a 2D array's slices, a cube's faces: FROM_LAYER and TO_LAYER, each layer_stride further
// on in its texture's memory. Served where the copy is one of bytes (one size, one pixel size; a compressed format block
// for block); a copy that has to be drawn (scaled, or big and into RGBA8/BGRA8 from another size) is refused for a layer
// other than the first, by name.
static int
blitLayers(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned from_layer, unsigned sx, unsigned sy,
    unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned to_layer, unsigned dx, unsigned dy,
    unsigned dw, unsigned dh, int linear);

int
hakios_metal_cb_blit(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned sx, unsigned sy,
    unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned dx, unsigned dy, unsigned dw, unsigned dh,
    int linear)
{
    return blitLayers(cb, from, from_level, 0, sx, sy, sw, sh, to, to_level, 0, dx, dy, dw, dh, linear);
}

int
hakios_metal_cb_blit_layer(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned from_layer,
    unsigned sx, unsigned sy, unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned to_layer,
    unsigned dx, unsigned dy)
{
    return blitLayers(cb, from, from_level, from_layer, sx, sy, sw, sh, to, to_level, to_layer, dx, dy, sw, sh, 0);
}

static int
blitLayers(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned from_layer, unsigned sx, unsigned sy,
    unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned to_layer, unsigned dx, unsigned dy,
    unsigned dw, unsigned dh, int linear)
{
    { hakios_metal_texture *rt_ = (from); while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent; if (rt_ != nullptr) rt_->reads++; }
    const char *why = nullptr;
    if (cb != nullptr && to != nullptr) {
        markBusy(cb, cb->queue->device, to);    // (a copy into a surface: the display waits for it too, as for a draw)
    }
    if (cb == nullptr || from == nullptr || to == nullptr) why = "a texture copy without its two textures";
    else if (from_layer >= (from->layers > 1 ? from->layers : 1u) || to_layer >= (to->layers > 1 ? to->layers : 1u))
        why = "a texture copy of a layer the texture has not got";
    else if ((from_layer != 0 || to_layer != 0) && (from->samples > 1 || to->samples > 1))
        why = "a copy between layers of multisample textures";
    else if (from->samples > 1 && to->samples == 1) {
        // a resolve: pixel for pixel, a colour format into an RGBA8/BGRA8 one
        if (from_level != 0 || sw != dw || sh != dh) why = "a multisample resolve other than pixel for pixel";
        else if (from->format == 250 || from->format == 252 || from->format == 253 || from->format == 260)
            why = "a depth/stencil multisample resolve (not on the Mali yet)";
        else if (!drawable(to->format)) why = "a multisample resolve into a format other than RGBA8/BGRA8/BGR5A1/B5G6R5";
        else if (to_level >= to->levels || sx + sw > from->width || sy + sh > from->height ||
                 dx + dw > to->levelWidth(to_level) || dy + dh > to->levelHeight(to_level))
            why = "a multisample resolve outside a texture";
        else if (sw == 0 || sh == 0) return 0;
        else if (resolveDraw(cb, from, sx, sy, to, to_level, dx, dy, dw, dh) != 0) {
            hakios_metal_cb_fail(cb, gError.c_str());
            return -1;
        } else return 0;
    }
    else if (from->samples > 1 || to->samples > 1) why = "a copy into or between multisample textures (not on the Mali yet)";
    else if (from_level >= from->levels || to_level >= to->levels) why = "a texture copy of a level the texture has not got";
    else if (sw == 0 || sh == 0 || dw == 0 || dh == 0) return 0;
    else if (sx + sw > from->levelWidth(from_level) || sy + sh > from->levelHeight(from_level) ||
             dx + dw > to->levelWidth(to_level) || dy + dh > to->levelHeight(to_level))
        why = "a texture copy outside a texture";
    if (why != nullptr) {
        fail("%s", why);
        if (cb) hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    bool same = sw == dw && sh == dh;
    bool blocks = from->bw > 1 || from->bh > 1 || to->bw > 1 || to->bh > 1;
    if (blocks && !same && (to->bw > 1 || to->bh > 1)) {
        fail("a scaled copy into a compressed texture (MTLPixelFormat %u)", to->format);
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    blocks = blocks && same;                            // (a scale from a compressed one: drawn, the GPU decodes it)
    if (blocks) {
        // a compressed format: copied block for block, between two of one block size, on block boundaries
        if (!same || from->bw != to->bw || from->bh != to->bh || from->bytes != to->bytes || sx % from->bw || sy % from->bh ||
            dx % to->bw || dy % to->bh) {
            fail("a copy of a compressed texture other than block for block (MTLPixelFormat %u into %u)", from->format,
                to->format);
            hakios_metal_cb_fail(cb, gError.c_str());
            return -1;
        }
        sx /= from->bw; dx /= from->bw; sw = (sw + from->bw - 1) / from->bw;
        sy /= from->bh; dy /= from->bh; sh = (sh + from->bh - 1) / from->bh;
    }
    if (same && from->bytes == to->bytes && (blocks || !drawable(to->format) || sh <= 16 || from_layer != 0 || to_layer != 0)) {
        // A BYTE COPY: one compute blit for rows that follow each other in both, else one a row
        uint64_t src = from->gpu() + from->layer_stride * from_layer + from->levelOffset(from_level);
        uint64_t dst = to->gpu() + to->layer_stride * to_layer + to->levelOffset(to_level);
        uint32_t ss = from->levelStride(from_level), ds = to->levelStride(to_level), row = sw * from->bytes;
        bool whole = sx == 0 && dx == 0 && ss == ds && row == ss;
        unsigned jobs = whole ? 1 : sh;
        if (jobs > 256) {
            fail("a texture copy of %u rows of MTLPixelFormat %u (one a row: a copy that big is drawn, RGBA8/BGRA8 only)",
                sh, to->format);
            hakios_metal_cb_fail(cb, gError.c_str());
            return -1;
        }
        for (unsigned r = 0; r < jobs; r++) {
            uint64_t s = src + (uint64_t)(sy + r) * ss + (uint64_t)sx * from->bytes;
            uint64_t t = dst + (uint64_t)(dy + r) * ds + (uint64_t)dx * to->bytes;
            uint64_t n = whole ? (uint64_t)row * sh : row;
            bool w = ((s | t | n) & 3) == 0;
            Job j = {};
            j.shader = w ? 0u : 2u;
            j.dst = t;
            j.src = s;
            j.count = (uint32_t)(w ? n / 4 : n);
            cb->blits.push_back(j);
        }
        from->retain();
        cb->held_textures.push_back(from);
        to->retain();
        cb->held_textures.push_back(to);
        return 0;
    }
    if (from_layer != 0 || to_layer != 0) {
        fail("a texture copy from layer %u into layer %u that is not one of bytes (%ux%u of MTLPixelFormat %u into %ux%u of %u: "
            "scaled, or between pixel sizes, is drawn, and a draw takes the first layer only)", from_layer, to_layer,
            sw, sh, from->format, dw, dh, to->format);
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    if (!drawable(to->format) || (same && from->bytes != to->bytes)) {
        fail("a %s texture copy from MTLPixelFormat %u into %u (into RGBA8/BGRA8 only on the Mali yet)", same ? "" : "scaled",
            from->format, to->format);
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    // A DRAW: destination pixel x (centre) -> source x' = sx + (x - dx) * sw / dw, normalized by the source level's width
    float W = (float)from->levelWidth(from_level), H = (float)from->levelHeight(from_level);
    float ku = (float)sw / dw / W, kv = (float)sh / dh / H;
    float m[6] = { ku, 0, (float)sx / W - dx * ku, 0, kv, (float)sy / H - dy * kv };
    if (blitDraw(cb, from, from_level, to, to_level, dx, dy, dw, dh, m, linear && !same, same) != 0) {
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    return 0;
}

int
hakios_metal_cb_blit_rotated(hakios_metal_cb *cb, hakios_metal_texture *from, hakios_metal_texture *to, unsigned rotation)
{
    { hakios_metal_texture *rt_ = (from); while (rt_ != nullptr && rt_->origin == 4) rt_ = rt_->parent; if (rt_ != nullptr) rt_->reads++; }
    // destination pixel (x, y) of TO (w x h) from FROM (W x H), normalized: 1 a quarter turn (TO's sides FROM's swapped):
    // u = y / H... each case the source position of the destination's, as a6xx_rotation turns
    if (cb != nullptr && to != nullptr) {
        markBusy(cb, cb->queue->device, to);    // (as hakios_metal_cb_blit's)
    }
    if (cb == nullptr || from == nullptr || to == nullptr || rotation > 5 || !drawable(to->format)) {
        fail("a rotated texture blit (%u) into MTLPixelFormat %u (RGBA8/BGRA8, rotations 0 to 5)", rotation, to ? to->format : 0);
        if (cb) hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    float W = (float)from->width, H = (float)from->height, w = (float)to->width, h = (float)to->height;
    bool swap = rotation == 1 || rotation == 3;
    if ((swap ? (to->width != from->height || to->height != from->width) : (to->width != from->width || to->height != from->height))) {
        fail("a rotated texture blit (%u) between textures of %ux%u and %ux%u", rotation, from->width, from->height, to->width,
            to->height);
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    float m[6];
    switch (rotation) {
    case 0: { float a[6] = { 1 / W, 0, 0, 0, 1 / H, 0 }; memcpy(m, a, sizeof(m)); break; }
    case 1: { float a[6] = { 0, 1 / W, 0, -1 / H, 0, w / H }; memcpy(m, a, sizeof(m)); break; }      // 90 clockwise
    case 2: { float a[6] = { -1 / W, 0, 1, 0, -1 / H, 1 }; memcpy(m, a, sizeof(m)); break; }         // 180
    case 3: { float a[6] = { 0, -1 / W, h / W, 1 / H, 0, 0 }; memcpy(m, a, sizeof(m)); break; }      // 270
    case 4: { float a[6] = { -1 / W, 0, 1, 0, 1 / H, 0 }; memcpy(m, a, sizeof(m)); break; }          // flip left-right
    default: { float a[6] = { 1 / W, 0, 0, 0, -1 / H, 1 }; memcpy(m, a, sizeof(m)); break; }         // flip top-bottom
    }
    if (blitDraw(cb, from, 0, to, 0, 0, 0, to->width, to->height, m, false, true) != 0) {
        hakios_metal_cb_fail(cb, gError.c_str());
        return -1;
    }
    return 0;
}

int
hakios_metal_cb_append(hakios_metal_cb *cb, hakios_metal_cb *other)
{
    (void)cb; (void)other;
    fail("a command buffer appended to another (not on the Mali yet)");
    return -1;
}

int
hakios_metal_cb_generate_mipmaps(hakios_metal_cb *cb, hakios_metal_texture *t)
{
    if (t == nullptr || t->levels <= 1) {
        return 0;                                       // one level: nothing to make
    }
    // each level from the one above, halved with linear filtering
    for (unsigned l = 1; l < t->levels; l++) {
        if (hakios_metal_cb_blit(cb, t, l - 1, 0, 0, t->levelWidth(l - 1), t->levelHeight(l - 1), t, l, 0, 0, t->levelWidth(l),
                t->levelHeight(l), 1) != 0) {
            return -1;
        }
    }
    return 0;
}

int
hakios_metal_cb_dispatch_indirect(hakios_metal_cb *cb, hakios_metal_kernel *kernel, hakios_metal_buffer *args,
    uint64_t offset, const unsigned threads[3], hakios_metal_buffer *const *buffers, const uint64_t *offsets,
    unsigned count, hakios_metal_texture *const *textures, unsigned texture_count, hakios_metal_sampler *const *samplers,
    unsigned sampler_count)
{
    (void)kernel; (void)args; (void)offset; (void)threads; (void)buffers; (void)offsets; (void)count; (void)textures;
    (void)texture_count; (void)samplers; (void)sampler_count;
    fail("an indirect dispatch (not on the Mali yet)");
    if (cb) hakios_metal_cb_fail(cb, gError.c_str());
    return -1;
}

hakios_metal_program *
hakios_metal_program_specialize(hakios_metal_program *base, const void *fs_bitcode, size_t fs_size,
    const hakios_metal_constant *fs_constants, unsigned fs_count, unsigned binding, const void *bytes, size_t length)
{
    // AN UBER SHADER SPECIALISED FOR ITS BUFFER'S BYTES (11-11): CoreAnimation draws widgets and cross-fades with its
    // generic fixed_frag_lph_cpf -- 160 KB of code, 64 registers, 144 bytes spilled a thread, ~30 cycles a pixel (a
    // 423x423 widget's 4 draws: 3.5 ms) -- whose path every fragment takes by a small constant buffer. HakiosMetalCA asks
    // for a program per content of that buffer (hakiosProgramFor:, in the background): the base's pipeline state with
    // the fragment stage compiled for those bytes (compile_mali.c's fold_known) -- 256 bytes to a few KB of code.
    if (base == nullptr || fs_bitcode == nullptr || fs_size == 0 || bytes == nullptr || length == 0 || length > 4096) {
        fail("a specialisation needs the program, its fragment function's bitcode and the buffer's bytes");
        return nullptr;
    }
    auto *p = new hakios_metal_program;
    {
        std::lock_guard<std::mutex> g(base->lock);
        p->color_format = base->color_format;
        p->samples = base->samples;
        p->blend_enabled = base->blend_enabled;
        p->vertex_descriptor = base->vertex_descriptor;
        p->blend_equation = base->blend_equation;
        p->vs_bitcode = base->vs_bitcode;
        p->vs_constants = base->vs_constants;
        p->fetch = base->fetch;
        p->fetch_described = base->fetch_described;
        p->shader_blend = base->shader_blend;
        for (unsigned i = 0; i < 4; i++) p->rt_blend_on[i] = base->rt_blend_on[i];
        for (unsigned i = 0; i < 4; i++) p->rt_format[i] = base->rt_format[i];
        p->blend_reads_dest = base->blend_reads_dest;
        p->blend_refusal = base->blend_refusal;
        for (unsigned i = 0; i < 3; i++) {
            p->more_blend_equation[i] = base->more_blend_equation[i];
            p->more_blend_reads_dest[i] = base->more_blend_reads_dest[i];
            p->more_blend_refusal[i] = base->more_blend_refusal[i];
        }
        memcpy(p->name, base->name, sizeof(p->name));
        memcpy(p->label, base->label, sizeof(p->label));
    }
    if (p->vs_bitcode.empty()) {
        fail("a specialisation of a program without its vertex function's bitcode");
        p->release();
        return nullptr;
    }
    p->fs_bitcode.assign((const uint8_t *)fs_bitcode, (const uint8_t *)fs_bitcode + fs_size);
    p->fs_constants.resize(fs_count);
    for (unsigned i = 0; i < fs_count; i++) {
        p->fs_constants[i].index = fs_constants[i].index;
        p->fs_constants[i].bits = fs_constants[i].bits;
    }
    p->known_binding = (int)binding;
    p->known.assign((const uint8_t *)bytes, (const uint8_t *)bytes + length);
    // both stages (the vertex stage again: the fragment stage is compiled against its varyings, which no cache keeps),
    // as compileFetch would at the first draw -- here, in the caller's background, and then not again
    bool blended = p->shader_blend.mask != 0 || p->shader_blend.format[0] || p->shader_blend.format[1] ||
        p->shader_blend.format[2] || p->shader_blend.format[3];   // (a float target's format: the fetch's conversion)
    mali_kernel_binary vs = {}, fs = {};
    char why[512] = "";
    uint64_t key = cacheKey(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants, p->fs_bitcode.data(),
                            p->fs_bitcode.size(), keyedConstants(p, p->fs_constants), p->color_format,
                            p->fetch_described ? &p->fetch : nullptr, blended ? &p->shader_blend : nullptr);
    if (!cacheLoad(key, &vs, &fs)) {
        KnownNow known(p);
        int r = p->fetch_described ?
            mali_vertex_compile(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants.data(), p->vs_constants.size(),
                                &p->fetch, &vs, why, sizeof(why)) :
            mali_stage_compile(p->vs_bitcode.data(), p->vs_bitcode.size(), p->vs_constants.data(), p->vs_constants.size(), 0,
                               nullptr, &vs, why, sizeof(why));
        if (r != 0 ||
            mali_fragment_compile(p->fs_bitcode.data(), p->fs_bitcode.size(), p->fs_constants.data(), p->fs_constants.size(),
                                  &vs, blended ? &p->shader_blend : nullptr, &fs, why, sizeof(why)) != 0) {
            mali_kernel_binary_free(&vs);
            mali_kernel_binary_free(&fs);
            fail("the specialised stages: %s", why);
            p->release();
            return nullptr;
        }
        cacheStore(key, vs, fs);
    }
    p->vs = vs;
    p->fs = fs;
    p->fetch_compiled = true;           // (compileFetch: nothing left to do)
    return p;
}

int
hakios_metal_argument_buffer_layout(const void *bitcode, size_t size, unsigned set, unsigned *ids, unsigned *offsets,
    unsigned max)
{
    (void)bitcode; (void)size; (void)set; (void)ids; (void)offsets; (void)max;
    fail("argument buffers (not on the Mali yet)");
    return -1;
}

void hakios_metal_cb_set_vertex_textures(hakios_metal_cb *cb, hakios_metal_texture *const *textures, unsigned texture_count,
    hakios_metal_sampler *const *samplers, unsigned sampler_count)
{
    // (11-13) kept for the draws that follow; a draw takes what its vertex function declares. VectorKit binds them for
    // every map draw: refused here, every command buffer of Maps and of Weather's radar map failed without a word and
    // the map stayed black (the tester's phone; night13/maps4)
    if (cb == nullptr) return;
    cb->vs_texture_count = texture_count < MALI_KERNEL_TEXTURES ? texture_count : MALI_KERNEL_TEXTURES;
    cb->vs_sampler_count = sampler_count < MALI_KERNEL_TEXTURES ? sampler_count : MALI_KERNEL_TEXTURES;
    for (unsigned i = 0; i < MALI_KERNEL_TEXTURES; i++) {
        cb->vs_textures[i] = i < cb->vs_texture_count ? textures[i] : nullptr;
        cb->vs_sampler_set[i] = i < cb->vs_sampler_count && samplers[i] != nullptr;
        if (cb->vs_sampler_set[i]) memcpy(cb->vs_samplers[i], samplers[i]->words, 32);
    }
}
void hakios_metal_cb_set_indirect(hakios_metal_cb *cb, hakios_metal_buffer *args, uint64_t offset)
{
    (void)offset;
    if (cb && args) {
        hakios_metal_cb_fail(cb, "an indirect draw (not on the Mali yet)");
    }
}
void hakios_metal_cb_set_null_fs_textures(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_cb_set_gmem(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_cb_set_gmem_binning(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_cb_set_gmem_min_draws(hakios_metal_cb *cb, unsigned draws) { (void)cb; (void)draws; }
void hakios_metal_cb_set_gmem_off(hakios_metal_cb *cb, unsigned bits) { (void)cb; (void)bits; }
void hakios_metal_cb_set_gmem_tuned(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_cb_set_profile(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_cb_set_profile_draws(hakios_metal_cb *cb, int on) { (void)cb; (void)on; }
void hakios_metal_queue_set_deferred(hakios_metal_queue *q, int on) { (void)q; (void)on; }
void hakios_metal_cb_gmem_counts(hakios_metal_cb *cb, unsigned *passes, unsigned *tiles)
{
    (void)cb;
    if (passes) *passes = 0;
    if (tiles) *tiles = 0;
}
const char *hakios_metal_cb_gmem_refusal(hakios_metal_cb *cb) { (void)cb; return "no GMEM on a Mali"; }
const char *hakios_metal_cb_pass_sysmem_why(hakios_metal_cb *cb, unsigned i) { (void)cb; (void)i; return "a Mali: tiled by itself"; }
int hakios_metal_cb_pass_tiles(hakios_metal_cb *cb, unsigned i) { (void)cb; (void)i; return -1; }
int hakios_metal_cb_pass_attachments(hakios_metal_cb *cb, unsigned i, unsigned *attached, unsigned *touched)
{
    (void)cb; (void)i; (void)attached; (void)touched;
    return -1;
}
int hakios_metal_cb_pass_time(hakios_metal_cb *cb, unsigned i, unsigned *kind, unsigned *width, unsigned *height,
    unsigned *draws, double *us)
{
    (void)cb; (void)i; (void)kind; (void)width; (void)height; (void)draws; (void)us;
    return -1;
}
int hakios_metal_cb_draw_time(hakios_metal_cb *cb, unsigned i, const char **name, double *us)
{
    (void)cb; (void)i; (void)name; (void)us;
    return -1;
}
int hakios_metal_cb_draw_place(hakios_metal_cb *cb, unsigned i, unsigned *pass, unsigned area[4])
{
    (void)cb; (void)i; (void)pass; (void)area;
    return -1;
}
void hakios_metal_cb_counts(hakios_metal_cb *cb, unsigned *passes, unsigned *draws, unsigned *dispatches, unsigned *copies,
    unsigned *clears)
{
    unsigned p = 0, dr = 0, di = 0, co = 0, cl = 0;
    if (cb != nullptr) {
        for (const Job &j : cb->blits) {
            if (j.shader == DRAW) {
                (cb->draws[j.dst].clear_only ? cl : dr)++;
                p++;
            } else if (j.shader == DISPATCH) {
                di++;
            } else {
                co++;
            }
        }
    }
    if (passes) *passes = p;
    if (draws) *draws = dr;
    if (dispatches) *dispatches = di;
    if (copies) *copies = co;
    if (clears) *clears = cl;
}
void hakios_metal_buffer_counts(unsigned *made, unsigned *reused, unsigned long long *bytes)
{
    if (made) *made = 0;
    if (reused) *reused = 0;
    if (bytes) *bytes = 0;
}

}  // extern "C"

static hakios_metal_texture *
nothingTexture(hakios_metal_device *d)
{
    std::lock_guard<std::mutex> g(d->scratch_lock);
    if (d->nothing_texture == nullptr) {
        d->nothing_texture = hakios_metal_texture_new(d, 2, 70, 4, 4, 3 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
        if (d->nothing_texture != nullptr) {
            memset(d->nothing_texture->pixels(), 0, (size_t)d->nothing_texture->stride * 4);
        }
    }
    return d->nothing_texture;
}
