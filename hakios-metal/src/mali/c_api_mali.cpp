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
    uint64_t spd[4] = {};
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

struct hakios_metal_buffer {
    hakios_metal_device *device;
    Bo bo;
    uint64_t length;
    std::atomic<int> refs{1};       // its owner's, and each command buffer's that uses it

    void retain() { refs.fetch_add(1); }
    void release()
    {
        if (refs.fetch_sub(1) == 1) {
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
    std::string fetch_refusal;
    bool blend_reads_dest = false;
    std::string blend_refusal;
    // colour attachments 1 to 3's (hakios_metal_program_set_blend_at): the same, for the multiple-target draws
    uint32_t more_blend_equation[3] = { 0x132 | 0x132 << 12 | 0xfu << 28, 0x132 | 0x132 << 12 | 0xfu << 28,
                                        0x132 | 0x132 << 12 | 0xfu << 28 };
    bool more_blend_reads_dest[3] = {};
    std::string more_blend_refusal[3];
    char name[300] = "";

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
        auto spd = [&](unsigned slot, unsigned stage, unsigned regs, uint64_t preload, uint64_t va) {
            auto *w = (uint32_t *)(bytes + spd_at + 0x40 * slot);
            w[0] = 8 | stage << 4 | (regs <= 32 ? 2u : 0u) << 30;
            w[1] = (uint32_t)(preload >> 48);
            w[2] = (uint32_t)va;
            w[3] = (uint32_t)(va >> 32);
            return code.va + spd_at + 0x40 * slot;
        };
        pos_spd = spd(0, 3, vs.work_regs, vs.preload, code.va);
        vary_spd = vs.secondary_offset ? spd(1, 3, vs.secondary_work_regs, vs.secondary_preload, code.va + vs.secondary_offset) : 0;
        frag_spd = spd(2, 2, fs.work_regs, fs.preload, code.va + fs_at);
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
    unsigned levels = 1, samples = 1;
    // a mipmapped texture (levels > 1): level n's rows at offset + level_off[n], level_stride[n] apart, its size
    // levelWidth/levelHeight; the levels' plane descriptors (32 bytes each, level 0 first) at planes_off in BO, written
    // once when the texture is made (a view of levels n.. points at the parent's, 32 * n further on)
    uint64_t level_off[16] = {};
    uint32_t level_stride[16] = {};
    uint64_t planes_off = 0;                    // 0: none (one level, described per use)
    uint32_t swizzle = 0;                       // 0: the identity
    hakios_metal_texture *parent = nullptr;     // a view's (held)
    hakios_metal_buffer *buffer = nullptr;      // a buffer texture's (held)
    std::atomic<int> refs{1};

    void retain() { refs.fetch_add(1); }
    void release();
    uint8_t *pixels() const { return (uint8_t *)bo.cpu + offset; }
    uint64_t gpu() const { return bo.va + offset; }
    unsigned levelWidth(unsigned l) const { return width >> l ? width >> l : 1; }
    unsigned levelHeight(unsigned l) const { return type == 0 ? 1 : height >> l ? height >> l : 1; }
    uint32_t levelStride(unsigned l) const { return l == 0 ? stride : level_stride[l]; }
    unsigned levelRows(unsigned l) const { return (levelHeight(l) + bh - 1) / bh; }    // rows of blocks
    uint64_t levelOffset(unsigned l) const { return l == 0 ? 0 : level_off[l]; }
    // the bytes from gpu() that its pixels (every level, every slice) span
    uint64_t span() const {
        if (levels > 1) return level_off[levels - 1] + (uint64_t)level_stride[levels - 1] * levelRows(levels - 1);
        return (uint64_t)stride * levelRows(0) * depth * samples;    // (a multisample texture: a plane a sample)
    }
};

struct hakios_metal_sampler {
    uint32_t words[8];                          // genxml v10 "Sampler"
};

void
hakios_metal_texture::release()
{
    if (refs.fetch_sub(1) != 1) {
        return;
    }
    if (origin == 0) {
        device->freeBo(bo);
    } else if (origin == 1) {
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
    // depth and stencil (render pass attachments; sampled as the plain format of their size): Depth16Unorm,
    // Depth32Float, Stencil8, Depth32Float_Stencil8 (its depth plane; the stencil plane after it, stencil_off)
    { 250, 164u << 12 | 16, 2 },
    { 252, 167u << 12 | 16, 4 },
    { 253, 163u << 12 | 16, 1 },
    { 260, 167u << 12 | 16, 4 },
    // ASTC, LDR (Metal 204..217) and sRGB (186..199), 16 bytes a block: on Valhall the format is the one it decodes to
    // (RGBA16F; sRGB RGBA8), the block size in the plane descriptor ("ASTC 2D Plane", describe())
#define ASTC(n, w, h) { 204 + n, 223u << 12 | 0, 16, w, h }, { 186 + n, 187u << 12 | 0 | 1u << 20, 16, w, h }
    ASTC(0, 4, 4), ASTC(1, 5, 4), ASTC(2, 5, 5), ASTC(3, 6, 5), ASTC(4, 6, 6), ASTC(5, 8, 5), ASTC(6, 8, 6), ASTC(7, 8, 8),
    ASTC(8, 10, 5), ASTC(9, 10, 6), ASTC(10, 10, 8), ASTC(11, 10, 10), ASTC(12, 12, 10), ASTC(13, 12, 12),
#undef ASTC
};
// A COLOUR ATTACHMENT'S FORMAT: the render target's word 1 (write-back format, linear block, sRGB, swizzle) for the
// formats the Mali build draws into; false for the rest. The tile buffer holds R8G8B8A8 for every one of them (the
// blend descriptors' and LD_TILE's conversion), converted at write-back. Metal names packed formats from the low bits:
// BGRA8, BGR5A1 and B5G6R5 all have blue first, hence the swizzle.
bool
rtFormat(unsigned f, uint32_t &rt1)
{
    uint32_t wb;
    bool bgra, srgb = f == 71 || f == 81;
    switch (f) {
    case 70: case 71: wb = 19; bgra = false; break;     // R8G8B8A8
    case 80: case 81: wb = 19; bgra = true; break;
    case 43: wb = 28; bgra = true; break;               // BGR5A1Unorm: R5G5B5A1
    case 40: wb = 21; bgra = true; break;               // B5G6R5Unorm: R5G6B5
    default: return false;
    }
    uint32_t swizzle = bgra ? (2 | 1 << 3 | 0 << 6 | 3 << 9) : (0 | 1 << 3 | 2 << 6 | 3 << 9);
    rt1 = wb << 3 | 2u << 8 | (srgb ? 1u << 14 : 0) | swizzle << 16;
    return true;
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
    tex[0] = 2 | (t->type == 7 ? 3u : t->type == 0 ? 1u : 2u) << 4 | t->mali_format << 10;
    tex[1] = (t->width - 1) | (t->height - 1) << 16;
    tex[2] = t->swizzle ? t->swizzle : kIdentitySwizzle;                    // one level, from level 0
    if (astc) tex[2] |= 1u << 12;                                           // texel interleave (compressed, as Mesa)
    if (t->levels > 1 && t->planes_off != 0) {
        // every level: the plane descriptors the texture keeps (levels - 1, and the maximum LOD as 8.8 fixed point)
        tex[2] |= (t->levels - 1) << 16;
        tex[3] = ((t->levels - 1) * 256) << 16;
        plane_va = t->bo.va + t->planes_off;
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
    hakios_metal_kernel *kernel;    // a dispatch's
    unsigned groups[3], threads[3], grid[3];
    uint64_t buffers[MALI_KERNEL_BUFFERS];
    uint32_t wls_size, wls_instances;   // a dispatch's threadgroup memory: bytes an instance (a power of two), instances
    hakios_metal_texture *textures[MALI_KERNEL_TEXTURES];  // bound at [[texture(n)]] (in the texture and image tables)
    uint32_t samplers[MALI_KERNEL_TEXTURES][8];
    unsigned texture_count, sampler_count;              // 0 0: no resource table
};
constexpr unsigned DISPATCH = 4, DRAW = 5;      // (a draw's Job: dst is its index in the command buffer's draws)

// One draw, everything resolved to what the GPU reads: one render pass of its own (begin, draw, end).
struct DrawJob {
    uint64_t pos_spd, vary_spd, frag_spd;
    unsigned vary_size;
    std::vector<uint32_t> vs_fau, fs_fau;               // the stages' push constants, whole
    uint64_t target_va;
    unsigned width, height, row_bytes;
    uint32_t rt1;                                       // the render target's word 1: write, format, block, swizzle
    uint32_t clear;                                     // (R8G8B8A8 in the tile buffer) and whether to write back
    bool write;
    uint32_t blend[4];
    unsigned mode, count, instances, vertex_start;
    uint32_t dcd0, dcd1, dcd2, tiler_flags, scissor_min, scissor_max;
    hakios_metal_texture *textures[MALI_KERNEL_TEXTURES];
    uint32_t samplers[MALI_KERNEL_TEXTURES][8];
    unsigned texture_count, sampler_count;
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
        bool write, load;
        bool missing;                                   // written by the fragment function, absent from the pass
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
    tc[2] = hierarchyMask(dj.width, dj.height, d->tiler_levels) | 1u << 17;    // (sample test off, as in mali_test)
    tc[3] = (dj.width - 1) | (dj.height - 1) << 16;
    *(uint64_t *)&tc[6] = d->heapdesc.va;
    *(uint64_t *)&tc[8] = (d->heapdesc.va + 0x1000) | (0x10000 >> 12);
    // sample positions: one, centred; four: the rotated 4x grid (descriptors + 0x2000)
    *(uint64_t *)&fb[4] = d->descriptors.va + (dj.samples > 1 ? 0x2000 : 0x1800);
    fb[8] = (dj.width - 1) | (dj.height - 1) << 16;
    fb[10] = (dj.width - 1) | (dj.height - 1) << 16;
    // 16x16 tiles, targets, 1 KiB a target a tile; four samples: 4 (log2 2), the rotated 4x grid (2), 8x8 tiles -- the
    // same kilobyte a target a tile (64 pixels of four samples)
    fb[11] = (dj.samples > 1 ? 2u | 2u << 3 : 0u) | 2 << 6 | (dj.samples > 1 ? 6u : 8u) << 9 | 1u << 15 | (dj.rts - 1) << 19 |
        dj.rts << 24;
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
    rt[0] = 1u << 26;                                                           // internal format R8G8B8A8
    rt[1] = dj.rt1 | (dj.write ? 1u : 0u) | 1u << 31 | (dj.samples > 1 ? 3u << 12 : 0u);   // clean tiles written too
    // (multisample: written back layered, a plane a sample, rt[11] apart)
    *(uint64_t *)&rt[8] = dj.target_va;
    rt[10] = dj.row_bytes;
    rt[11] = dj.row_bytes * dj.height;
    rt[12] = rt[13] = rt[14] = rt[15] = dj.clear;
    memcpy(bl, dj.blend, 16);
    for (unsigned i = 1; i < dj.rts; i++) {
        // target i: its kilobyte of the tile buffer (internal buffer offset, 16-byte units), its own clear and memory
        const DrawJob::More &m = dj.more[i - 1];
        auto *r = rt + 16 * i;
        r[0] = 1u << 26 | (1024 * i / 16) << 4;
        r[1] = m.rt1 | (m.write ? 1u : 0u) | 1u << 31 | (dj.samples > 1 ? 3u << 12 : 0u);
        *(uint64_t *)&r[8] = m.va;
        r[10] = m.row_bytes;
        r[11] = m.row_bytes * dj.height;
        r[12] = r[13] = r[14] = r[15] = m.clear;
        memcpy(bl + 4 * i, m.blend, 16);
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
    uint64_t vs_fau_va = area_va + vs_at, fs_fau_va = area_va + fs_at;
    if (first && ir != nullptr) {
        *(uint64_t *)(ir + 0x10) = 0;
        if (!dj.clear_only && dj.ir_frag_spd != 0) {
            *(uint64_t *)(ir + 0x10) = 1;                                       // (emitIr when the pass ends)
            s[n++] = move48(76, ir_va + IR_HANDLER);
            s[n++] = move32(78, IR_HANDLER_BYTES);
            s[n++] = 25ULL << 56 | 76ULL << 40 | 78ULL << 32 | 2;              // SET_EXCEPTION_HANDLER: tiler OOM
        }
    }
    if (dj.load && first) {
        // THE LOAD ACTION: the target's own pixels drawn first, one triangle over it whose fragment is the target's
        // texel (texture 0, txf), into the same tiler context as the draw -- the tile buffer preloaded through the
        // ordinary draw path. Caches first, as at a command buffer's start: what an earlier pass wrote back reaches
        // L2, and the texture cache drops what an earlier preload read of the same target.
        uint64_t end_at = srt_at + (dj.texture_count || dj.sampler_count ? 0x80 + (dj.texture_count * 3 + dj.sampler_count) * 32 : 0);
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
            bool loads = i == 0 ? dj.load0 : (dj.more[i - 1].load && !dj.more[i - 1].missing);
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
    s[n++] = move48(0, 0);                                                      // vertex SRT: none
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
            prt[16 * i + 1] = (prt[16 * i + 1] & ~1u) | (dj.more[i - 1].write ? 1u : 0u);
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
constexpr unsigned DRAW_WORDS = 62 + 45 + 10; // (instructions emitDraw writes, at most: the preload's 45 and the
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
            (!b.more[i - 1].missing && !b.more[i - 1].load)) {
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
bool bootArg(const char *word);
}  // namespace

struct hakios_metal_cb {
    hakios_metal_queue *queue;
    std::chrono::steady_clock::time_point committed_at; // (boot-arg metal_stats)
    Bo timestamps;                                      // ... the GPU's timestamp at each Mali pass's start and end
    std::chrono::steady_clock::time_point bo_at, submitted_at, waited_at;   // (boot-arg metal_cbtime)
    unsigned timestamp_count = 0;
    std::vector<Job> blits;
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
                int r;
                {
                    std::lock_guard<std::mutex> g(device->lock);
                    r = device->call(SYNCOBJ_WAIT, &w, sizeof(w));
                }
                cb->waited_at = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> g(cb->lock);
                if (r != 0) {
                    cb->error = "the GPU did not finish the command buffer in 10 s";
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
            static const bool stats = bootArg("metal_stats"), cbtime = bootArg("metal_cbtime");
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
    struct { uint32_t vm_id, initial, chunk_size, max_chunks, in_flight, handle; uint64_t ctx, first; } th =
        { d->vm, 5, 2u << 20, max_chunks, 65535, 0, 0, 0 };
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
    if (!d->newBo(b->bo, length ? length : 1, BIND_NOEXEC)) {
        delete b;
        return nullptr;
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
    q->thread = std::thread([q] { q->run(); });
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

void
hakios_metal_cb_commit(hakios_metal_cb *cb, void (*callback)(hakios_metal_cb *cb, void *context), void *context)
{
    hakios_metal_device *d = cb->queue->device;

    cb->callback = callback;
    cb->context = context;
    cb->committed_at = std::chrono::steady_clock::now();
    glitchScan(cb, true);
    cb->status = HAKIOS_METAL_COMMITTED;
    if (!cb->error.empty()) {
        cb->status = HAKIOS_METAL_ERROR;
    } else if (!cb->blits.empty()) {
        // the stream (from 0), then each job's push constants: 32 bytes for a blit, a kernel's fau_words for a dispatch
        static const bool stats = bootArg("metal_stats");
        size_t words = 18 + cb->blits.size() * 20 + cb->draws.size() * (DRAW_WORDS + (stats ? 4 : 0));
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
                    spd = d->spd[b.shader];
                    fau_entries = 4;                                                // 8 words: 4 of 64 bits
                    size[0] = 64, size[1] = size[2] = 1;                            // 64 invocations a workgroup
                    groups[0] = (b.count + 63) / 64, groups[1] = groups[2] = 1;
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
    {
        std::lock_guard<std::mutex> l(cb->queue->lock);
        cb->queue->pending.push_back(cb);
    }
    cb->queue->work.notify_all();
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
            fail("dispatch of %s: nothing bound at [[buffer(%u)]]", k->binary.name, res.index);
            return -1;
        }
        hakios_metal_buffer *b = buffers[res.index];
        uint64_t offset = offsets ? offsets[res.index] : 0;
        if (offset > b->bo.size || (res.size != 0 && offset + res.size > b->bo.size)) {
            fail("dispatch of %s: [[buffer(%u)]] at offset %llu is past its buffer", k->binary.name, res.index,
                (unsigned long long)offset);
            return -1;
        }
        j.buffers[res.index] = b->bo.va + offset;
    }
    // textures and samplers the kernel declares: bound, and of a kind it can use
    for (unsigned r = 0; r < k->binary.num_resources; r++) {
        const mali_kernel_resource &res = k->binary.resources[r];
        if (res.kind == HAKIOS_METAL_RESOURCE_TEXTURE || res.kind == HAKIOS_METAL_RESOURCE_IMAGE) {
            if (res.index >= texture_count || textures[res.index] == nullptr) {
                fail("dispatch of %s: nothing bound at [[texture(%u)]]", k->binary.name, res.index);
                return -1;
            }
            if (res.kind == HAKIOS_METAL_RESOURCE_IMAGE && !(textures[res.index]->usage & 2)) {
                fail("dispatch of %s: [[texture(%u)]] is written and was made without MTLTextureUsageShaderWrite", k->binary.name,
                    res.index);
                return -1;
            }
            j.texture_count = res.index + 1 > j.texture_count ? res.index + 1 : j.texture_count;
        } else if (res.kind == HAKIOS_METAL_RESOURCE_SAMPLER) {
            if (res.index >= sampler_count || samplers[res.index] == nullptr) {
                fail("dispatch of %s: nothing bound at [[sampler(%u)]]", k->binary.name, res.index);
                return -1;
            }
            j.sampler_count = res.index + 1 > j.sampler_count ? res.index + 1 : j.sampler_count;
        }
    }
    if (j.texture_count != 0 && j.sampler_count == 0) {
        j.sampler_count = 1;                // (a texel fetch wants a valid sampler 0: a nearest, clamped one)
        j.samplers[0][0] = 1 | 9u << 8 | 9u << 12 | 9u << 16 | 1u << 21 | 1u << 25 | 1u << 26 | 1u << 27 | 1u << 28;
    }
    for (unsigned t = 0; t < j.texture_count; t++) {
        j.textures[t] = t < texture_count ? textures[t] : nullptr;
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
        if (res.kind == HAKIOS_METAL_RESOURCE_CONSTANT_BUFFER || res.kind == HAKIOS_METAL_RESOURCE_DEVICE_BUFFER) {
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
        for (unsigned y = 0; y < t->height; y += 16) {
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

void
passStats(hakios_metal_cb *cb)
{
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cb->committed_at).count();
    if (ms < 40) {
        return;
    }
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
        double gpu_us = -1;
        if (cb->timestamps.size != 0 && 2 * passes + 1 < cb->timestamp_count) {
            auto *ts = (volatile uint64_t *)cb->timestamps.cpu;
            gpu_us = (double)(ts[2 * passes + 1] - ts[2 * passes]) / 24.576;    // (the system counter: 24.576 MHz)
        }
        passes++;
        draws += n;
        loads += dj.load;
        pixels += (uint64_t)dj.width * dj.height;
        if (list.size() < 1500) {
            char one[160];
            snprintf(one, sizeof(one), " [%.0fus %ux%u %u%s%s %s rts %u%s%s%s]", gpu_us, dj.width, dj.height, n, dj.load ? " L" : "",
                dj.clear_only ? " C" : "", dj.name ? dj.name : "", dj.rts,
                dj.rts > 1 ? (dj.more[0].missing ? "m" : dj.more[0].load ? "l" : dj.more[0].write ? "w" : "-") : "",
                why[0] ? " | " : "", why);
            list += one;
        }
        i = k;
    }
    fprintf(stderr, "hakios-metal: stats: %.1f ms: %u Mali passes (%u preload), %u draws, %u copies/dispatches, %.1f MP of "
        "targets:%s\n", ms, passes, loads, draws, other, pixels / 1e6, list.c_str());
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

uint64_t
cacheKey(const void *vs, size_t vs_size, const std::vector<mali_constant> &vc, const void *fs, size_t fs_size,
    const std::vector<mali_constant> &fc, unsigned format, const mali_vertex_fetch *fetch,
    const mali_shader_blend *blend = nullptr)
{
    uint64_t h = 14695981039346656037ULL;
    fnv(h, kCacheMagic, sizeof(kCacheMagic));
    fnvImage(h);
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

// both stages from the file KEY names; false (and nothing left allocated) when there is none or it is not whole
bool
cacheLoad(uint64_t key, mali_kernel_binary *vs, mali_kernel_binary *fs)
{
    if (!cacheOn() || cacheDir().empty()) return false;
    int fd = open(cachePath(key).c_str(), O_RDONLY);
    if (fd < 0) return false;
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
    if (fd < 0) return;
    uint32_t struct_size = sizeof(mali_kernel_binary);
    bool ok = writeAll(fd, kCacheMagic, 8) && writeAll(fd, &key, 8) && writeAll(fd, &struct_size, 4);
    const mali_kernel_binary *b[2] = { &vs, &fs };
    for (int i = 0; i < 2 && ok; i++) {
        ok = writeAll(fd, b[i], sizeof(mali_kernel_binary)) && writeAll(fd, b[i]->code, b[i]->size);
    }
    close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
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
    uint64_t key = cacheKey(vs_bitcode, vs_size, vc, fs_bitcode, fs_size, fc, color_pixel_format, nullptr);
    if (!cacheLoad(key, &p->vs, &p->fs)) {
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
// the stages again with the vertex descriptor's attributes fetched in the shader, before the program's first upload;
// true when there is nothing to do or it was done, false with p->fetch_refusal said
bool
compileFetch(hakios_metal_program *p)
{
    std::lock_guard<std::mutex> g(p->lock);
    bool blended = p->shader_blend.mask != 0;
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
                            p->fs_bitcode.size(), p->fs_constants, p->color_format, p->fetch_described ? &p->fetch : nullptr,
                            blended ? &p->shader_blend : nullptr);
    if (!cacheLoad(key, &vs, &fs)) {
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
    if (why != nullptr && shader != nullptr && rgb_op <= 4 && alpha_op <= 4 && shaderable(src_rgb) && shaderable(dst_rgb) &&
        shaderable(src_alpha) && shaderable(dst_alpha)) {
        *shader = { rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask };
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

void
hakios_metal_program_set_blend(hakios_metal_program *p, int enabled, unsigned rgb_op, unsigned alpha_op, unsigned src_rgb,
    unsigned dst_rgb, unsigned src_alpha, unsigned dst_alpha, unsigned write_mask)
{
    if (p == nullptr) {
        return;
    }
    p->blend_enabled = enabled != 0;
    bool in_shader = false;
    encodeBlend(enabled, rgb_op, alpha_op, src_rgb, dst_rgb, src_alpha, dst_alpha, write_mask, p->blend_equation,
                p->blend_reads_dest, p->blend_refusal, &p->shader_blend.rt[0], &in_shader);
    p->shader_blend.mask = (p->shader_blend.mask & ~1u) | (in_shader ? 1u : 0u);
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
            fail("draw with %s: nothing bound at [[buffer(%u)]]", b.name, res.index);
            return false;
        }
        uint64_t va = buffers[res.index]->bo.va + (offsets ? offsets[res.index] : 0);
        fau[2 * res.index] = (uint32_t)va;
        fau[2 * res.index + 1] = (uint32_t)(va >> 32);
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
    else if (!drawable(t->format)) refused = "a colour attachment not RGBA8/BGRA8/BGR5A1/B5G6R5 (not on the Mali yet)";
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
        else if (!drawable(m.texture->format))
            refused = "a colour attachment past 0 not RGBA8/BGRA8/BGR5A1/B5G6R5 (not on the Mali yet)";
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
        loads = loads || (pass->more[i - 1].texture != nullptr &&
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
                fprintf(stderr, "hakios-metal: first draw of %s at %.3f s: fs push words %u, textures %u, samplers %u, blend "
                    "in shader 0x%x, written 0x%x, read 0x%x, target %ux%u format %u, samples %u, %u vertices x %u\n",
                    p->name, t, p->fs.fau_words, fs_texture_count, fs_sampler_count, p->shader_blend.mask, p->fs.rt_written,
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
    std::vector<hakios_metal_buffer *> used;
    struct hakios_metal_viewport vp = viewport ? *viewport :
        hakios_metal_viewport{ 0, 0, (double)t->width, (double)t->height, 0, 1 };
    // Metal's viewport: origin top left, NDC y up, z in [0, 1]
    float scale[3] = { (float)(vp.width / 2), (float)(-vp.height / 2), (float)(vp.zfar - vp.znear) };
    float offset[3] = { (float)(vp.x + vp.width / 2), (float)(vp.y + vp.height / 2), (float)vp.znear };
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
    for (unsigned r = 0; r < p->fs.num_resources; r++) {
        const mali_kernel_resource &res = p->fs.resources[r];
        if (res.kind == HAKIOS_METAL_RESOURCE_TEXTURE) {
            if (res.index >= fs_texture_count || fs_textures[res.index] == nullptr) {
                fail("draw with %s: nothing bound at [[texture(%u)]]", p->name, res.index);
                return -1;
            }
            dj.texture_count = res.index + 1 > dj.texture_count ? res.index + 1 : dj.texture_count;
        } else if (res.kind == HAKIOS_METAL_RESOURCE_SAMPLER) {
            if (res.inline_sampler) {
                // the function's own (constexpr sampler): made from its state word below, nothing bound
            } else if (res.index >= fs_sampler_count || fs_samplers[res.index] == nullptr) {
                fail("draw with %s: nothing bound at [[sampler(%u)]]", p->name, res.index);
                return -1;
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
        dj.textures[i] = i < fs_texture_count ? fs_textures[i] : nullptr;
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
    dj.name = p->name;
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
        dj.clear = unorm8(pass->clear_color[0]) | unorm8(pass->clear_color[1]) << 8 | unorm8(pass->clear_color[2]) << 16 |
            (uint32_t)unorm8(pass->clear_color[3]) << 24;
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
    dj.blend[2] = (reads0 ? 2u : 1u) | 3 << 3;                // mode, 4 components, RT 0
    dj.blend[3] = 237u << 12;                           // the tile buffer's RGBA8
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
            m.blend[3] = 237u << 12;
            continue;
        }
        m.va = a.texture->gpu();
        m.row_bytes = a.texture->stride;
        rtFormat(a.texture->format, m.rt1);
        m.write = a.store_action == 1;
        m.load = a.load_action == 1;
        m.clear = a.load_action == 2 ? (unorm8(a.clear_color[0]) | unorm8(a.clear_color[1]) << 8 |
                                        unorm8(a.clear_color[2]) << 16 | (uint32_t)unorm8(a.clear_color[3]) << 24) : 0;
        m.texture = a.texture;
        if (!(p->fs.rt_written & 1u << i)) {
            // not written by this draw: in the pass only to keep the pass whole, exactly as it is -- loaded and stored
            // back unchanged, never cleared here (as when it was left out: CoreAnimation's Settings showed another
            // cell's label when this draw applied the attachment's Clear/DontCare)
            m.load = true;
            m.write = true;
            m.clear = 0;
            m.blend[0] = 1u << 9 | 1u;
            m.blend[1] = 0x132 | 0x132 << 12;           // colour mask 0
            m.blend[2] = 2u | 3 << 3 | i << 16;
            m.blend[3] = 237u << 12;
            continue;
        }
        bool reads = p->more_blend_reads_dest[i - 1] || (p->fs.rt_read & 1u << i);
        m.blend[0] = 1u << 9 | (reads ? 1u : 0);
        m.blend[1] = p->more_blend_equation[i - 1];
        m.blend[2] = (reads ? 2u : 1u) | 3 << 3 | i << 16;
        m.blend[3] = 237u << 12;
    }
    // Metal's point, line, line strip, triangle, triangle strip as Mali's draw modes (genxml v10 "Draw Mode")
    static const unsigned kDrawModes[5] = { 1, 2, 4, 8, 10 };
    dj.mode = kDrawModes[primitive];
    dj.count = vertex_count;
    dj.instances = instance_count;
    dj.vertex_start = ix ? (unsigned)ix->base_vertex : vertex_start;
    // DCD0: scissor to the bounding box, cull zero area (v11); the face culled and the front's winding
    dj.dcd0 = 1u << 23 | 1u << 26 | (pass->winding == 1 ? 1u << 16 : 0) | (pass->cull_mode == 1 ? 1u << 17 : 0) |
        (pass->cull_mode == 2 ? 1u << 18 : 0) | (dj.samples > 1 ? 1u << 19 : 0);  // (19: multisample)
    if (dj.zs && ((dj.zsd[4] >> 29) != 7 || (dj.zsd[4] & 1u << 27) || (dj.zsd[0] >> 31))) {
        // a depth or stencil test or write: pixels killed weakly early, Z/S updated late (a fragment the shader
        // discards must not write them; whether a shader discards is not tracked here)
        dj.dcd0 |= 2u << 2 | 3u << 4;
    }
    dj.dcd1 = 0xffff | 1u << 16;
    dj.dcd2 = 1u << 12 | 1u << 24 | 1u << 25;
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
levelsOf(hakios_metal_texture *p, const MaliFormat *f, unsigned pixel_format, unsigned type, unsigned base, unsigned count)
{
    hakios_metal_texture *t = textureOver(p->device, p->bo, p->offset + p->levelOffset(base), p->levelStride(base), type, f,
        pixel_format, p->levelWidth(base), p->levelHeight(base), p->usage, 4);
    t->depth = p->depth;
    t->levels = count;
    for (unsigned l = 0; l < count; l++) {
        t->level_off[l] = p->levelOffset(base + l) - p->levelOffset(base);
        t->level_stride[l] = p->levelStride(base + l);
    }
    t->planes_off = count > 1 && p->planes_off != 0 ? p->planes_off + 32 * base : 0;
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
    if (level >= t->levels || slice >= (t->type == 7 ? t->depth : 1)) {
        fail("an attachment at level %u, slice %u of a texture of %u level(s) and %u slice(s)", level, slice, t->levels,
            t->type == 7 ? t->depth : 1);
        return nullptr;
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
    fail("a layered texture: type %u, %u layers (not on the Mali yet)", type, layers);
    return nullptr;
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
unsigned hakios_metal_texture_depth(hakios_metal_texture *t) { return t ? t->depth : 0; }
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
    if (f == nullptr || (type != 2 && type != 4) || depth > 1 || (samples != 1 && samples != 4) || (samples > 1 && levels > 1) ||
        width == 0 || height == 0) {
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
    if (l > 1) at += 32 * l;
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
    (void)bytes_per_image;
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
    (void)bytes_per_image;
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
    auto colourOf = [](const double *k) {
        return unorm8(k[0]) | unorm8(k[1]) << 8 | unorm8(k[2]) << 16 | (uint32_t)unorm8(k[3]) << 24;
    };
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
        dj.clear = colourOf(c[at].color);
        dj.blend[0] = 1u << 9;
        dj.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;
        dj.blend[2] = 1 | 3 << 3;
        dj.blend[3] = 237u << 12;
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
            m.clear = colourOf(c[at + i].color);
            m.write = c[at + i].store == 1;
            m.texture = u;
            m.blend[0] = 1u << 9;
            m.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;
            m.blend[2] = 1 | 3 << 3 | i << 16;
            m.blend[3] = 237u << 12;
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

int
hakios_metal_cb_blit(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned sx, unsigned sy,
    unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned dx, unsigned dy, unsigned dw, unsigned dh,
    int linear)
{
    const char *why = nullptr;
    if (cb == nullptr || from == nullptr || to == nullptr) why = "a texture copy without its two textures";
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
    if (same && from->bytes == to->bytes && (blocks || !drawable(to->format) || sh <= 16)) {
        // A BYTE COPY: one compute blit for rows that follow each other in both, else one a row
        uint64_t src = from->gpu() + from->levelOffset(from_level), dst = to->gpu() + to->levelOffset(to_level);
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
    // destination pixel (x, y) of TO (w x h) from FROM (W x H), normalized: 1 a quarter turn (TO's sides FROM's swapped):
    // u = y / H... each case the source position of the destination's, as a6xx_rotation turns
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
    (void)base; (void)fs_bitcode; (void)fs_size; (void)fs_constants; (void)fs_count; (void)binding; (void)bytes;
    (void)length;
    fail("a program specialized for its buffer's bytes (not on the Mali yet)");
    return nullptr;
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
    (void)textures; (void)samplers;
    if (cb && (texture_count || sampler_count)) {
        hakios_metal_cb_fail(cb, "vertex stage textures and samplers (not on the Mali yet)");
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
