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
    uint32_t mali_format, bytes, stride;        // per texel; per row (64-byte aligned for its own)
    uint64_t offset = 0;                        // where its pixels start in BO
    unsigned origin = 0;                        // 0 its own memory, 1 an IOSurface's, 2 a buffer's, 4 a view's
    unsigned surface = 0;                       // the IOSurface's ID (origin 1)
    unsigned levels = 1, samples = 1;
    uint32_t swizzle = 0;                       // 0: the identity
    hakios_metal_texture *parent = nullptr;     // a view's (held)
    hakios_metal_buffer *buffer = nullptr;      // a buffer texture's (held)
    std::atomic<int> refs{1};

    void retain() { refs.fetch_add(1); }
    void release();
    uint8_t *pixels() const { return (uint8_t *)bo.cpu + offset; }
    uint64_t gpu() const { return bo.va + offset; }
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
struct MaliFormat { unsigned metal; uint32_t mali; uint32_t bytes; };
const MaliFormat kFormats[] = {
    { 10, 163u << 12 | 16, 1 },                 // R8Unorm (R001 -> RGB1)
    { 30, 171u << 12 | 16, 2 },                 // RG8Unorm
    { 53, 133u << 12 | 16, 4 },                 // R32Uint
    { 55, 167u << 12 | 16, 4 },                 // R32Float
    { 70, 187u << 12 | 0, 4 },                  // RGBA8Unorm
    { 71, 187u << 12 | 0 | 1u << 20, 4 },       // RGBA8Unorm_sRGB
    { 80, 187u << 12 | 4, 4 },                  // BGRA8Unorm
    { 81, 187u << 12 | 4 | 1u << 20, 4 },       // BGRA8Unorm_sRGB
    { 115, 223u << 12 | 0, 8 },                 // RGBA16Float
    { 123, 157u << 12 | 0, 16 },                // RGBA32Uint
    { 125, 191u << 12 | 0, 16 },                // RGBA32Float
};
constexpr uint32_t kIdentitySwizzle = 0 | 1 << 3 | 2 << 6 | 3 << 9;

// a texture's descriptor (genxml v10 "Texture", 32 bytes) and its plane's ("Generic Plane"), the plane at PLANE_VA
void
describe(const hakios_metal_texture *t, uint32_t *tex, uint32_t *plane, uint64_t plane_va)
{
    uint32_t clump = t->bytes == 1 ? 0 : t->bytes == 2 ? 1 : t->bytes == 4 ? 2 : t->bytes == 8 ? 3 : 4;   // RAW8..RAW128
    memset(tex, 0, 32);
    memset(plane, 0, 32);
    tex[0] = 2 | (t->type == 7 ? 3u : 2u) << 4 | t->mali_format << 10;      // a texture, 2D (or 3D: MTLTextureType3D 7)
    tex[1] = (t->width - 1) | (t->height - 1) << 16;
    tex[2] = t->swizzle ? t->swizzle : kIdentitySwizzle;                    // one level, from level 0
    tex[4] = (uint32_t)plane_va;
    tex[5] = (uint32_t)(plane_va >> 32);
    tex[7] = t->type == 7 ? t->depth - 1 : 0;
    plane[0] = 10 | 1 << 4 | 2 << 8 | clump << 24;                          // a plane, generic, linear
    plane[1] = (uint32_t)(t->bo.size - t->offset);
    plane[2] = (uint32_t)t->gpu();
    plane[3] = (uint32_t)(t->gpu() >> 32);
    plane[4] = t->stride;
    plane[6] = t->stride * t->height;                                       // (a slice: a 3D texture's depth)
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
void
emitDraw(hakios_metal_device *d, const DrawJob &dj, uint8_t *area, uint64_t area_va, uint64_t *s, unsigned &n)
{
    auto *tc = (uint32_t *)area, *fb = (uint32_t *)(area + 0x80), *rt = (uint32_t *)(area + 0x100);
    auto *bl = (uint32_t *)(area + 0x180);
    uint64_t vs_at = 0x1c0, fs_at = vs_at + ((dj.vs_fau.size() * 4 + 0x3f) & ~0x3fULL);
    uint64_t srt_at = fs_at + ((dj.fs_fau.size() * 4 + 0x3f) & ~0x3fULL), srt = 0;

    memset(area, 0, 0x1c0);
    tc[2] = hierarchyMask(dj.width, dj.height, d->tiler_levels) | 1u << 17;    // (sample test off, as in mali_test)
    tc[3] = (dj.width - 1) | (dj.height - 1) << 16;
    *(uint64_t *)&tc[6] = d->heapdesc.va;
    *(uint64_t *)&tc[8] = (d->heapdesc.va + 0x1000) | (0x10000 >> 12);
    *(uint64_t *)&fb[4] = d->descriptors.va + 0x1800;                          // sample positions: one, centred
    fb[8] = (dj.width - 1) | (dj.height - 1) << 16;
    fb[10] = (dj.width - 1) | (dj.height - 1) << 16;
    fb[11] = 2 << 6 | 8 << 9 | 1u << 15 | 1u << 24;                            // 16x16 tiles, 1 KiB of colour a tile
    fb[12] = 1u << 29;                                                          // empty tiles written (the clear)
    *(uint64_t *)&fb[14] = area_va;
    rt[0] = 1u << 26;                                                           // internal format R8G8B8A8
    rt[1] = dj.rt1 | (dj.write ? 1u : 0u) | 1u << 31;                          // clean tiles written too
    *(uint64_t *)&rt[8] = dj.target_va;
    rt[10] = dj.row_bytes;
    rt[11] = dj.row_bytes * dj.height;
    rt[12] = rt[13] = rt[14] = rt[15] = dj.clear;
    memcpy(bl, dj.blend, 16);
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
    s[n++] = move48(0, 0);                                                      // vertex SRT: none
    s[n++] = move48(4, srt);                                                    // fragment SRT
    s[n++] = move48(8, vs_fau_va);
    s[n++] = move32(9, (uint32_t)(vs_fau_va >> 32) | (uint32_t)((dj.vs_fau.size() + 1) / 2) << 24);
    s[n++] = move48(12, fs_fau_va);
    s[n++] = move32(13, (uint32_t)(fs_fau_va >> 32) | (uint32_t)((dj.fs_fau.size() + 1) / 2) << 24);
    s[n++] = move48(16, dj.pos_spd);
    s[n++] = move48(18, dj.vary_spd);
    s[n++] = move48(20, dj.frag_spd);
    s[n++] = move48(24, d->descriptors.va + 0x1200);                           // local storage: none
    s[n++] = move32(32, 0);
    s[n++] = move32(33, dj.count);
    s[n++] = move32(34, dj.instances);
    s[n++] = move32(35, 0);
    s[n++] = move32(36, dj.vertex_start);
    s[n++] = move32(37, 0);
    s[n++] = move32(38, dj.dcd2);
    s[n++] = move32(39, (uint32_t)dj.index_size);                               // the index buffer's bytes
    s[n++] = move48(40, area_va);                                               // the tiler context
    s[n++] = move32(42, dj.scissor_min);
    s[n++] = move32(43, dj.scissor_max);
    s[n++] = move32(44, 0);                                                     // depth clamps 0 .. 1
    s[n++] = move32(45, 0x3f800000);
    s[n++] = move48(46, 0);                                                     // no occlusion query
    s[n++] = move32(48, dj.vary_size);
    s[n++] = move48(50, (area_va + 0x180) | 1);                                // one blend descriptor
    s[n++] = move48(52, d->descriptors.va + 0x1980);                           // depth/stencil: all pass
    s[n++] = move48(54, dj.index_va);                                           // the index buffer (0: none)
    s[n++] = move32(56, dj.tiler_flags);
    s[n++] = move32(57, dj.dcd0);
    s[n++] = move32(58, dj.dcd1);
    s[n++] = move32(60, 0x3f800000);                                            // primitive size 1.0
    s[n++] = move32(61, 0);                                                     // TILER_FLAGS2: VRS 1x1
    if (!dj.clear_only) {                                                       // (a clear: the fragment job alone)
        s[n++] = 49ULL << 56 | 0ULL << 32;                                      // HEAP_OPERATION: vertex/tiler started
        s[n++] = 6ULL << 56 | 1ULL << 33 | 1ULL << 38;                          // RUN_IDVS: malloc, the fragment's SRT
        s[n++] = WAIT_ALL;
        s[n++] = 9ULL << 56;                                                    // FINISH_TILING
        s[n++] = WAIT_ALL;
        s[n++] = 49ULL << 56 | 1ULL << 32;                                      // HEAP_OPERATION: vertex/tiler completed
        s[n++] = WAIT_ALL;
    }
    s[n++] = move48(40, (area_va + 0x80) | 1);                                 // the framebuffer descriptor, one RT
    s[n++] = move32(42, 0);
    s[n++] = move32(43, (dj.width - 1) | (dj.height - 1) << 16);
    s[n++] = move48(44, 0);                                                     // no tile enable map (r44 held the
    s[n++] = move32(46, 0);                                                     // depth clamps), its row stride
    s[n++] = 7ULL << 56;                                                        // RUN_FRAGMENT
    s[n++] = WAIT_ALL;
    // the pass finished, as panvk ends one: the tiler context's completed top and bottom (byte 40, four words) into
    // r70-r73, and FINISH_FRAGMENT gives that heap memory back and counts the pass done. Without it a queue drew one
    // pass and then binned nothing (the heap's in-flight passes never went down).
    s[n++] = move48(74, area_va);
    s[n++] = 20ULL << 56 | 70ULL << 48 | 74ULL << 40 | 0xfULL << 16 | 40;      // LOAD_MULTIPLE r70-73 <- [r74 + 40]
    s[n++] = WAIT_ALL;
    s[n++] = 11ULL << 56 | 70ULL << 40 | 72ULL << 32 | 1;                       // FINISH_FRAGMENT: chunks r70..r72, count it
    s[n++] = WAIT_ALL;
}
constexpr unsigned DRAW_WORDS = 62;     // (instructions emitDraw writes, at most)

// the bytes a draw needs in its command buffer's stream BO, past the instructions
uint64_t
drawBytes(const DrawJob &dj)
{
    uint64_t bytes = 0x80 + 0x100 + 0x40 + ((dj.vs_fau.size() * 4 + 0x3f) & ~0x3fULL) + ((dj.fs_fau.size() * 4 + 0x3f) & ~0x3fULL);
    if (dj.texture_count || dj.sampler_count) {
        bytes += 0x80 + (dj.texture_count * 3 + dj.sampler_count) * 32 + 0x40;
    }
    return bytes;
}
}  // namespace

struct hakios_metal_cb {
    hakios_metal_queue *queue;
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
            device->freeBo(cb->stream);
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
    struct { uint32_t vm_id, initial, chunk_size, max_chunks, in_flight, handle; uint64_t ctx, first; } th =
        { d->vm, 5, 2u << 20, 64, 65535, 0, 0, 0 };
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
    memset(zs, 0, 32);
    zs[0] = 7 | 7 << 4 | 7 << 16;                       // stencil faces: always
    zs[4] = 1u << 21 | 1u << 22 | 7u << 29;             // depth: always, nothing written
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

hakios_metal_device *
hakios_metal_open(void)
{
    io_service_t service = IOServiceGetMatchingService(0, IOServiceMatching("HakiosGPU"));
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
    cb->status = HAKIOS_METAL_COMMITTED;
    if (!cb->error.empty()) {
        cb->status = HAKIOS_METAL_ERROR;
    } else if (!cb->blits.empty()) {
        // the stream (from 0), then each job's push constants: 32 bytes for a blit, a kernel's fau_words for a dispatch
        size_t words = 18 + cb->blits.size() * 20 + cb->draws.size() * DRAW_WORDS;
        uint64_t fau_area = (words * 8 + 0xff) & ~0xffULL, fau_size = 0;
        std::vector<uint64_t> fau_at(cb->blits.size());
        for (size_t i = 0; i < cb->blits.size(); i++) {
            fau_at[i] = fau_area + fau_size;
            if (cb->blits[i].shader == DRAW) {
                fau_size += (drawBytes(cb->draws[cb->blits[i].dst]) + 0x3f) & ~0x3fULL;
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
        if (!d->newBo(cb->stream, fau_area + fau_size, BIND_NOEXEC)) {
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
            for (size_t i = 0; i < cb->blits.size(); i++) {
                const Job &b = cb->blits[i];
                auto *pc = (uint8_t *)cb->stream.cpu + fau_at[i];
                if (b.shader == DRAW) {
                    emitDraw(d, cb->draws[b.dst], pc, cb->stream.va + fau_at[i], s, n);
                    cb->draw_areas.push_back(fau_at[i]);
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
                if (d->call(GROUP_SUBMIT, &gs, sizeof(gs)) != 0) {
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

hakios_metal_texture *
hakios_metal_texture_new(hakios_metal_device *d, unsigned type, unsigned pixel_format, unsigned width, unsigned height,
    unsigned usage)
{
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
    if (type != 2 || width == 0 || height == 0 || width > 16384 || height > 16384) {
        fail("a texture of type %u, %u x %u (the Mali build makes 2D ones up to 16384)", type, width, height);
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
    t->bytes = f->bytes;
    t->stride = (width * f->bytes + 63) & ~63u;
    if (!d->newBo(t->bo, (uint64_t)t->stride * height, BIND_NOEXEC)) {
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
    bool in)
{
    if (t == nullptr || x + w > t->width || y + h > t->height || bytes_per_row < (uint64_t)w * t->bytes) {
        fail("a texture region outside the texture, or rows shorter than its width");
        return -1;
    }
    for (unsigned row = 0; row < h; row++) {
        uint8_t *gpu = t->pixels() + (uint64_t)(y + row) * t->stride + (uint64_t)x * t->bytes;
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
    return copyRows(t, x, y, w, h, (uint8_t *)bytes, bytes_per_row, true);
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
    if (mali_stage_compile(vs_bitcode, vs_size, vc.data(), vs_count, 0, nullptr, &p->vs, why, sizeof(why)) != 0 ||
        mali_stage_compile(fs_bitcode, fs_size, fc.data(), fs_count, 1, &p->vs, &p->fs, why, sizeof(why)) != 0) {
        fail("%s", why);
        p->release();
        return nullptr;
    }
    p->color_format = color_pixel_format;
    snprintf(p->name, sizeof(p->name), "%s+%s", p->vs.name, p->fs.name);
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
hakios_metal_program_set_vertex_attribute(hakios_metal_program *p, unsigned index, unsigned, unsigned, unsigned)
{
    if (p == nullptr || index > 30) {
        fail("a vertex attribute past 30");
        return -1;
    }
    p->vertex_descriptor = true;
    return 0;
}

int
hakios_metal_program_set_vertex_layout(hakios_metal_program *p, unsigned buffer_index, unsigned, unsigned, unsigned)
{
    if (p == nullptr || buffer_index > 30) {
        fail("a vertex layout past 30");
        return -1;
    }
    p->vertex_descriptor = true;
    return 0;
}

void
hakios_metal_program_set_blend(hakios_metal_program *p, int enabled, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned)
{
    if (p != nullptr) {
        p->blend_enabled = enabled != 0;
    }
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

int
drawImpl(hakios_metal_cb *cb, hakios_metal_program *p, const struct hakios_metal_render_pass *pass,
    const struct hakios_metal_viewport *viewport, const struct hakios_metal_scissor *scissor, unsigned primitive,
    unsigned vertex_start, unsigned vertex_count, unsigned instance_count, unsigned base_instance,
    hakios_metal_buffer *const *vs_buffers, const uint64_t *vs_offsets, unsigned vs_count,
    hakios_metal_buffer *const *fs_buffers, const uint64_t *fs_offsets, unsigned fs_count,
    hakios_metal_texture *const *fs_textures, unsigned fs_texture_count, hakios_metal_sampler *const *fs_samplers,
    unsigned fs_sampler_count, const IndexInfo *ix = nullptr)
{
    hakios_metal_device *d = cb->queue->device;
    const char *refused = nullptr;
    hakios_metal_texture *t = pass ? pass->color0 : nullptr;

    // what the Mali build draws so far; the rest by name
    if (!d->heap_ok) refused = "draws (no tiler heap on this device: hakios_metal_open said why)";
    else if (p == nullptr || t == nullptr) refused = "a draw without a program or a colour attachment";
    else if (t->format != 70 && t->format != 71 && t->format != 80 && t->format != 81) refused = "a colour attachment not RGBA8/BGRA8 (not on the Mali yet)";
    else if (!(t->usage & HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET)) refused = "a colour attachment made without render-target usage";
    else if (pass->load_action == 1) refused = "load action Load (the tile buffer's preload: not on the Mali yet)";
    else if (pass->store_action > 1) refused = "a store action other than DontCare and Store";
    else if (pass->depth || pass->stencil || pass->depth_state_set || pass->stencil_state_set) refused = "depth and stencil (not on the Mali yet)";
    else if (pass->visibility && pass->visibility_mode) refused = "visibility results (not on the Mali yet)";
    else if (pass->color0_level || pass->color0_slice) refused = "a mip level or slice of the colour attachment";
    else if (p->blend_enabled) refused = "blending (not on the Mali yet)";
    else if (p->vertex_descriptor) refused = "a vertex descriptor (stage_in attributes: not on the Mali yet)";
    else if (p->samples != 1) refused = "multisampling (not on the Mali yet)";
    else if (primitive != 3 && primitive != 4) refused = "a primitive other than triangles and triangle strips";
    else if (instance_count == 0) refused = "no instances";
    else if (ix && (ix->buffer == nullptr || ix->type > 1 || ix->base_vertex < 0 ||
                    ix->offset + (uint64_t)vertex_count * (ix->type ? 4 : 2) > ix->buffer->length))
        refused = "an indexed draw past its index buffer, of an unknown index type or with a negative base vertex";
    for (unsigned i = 0; refused == nullptr && i < 7; i++) {
        if (pass->more[i].texture) refused = "colour attachments past 0 (not on the Mali yet)";
    }
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
    DrawJob dj = {};
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
            if (res.index >= fs_sampler_count || fs_samplers[res.index] == nullptr) {
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
    dj.pos_spd = p->pos_spd;
    dj.vary_spd = p->vary_spd;
    dj.frag_spd = p->frag_spd;
    dj.vary_size = p->vs.varying_bytes;
    dj.target_va = t->gpu();
    dj.width = t->width;
    dj.height = t->height;
    dj.row_bytes = t->stride;
    bool bgra = t->format == 80 || t->format == 81, srgb = t->format == 71 || t->format == 81;
    uint32_t swizzle = bgra ? (2 | 1 << 3 | 0 << 6 | 3 << 9) : (0 | 1 << 3 | 2 << 6 | 3 << 9);
    dj.rt1 = 19 << 3 | 2 << 8 | (srgb ? 1u << 14 : 0) | swizzle << 16;
    dj.write = pass->store_action == 1;
    if (pass->load_action == 2) {
        dj.clear = unorm8(pass->clear_color[0]) | unorm8(pass->clear_color[1]) << 8 | unorm8(pass->clear_color[2]) << 16 |
            (uint32_t)unorm8(pass->clear_color[3]) << 24;
    }
    dj.blend[0] = 1u << 9;
    dj.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;     // replace
    dj.blend[2] = 1 | 3 << 3;                           // opaque, 4 components, RT 0
    dj.blend[3] = 237u << 12;                           // the tile buffer's RGBA8
    dj.mode = primitive == 4 ? 10 : 8;
    dj.count = vertex_count;
    dj.instances = instance_count;
    dj.vertex_start = ix ? (unsigned)ix->base_vertex : vertex_start;
    // DCD0: scissor to the bounding box, cull zero area (v11); the face culled and the front's winding
    dj.dcd0 = 1u << 23 | 1u << 26 | (pass->winding == 1 ? 1u << 16 : 0) | (pass->cull_mode == 1 ? 1u << 17 : 0) |
        (pass->cull_mode == 2 ? 1u << 18 : 0);
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
    t->bytes = f->bytes;
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

hakios_metal_texture *
hakios_metal_texture_new_view(hakios_metal_texture *p, unsigned pixel_format, unsigned type, unsigned base_level,
    unsigned levels, unsigned base_slice, unsigned slices)
{
    const MaliFormat *f = findFormat(pixel_format);
    if (p == nullptr || f == nullptr || f->bytes != p->bytes || type != p->type || base_level != 0 || levels != 1 ||
        base_slice != 0 || slices != 1) {
        fail("a texture view of MTLPixelFormat %u, type %u, levels %u+%u, slices %u+%u (the Mali build views level 0 of "
            "a 2D texture, as a format of the same size)", pixel_format, type, base_level, levels, base_slice, slices);
        return nullptr;
    }
    hakios_metal_texture *t = textureOver(p->device, p->bo, p->offset, p->stride, type, f, pixel_format, p->width,
        p->height, p->usage, 4);
    p->retain();
    t->parent = p;
    t->surface = p->surface;
    return t;
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
    if (levels <= 1) {
        return hakios_metal_texture_new(d, type, pixel_format, width, height, usage);
    }
    fail("a texture of %u mip levels (one level only on the Mali yet)", levels);
    return nullptr;
}

hakios_metal_texture *
hakios_metal_texture_new_multisample(hakios_metal_device *d, unsigned pixel_format, unsigned width, unsigned height,
    unsigned samples, unsigned usage)
{
    if (samples <= 1) {
        return hakios_metal_texture_new(d, 2, pixel_format, width, height, usage);
    }
    fail("a multisample texture of %u samples (not on the Mali yet)", samples);
    return nullptr;
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
    (void)d; (void)usage;
    fail("a 3D texture of MTLPixelFormat %u, %u x %u x %u (not on the Mali yet)", pixel_format, width, height, depth);
    return nullptr;
}

hakios_metal_texture *
hakios_metal_texture_new_compressed(hakios_metal_device *d, unsigned pixel_format, unsigned width, unsigned height,
    unsigned usage)
{
    (void)d; (void)usage;
    fail("a compressed texture of MTLPixelFormat %u, %u x %u (not on the Mali yet)", pixel_format, width, height);
    return nullptr;
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
    *block_w = *block_h = 1;
    *bytes = f->bytes;
    return 1;
}

int
hakios_metal_texture_plan_bytes(unsigned type, unsigned pixel_format, unsigned width, unsigned height, unsigned depth,
    unsigned levels, unsigned samples, unsigned usage, uint64_t *bytes)
{
    (void)usage;
    const MaliFormat *f = findFormat(pixel_format);
    if (f == nullptr || type != 2 || depth > 1 || levels > 1 || samples > 1) {
        fail("a texture of type %u, MTLPixelFormat %u, %u levels, %u samples (not on the Mali yet)", type, pixel_format,
            levels, samples);
        return -1;
    }
    *bytes = (((uint64_t)((width * f->bytes + 63) & ~63u) * height) + 0x3fff) & ~0x3fffULL;
    return 0;
}

int
hakios_metal_texture_read_level(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned w, unsigned h,
    void *bytes, uint64_t bytes_per_row)
{
    if (level != 0) {
        fail("a texture's level %u (one level only on the Mali yet)", level);
        return -1;
    }
    return hakios_metal_texture_read(t, x, y, w, h, bytes, bytes_per_row);
}

int
hakios_metal_texture_replace_level(hakios_metal_texture *t, unsigned level, unsigned x, unsigned y, unsigned w,
    unsigned h, const void *bytes, uint64_t bytes_per_row)
{
    if (level != 0) {
        fail("a texture's level %u (one level only on the Mali yet)", level);
        return -1;
    }
    return hakios_metal_texture_replace(t, x, y, w, h, bytes, bytes_per_row);
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
    hakios_metal_texture *t = pass ? pass->color0 : nullptr;
    if (cb == nullptr || t == nullptr) {
        fail("a clear without a colour attachment");
        return -1;
    }
    if (pass->load_action != 2) {
        return 0;                                       // nothing cleared, nothing drawn: nothing to do
    }
    if (!cb->queue->device->heap_ok || (t->format != 70 && t->format != 71 && t->format != 80 && t->format != 81) ||
        !(t->usage & HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET) || pass->depth || pass->stencil) {
        fail("a clear of MTLPixelFormat %u%s (RGBA8/BGRA8 colour alone on the Mali yet)", t->format,
            pass->depth || pass->stencil ? " with depth or stencil" : "");
        return -1;
    }
    for (unsigned i = 0; i < 7; i++) {
        if (pass->more[i].texture) {
            fail("a clear of colour attachments past 0 (not on the Mali yet)");
            return -1;
        }
    }
    DrawJob dj = {};
    dj.clear_only = true;
    dj.target_va = t->gpu();
    dj.width = t->width;
    dj.height = t->height;
    dj.row_bytes = t->stride;
    bool bgra = t->format == 80 || t->format == 81, srgb = t->format == 71 || t->format == 81;
    uint32_t swizzle = bgra ? (2 | 1 << 3 | 0 << 6 | 3 << 9) : (0 | 1 << 3 | 2 << 6 | 3 << 9);
    dj.rt1 = 19 << 3 | 2 << 8 | (srgb ? 1u << 14 : 0) | swizzle << 16;
    dj.write = pass->store_action == 1;
    dj.clear = unorm8(pass->clear_color[0]) | unorm8(pass->clear_color[1]) << 8 | unorm8(pass->clear_color[2]) << 16 |
        (uint32_t)unorm8(pass->clear_color[3]) << 24;
    dj.blend[0] = 1u << 9;
    dj.blend[1] = 0x132 | 0x132 << 12 | 0xfu << 28;
    dj.blend[2] = 1 | 3 << 3;
    dj.blend[3] = 237u << 12;
    dj.mode = 8;
    dj.instances = 1;
    dj.scissor_max = (t->width - 1) | (t->height - 1) << 16;
    Job j = {};
    j.shader = DRAW;
    j.dst = cb->draws.size();
    cb->draws.push_back(dj);
    cb->blits.push_back(j);
    t->retain();
    cb->held_textures.push_back(t);
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

// the rest: Adreno's GMEM and its tuning, profiles and counts (none here), and what is refused by name
int
hakios_metal_cb_blit(hakios_metal_cb *cb, hakios_metal_texture *from, unsigned from_level, unsigned sx, unsigned sy,
    unsigned sw, unsigned sh, hakios_metal_texture *to, unsigned to_level, unsigned dx, unsigned dy, unsigned dw, unsigned dh,
    int linear)
{
    (void)from; (void)from_level; (void)sx; (void)sy; (void)sw; (void)sh; (void)to; (void)to_level; (void)dx; (void)dy;
    (void)dw; (void)dh; (void)linear;
    fail("a scaled texture blit (not on the Mali yet)");
    if (cb) hakios_metal_cb_fail(cb, gError.c_str());
    return -1;
}

int
hakios_metal_cb_blit_rotated(hakios_metal_cb *cb, hakios_metal_texture *from, hakios_metal_texture *to, unsigned rotation)
{
    (void)from; (void)to;
    fail("a rotated texture blit (%u, not on the Mali yet)", rotation);
    if (cb) hakios_metal_cb_fail(cb, gError.c_str());
    return -1;
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
    (void)cb;
    if (t != nullptr && t->levels <= 1) {
        return 0;                                       // one level: nothing to make
    }
    fail("mipmaps (not on the Mali yet)");
    return -1;
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
