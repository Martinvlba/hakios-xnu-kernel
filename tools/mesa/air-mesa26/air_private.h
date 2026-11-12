/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/* Shared between air_to_nir.c and air_intrinsics.c. */

#ifndef AIR_PRIVATE_H
#define AIR_PRIVATE_H

#include "air_module.h"
#include "nir_air.h"

#include "nir.h"
#include "nir_builder.h"
#include "util/set.h"

#ifdef AIR_MESA_NEW_NIR         /* (nir_air.h) */
#define AIR_B_EXACT(b)                  (((b).fp_math_ctrl & nir_fp_exact) == nir_fp_exact)
#define AIR_B_SET_EXACT(b, on)          ((b).fp_math_ctrl = (on) ? nir_fp_exact : nir_fp_fast_math)
#define AIR_SHADER_CREATE(ctx, st, opt) nir_shader_create((ctx), (st), (opt))
#define AIR_BLOCK_CREATE(shader, impl)  nir_block_create(impl)
#else
#define AIR_B_EXACT(b)                  ((b).exact)
#define AIR_B_SET_EXACT(b, on)          ((b).exact = (on))
#define AIR_SHADER_CREATE(ctx, st, opt) nir_shader_create((ctx), (st), (opt), NULL)
#define AIR_BLOCK_CREATE(shader, impl)  nir_block_create(shader)
#endif

/* Where a pointer points. AIR's address spaces say which kind of memory;
 * which buffer, and the byte offset into it, the translator tracks itself
 * (docs/air.md section 3): no pointer in the corpus is ever stored to
 * memory or cast to an integer, so a pointer never has to become a number.
 */
enum air_ptr_class {
   AIR_PTR_NONE,
   AIR_PTR_UBO,           /* a constant buffer: .binding is its index */
   AIR_PTR_SSBO,          /* a device buffer */
   AIR_PTR_SHARED,        /* threadgroup memory */
   AIR_PTR_SCRATCH,       /* an alloca, or a program-scope variable */
   AIR_PTR_CONSTANT_DATA, /* a constant global: nir->constant_data */
   AIR_PTR_FUNCTION_CONSTANT, /* a function constant: .binding is its index */
   AIR_PTR_TEXTURE,       /* a texture handle; offset unused */
   AIR_PTR_SAMPLER,       /* a sampler handle */
};

struct air_ptr {
   enum air_ptr_class class;
   unsigned binding;
   bool is_image;                    /* a texture the shader writes: an image, not a view */
   nir_def *offset;                  /* bytes, 32-bit; NULL means zero */
   nir_def *binding_def;             /* a buffer chosen while the shader runs (a phi or select between buffers): its
                                      * index, 32-bit; NULL: BINDING as it stands */
   const struct air_type *pointee;   /* for sizes, and for loads without a type */
};

enum air_val_kind {
   AIR_VAL_NONE,
   AIR_VAL_SSA,   /* a scalar or vector */
   AIR_VAL_AGG,   /* a struct or array, as a tree of values */
   AIR_VAL_PTR,
};

struct air_val {
   enum air_val_kind kind;
   nir_def *def;
   struct air_ptr ptr;
   unsigned num_elems;
   struct air_val **elems;
};

struct air_ctx {
   void *mem_ctx;
   const struct air_module *m;
   const struct air_function *fn;
   nir_shader *nir;
   nir_builder b;

   const struct air_to_nir_options *options;
   struct air_shader_info *info;

   /* translated values, keyed on const struct air_value * */
   struct hash_table *values;
   /* air_block * -> nir_block * */
   struct hash_table *blocks;
   /* a cursor in each block, before its terminator, where phi stores go */
   struct hash_table *phi_cursors;
   /* phi -> nir register decl */
   struct hash_table *phi_regs;
   struct hash_table *phi_bind_regs;  /* a pointer phi's second register: the buffer's index (binding_def) */
   /* the blocks the entry can reach, branches on known values followed only
    * where they go (translate_body): what a phi between pointers chooses from
    */
   struct set *reachable;

   /* A call of another function in the module becomes a nir_call, and
    * nir_inline_functions folds it in afterwards. These two carry the state
    * that needs: the functions already translated, so a callee shared by two
    * call sites is translated once, and -- while a callee is being
    * translated -- the deref its `ret` has to store through, since a NIR call
    * returns nothing and the convention is an out-parameter at index 0.
    * ret_deref is NULL while the entry point is being translated, which is how
    * emit_return tells the two apart.
    */
   struct hash_table *fns;
   nir_deref_instr *ret_deref;
   /* a callee that returns a struct or array (11-12): an out-parameter for each scalar or vector in it, in the type's
    * order -- ret_derefs[0 .. num_ret_derefs - 1], ret_type the aggregate's; ret_deref is then the first of them */
   nir_deref_instr **ret_derefs;
   unsigned num_ret_derefs;
   const struct air_type *ret_type;

   /* What each module global resolves to, decided once (plan_globals) and
    * turned into a value in whichever function needs it (materialise_globals).
    * The two are separate because the deciding half has side effects that must
    * happen once -- it appends inline samplers to info->resources and lays out
    * shared and scratch memory -- while the making half emits an nir_def for
    * the byte offset, and a def belongs to the impl that created it. Copying
    * the entry point's values into a callee instead left the callee using the
    * caller's defs, which is a cross-function def/use: nir_repair_ssa then
    * asks the wrong function for its dominance metadata and asserts.
    */
   struct air_global_plan {
      enum air_ptr_class class;
      unsigned binding;
      unsigned offset;      /* bytes, when has_offset */
      bool has_offset;
      bool skip;            /* llvm.* lists, and anything that failed */
      /* A program-scope variable that is a VALUE (plan_global_folding): a
       * scalar integer loaded and stored only whole, stored once, in its
       * air.static_init constructor. When that store's value is a constant
       * (it is when it came from function constants) every load answers with
       * it -- the generic uber shaders' derived flags (fc_has_texture_matrix)
       * decide branches and pointer selects at translation, as Metal's
       * specialisation does. (Ported from tools/mesa/air.)
       */
      bool fold;
      bool known;
      uint64_t bits;
      unsigned bit_size;
   } *global_plan;

   /* what the entry point's parameters are */
   unsigned num_params;
   struct air_val **params;

   /* outputs, in the order of the returned struct's fields */
   unsigned num_outputs;
   struct {
      unsigned location;      /* gl_varying_slot or gl_frag_result */
      unsigned index;         /* dual-source index */
      nir_variable *var;
   } *outputs;

   unsigned scratch_size;
   /* pointers stored into scratch at a known offset (a local `alloca ptr`:
    * CoreAnimation's generic uber shaders keep a sampler argument so and read
    * it back): kept as the values they are, since memory cannot hold them
    */
   /* what the static initializers stored, as constants, into program-scope
    * variables (bytes at a scratch offset): a load of exactly those bytes is
    * that constant -- the generic uber shaders' fc_fragment_shader_config,
    * filled field by field from function constants. Not for a variable the
    * body itself stores into (scratch_dirty: those offsets' ranges).
    */
   bool in_static_init;
   unsigned num_known_mem;
   struct { uint64_t offset; unsigned comps, bit_size; uint64_t bits[4]; } known_mem[256];
   unsigned num_scratch_dirty;
   struct { uint64_t lo, hi; } scratch_dirty[64];
   bool scratch_dirty_all;
   unsigned num_scratch_ptrs;
   struct { uint64_t offset; struct air_val *val; } scratch_ptrs[64];
   unsigned shared_size;
   unsigned tg_arg_param_plus1;   /* the [[threadgroup(0)]] parameter's index + 1, 0 when there is none */

   /* module-scope variables, by their air_global */
   struct hash_table *globals;

   char error[256];
};

bool air_error(struct air_ctx *c, const char *fmt, ...) PRINTFLIKE(2, 3);

/* value bookkeeping */
struct air_val *air_val_ssa(struct air_ctx *c, nir_def *def);
struct air_val *air_val_ptr(struct air_ctx *c, struct air_ptr ptr);
struct air_val *air_get(struct air_ctx *c, const struct air_value *v);
void air_set(struct air_ctx *c, const struct air_value *v, struct air_val *val);

/* an SSA value, with aggregates and pointers refused */
nir_def *air_get_ssa(struct air_ctx *c, const struct air_value *v);

/* the NIR type of an AIR type: bit size and components */
unsigned air_nir_bit_size(struct air_ctx *c, const struct air_type *t);
unsigned air_nir_components(const struct air_type *t);

/* memory, by the pointer's class */
nir_def *air_load_ptr(struct air_ctx *c, struct air_ptr ptr, const struct air_type *type,
                      unsigned align);
bool air_store_ptr(struct air_ctx *c, struct air_ptr ptr, nir_def *value, unsigned align);
nir_def *ptr_binding(struct air_ctx *c, struct air_ptr ptr);
struct air_ptr air_ptr_add(struct air_ctx *c, struct air_ptr ptr, nir_def *bytes);

/* air.* and llvm.* calls; the result is set on the instruction by the callee */
bool air_translate_call(struct air_ctx *c, const struct air_inst *in, const char *name);
/* a call of a function defined in this module, folded in by nir_inline_functions */
bool air_call_in_module(struct air_ctx *c, const struct air_inst *in, const char *name);

#endif /* AIR_PRIVATE_H */
