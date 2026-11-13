/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * One AIR module -> one nir_shader. The mapping from AIR's named metadata to
 * NIR's variables, bindings and system values is the table in docs/air.md
 * section 3.1; this file is that table in code, and nothing here decides
 * anything the table does not state.
 */

#include "air_private.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "util/hash_table.h"
#include "util/u_math.h"

static struct air_global_plan *foldable_global_plan(struct air_ctx *c, const struct air_value *v);

/* A scratch pointer's byte offset, when it is known at translation */
static bool
scratch_const_offset(const struct air_ptr *p, uint64_t *off)
{
   if (p->class != AIR_PTR_SCRATCH)
      return false;
   if (!p->offset) {
      *off = 0;
      return true;
   }
   if (!nir_def_is_const(p->offset))
      return false;
   *off = nir_const_value_as_uint(nir_def_as_load_const(p->offset)->value[0], p->offset->bit_size);
   return true;
}

/* a constant the static initializers stored at scratch offset OFF */
static void
known_mem_record(struct air_ctx *c, uint64_t off, nir_def *v)
{
   unsigned bytes = v->num_components * v->bit_size / 8;
   /* a later store over the same bytes replaces what was known of them */
   for (unsigned k = 0; k < c->num_known_mem; k++) {
      uint64_t kb = c->known_mem[k].comps * c->known_mem[k].bit_size / 8;
      if (c->known_mem[k].offset < off + bytes && off < c->known_mem[k].offset + kb)
         c->known_mem[k] = c->known_mem[--c->num_known_mem], k--;
   }
   if (c->num_known_mem == ARRAY_SIZE(c->known_mem))
      return;
   unsigned n = c->num_known_mem++;
   c->known_mem[n].offset = off;
   c->known_mem[n].comps = v->num_components;
   c->known_mem[n].bit_size = v->bit_size;
   for (unsigned i = 0; i < v->num_components; i++)
      c->known_mem[n].bits[i] = nir_const_value_as_uint(nir_def_as_load_const(v)->value[i], v->bit_size);
}

/* a load of TYPE at scratch offset OFF that is exactly a known store: its
 * constant (NULL otherwise, or when the bytes are a variable the body writes)
 */
static nir_def *
known_mem_load(struct air_ctx *c, uint64_t off, const struct air_type *type)
{
   if (c->in_static_init || c->scratch_dirty_all || !type)
      return NULL;
   unsigned comps = air_nir_components(type), bits = air_nir_bit_size(c, type);
   if (bits < 8 || comps > 4)
      return NULL;
   for (unsigned d = 0; d < c->num_scratch_dirty; d++)
      if (off >= c->scratch_dirty[d].lo && off < c->scratch_dirty[d].hi)
         return NULL;
   for (unsigned k = 0; k < c->num_known_mem; k++) {
      if (c->known_mem[k].offset != off || c->known_mem[k].comps != comps || c->known_mem[k].bit_size != bits)
         continue;
      nir_def *parts[4];
      for (unsigned i = 0; i < comps; i++)
         parts[i] = nir_imm_intN_t(&c->b, c->known_mem[k].bits[i], bits);
      return comps == 1 ? parts[0] : nir_vec(&c->b, parts, comps);
   }
   return NULL;
}

bool
air_error(struct air_ctx *c, const char *fmt, ...)
{
   if (!c->error[0]) {
      va_list ap;
      va_start(ap, fmt);
      vsnprintf(c->error, sizeof(c->error), fmt, ap);
      va_end(ap);
   }
   return false;
}

/* -- values ------------------------------------------------------------- */

struct air_val *
air_val_ssa(struct air_ctx *c, nir_def *def)
{
   struct air_val *v = rzalloc(c->mem_ctx, struct air_val);
   v->kind = AIR_VAL_SSA;
   v->def = def;
   return v;
}

struct air_val *
air_val_ptr(struct air_ctx *c, struct air_ptr ptr)
{
   struct air_val *v = rzalloc(c->mem_ctx, struct air_val);
   v->kind = AIR_VAL_PTR;
   v->ptr = ptr;
   return v;
}

static struct air_val *
air_val_agg(struct air_ctx *c, unsigned n)
{
   struct air_val *v = rzalloc(c->mem_ctx, struct air_val);
   v->kind = AIR_VAL_AGG;
   v->num_elems = n;
   v->elems = rzalloc_array(c->mem_ctx, struct air_val *, MAX2(n, 1));
   return v;
}

void
air_set(struct air_ctx *c, const struct air_value *v, struct air_val *val)
{
   _mesa_hash_table_insert(c->values, v, val);
}

unsigned
air_nir_bit_size(struct air_ctx *c, const struct air_type *t)
{
   unsigned bits = air_type_bit_size(air_type_scalar(t));
   /* NIR has no i1 in SSA: booleans are 1-bit defs, which it does have */
   return bits;
}

unsigned
air_nir_components(const struct air_type *t)
{
   return t->kind == AIR_TYPE_VECTOR ? t->count : 1;
}

static struct air_val *air_constant_val(struct air_ctx *c, const struct air_constant *k);

struct air_val *
air_get(struct air_ctx *c, const struct air_value *v)
{
   struct hash_entry *e = _mesa_hash_table_search(c->values, v);
   if (e)
      return e->data;
   const struct air_constant *k = air_value_as_constant(v);
   if (k) {
      /* KNOWN, NOT FIXED (11-13; devices/akita TODO 18c): the constant is made at its first use and kept, its
       * load_const in whatever block that use was in. A later use on a path not through that block -- `if (a) x = -x;
       * if (b) y = -y;`, both negations LLVM's `fsub -0.0, v` (ANGLE's blit vertex function) -- gets, from
       * nir_repair_ssa, a phi of the constant and UNDEFINED. The Mali reads undefined as 0, so -0.0 and 0 survive it
       * and any other constant would not. Making it anew at each use, or once at the function's start, both stopped
       * the compile of CoreAnimation's uber_fragment for the Calendar widget ("Unhandled ALU op i2i1": a block with
       * integers of 3 bits, which this translator does not really serve, is then no longer folded away) -- so the
       * integers of odd widths first.
       */
      struct air_val *val = air_constant_val(c, k);
      if (val)
         air_set(c, v, val);
      return val;
   }
   air_error(c, "instruction: a %s value used before it is defined%s%s",
             v->kind == AIR_VALUE_GLOBAL ? "global" :
             v->kind == AIR_VALUE_FUNCTION ? "function" :
             v->kind == AIR_VALUE_ARGUMENT ? "argument" :
             v->kind == AIR_VALUE_INSTRUCTION ? "computed" : "metadata",
             v->name ? " " : "", v->name ? v->name : "");
   if (v->kind == AIR_VALUE_INSTRUCTION && !c->fn && strlen(c->error) + 40 < sizeof(c->error))
      strcat(c->error, " (outside any function's translation)");
   if (v->kind == AIR_VALUE_INSTRUCTION && c->fn) {
      /* which one, and where it stands in the function being translated (11-12) */
      const struct air_inst *vi = (const struct air_inst *)v;
      char more[120];
      int vb = -1, vn = -1;

      for (unsigned bi = 0; bi < c->fn->num_blocks && vb < 0; bi++)
         for (unsigned i = 0; i < c->fn->blocks[bi].num_insts; i++)
            if (c->fn->blocks[bi].insts[i] == vi) {
               vb = (int)bi;
               vn = (int)i;
               break;
            }
      snprintf(more, sizeof(more), " (the value: opcode %u sub %u, block %d instruction %d of %s)", vi->op, vi->sub, vb, vn,
               c->fn->base.name ? c->fn->base.name : "?");
      if (strlen(c->error) + strlen(more) < sizeof(c->error))
         strcat(c->error, more);
   }
   return NULL;
}

nir_def *
air_get_ssa(struct air_ctx *c, const struct air_value *v)
{
   struct air_val *val = air_get(c, v);
   if (!val)
      return NULL;
   if (val->kind != AIR_VAL_SSA) {
      air_error(c, "instruction: %s where a number belongs",
                val->kind == AIR_VAL_PTR ? "a pointer" : "an aggregate");
      return NULL;
   }
   return val->def;
}

/* -- constants ---------------------------------------------------------- */

static nir_def *
scalar_constant(struct air_ctx *c, const struct air_type *t, uint64_t bits)
{
   unsigned n = air_type_bit_size(t);
   if (t->kind == AIR_TYPE_INT && t->bits == 1)
      return nir_imm_bool(&c->b, bits & 1);
   if (air_type_is_float(t)) {
      switch (n) {
      case 16: return nir_imm_intN_t(&c->b, bits, 16);
      case 32: return nir_imm_int(&c->b, (uint32_t)bits);
      case 64: return nir_imm_int64(&c->b, bits);
      default: break;
      }
   }
   return nir_imm_intN_t(&c->b, bits, MAX2(n, 1));
}

/* The float bits of a constant have to come through unchanged: a float
 * constant is built as an integer of the same bits and reinterpreted, so
 * that no NaN payload or denormal is rewritten on the way.
 */
static nir_def *
reinterpret_float(struct air_ctx *c, const struct air_type *t, nir_def *bits)
{
   if (!air_type_is_float(t))
      return bits;
   return nir_mov(&c->b, bits); /* same bits, and NIR tracks the type by use */
}

static struct air_val *
air_constant_val(struct air_ctx *c, const struct air_constant *k)
{
   const struct air_type *t = k->base.type;

   switch (k->ckind) {
   case AIR_CONST_INT:
   case AIR_CONST_FLOAT:
      return air_val_ssa(c, reinterpret_float(c, t, scalar_constant(c, air_type_scalar(t), k->u)));

   case AIR_CONST_NULL:
      if (t->kind == AIR_TYPE_POINTER) {
         struct air_ptr p = { .class = AIR_PTR_NONE, .pointee = t->elem };
         return air_val_ptr(c, p);
      }
      if (t->kind == AIR_TYPE_STRUCT || t->kind == AIR_TYPE_ARRAY) {
         unsigned n = t->kind == AIR_TYPE_STRUCT ? t->num_members : t->count;
         struct air_val *v = air_val_agg(c, n);
         for (unsigned i = 0; i < n; i++) {
            const struct air_type *et = t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem;
            struct air_constant z = { .base = { .kind = AIR_VALUE_CONSTANT, .type = et },
                                      .ckind = AIR_CONST_NULL };
            if (!(v->elems[i] = air_constant_val(c, &z)))
               return NULL;
         }
         return v;
      }
      return air_val_ssa(c, nir_imm_zero(&c->b, air_nir_components(t), air_nir_bit_size(c, t)));

   case AIR_CONST_UNDEF:
   case AIR_CONST_POISON:
      if (t->kind == AIR_TYPE_STRUCT || t->kind == AIR_TYPE_ARRAY) {
         unsigned n = t->kind == AIR_TYPE_STRUCT ? t->num_members : t->count;
         struct air_val *v = air_val_agg(c, n);
         /* fields stay NONE until insertvalue puts something there */
         return v;
      }
      if (t->kind == AIR_TYPE_POINTER) {
         struct air_ptr p = { .class = AIR_PTR_NONE, .pointee = t->elem };
         return air_val_ptr(c, p);
      }
      return air_val_ssa(c, nir_undef(&c->b, air_nir_components(t), air_nir_bit_size(c, t)));

   case AIR_CONST_AGGREGATE: {
      if (t->kind == AIR_TYPE_VECTOR) {
         nir_def *comps[NIR_MAX_VEC_COMPONENTS];
         if (k->num_elems > NIR_MAX_VEC_COMPONENTS)
            return air_error(c, "instruction: a vector constant of %u elements", k->num_elems), NULL;
         for (unsigned i = 0; i < k->num_elems; i++)
            if (!(comps[i] = air_get_ssa(c, k->elems[i])))
               return NULL;
         return air_val_ssa(c, nir_vec(&c->b, comps, k->num_elems));
      }
      struct air_val *v = air_val_agg(c, k->num_elems);
      for (unsigned i = 0; i < k->num_elems; i++)
         if (!(v->elems[i] = air_get(c, k->elems[i])))
            return NULL;
      return v;
   }

   case AIR_CONST_DATA: {
      const struct air_type *et = t->kind == AIR_TYPE_STRUCT ? NULL : t->elem;
      if (!et)
         return air_error(c, "instruction: a data constant of a struct type"), NULL;
      if (t->kind == AIR_TYPE_VECTOR) {
         nir_def *comps[NIR_MAX_VEC_COMPONENTS];
         if (k->num_elems > NIR_MAX_VEC_COMPONENTS)
            return air_error(c, "instruction: a vector constant of %u elements", k->num_elems), NULL;
         for (unsigned i = 0; i < k->num_elems; i++)
            comps[i] = reinterpret_float(c, et, scalar_constant(c, et, k->data[i]));
         return air_val_ssa(c, nir_vec(&c->b, comps, k->num_elems));
      }
      struct air_val *v = air_val_agg(c, k->num_elems);
      for (unsigned i = 0; i < k->num_elems; i++)
         v->elems[i] = air_val_ssa(c, reinterpret_float(c, et, scalar_constant(c, et, k->data[i])));
      return v;
   }

   case AIR_CONST_EXPR:
      /* with opaque pointers the only constant expression left in the corpus
       * is a pointer bitcast, which is the pointer itself
       */
      if (k->expr_op == AIR_OP_CAST && k->expr_sub == AIR_CAST_BITCAST &&
          k->base.type->kind == AIR_TYPE_POINTER)
         return air_get(c, k->operands[0]);
      /* ... and a constant getelementptr: the base pointer with a constant
       * byte offset and a narrowed pointee (CoreAnimation's generic uber
       * shaders name a module constant's field so; ported from tools/mesa/air)
       */
      if (k->expr_op == AIR_OP_GEP) {
         if (k->num_operands < 2 || !k->src_type)
            return air_error(c, "instruction: a constant getelementptr with no base "
                                "or no source type"), NULL;
         struct air_val *base = air_get(c, k->operands[0]);
         if (!base)
            return NULL;
         if (base->kind != AIR_VAL_PTR)
            return air_error(c, "instruction: a constant getelementptr on something "
                                "that is not a pointer"), NULL;
         struct air_ptr p = base->ptr;
         const struct air_type *t = k->src_type;
         uint64_t off = 0, idx = 0;
         if (!air_const_uint(k->operands[1], &idx))
            return air_error(c, "instruction: a constant getelementptr whose first "
                                "index is not a constant"), NULL;
         off += idx * air_type_alloc_size(c->m, t);
         for (unsigned i = 2; i < k->num_operands; i++) {
            if (!air_const_uint(k->operands[i], &idx))
               return air_error(c, "instruction: a constant getelementptr with an index "
                                   "that is not a constant"), NULL;
            if (t->kind == AIR_TYPE_STRUCT) {
               if (idx >= t->num_members)
                  return air_error(c, "instruction: a constant getelementptr to struct "
                                      "field %u of %u", (unsigned)idx, t->num_members), NULL;
               off += air_struct_member_offset(c->m, t, idx);
               t = t->members[idx];
            } else if (t->kind == AIR_TYPE_ARRAY || t->kind == AIR_TYPE_VECTOR) {
               off += idx * (t->kind == AIR_TYPE_ARRAY ? air_type_alloc_size(c->m, t->elem)
                                                       : MAX2(air_type_bit_size(t->elem) / 8, 1));
               t = t->elem;
            } else {
               return air_error(c, "instruction: a constant getelementptr into a scalar"), NULL;
            }
         }
         p.pointee = t;
         if (off)
            p = air_ptr_add(c, p, nir_imm_int(&c->b, off));
         return air_val_ptr(c, p);
      }
      return air_error(c, "instruction: constant expression %u", k->expr_op), NULL;
   }
   return air_error(c, "instruction: constant kind %u", k->ckind), NULL;
}

/* -- memory ------------------------------------------------------------- */

struct air_ptr
air_ptr_add(struct air_ctx *c, struct air_ptr ptr, nir_def *bytes)
{
   if (!bytes)
      return ptr;
   ptr.offset = ptr.offset ? nir_iadd(&c->b, ptr.offset, bytes) : bytes;
   return ptr;
}

/* the buffer's index: as it stands, or the one chosen while the shader runs */
nir_def *
ptr_binding(struct air_ctx *c, struct air_ptr ptr)
{
   return ptr.binding_def ? ptr.binding_def : nir_imm_int(&c->b, ptr.binding);
}

static nir_def *
ptr_offset(struct air_ctx *c, struct air_ptr ptr)
{
   return ptr.offset ? ptr.offset : nir_imm_int(&c->b, 0);
}

nir_def *
air_load_ptr(struct air_ctx *c, struct air_ptr ptr, const struct air_type *type, unsigned align)
{
   unsigned comps = air_nir_components(type);
   unsigned bits = air_nir_bit_size(c, type);
   if (type->kind == AIR_TYPE_POINTER)
      return air_error(c, "instruction: a pointer loaded from memory (argument buffers)"), NULL;
   if (!align)
      align = MAX2(bits / 8, 1);

   /* a 1-bit load is a byte in memory */
   bool as_bool = bits == 1;
   if (as_bool)
      bits = 8;

   nir_def *off = ptr_offset(c, ptr);
   nir_def *res;
   switch (ptr.class) {
   case AIR_PTR_UBO:
      res = nir_load_ubo(&c->b, comps, bits, ptr_binding(c, ptr), off,
                         .align_mul = align, .align_offset = 0, .range_base = 0, .range = ~0);
      break;
   case AIR_PTR_SSBO:
      res = nir_load_ssbo(&c->b, comps, bits, ptr_binding(c, ptr), off,
                          .align_mul = align, .align_offset = 0);
      break;
   case AIR_PTR_SHARED:
      res = nir_load_shared(&c->b, comps, bits, off, .base = 0, .align_mul = align);
      break;
   case AIR_PTR_SCRATCH:
      res = nir_load_scratch(&c->b, comps, bits, off, .align_mul = align);
      break;
   case AIR_PTR_CONSTANT_DATA:
      res = nir_load_constant(&c->b, comps, bits, off, .base = 0, .range = c->nir->constant_data_size,
                              .align_mul = align);
      break;
   case AIR_PTR_FUNCTION_CONSTANT: {
      /* its value is known at pipeline creation, so the load is that value */
      const struct air_constant_value *v = NULL;
      for (unsigned i = 0; c->options && i < c->options->num_constants; i++)
         if (c->options->constants[i].index == ptr.binding)
            v = &c->options->constants[i];
      /* one the pipeline did not supply reads as zero (Metal: undefined; a
       * shader asks air.is_function_constant_defined first); a vector's
       * component i is at bit i * its width (fc_color_opcodes0, a uchar3)
       */
      uint64_t value = v ? v->bits : 0;
      unsigned width = as_bool ? 8 : bits;
      if (comps * width > 64)
         return air_error(c, "interface: a function constant of %u x %u bits", comps, width), NULL;
      nir_def *parts[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < comps; i++)
         parts[i] = nir_imm_intN_t(&c->b, width == 64 ? value : (value >> (i * width)) & ((1ull << width) - 1), width);
      res = comps == 1 ? parts[0] : nir_vec(&c->b, parts, comps);
      break;
   }
   default:
      return air_error(c, "instruction: a load through a %s pointer",
                       ptr.class == AIR_PTR_NONE ? "null" : "texture or sampler"), NULL;
   }
   return as_bool ? nir_ine_imm(&c->b, res, 0) : res;
}

bool
air_store_ptr(struct air_ctx *c, struct air_ptr ptr, nir_def *value, unsigned align)
{
   unsigned comps = value->num_components;
   if (value->bit_size == 1) {
      value = nir_b2iN(&c->b, value, 8);
   }
   if (!align)
      align = MAX2(value->bit_size / 8, 1);

   nir_def *off = ptr_offset(c, ptr);
   switch (ptr.class) {
   case AIR_PTR_SSBO:
      nir_store_ssbo(&c->b, value, ptr_binding(c, ptr), off,
                     .write_mask = BITFIELD_MASK(comps), .align_mul = align);
      return true;
   case AIR_PTR_SHARED:
      nir_store_shared(&c->b, value, off, .base = 0, .write_mask = BITFIELD_MASK(comps),
                       .align_mul = align);
      return true;
   case AIR_PTR_SCRATCH:
      nir_store_scratch(&c->b, value, off, .align_mul = align,
                        .write_mask = BITFIELD_MASK(comps));
      return true;
   case AIR_PTR_UBO:
   case AIR_PTR_CONSTANT_DATA:
      return air_error(c, "instruction: a store into constant memory");
   default:
      return air_error(c, "instruction: a store through a %s pointer",
                       ptr.class == AIR_PTR_NONE ? "null" : "texture or sampler");
   }
}

/* -- metadata helpers --------------------------------------------------- */

static const char *
md_str(const struct air_md *node, unsigned i)
{
   if (!node || i >= node->num_ops || !node->ops[i] || node->ops[i]->kind != AIR_MD_STRING)
      return NULL;
   return air_md_string(node->ops[i]);
}

static bool
md_int(const struct air_md *node, unsigned i, uint64_t *out)
{
   if (!node || i >= node->num_ops || !node->ops[i] || node->ops[i]->kind != AIR_MD_VALUE)
      return false;
   /* zero is written as a null constant, as everywhere else in the bitcode */
   return air_const_uint(node->ops[i]->value, out);
}

/* the index of the operand after the key, or -1 */
static int
md_key(const struct air_md *node, const char *key)
{
   for (unsigned i = 0; i < (node ? node->num_ops : 0); i++) {
      const char *s = md_str(node, i);
      if (s && !strcmp(s, key))
         return i;
   }
   return -1;
}

static bool
md_key_int(const struct air_md *node, const char *key, uint64_t *out)
{
   int i = md_key(node, key);
   return i >= 0 && md_int(node, i + 1, out);
}

static const char *
md_key_str(const struct air_md *node, const char *key)
{
   int i = md_key(node, key);
   return i >= 0 ? md_str(node, i + 1) : NULL;
}

static const struct air_named_md *
named_md(const struct air_module *m, const char *name)
{
   for (unsigned i = 0; i < m->num_named_md; i++)
      if (!strcmp(m->named_md[i].name, name))
         return &m->named_md[i];
   return NULL;
}

/* -- stage -------------------------------------------------------------- */

struct stage_md {
   gl_shader_stage stage;
   const char *md_name;
};

static const struct stage_md stages[] = {
   { MESA_SHADER_VERTEX, "air.vertex" },
   { MESA_SHADER_FRAGMENT, "air.fragment" },
   { MESA_SHADER_COMPUTE, "air.kernel" },
   { MESA_SHADER_NONE, "air.visible" },
};

static int
module_stage(const struct air_module *m, const struct air_md **entry)
{
   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      const struct air_named_md *n = named_md(m, stages[i].md_name);
      if (n && n->num_ops >= 1 && n->ops[0] && n->ops[0]->kind == AIR_MD_NODE) {
         if (entry)
            *entry = n->ops[0];
         return stages[i].stage;
      }
   }
   return -2; /* no entry point at all */
}

int
air_module_stage(void *mem_ctx, const void *bitcode, size_t size, const char **calls,
                 const char **error)
{
   void *ctx = ralloc_context(mem_ctx);
   struct air_module *m = air_module_read(ctx, bitcode, size, error);
   if (!m) {
      ralloc_free(ctx);
      return -1;
   }

   if (calls) {
      size_t len = 1;
      for (unsigned i = 0; i < m->num_functions; i++) {
         const char *n = m->functions[i]->base.name;
         if (m->functions[i]->is_declaration && n && !strncmp(n, "air.", 4))
            len += strlen(n) + 1;
      }
      char *buf = rzalloc_size(mem_ctx, len), *p = buf;
      for (unsigned i = 0; i < m->num_functions; i++) {
         const char *n = m->functions[i]->base.name;
         if (m->functions[i]->is_declaration && n && !strncmp(n, "air.", 4)) {
            memcpy(p, n, strlen(n) + 1);
            p += strlen(n) + 1;
         }
      }
      *calls = buf;
   }

   int stage = module_stage(m, NULL);
   ralloc_free(ctx);
   return stage;
}

unsigned
air_module_function_constants(void *mem_ctx, const void *bitcode, size_t size,
                              struct air_constant_value **out)
{
   void *ctx = ralloc_context(mem_ctx);
   const char *err;
   struct air_module *m = air_module_read(ctx, bitcode, size, &err);
   *out = NULL;
   if (!m) {
      ralloc_free(ctx);
      return 0;
   }
   const struct air_named_md *n = named_md(m, "air.function_constants");
   unsigned num = 0;
   if (n && n->num_ops) {
      *out = rzalloc_array(mem_ctx, struct air_constant_value, n->num_ops);
      for (unsigned i = 0; i < n->num_ops; i++) {
         const struct air_md *node = n->ops[i];
         for (unsigned k = 1; k < (node ? node->num_ops : 0); k++) {
            uint64_t idx;
            if (md_int(node, k, &idx)) {
               (*out)[num++].index = idx;
               break;
            }
         }
      }
   }
   ralloc_free(ctx);
   return num;
}

/* -- the interface ------------------------------------------------------ */

/* AIR's names for what NIR calls system values */
enum air_sysval {
   AIR_SV_VERTEX_ID, AIR_SV_INSTANCE_ID, AIR_SV_BASE_VERTEX, AIR_SV_BASE_INSTANCE,
   AIR_SV_FRONT_FACING, AIR_SV_SAMPLE_ID, AIR_SV_SAMPLE_MASK_IN, AIR_SV_POINT_COORD,
   AIR_SV_THREAD_POS_GRID, AIR_SV_THREAD_POS_TG, AIR_SV_TG_POS_GRID,
   AIR_SV_THREADS_PER_TG, AIR_SV_TGS_PER_GRID, AIR_SV_THREAD_INDEX_TG,
   AIR_SV_THREADS_PER_GRID, AIR_SV_SIMD_LANE, AIR_SV_SIMD_INDEX, AIR_SV_SIMD_WIDTH,
};

static const struct { const char *key; enum air_sysval sv; } sysvals[] = {
   { "air.vertex_id", AIR_SV_VERTEX_ID },
   { "air.instance_id", AIR_SV_INSTANCE_ID },
   { "air.base_vertex", AIR_SV_BASE_VERTEX },
   { "air.base_instance", AIR_SV_BASE_INSTANCE },
   { "air.front_facing", AIR_SV_FRONT_FACING },
   { "air.sample_id", AIR_SV_SAMPLE_ID },
   { "air.sample_mask_in", AIR_SV_SAMPLE_MASK_IN },
   { "air.point_coord", AIR_SV_POINT_COORD },
   { "air.thread_position_in_grid", AIR_SV_THREAD_POS_GRID },
   { "air.thread_position_in_threadgroup", AIR_SV_THREAD_POS_TG },
   { "air.threadgroup_position_in_grid", AIR_SV_TG_POS_GRID },
   { "air.threads_per_threadgroup", AIR_SV_THREADS_PER_TG },
   { "air.threadgroups_per_grid", AIR_SV_TGS_PER_GRID },
   { "air.thread_index_in_threadgroup", AIR_SV_THREAD_INDEX_TG },
   { "air.threads_per_grid", AIR_SV_THREADS_PER_GRID },
   { "air.thread_index_in_simdgroup", AIR_SV_SIMD_LANE },
   { "air.simdgroup_index_in_threadgroup", AIR_SV_SIMD_INDEX },
   { "air.threads_per_simdgroup", AIR_SV_SIMD_WIDTH },
};

static nir_def *
load_sysval(struct air_ctx *c, enum air_sysval sv)
{
   switch (sv) {
   case AIR_SV_VERTEX_ID: return nir_load_vertex_id(&c->b);
   /* Metal's instance id counts from the base instance, GL's does not */
   case AIR_SV_INSTANCE_ID:
      return nir_iadd(&c->b, nir_load_instance_id(&c->b), nir_load_base_instance(&c->b));
   case AIR_SV_BASE_VERTEX: return nir_load_first_vertex(&c->b);
   case AIR_SV_BASE_INSTANCE: return nir_load_base_instance(&c->b);
   case AIR_SV_FRONT_FACING: return nir_load_front_face(&c->b, 1);
   case AIR_SV_SAMPLE_ID: return nir_load_sample_id(&c->b);
   case AIR_SV_SAMPLE_MASK_IN: return nir_load_sample_mask_in(&c->b);
   case AIR_SV_POINT_COORD: return nir_load_point_coord(&c->b);
   case AIR_SV_THREAD_POS_GRID: return nir_load_global_invocation_id(&c->b, 32);
   case AIR_SV_THREAD_POS_TG: return nir_load_local_invocation_id(&c->b);
   case AIR_SV_TG_POS_GRID: return nir_load_workgroup_id(&c->b);
   case AIR_SV_THREADS_PER_TG: return nir_load_workgroup_size(&c->b);
   case AIR_SV_TGS_PER_GRID: return nir_load_num_workgroups(&c->b);
   case AIR_SV_THREAD_INDEX_TG: {
      /* (not nir_load_local_invocation_index: the Mali's compiler has no such input -- "Unhandled intrinsic
       * load_local_invocation_index", and mediaserverd died in the compile) (z * height + y) * width + x */
      nir_def *id = nir_load_local_invocation_id(&c->b), *size = nir_load_workgroup_size(&c->b);

      return nir_iadd(&c->b, nir_imul(&c->b, nir_iadd(&c->b, nir_imul(&c->b, nir_channel(&c->b, id, 2),
          nir_channel(&c->b, size, 1)), nir_channel(&c->b, id, 1)), nir_channel(&c->b, size, 0)), nir_channel(&c->b, id, 0));
   }
   case AIR_SV_THREADS_PER_GRID:
      return nir_imul(&c->b, nir_load_num_workgroups(&c->b), nir_load_workgroup_size(&c->b));
   case AIR_SV_SIMD_LANE: return nir_load_subgroup_invocation(&c->b);
   case AIR_SV_SIMD_INDEX: return nir_load_subgroup_id(&c->b);
   case AIR_SV_SIMD_WIDTH: return nir_load_subgroup_size(&c->b);
   }
   return NULL;
}

/* a system value cut or widened to the type the shader declared */
static nir_def *
fit(struct air_ctx *c, nir_def *def, const struct air_type *t)
{
   unsigned want = air_nir_components(t);
   unsigned bits = air_nir_bit_size(c, t);
   if (def->num_components > want)
      def = nir_trim_vector(&c->b, def, want);
   else if (def->num_components < want)
      def = nir_pad_vector(&c->b, def, want);
   if (def->bit_size != bits)
      def = air_type_is_float(air_type_scalar(t))
               ? nir_u2fN(&c->b, def, bits)
               : nir_u2uN(&c->b, def, bits);
   return def;
}

static void
add_resource(struct air_ctx *c, enum air_resource_kind kind, unsigned index, unsigned size,
             bool read, bool write, unsigned param, const struct air_md *node)
{
   struct air_shader_info *info = c->info;
   info->resources = reralloc(c->mem_ctx, info->resources, struct air_resource,
                              info->num_resources + 1);
   info->resources[info->num_resources++] = (struct air_resource) {
      .kind = kind, .index = index, .size = size, .read = read, .write = write,
      .param = param,
      .name = md_key_str(node, "air.arg_name"),
      .type_name = md_key_str(node, "air.arg_type_name"),
   };
}

static nir_variable *
make_io_var(struct air_ctx *c, nir_variable_mode mode, const struct air_type *t,
            unsigned location, const char *name)
{
   const struct glsl_type *gt;
   unsigned comps = air_nir_components(t);
   unsigned bits = air_nir_bit_size(c, t);
   if (air_type_is_float(air_type_scalar(t)))
      gt = bits == 16 ? glsl_f16vec_type(comps) : glsl_vec_type(comps);
   else
      gt = glsl_vector_type(bits == 16 ? GLSL_TYPE_INT16 : GLSL_TYPE_INT, comps);

   nir_variable *var = nir_variable_create(c->nir, mode, gt, name ? name : "io");
   var->data.location = location;
   return var;
}

/* fragment inputs and vertex outputs take the slots in the order they appear;
 * the runtime links the two sides by name (docs/air.md section 7.5)
 */
static void
add_varying(struct air_ctx *c, const char *name, unsigned slot, unsigned comps,
            const struct air_md *node)
{
   struct air_shader_info *info = c->info;
   info->varyings = reralloc(c->mem_ctx, info->varyings, struct air_varying,
                             info->num_varyings + 1);
   info->varyings[info->num_varyings++] = (struct air_varying) {
      .name = name, .slot = slot, .components = comps,
      .flat = md_key(node, "air.flat") >= 0,
      .centroid = md_key(node, "air.centroid") >= 0,
      .sample = md_key(node, "air.sample") >= 0,
      .no_perspective = md_key(node, "air.no_perspective") >= 0,
   };
}

static bool
translate_input(struct air_ctx *c, const struct air_md *node, unsigned *next_varying)
{
   uint64_t param;
   if (!md_int(node, 0, &param))
      return air_error(c, "interface: an input without a parameter number");
   if (param >= c->fn->num_args)
      return air_error(c, "interface: input for parameter %u of %u", (unsigned)param,
                       c->fn->num_args);
   const struct air_type *t = c->fn->args[param].type;

   /* resources */
   if (md_key(node, "air.buffer") >= 0) {
      uint64_t as = 0, index = 0, size = 0;
      md_key_int(node, "air.address_space", &as);
      md_key_int(node, "air.location_index", &index);
      md_key_int(node, "air.buffer_size", &size);
      bool read = md_key(node, "air.read") >= 0 || md_key(node, "air.read_write") >= 0;
      bool write = md_key(node, "air.write") >= 0 || md_key(node, "air.read_write") >= 0;

      struct air_ptr p = { .binding = index, .pointee = t->elem };
      enum air_resource_kind rk;
      if (as == 0) {
         /* no air.address_space in the metadata (VectorKit's Clear::clear functions): the pointer's own, and failing
          * that a buffer the function only reads is a constant buffer, any other a device buffer -- one thing to the
          * back end either way */
         as = t->kind == AIR_TYPE_POINTER ? t->addrspace : 0;
         if (as == 0 || as > 3)
            as = write ? 1 : 2;
      }
      switch (as) {
      case 1: p.class = AIR_PTR_SSBO; rk = AIR_RESOURCE_DEVICE_BUFFER; break;
      case 2: p.class = AIR_PTR_UBO; rk = AIR_RESOURCE_CONSTANT_BUFFER; break;
      case 3:
         p.class = AIR_PTR_SHARED;
         rk = AIR_RESOURCE_THREADGROUP_BUFFER;
         /* a threadgroup argument is sized at dispatch and starts past the
          * static threadgroup arrays: its offset is set once plan_globals has
          * laid those out (see air_to_nir). One is all the corpus uses, so
          * anything else is refused.
          */
         if (index != 0)
            return air_error(c, "interface: a second threadgroup buffer (index %u)",
                             (unsigned)index);
         c->tg_arg_param_plus1 = param + 1;
         break;
      default:
         return air_error(c, "interface: a buffer in address space %u", (unsigned)as);
      }
      add_resource(c, rk, index, size, read, write, param, node);
      c->params[param] = air_val_ptr(c, p);
      return true;
   }

   if (md_key(node, "air.texture") >= 0) {
      uint64_t index = 0;
      md_key_int(node, "air.location_index", &index);
      bool sample = md_key(node, "air.sample") >= 0;
      bool read = md_key(node, "air.read") >= 0;
      bool write = md_key(node, "air.write") >= 0 || md_key(node, "air.read_write") >= 0;
      /* A texture the shader writes is an image to Gallium, with its own
       * binding space; one it only samples or reads is a sampler view.
       */
      struct air_ptr p = {
         .class = AIR_PTR_TEXTURE,
         .binding = index,
         .is_image = write,
      };
      add_resource(c, write ? AIR_RESOURCE_IMAGE : AIR_RESOURCE_TEXTURE, index, 0,
                   sample || read, write, param, node);
      c->params[param] = air_val_ptr(c, p);
      return true;
   }

   if (md_key(node, "air.sampler") >= 0) {
      uint64_t index = 0;
      md_key_int(node, "air.location_index", &index);
      struct air_ptr p = { .class = AIR_PTR_SAMPLER, .binding = index };
      add_resource(c, AIR_RESOURCE_SAMPLER, index, 0, true, false, param, node);
      c->params[param] = air_val_ptr(c, p);
      return true;
   }

   /* system values */
   for (unsigned i = 0; i < ARRAY_SIZE(sysvals); i++) {
      if (md_key(node, sysvals[i].key) < 0)
         continue;
      nir_def *def = load_sysval(c, sysvals[i].sv);
      if (!def)
         return air_error(c, "interface: %s", sysvals[i].key);
      c->params[param] = air_val_ssa(c, fit(c, def, t));
      return true;
   }

   /* interpolated inputs of a fragment shader, and vertex attributes */
   if (md_key(node, "air.position") >= 0) {
      if (c->nir->info.stage == MESA_SHADER_FRAGMENT) {
         nir_def *pos = nir_load_frag_coord(&c->b);
         c->params[param] = air_val_ssa(c, fit(c, pos, t));
         c->nir->info.fs.origin_upper_left = true;
         c->nir->info.fs.pixel_center_integer = false;
         return true;
      }
      return air_error(c, "interface: air.position as an input of a %s shader",
                       gl_shader_stage_name(c->nir->info.stage));
   }
   if (md_key(node, "air.fragment_input") >= 0 || md_key(node, "air.vertex_input") >= 0 ||
       md_key(node, "air.stage_in") >= 0) {
      unsigned slot;
      if (c->nir->info.stage == MESA_SHADER_VERTEX) {
         uint64_t index = 0;
         md_key_int(node, "air.location_index", &index);
         slot = VERT_ATTRIB_GENERIC0 + index;
      } else {
         slot = VARYING_SLOT_VAR0 + (*next_varying)++;
      }
      const char *name = md_key_str(node, "air.arg_name");
      nir_variable *var = make_io_var(c, nir_var_shader_in, t, slot, name);
      var->data.driver_location = c->nir->num_inputs++;
      if (md_key(node, "air.flat") >= 0)
         var->data.interpolation = INTERP_MODE_FLAT;
      else if (md_key(node, "air.no_perspective") >= 0)
         var->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      else
         var->data.interpolation = INTERP_MODE_SMOOTH;
      var->data.centroid = md_key(node, "air.centroid") >= 0;
      var->data.sample = md_key(node, "air.sample") >= 0;
      if (c->nir->info.stage == MESA_SHADER_FRAGMENT)
         add_varying(c, name, slot, air_nir_components(t), node);
      c->params[param] = air_val_ssa(c, nir_load_var(&c->b, var));
      return true;
   }

   /* A COLOUR ATTACHMENT READ AS AN INPUT -- `[[color(n)]]` on a fragment argument, programmable blending: what the
    * primitives already drawn left at this pixel (from upstream's tools/mesa/air). NIR's form is a read of the output,
    * load_output with io_semantics.fb_fetch_output (what nir_lower_io makes of a GLSL inout colour); the Mali compiler
    * lowers it to LD_TILE (bifrost_nir_lower_load_output). Read as four 32-bit floats and narrowed after. The metadata
    * has ONE integer after the key (the target), where an output has two:
    *   input   !{i32 1, !"air.render_target", i32 2, !"air.arg_type_name", !"float", ...}
    * Only float targets: an int or uint [[color(n)]] would come back converted from the attachment's format.
    */
   int rt_in = md_key(node, "air.render_target");
   if (rt_in >= 0) {
      if (c->nir->info.stage != MESA_SHADER_FRAGMENT)
         return air_error(c, "interface: air.render_target as an input of a %s shader",
                          gl_shader_stage_name(c->nir->info.stage));
      if (md_key(node, "air.arg_unused") >= 0) {
         c->params[param] = air_val_ssa(c, nir_undef(&c->b, air_nir_components(t), air_nir_bit_size(c, t)));
         return true;
      }
      uint64_t n = 0;
      if (!md_int(node, rt_in + 1, &n))
         return air_error(c, "interface: an air.render_target input without a target");
      if (n >= MAX_DRAW_BUFFERS)
         return air_error(c, "interface: air.render_target input %u", (unsigned)n);
      unsigned bits = air_nir_bit_size(c, t);
      if (!air_type_is_float(air_type_scalar(t)) || (bits != 16 && bits != 32)) {
         const char *tn = md_key_str(node, "air.arg_type_name");
         return air_error(c, "interface: an air.render_target input of type %s", tn ? tn : "?");
      }
      nir_io_semantics sem = {
         .location = FRAG_RESULT_DATA0 + n,
         .num_slots = 1,
         .fb_fetch_output = true,
         .fb_fetch_output_coherent = true,      /* Metal's guarantee: what earlier primitives left */
      };
      nir_def *fetched = nir_load_output(&c->b, 4, 32, nir_imm_int(&c->b, 0),
                                         .dest_type = nir_type_float32, .io_semantics = sem);
      unsigned want = air_nir_components(t);
      if (want < 4)
         fetched = nir_trim_vector(&c->b, fetched, want);
      if (bits == 16)
         fetched = nir_f2f16(&c->b, fetched);
      c->nir->info.outputs_read |= BITFIELD64_BIT(sem.location);
      c->nir->info.fs.uses_fbfetch_output = true;
      c->nir->info.fs.fbfetch_coherent = true;
      c->params[param] = air_val_ssa(c, fetched);
      return true;
   }

   /* the one thing left that is not an error: an argument nothing reads */
   if (md_key(node, "air.arg_unused") >= 0) {
      c->params[param] = air_val_ssa(c, nir_undef(&c->b, air_nir_components(t),
                                                  air_nir_bit_size(c, t)));
      return true;
   }

   for (unsigned i = 0; i < node->num_ops; i++) {
      const char *s = md_str(node, i);
      if (s && !strncmp(s, "air.", 4) && strcmp(s, "air.arg_name") &&
          strcmp(s, "air.arg_type_name"))
         return air_error(c, "interface: input %s", s);
   }
   return air_error(c, "interface: an input with no key this translator knows");
}

static bool
translate_output(struct air_ctx *c, const struct air_md *node, unsigned field,
                 const struct air_type *t, unsigned *next_varying)
{
   unsigned location = ~0u, index = 0;
   const char *name = md_key_str(node, "air.arg_name");

   int rt = md_key(node, "air.render_target");
   if (rt >= 0) {
      uint64_t n = 0, dual = 0;
      md_int(node, rt + 1, &n);
      md_int(node, rt + 2, &dual);
      location = FRAG_RESULT_DATA0 + n;
      index = dual;
      c->info->num_render_targets = MAX2(c->info->num_render_targets, n + 1);
   } else if (md_key(node, "air.depth") >= 0) {
      location = FRAG_RESULT_DEPTH;
      const char *q = md_key_str(node, "air.depth_qualifier");
      c->nir->info.fs.depth_layout =
         !q || !strcmp(q, "air.any") ? FRAG_DEPTH_LAYOUT_ANY :
         !strcmp(q, "air.greater") ? FRAG_DEPTH_LAYOUT_GREATER :
         !strcmp(q, "air.less") ? FRAG_DEPTH_LAYOUT_LESS : FRAG_DEPTH_LAYOUT_UNCHANGED;
   } else if (md_key(node, "air.sample_mask") >= 0) {
      location = FRAG_RESULT_SAMPLE_MASK;
   } else if (md_key(node, "air.stencil") >= 0) {
      location = FRAG_RESULT_STENCIL;
   } else if (md_key(node, "air.position") >= 0) {
      location = VARYING_SLOT_POS;
   } else if (md_key(node, "air.point_size") >= 0) {
      location = VARYING_SLOT_PSIZ;
   } else if (md_key(node, "air.clip_distance") >= 0) {
      location = VARYING_SLOT_CLIP_DIST0;
   } else if (md_key(node, "air.render_target_array_index") >= 0) {
      location = VARYING_SLOT_LAYER;
   } else if (md_key(node, "air.viewport_array_index") >= 0) {
      location = VARYING_SLOT_VIEWPORT;
   } else if (md_key(node, "air.vertex_output") >= 0) {
      location = VARYING_SLOT_VAR0 + (*next_varying)++;
      add_varying(c, md_str(node, md_key(node, "air.vertex_output") + 1), location,
                  air_nir_components(t), node);
   } else {
      for (unsigned i = 0; i < node->num_ops; i++) {
         const char *s = md_str(node, i);
         if (s && !strncmp(s, "air.", 4) && strcmp(s, "air.arg_name") &&
             strcmp(s, "air.arg_type_name"))
            return air_error(c, "interface: output %s", s);
      }
      return air_error(c, "interface: an output with no key this translator knows");
   }

   nir_variable *var = make_io_var(c, nir_var_shader_out, t, location, name);
   var->data.index = index;
   var->data.driver_location = c->nir->num_outputs++;
   c->outputs[field].location = location;
   c->outputs[field].index = index;
   c->outputs[field].var = var;
   return true;
}

/* -- instructions ------------------------------------------------------- */

static bool
translate_gep(struct air_ctx *c, const struct air_inst *in)
{
   struct air_val *base = air_get(c, in->operands[0]);
   if (!base)
      return false;
   if (base->kind != AIR_VAL_PTR)
      return air_error(c, "instruction: getelementptr on something that is not a pointer");

   struct air_ptr p = base->ptr;
   const struct air_type *t = in->aux_type;

   /* index 0 steps over objects of the source type */
   uint64_t stride = air_type_alloc_size(c->m, t);
   uint64_t first_const;
   if (!air_const_uint(in->operands[1], &first_const) || first_const != 0) {
      nir_def *first = air_get_ssa(c, in->operands[1]);
      if (!first)
         return false;
      p = air_ptr_add(c, p, nir_imul_imm(&c->b, nir_u2u32(&c->b, first), stride));
   }

   for (unsigned i = 2; i < in->num_operands; i++) {
      nir_def *idx = air_get_ssa(c, in->operands[i]);
      if (!idx)
         return false;
      if (t->kind == AIR_TYPE_STRUCT) {
         uint64_t field;
         if (!air_const_uint(in->operands[i], &field))
            return air_error(c, "instruction: a struct field chosen at run time");
         unsigned n = field;
         if (n >= t->num_members)
            return air_error(c, "instruction: struct field %u of %u", n, t->num_members);
         uint64_t off = air_struct_member_offset(c->m, t, n);
         if (off)
            p = air_ptr_add(c, p, nir_imm_int(&c->b, off));
         t = t->members[n];
      } else if (t->kind == AIR_TYPE_ARRAY || t->kind == AIR_TYPE_VECTOR) {
         uint64_t es = t->kind == AIR_TYPE_ARRAY ? air_type_alloc_size(c->m, t->elem)
                                                 : MAX2(air_type_bit_size(t->elem) / 8, 1);
         p = air_ptr_add(c, p, nir_imul_imm(&c->b, nir_u2u32(&c->b, idx), es));
         t = t->elem;
      } else {
         return air_error(c, "instruction: getelementptr into a scalar");
      }
   }
   p.pointee = t;
   air_set(c, &in->base, air_val_ptr(c, p));
   return true;
}

static bool
translate_binop(struct air_ctx *c, const struct air_inst *in)
{
   nir_def *x = air_get_ssa(c, in->operands[0]);
   nir_def *y = air_get_ssa(c, in->operands[1]);
   if (!x || !y)
      return false;
   bool is_float = air_type_is_float(air_type_scalar(in->base.type));
   nir_def *res;

   /* Metal compiles with fast math on, and the modules say so per
    * instruction. Without the flags the arithmetic must not be reassociated,
    * so the builder is told to keep it exact.
    */
   bool was_exact = AIR_B_EXACT(c->b);
   if (is_float)
      AIR_B_SET_EXACT(c->b, !(in->flags & (AIR_FMF_UNSAFE_ALGEBRA | AIR_FMF_REASSOC)));

   switch (in->sub) {
   case AIR_BINOP_ADD: res = is_float ? nir_fadd(&c->b, x, y) : nir_iadd(&c->b, x, y); break;
   case AIR_BINOP_SUB: res = is_float ? nir_fsub(&c->b, x, y) : nir_isub(&c->b, x, y); break;
   case AIR_BINOP_MUL: res = is_float ? nir_fmul(&c->b, x, y) : nir_imul(&c->b, x, y); break;
   case AIR_BINOP_UDIV: res = is_float ? nir_fdiv(&c->b, x, y) : nir_udiv(&c->b, x, y); break;
   case AIR_BINOP_SDIV: res = is_float ? nir_fdiv(&c->b, x, y) : nir_idiv(&c->b, x, y); break;
   case AIR_BINOP_UREM: res = is_float ? nir_frem(&c->b, x, y) : nir_umod(&c->b, x, y); break;
   case AIR_BINOP_SREM: res = is_float ? nir_frem(&c->b, x, y) : nir_irem(&c->b, x, y); break;
   case AIR_BINOP_SHL: res = nir_ishl(&c->b, x, nir_u2u32(&c->b, y)); break;
   case AIR_BINOP_LSHR: res = nir_ushr(&c->b, x, nir_u2u32(&c->b, y)); break;
   case AIR_BINOP_ASHR: res = nir_ishr(&c->b, x, nir_u2u32(&c->b, y)); break;
   case AIR_BINOP_AND: res = nir_iand(&c->b, x, y); break;
   case AIR_BINOP_OR: res = nir_ior(&c->b, x, y); break;
   case AIR_BINOP_XOR: res = nir_ixor(&c->b, x, y); break;
   default:
      AIR_B_SET_EXACT(c->b, was_exact);
      return air_error(c, "instruction: binary operator %u", in->sub);
   }
   AIR_B_SET_EXACT(c->b, was_exact);
   air_set(c, &in->base, air_val_ssa(c, res));
   return true;
}

static bool
translate_cast(struct air_ctx *c, const struct air_inst *in)
{
   const struct air_type *dt = in->base.type;
   const struct air_type *st = in->operands[0]->type;

   if (in->sub == AIR_CAST_BITCAST && dt->kind == AIR_TYPE_POINTER) {
      struct air_val *v = air_get(c, in->operands[0]);
      if (!v || v->kind != AIR_VAL_PTR)
         return air_error(c, "instruction: a bitcast of something that is not a pointer");
      struct air_ptr p = v->ptr;
      p.pointee = dt->elem;
      air_set(c, &in->base, air_val_ptr(c, p));
      return true;
   }

   nir_def *x = air_get_ssa(c, in->operands[0]);
   if (!x)
      return false;
   unsigned bits = air_nir_bit_size(c, dt);
   nir_def *res;
   switch (in->sub) {
   case AIR_CAST_TRUNC:
      res = bits == 1 ? nir_i2b(&c->b, nir_iand_imm(&c->b, x, 1)) : nir_u2uN(&c->b, x, bits);
      break;
   case AIR_CAST_ZEXT:
      res = x->bit_size == 1 ? nir_b2iN(&c->b, x, bits) : nir_u2uN(&c->b, x, bits);
      break;
   case AIR_CAST_SEXT:
      res = x->bit_size == 1 ? nir_ineg(&c->b, nir_b2iN(&c->b, x, bits))
                             : nir_i2iN(&c->b, x, bits);
      break;
   /* to i1: NIR's f2u1/f2i1 have no instruction on the Mali ("Unhandled ALU op f2u1": bi_emit_alu trapped, and the
    * window server died with it); the only results in range are 0 and 1 (-1 for a signed i1), so the value is
    * x != 0 -- out-of-range inputs are poison in LLVM, any answer refines them */
   case AIR_CAST_FPTOUI: res = bits == 1 ? nir_fneu_imm(&c->b, x, 0.0) : nir_f2uN(&c->b, x, bits); break;
   case AIR_CAST_FPTOSI: res = bits == 1 ? nir_fneu_imm(&c->b, x, 0.0) : nir_f2iN(&c->b, x, bits); break;
   case AIR_CAST_UITOFP:
      res = x->bit_size == 1 ? nir_b2fN(&c->b, x, bits) : nir_u2fN(&c->b, x, bits);
      break;
   case AIR_CAST_SITOFP:
      res = x->bit_size == 1 ? nir_fneg(&c->b, nir_b2fN(&c->b, x, bits))
                             : nir_i2fN(&c->b, x, bits);
      break;
   case AIR_CAST_FPTRUNC: res = nir_f2fN(&c->b, x, bits); break;
   case AIR_CAST_FPEXT: res = nir_f2fN(&c->b, x, bits); break;
   case AIR_CAST_BITCAST:
      if (air_nir_components(dt) == air_nir_components(st) &&
          air_nir_bit_size(c, dt) == air_nir_bit_size(c, st)) {
         res = nir_mov(&c->b, x);
      } else if (bits >= 8 && x->bit_size >= 8 &&
                 air_nir_components(dt) * bits == x->num_components * x->bit_size &&
                 air_nir_components(dt) <= NIR_MAX_VEC_COMPONENTS) {
         /* the same bits as another vector (<2 x i64> as <4 x i32>, a float4 as 16 bytes, an i32 as <4 x i8>...):
          * the low element first, as LLVM on a little-endian target -- mediaserverd's video kernels (VTMTS,
          * pyr_downscale, hbf) */
         res = nir_bitcast_vector(&c->b, x, bits);
      } else {
         return air_error(c, "instruction: a bitcast that changes the size of its value (%u x %u bits to %u x %u)",
                          x->num_components, x->bit_size, air_nir_components(dt), bits);
      }
      break;
   default:
      return air_error(c, "instruction: cast %u", in->sub);
   }
   air_set(c, &in->base, air_val_ssa(c, res));
   return true;
}

static bool
translate_cmp(struct air_ctx *c, const struct air_inst *in)
{
   nir_def *x = air_get_ssa(c, in->operands[0]);
   nir_def *y = air_get_ssa(c, in->operands[1]);
   if (!x || !y)
      return false;
   nir_def *res;
   switch (in->sub) {
   /* NaN never appears in these shaders (nnan is set on every one), so the
    * ordered and unordered forms of a comparison are the same comparison;
    * what differs is only the answer when an operand is NaN.
    */
   case AIR_FCMP_OEQ: case AIR_FCMP_UEQ: res = nir_feq(&c->b, x, y); break;
   case AIR_FCMP_ONE: case AIR_FCMP_UNE: res = nir_fneu(&c->b, x, y); break;
   case AIR_FCMP_OGT: case AIR_FCMP_UGT: res = nir_flt(&c->b, y, x); break;
   case AIR_FCMP_OGE: case AIR_FCMP_UGE: res = nir_fge(&c->b, x, y); break;
   case AIR_FCMP_OLT: case AIR_FCMP_ULT: res = nir_flt(&c->b, x, y); break;
   case AIR_FCMP_OLE: case AIR_FCMP_ULE: res = nir_fge(&c->b, y, x); break;
   case AIR_FCMP_ORD: res = nir_iand(&c->b, nir_feq(&c->b, x, x), nir_feq(&c->b, y, y)); break;
   case AIR_FCMP_UNO: res = nir_ior(&c->b, nir_fneu(&c->b, x, x), nir_fneu(&c->b, y, y)); break;
   case AIR_FCMP_FALSE: res = nir_imm_false(&c->b); break;
   case AIR_FCMP_TRUE: res = nir_imm_true(&c->b); break;
   case AIR_ICMP_EQ: res = nir_ieq(&c->b, x, y); break;
   case AIR_ICMP_NE: res = nir_ine(&c->b, x, y); break;
   case AIR_ICMP_UGT: res = nir_ult(&c->b, y, x); break;
   case AIR_ICMP_UGE: res = nir_uge(&c->b, x, y); break;
   case AIR_ICMP_ULT: res = nir_ult(&c->b, x, y); break;
   case AIR_ICMP_ULE: res = nir_uge(&c->b, y, x); break;
   case AIR_ICMP_SGT: res = nir_ilt(&c->b, y, x); break;
   case AIR_ICMP_SGE: res = nir_ige(&c->b, x, y); break;
   case AIR_ICMP_SLT: res = nir_ilt(&c->b, x, y); break;
   case AIR_ICMP_SLE: res = nir_ige(&c->b, y, x); break;
   default:
      return air_error(c, "instruction: comparison %u", in->sub);
   }
   air_set(c, &in->base, air_val_ssa(c, res));
   return true;
}

/* a struct or array at PTR as a value: its members loaded where the layout puts them (a pointer member is refused by
 * air_load_ptr, by name) */
static struct air_val *
load_aggregate(struct air_ctx *c, struct air_ptr ptr, const struct air_type *t)
{
   if (t->kind != AIR_TYPE_STRUCT && t->kind != AIR_TYPE_ARRAY) {
      nir_def *v = air_load_ptr(c, ptr, t, 0);
      return v ? air_val_ssa(c, v) : NULL;
   }
   unsigned n = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   if (n > 4096)
      return air_error(c, "instruction: a load of an aggregate of %u members", n), NULL;
   struct air_val *agg = air_val_agg(c, n);
   uint64_t stride = t->kind == AIR_TYPE_ARRAY ? air_type_alloc_size(c->m, t->elem) : 0;
   for (unsigned i = 0; i < n; i++) {
      const struct air_type *mt = t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem;
      uint64_t off = t->kind == AIR_TYPE_STRUCT ? air_struct_member_offset(c->m, t, i) : stride * i;
      struct air_ptr mp = air_ptr_add(c, ptr, off ? nir_imm_int(&c->b, (int)off) : NULL);
      mp.pointee = mt;
      if (!(agg->elems[i] = load_aggregate(c, mp, mt)))
         return NULL;
   }
   return agg;
}

/* ... and written: VALUE's members stored where the layout puts them (an undefined member is left as it is) */
static bool
store_aggregate(struct air_ctx *c, struct air_ptr ptr, struct air_val *value, const struct air_type *t)
{
   if (!value)
      return true;
   if (value->kind == AIR_VAL_SSA)
      return air_store_ptr(c, ptr, value->def, 0);
   if (value->kind != AIR_VAL_AGG || !t || (t->kind != AIR_TYPE_STRUCT && t->kind != AIR_TYPE_ARRAY))
      return air_error(c, "instruction: a store of an aggregate holding a pointer");
   uint64_t stride = t->kind == AIR_TYPE_ARRAY ? air_type_alloc_size(c->m, t->elem) : 0;
   for (unsigned i = 0; i < value->num_elems; i++) {
      const struct air_type *mt = t->kind == AIR_TYPE_STRUCT ? (i < t->num_members ? t->members[i] : NULL) : t->elem;
      uint64_t off = t->kind == AIR_TYPE_STRUCT ? air_struct_member_offset(c->m, t, i) : stride * i;
      struct air_ptr mp = air_ptr_add(c, ptr, off ? nir_imm_int(&c->b, (int)off) : NULL);
      mp.pointee = mt;
      if (!mt || !store_aggregate(c, mp, value->elems[i], mt))
         return false;
   }
   return true;
}

/* an aggregate value's field, by the index path of extract/insertvalue */
static struct air_val **
agg_slot(struct air_ctx *c, struct air_val *agg, const uint32_t *idx, unsigned n)
{
   struct air_val **slot = &agg;
   for (unsigned i = 0; i < n; i++) {
      if (!*slot || (*slot)->kind != AIR_VAL_AGG || idx[i] >= (*slot)->num_elems) {
         air_error(c, "instruction: an aggregate index out of its value (index %u of %u: %u into %s of %u)", i, n, idx[i],
                   !*slot ? "nothing" : (*slot)->kind == AIR_VAL_AGG ? "an aggregate" : (*slot)->kind == AIR_VAL_SSA ?
                   "a scalar or vector" : (*slot)->kind == AIR_VAL_PTR ? "a pointer" : "no value",
                   *slot && (*slot)->kind == AIR_VAL_AGG ? (*slot)->num_elems : *slot && (*slot)->kind == AIR_VAL_SSA &&
                   (*slot)->def ? (*slot)->def->num_components : 0);
         return NULL;
      }
      slot = &(*slot)->elems[idx[i]];
   }
   return slot;
}

static struct air_val *
clone_val(struct air_ctx *c, const struct air_val *v)
{
   if (!v)
      return NULL;
   struct air_val *n = rzalloc(c->mem_ctx, struct air_val);
   *n = *v;
   if (v->kind == AIR_VAL_AGG) {
      n->elems = rzalloc_array(c->mem_ctx, struct air_val *, MAX2(v->num_elems, 1));
      for (unsigned i = 0; i < v->num_elems; i++)
         n->elems[i] = clone_val(c, v->elems[i]);
   }
   return n;
}

/* What a value turned out to be, for a message that has to be read without the
 * module in hand. A refusal that only says "not constant" sends the reader to
 * llvm-dis to find out which of six things it was, and the three kernels that
 * stopped on a shuffle mask cost exactly that detour.
 */
static const char *
air_value_kind_name(const struct air_value *v)
{
   if (!v)
      return "nothing";
   const struct air_constant *c = air_value_as_constant(v);
   if (!c) {
      switch (v->kind) {
      case AIR_VALUE_GLOBAL:      return "a global";
      case AIR_VALUE_FUNCTION:    return "a function";
      case AIR_VALUE_ARGUMENT:    return "a function argument";
      case AIR_VALUE_INSTRUCTION: return "an instruction";
      case AIR_VALUE_METADATA:    return "metadata";
      default:                    return "a value of an unknown kind";
      }
   }
   switch (c->ckind) {
   case AIR_CONST_INT:       return "an integer";
   case AIR_CONST_FLOAT:     return "a float";
   case AIR_CONST_NULL:      return "zeroinitializer";
   case AIR_CONST_UNDEF:     return "undef";
   case AIR_CONST_POISON:    return "poison";
   case AIR_CONST_AGGREGATE: return "an aggregate of constants";
   case AIR_CONST_DATA:      return "a vector of plain integers";
   case AIR_CONST_EXPR:      return "a constant expression";
   }
   return "a constant of an unknown kind";
}

static bool
translate_instruction(struct air_ctx *c, const struct air_inst *in)
{
   switch (in->op) {
   case AIR_OP_BINOP:
      return translate_binop(c, in);
   case AIR_OP_UNOP: {
      nir_def *x = air_get_ssa(c, in->operands[0]);
      if (!x)
         return false;
      air_set(c, &in->base, air_val_ssa(c, nir_fneg(&c->b, x)));
      return true;
   }
   case AIR_OP_CAST:
      return translate_cast(c, in);
   case AIR_OP_CMP:
      return translate_cmp(c, in);
   case AIR_OP_GEP:
      return translate_gep(c, in);
   case AIR_OP_SELECT: {
      /* a select between two POINTERS (a module constant when the pipeline's
       * function constant says no buffer is bound): answered when its
       * condition is known, which a function constant's is; kept as a choice
       * at run time it is not, yet
       */
      struct air_val *pa = air_get(c, in->operands[1]);
      struct air_val *pb = air_get(c, in->operands[2]);
      if (!pa || !pb)
         return false;
      if (pa->kind == AIR_VAL_PTR || pb->kind == AIR_VAL_PTR) {
         nir_def *pcond = air_get_ssa(c, in->operands[0]);
         if (!pcond)
            return false;
         if (pcond->num_components == 1 && !nir_def_is_const(pcond) && pa->kind == AIR_VAL_PTR && pb->kind == AIR_VAL_PTR &&
             pa->ptr.class == pb->ptr.class && pa->ptr.is_image == pb->ptr.is_image &&
             ((pa->ptr.binding == pb->ptr.binding && !pa->ptr.binding_def && !pb->ptr.binding_def) ||
              ((pa->ptr.class == AIR_PTR_SSBO || pa->ptr.class == AIR_PTR_UBO) && !pa->ptr.is_image))) {
            /* two places in ONE buffer, chosen while the shader runs: the offset is what is chosen (VectorKit's
             * PolygonStroke: one of two members of its constants) */
            struct air_ptr p = pa->ptr;
            p.offset = nir_bcsel(&c->b, pcond, pa->ptr.offset ? pa->ptr.offset : nir_imm_int(&c->b, 0),
                                 pb->ptr.offset ? pb->ptr.offset : nir_imm_int(&c->b, 0));
            if (pa->ptr.binding != pb->ptr.binding || pa->ptr.binding_def || pb->ptr.binding_def)
               p.binding_def = nir_bcsel(&c->b, pcond, ptr_binding(c, pa->ptr), ptr_binding(c, pb->ptr));   /* two buffers */
            air_set(c, &in->base, air_val_ptr(c, p));
            return true;
         }
         if (pcond->num_components != 1 || !nir_def_is_const(pcond))
            return air_error(c, "instruction: a select between pointers into different buffers decided while the shader runs");
         air_set(c, &in->base, nir_const_value_as_bool(nir_def_as_load_const(pcond)->value[0], pcond->bit_size) ? pa : pb);
         return true;
      }
      nir_def *cond = air_get_ssa(c, in->operands[0]);
      nir_def *x = air_get_ssa(c, in->operands[1]);
      nir_def *y = air_get_ssa(c, in->operands[2]);
      if (!cond || !x || !y)
         return false;
      if (cond->num_components > 1 && cond->num_components != x->num_components)
         return air_error(c, "instruction: a select whose condition does not match its values");
      air_set(c, &in->base, air_val_ssa(c, nir_bcsel(&c->b, cond, x, y)));
      return true;
   }
   case AIR_OP_EXTRACTELT: {
      nir_def *v = air_get_ssa(c, in->operands[0]);
      nir_def *i = air_get_ssa(c, in->operands[1]);
      if (!v || !i)
         return false;
      air_set(c, &in->base, air_val_ssa(c, nir_vector_extract(&c->b, v, nir_u2u32(&c->b, i))));
      return true;
   }
   case AIR_OP_INSERTELT: {
      nir_def *v = air_get_ssa(c, in->operands[0]);
      nir_def *x = air_get_ssa(c, in->operands[1]);
      nir_def *i = air_get_ssa(c, in->operands[2]);
      if (!v || !x || !i)
         return false;
      /* (from upstream's tools/mesa/air) an index decided while the shader runs: nir_vector_insert builds
       * bcsel(i == {0,1,2,3}, scalar, vec), and takes the constant path for a constant index. An index past the
       * vector leaves it unchanged, a refinement of LLVM's poison. fixed_frag_lph_cpf reaches it. */
      air_set(c, &in->base, air_val_ssa(c, nir_vector_insert(&c->b, v, x, nir_u2u32(&c->b, i))));
      return true;
   }
   case AIR_OP_SHUFFLEVEC: {
      nir_def *x = air_get_ssa(c, in->operands[0]);
      nir_def *y = air_get_ssa(c, in->operands[1]);
      struct air_val *mv = air_get(c, in->operands[2]);
      if (!x || !y || !mv)
         return false;
      unsigned n = air_nir_components(in->base.type);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      if (n > NIR_MAX_VEC_COMPONENTS)
         return air_error(c, "instruction: a shuffle of %u components", n);
      /* the mask is a constant vector, or zeroinitializer, or undef */
      const struct air_constant *mk = air_value_as_constant(in->operands[2]);
      for (unsigned i = 0; i < n; i++) {
         int64_t sel = 0;
         bool undef = false;
         if (mk && mk->ckind == AIR_CONST_DATA && i < mk->num_elems) {
            sel = (int32_t)mk->data[i];
         } else if (mk && mk->ckind == AIR_CONST_AGGREGATE && i < mk->num_elems) {
            const struct air_constant *e = air_value_as_constant(mk->elems[i]);
            /* The mask is an aggregate rather than plain data whenever a lane
             * is poison, so <i32 0, i32 poison, ...> -- what a Metal
             * half2-to-half4 widening compiles to -- mixes LLVM's two
             * spellings of a constant integer. air_const_uint reads both; this
             * code used to test ckind == AIR_CONST_INT by hand instead, which
             * is the whole bug.
             */
            uint64_t lane = 0;
            bool have = air_const_uint(mk->elems[i], &lane);
            if (!e || (!have && e->ckind != AIR_CONST_UNDEF &&
                       e->ckind != AIR_CONST_POISON))
               return air_error(c, "instruction: element %u of a shuffle mask is %s",
                                i, air_value_kind_name(mk->elems[i]));
            undef = !have;
            sel = have ? (int32_t)lane : 0;
         } else if (mk && (mk->ckind == AIR_CONST_NULL)) {
            sel = 0;
         } else if (mk && (mk->ckind == AIR_CONST_UNDEF || mk->ckind == AIR_CONST_POISON)) {
            undef = true;
         } else {
            return air_error(c, "instruction: a shuffle mask that is %s, with %u of the "
                                "%u elements it needs",
                             air_value_kind_name(in->operands[2]),
                             mk ? mk->num_elems : 0, n);
         }
         if (undef) {
            comps[i] = nir_undef(&c->b, 1, x->bit_size);
         } else if (sel < x->num_components) {
            comps[i] = nir_channel(&c->b, x, sel);
         } else if (sel - x->num_components < y->num_components) {
            comps[i] = nir_channel(&c->b, y, sel - x->num_components);
         } else {
            return air_error(c, "instruction: a shuffle index past its inputs");
         }
      }
      air_set(c, &in->base, air_val_ssa(c, nir_vec(&c->b, comps, n)));
      return true;
   }
   case AIR_OP_EXTRACTVAL: {
      struct air_val *agg = air_get(c, in->operands[0]);
      if (!agg)
         return false;
      /* (11-12) a field INSIDE a member that is itself undefined (an undef struct's member's member: the widgets'
       * uber_fragment): undefined too -- agg_slot refused the second index "into nothing" */
      struct air_val *walk = agg;
      bool undefined = false;
      for (unsigned i = 0; i < in->num_indices && !undefined; i++) {
         if (!walk)
            undefined = true;
         else if (walk->kind == AIR_VAL_SSA && walk->def && nir_def_instr(walk->def)->type == nir_instr_type_undef)
            undefined = true;        /* (an undef aggregate that came as one undefined value: a callee's argument) */
         else if (walk->kind != AIR_VAL_AGG || in->indices[i] >= walk->num_elems)
            break;
         else
            walk = walk->elems[in->indices[i]];
      }
      struct air_val **slot = undefined ? NULL : agg_slot(c, agg, in->indices, in->num_indices);
      if (!slot && !undefined)
         return false;
      if (undefined || !*slot) {
         /* a field of an undef aggregate */
         const struct air_type *rt = in->base.type;
         if (rt->kind == AIR_TYPE_STRUCT || rt->kind == AIR_TYPE_ARRAY) {
            air_set(c, &in->base, air_val_agg(c, rt->kind == AIR_TYPE_STRUCT ? rt->num_members : rt->count));
            return true;
         }
         air_set(c, &in->base,
                 air_val_ssa(c, nir_undef(&c->b, air_nir_components(in->base.type),
                                          air_nir_bit_size(c, in->base.type))));
         return true;
      }
      air_set(c, &in->base, *slot);
      return true;
   }
   case AIR_OP_INSERTVAL: {
      struct air_val *agg = air_get(c, in->operands[0]);
      struct air_val *val = air_get(c, in->operands[1]);
      if (!agg || !val)
         return false;
      struct air_val *copy = clone_val(c, agg);
      {
         /* (11-12) a member on the way that is still undefined (insertvalue into an undef struct's member struct:
          * the widgets' uber_fragment, "index 1 of 2: 0 into nothing") is made, empty, as its type says */
         const struct air_type *t = in->operands[0]->type;
         struct air_val **at = &copy;

         for (unsigned i = 0; i < in->num_indices && t; i++) {
            if (!*at && (t->kind == AIR_TYPE_STRUCT || t->kind == AIR_TYPE_ARRAY))
               *at = air_val_agg(c, t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count);
            if (!*at || (*at)->kind != AIR_VAL_AGG || in->indices[i] >= (*at)->num_elems)
               break;
            at = &(*at)->elems[in->indices[i]];
            t = t->kind == AIR_TYPE_STRUCT ? (in->indices[i] < t->num_members ? t->members[in->indices[i]] : NULL) :
                t->kind == AIR_TYPE_ARRAY ? t->elem : NULL;
         }
      }
      struct air_val **slot = agg_slot(c, copy, in->indices, in->num_indices);
      if (!slot)
         return false;
      *slot = val;
      air_set(c, &in->base, copy);
      return true;
   }
   case AIR_OP_FREEZE:
      air_set(c, &in->base, air_get(c, in->operands[0]));
      return c->error[0] == 0;
   case AIR_OP_ALLOCA: {
      uint64_t count;
      if (!air_const_uint(in->operands[0], &count))
         return air_error(c, "instruction: an alloca of a run-time size");
      uint64_t size = air_type_alloc_size(c->m, in->aux_type) * count;
      unsigned align = MAX2(in->align, air_type_abi_align(c->m, in->aux_type));
      c->scratch_size = ALIGN_POT(c->scratch_size, MAX2(align, 1));
      struct air_ptr p = {
         .class = AIR_PTR_SCRATCH,
         .offset = nir_imm_int(&c->b, c->scratch_size),
         .pointee = in->aux_type,
      };
      c->scratch_size += size;
      air_set(c, &in->base, air_val_ptr(c, p));
      return true;
   }
   case AIR_OP_LOAD: {
      /* a program-scope variable whose one static-initialiser store was a
       * constant: the load is that constant (plan_global_folding)
       */
      struct air_global_plan *gpl = foldable_global_plan(c, in->operands[0]);
      if (gpl && gpl->known) {
         air_set(c, &in->base, air_val_ssa(c, nir_imm_intN_t(&c->b, gpl->bits, gpl->bit_size)));
         return true;
      }
      struct air_val *p = air_get(c, in->operands[0]);
      if (!p)
         return false;
      if (p->kind != AIR_VAL_PTR)
         return air_error(c, "instruction: a load from something that is not a pointer");
      if (in->base.type && in->base.type->kind == AIR_TYPE_POINTER) {
         /* a pointer read back from a local it was stored into */
         uint64_t off;
         if (scratch_const_offset(&p->ptr, &off)) {
            for (unsigned k = 0; k < c->num_scratch_ptrs; k++) {
               if (c->scratch_ptrs[k].offset == off) {
                  air_set(c, &in->base, c->scratch_ptrs[k].val);
                  return true;
               }
            }
            return air_error(c, "instruction: a pointer loaded from a local nothing stored one into");
         }
      }
      {
         uint64_t loff;
         nir_def *known;
         if (scratch_const_offset(&p->ptr, &loff) && (known = known_mem_load(c, loff, in->base.type))) {
            air_set(c, &in->base, air_val_ssa(c, known));
            return true;
         }
      }
      if (in->base.type && (in->base.type->kind == AIR_TYPE_STRUCT || in->base.type->kind == AIR_TYPE_ARRAY)) {
         /* A WHOLE STRUCT OR ARRAY LOADED (11-12: the widgets' uber_fragment hands one to a function, which takes its
          * members with extractvalue -- "an aggregate index ... into a scalar"): member by member, where each lies */
         struct air_val *agg = load_aggregate(c, p->ptr, in->base.type);
         if (!agg)
            return false;
         air_set(c, &in->base, agg);
         return true;
      }
      nir_def *res = air_load_ptr(c, p->ptr, in->base.type, in->align);
      if (!res)
         return false;
      air_set(c, &in->base, air_val_ssa(c, res));
      return true;
   }
   case AIR_OP_STORE: {
      struct air_val *p = air_get(c, in->operands[1]);
      struct air_val *sv = air_get(c, in->operands[0]);
      if (!p || !sv)
         return false;
      if (sv->kind == AIR_VAL_PTR) {
         /* a pointer into a local: kept as the value it is */
         uint64_t off;
         if (p->kind != AIR_VAL_PTR || !scratch_const_offset(&p->ptr, &off))
            return air_error(c, "instruction: a store of a pointer into memory that is not a local");
         unsigned k = 0;
         while (k < c->num_scratch_ptrs && c->scratch_ptrs[k].offset != off)
            k++;
         if (k == ARRAY_SIZE(c->scratch_ptrs))
            return air_error(c, "instruction: more than %u pointers stored into locals",
                             (unsigned)ARRAY_SIZE(c->scratch_ptrs));
         if (k == c->num_scratch_ptrs)
            c->num_scratch_ptrs++;
         c->scratch_ptrs[k].offset = off;
         c->scratch_ptrs[k].val = sv;
         return true;
      }
      if (sv->kind == AIR_VAL_AGG && p->kind == AIR_VAL_PTR)
         return store_aggregate(c, p->ptr, sv, in->operands[0]->type);   /* a whole struct or array, member by member */
      nir_def *v = air_get_ssa(c, in->operands[0]);
      if (!v)
         return false;
      if (p->kind != AIR_VAL_PTR)
         return air_error(c, "instruction: a store to something that is not a pointer");
      /* the one store into a foldable variable: its value kept when it is a
       * constant; the memory is written either way (dead when the fold answers)
       */
      uint64_t soff;
      if (c->in_static_init && scratch_const_offset(&p->ptr, &soff) && nir_def_is_const(v) &&
          v->num_components <= 4 && v->bit_size >= 8)
         known_mem_record(c, soff, v);
      struct air_global_plan *gpl = foldable_global_plan(c, in->operands[1]);
      if (gpl && v->num_components == 1 && nir_def_is_const(v)) {
         gpl->known = true;
         gpl->bits = nir_const_value_as_uint(nir_def_as_load_const(v)->value[0], v->bit_size);
         gpl->bit_size = v->bit_size;
      }
      return air_store_ptr(c, p->ptr, v, in->align);
   }
   case AIR_OP_CALL: {
      const struct air_value *callee = in->operands[0];
      if (callee->kind != AIR_VALUE_FUNCTION || !callee->name)
         return air_error(c, "instruction: a call through a pointer");
      return air_translate_call(c, in, callee->name);
   }
   case AIR_OP_RET:
   case AIR_OP_BR:
   case AIR_OP_SWITCH:
   case AIR_OP_UNREACHABLE:
   case AIR_OP_PHI:
      return true; /* handled by the CFG walk */
   default:
      return air_error(c, "instruction: opcode %u", in->op);
   }
}

/* -- the CFG ------------------------------------------------------------ */

static nir_block *
nir_block_of(struct air_ctx *c, const struct air_block *b)
{
   struct hash_entry *e = _mesa_hash_table_search(c->blocks, b);
   if (e)
      return e->data;
   nir_block *n = AIR_BLOCK_CREATE(c->nir, c->b.impl);
   exec_list_push_tail(&c->b.impl->body, &n->cf_node.node);
   n->cf_node.parent = &c->b.impl->cf_node;
   _mesa_hash_table_insert(c->blocks, b, n);
   return n;
}

static void aggregate_store_return(struct air_ctx *c, const struct air_val *value, const struct air_type *t, unsigned *k);

static bool
emit_return(struct air_ctx *c, const struct air_inst *in)
{
   if (in->num_operands == 0)
      return true;
   struct air_val *v = air_get(c, in->operands[0]);
   if (!v)
      return false;

   /* Inside a callee the return value goes through the out-parameter the
    * caller passed, because a NIR call has no destination. The entry point
    * has no such parameter and its return is its render targets, below.
    */
   if (c->ret_derefs) {
      /* a struct or array returned from a function called in the module: member by member (WebKit's path_cubic) */
      unsigned k = 0;
      aggregate_store_return(c, v, c->ret_type, &k);
      return true;
   }
   if (c->ret_deref) {
      nir_def *d = air_get_ssa(c, in->operands[0]);
      if (!d)
         return air_error(c, "instruction: a return of a value this cannot put in a "
                             "register, from a function called in the module");
      nir_store_deref(&c->b, c->ret_deref, d, nir_component_mask(d->num_components));
      return true;
   }

   const struct air_type *rt = in->operands[0]->type;
   if (rt->kind == AIR_TYPE_STRUCT) {
      if (v->kind != AIR_VAL_AGG || v->num_elems != rt->num_members)
         return air_error(c, "instruction: a returned struct that was never built");
      for (unsigned i = 0; i < v->num_elems; i++) {
         if (!c->outputs[i].var)
            return air_error(c, "interface: field %u of the result has no output", i);
         if (!v->elems[i] || v->elems[i]->kind != AIR_VAL_SSA)
            continue; /* a field left undef is a target nothing writes */
         nir_store_var(&c->b, c->outputs[i].var, v->elems[i]->def,
                       BITFIELD_MASK(v->elems[i]->def->num_components));
      }
      return true;
   }
   if (v->kind != AIR_VAL_SSA)
      return air_error(c, "instruction: a returned value that is not a number");
   if (!c->num_outputs || !c->outputs[0].var)
      return air_error(c, "interface: a returned value with no output");
   nir_store_var(&c->b, c->outputs[0].var, v->def, BITFIELD_MASK(v->def->num_components));
   return true;
}

/* The blocks in reverse post-order, so that every value is translated before
 * it is used: LLVM lays blocks out in its own order, and a block whose only
 * predecessor comes after it in that order is ordinary (a loop's exit, say).
 * Reverse post-order respects dominance, which is exactly the guarantee the
 * translation needs. Blocks the entry cannot reach are left out, and the phi
 * pass below ignores edges from them.
 */
static unsigned
reverse_post_order(struct air_ctx *c, const struct air_function *f, unsigned *order)
{
   unsigned n = f->num_blocks;
   uint8_t *state = rzalloc_array(c->mem_ctx, uint8_t, MAX2(n, 1)); /* 0 new, 1 open, 2 done */
   unsigned *stack = rzalloc_array(c->mem_ctx, unsigned, MAX2(n, 1));
   unsigned *next_succ = rzalloc_array(c->mem_ctx, unsigned, MAX2(n, 1));
   unsigned sp = 0, num_post = 0;

   stack[sp++] = 0;
   state[0] = 1;
   while (sp) {
      unsigned bi = stack[sp - 1];
      const struct air_block *blk = &f->blocks[bi];
      const struct air_inst *term = blk->insts[blk->num_insts - 1];
      if (next_succ[bi] < term->num_targets) {
         unsigned to = term->targets[next_succ[bi]++]->index;
         if (state[to] == 0) {
            state[to] = 1;
            next_succ[to] = 0;
            stack[sp++] = to;
         }
         continue;
      }
      state[bi] = 2;
      order[num_post++] = bi; /* post-order */
      sp--;
   }
   /* reverse it in place */
   for (unsigned i = 0; i < num_post / 2; i++) {
      unsigned t = order[i];
      order[i] = order[num_post - 1 - i];
      order[num_post - 1 - i] = t;
   }
   return num_post;
}

static bool translate_body(struct air_ctx *c);
static void materialise_globals(struct air_ctx *c);

/* A function of the module, translated as a nir_function so that
 * nir_inline_functions can fold it into its callers afterwards.
 *
 * Translating it in place at the call site was the other option and is worse:
 * the callees here are not straight-line code -- FilmGrainLuma has 17 blocks
 * and 24 branches -- so inlining by hand would mean reproducing a CFG inside
 * another one, with its own block map and phis. NIR already has that pass.
 *
 * The return goes through an out-parameter at index 0, with is_return set,
 * because a NIR call has no destination; this is the convention spirv_to_nir
 * uses (vtn_cfg.c) and the one nir_inline_functions expects.
 *
 * The per-function tables are swapped rather than shared: values, blocks and
 * phis are all keyed on this function's own objects, and leaving the caller's
 * entries in place would let a callee resolve a value it cannot see.
 */
/* (from upstream's tools/mesa/air, ptr_offset_to_pass) a pointer argument whose byte offset is computed at run time:
 * the pointer is bound from the caller's resolution as any, its offset passed as a parameter of its own, after the
 * ordinary ones -- the callee reads it with load_param, so no def of the caller's is used in the callee */
static bool
offset_passed(const struct air_val *v)
{
   return v && v->kind == AIR_VAL_PTR && v->ptr.offset && !nir_src_is_const(nir_src_for_ssa(v->ptr.offset));
}

/* AN ARGUMENT THAT IS A WHOLE STRUCT OR ARRAY (11-12: the widgets' uber_fragment passes one by value; the callee's
 * first instruction is an extractvalue of it): each scalar or vector in it is a NIR parameter of its own, in the
 * order of the type -- the callee puts them together again as the aggregate. */
static bool
type_is_aggregate(const struct air_type *t)
{
   return t && (t->kind == AIR_TYPE_STRUCT || t->kind == AIR_TYPE_ARRAY);
}

static unsigned
aggregate_leaves(const struct air_type *t)
{
   if (!type_is_aggregate(t))
      return 1;
   unsigned n = 0, members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   for (unsigned i = 0; i < members && n < 4096; i++)
      n += aggregate_leaves(t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem);
   return n;
}

/* the scalars and vectors of T, in order, into OUT (at most MOST) */
static void
aggregate_leaf_types(const struct air_type *t, const struct air_type **out, unsigned *n, unsigned most)
{
   if (!type_is_aggregate(t)) {
      if (*n < most)
         out[*n] = t;
      (*n)++;
      return;
   }
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   for (unsigned i = 0; i < members; i++)
      aggregate_leaf_types(t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, out, n, most);
}

/* the variable type a value of T is returned through */
static const struct glsl_type *
return_glsl_type(struct air_ctx *c, const struct air_type *t)
{
   unsigned bits = t->kind == AIR_TYPE_POINTER ? 32 : air_nir_bit_size(c, t);
   return glsl_vector_type(bits == 1 ? GLSL_TYPE_BOOL : bits == 8 ? GLSL_TYPE_UINT8 : bits == 16 ? GLSL_TYPE_FLOAT16 :
                           bits == 64 ? GLSL_TYPE_UINT64 : GLSL_TYPE_UINT, air_nir_components(t));
}

#define AIR_MAX_RETURN_LEAVES 64

/* a returned aggregate's leaves stored through the callee's out-parameters (emit_return); *K counts them */
static void
aggregate_store_return(struct air_ctx *c, const struct air_val *value, const struct air_type *t, unsigned *k)
{
   if (!type_is_aggregate(t)) {
      if (*k < c->num_ret_derefs && value && value->kind == AIR_VAL_SSA && value->def &&
          value->def->num_components == air_nir_components(t))
         nir_store_deref(&c->b, c->ret_derefs[*k], value->def, nir_component_mask(value->def->num_components));
      (*k)++;                           /* (an undefined member: its variable is left as it is) */
      return;
   }
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   for (unsigned i = 0; i < members; i++)
      aggregate_store_return(c, value && value->kind == AIR_VAL_AGG && i < value->num_elems ? value->elems[i] : NULL,
                             t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, k);
}

/* the caller's side: the aggregate put together from the variables the callee stored into */
static struct air_val *
aggregate_load_return(struct air_ctx *c, const struct air_type *t, nir_variable **vars, unsigned *k)
{
   if (!type_is_aggregate(t))
      return air_val_ssa(c, nir_load_var(&c->b, vars[(*k)++]));
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   struct air_val *agg = air_val_agg(c, members);
   for (unsigned i = 0; i < members; i++)
      agg->elems[i] = aggregate_load_return(c, t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, vars, k);
   return agg;
}

static void
aggregate_declare_params(struct air_ctx *c, const struct air_type *t, nir_function *fn, unsigned *idx)
{
   if (!type_is_aggregate(t)) {
      fn->params[(*idx)++] = (nir_parameter){
         .num_components = (uint8_t)air_nir_components(t),
         .bit_size = (uint8_t)(t->kind == AIR_TYPE_POINTER ? 32 : air_nir_bit_size(c, t)),
      };
      return;
   }
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   for (unsigned i = 0; i < members; i++)
      aggregate_declare_params(c, t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, fn, idx);
}

static struct air_val *
aggregate_from_params(struct air_ctx *c, const struct air_type *t, unsigned *pidx)
{
   if (!type_is_aggregate(t))
      return air_val_ssa(c, nir_load_param(&c->b, (*pidx)++));
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   struct air_val *agg = air_val_agg(c, members);
   for (unsigned i = 0; i < members; i++)
      agg->elems[i] = aggregate_from_params(c, t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, pidx);
   return agg;
}

static bool
aggregate_holds_pointer(const struct air_val *value)
{
   if (!value)
      return false;
   if (value->kind == AIR_VAL_PTR)
      return true;
   if (value->kind == AIR_VAL_AGG)
      for (unsigned i = 0; i < value->num_elems; i++)
         if (aggregate_holds_pointer(value->elems[i]))
            return true;
   return false;
}

/* the caller's side: VALUE's leaves as the call's sources (one that is undefined, or not a plain value: undefined) */
static void
aggregate_pass_params(struct air_ctx *c, const struct air_val *value, const struct air_type *t, nir_call_instr *call,
                      unsigned *idx)
{
   if (!type_is_aggregate(t)) {
      unsigned comps = air_nir_components(t), bits = t->kind == AIR_TYPE_POINTER ? 32 : air_nir_bit_size(c, t);
      nir_def *d = value && value->kind == AIR_VAL_SSA && value->def && value->def->num_components == comps &&
                   value->def->bit_size == bits ? value->def : nir_undef(&c->b, comps, bits);
      call->params[(*idx)++] = nir_src_for_ssa(d);
      return;
   }
   unsigned members = t->kind == AIR_TYPE_STRUCT ? t->num_members : (unsigned)t->count;
   for (unsigned i = 0; i < members; i++)
      aggregate_pass_params(c, value && value->kind == AIR_VAL_AGG && i < value->num_elems ? value->elems[i] : NULL,
                            t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem, call, idx);
}

static nir_function *
translate_callee(struct air_ctx *c, const struct air_function *f,
                 struct air_val **argv, bool *is_param)
{
   /* A texture, sampler or buffer argument is not a value in this frontend: it
    * is resolved at translation time into a binding, and tex_handle wants an
    * AIR_VAL_PTR rather than an SSA def. Such an argument is bound from the
    * caller's own resolution and takes no NIR parameter, which specialises the
    * callee to its call site -- so it is not cached. Each of the four modules
    * that needs this has exactly one call site, counted beforehand.
    *
    * A pointer carrying a byte offset is refused instead: that offset is an
    * nir_def of the CALLER's impl, and using it here would be the same
    * cross-function def/use that the globals used to cause.
    */
   bool specialised = false;
   for (unsigned i = 0; i < f->num_args; i++) {
      bool ptr = argv && argv[i] && argv[i]->kind == AIR_VAL_PTR;
      /* (from upstream's tools/mesa/air, rebind_here) a CONSTANT byte offset is rebuilt in the callee's own builder
       * when the argument is bound below; one computed at run time is passed as a parameter (offset_passed) */
      is_param[i] = !ptr;
      specialised |= ptr;
   }

   struct hash_entry *e = specialised ? NULL : _mesa_hash_table_search(c->fns, f);
   if (e)
      return e->data;

   if (f->is_declaration || !f->num_blocks) {
      air_error(c, "intrinsic: a call of %s, which the module only declares",
                f->base.name ? f->base.name : "a function");
      return NULL;
   }

   const struct air_type *rt = f->fn_type->elem;   /* FUNCTION: the return type */
   bool has_ret = rt && rt->kind != AIR_TYPE_VOID;
   unsigned nparams = 2 * f->num_args + (has_ret ? 1 : 0);   /* (at most: an offset parameter for each argument) */
   const bool agg_ret = has_ret && type_is_aggregate(rt);
   const struct air_type *ret_leaf[AIR_MAX_RETURN_LEAVES];
   unsigned ret_leaves = 0;
   if (agg_ret) {
      aggregate_leaf_types(rt, ret_leaf, &ret_leaves, AIR_MAX_RETURN_LEAVES);
      if (ret_leaves == 0 || ret_leaves > AIR_MAX_RETURN_LEAVES) {
         air_error(c, "intrinsic: %s returns an aggregate of %u members", f->base.name ? f->base.name : "a function",
                   ret_leaves);
         return NULL;
      }
      nparams += ret_leaves;
   }
   for (unsigned i = 0; i < f->num_args; i++)
      if (type_is_aggregate(f->args[i].type))
         nparams += aggregate_leaves(f->args[i].type);         /* (an aggregate's members, a parameter each) */
   nir_function *fn = nir_function_create(c->nir, f->base.name ? f->base.name : "callee");
   fn->num_params = nparams;
   fn->params = rzalloc_array(c->nir, nir_parameter, MAX2(nparams, 1));
   fn->should_inline = true;

   unsigned idx = 0;
   const struct glsl_type *ret_glsl = NULL;
   if (agg_ret) {
      for (unsigned k = 0; k < ret_leaves; k++)
         fn->params[idx++] = (nir_parameter){
            .num_components = 1, .bit_size = 32,
            .is_return = true,
            .mode = nir_var_function_temp,
            .type = return_glsl_type(c, ret_leaf[k]),
         };
   } else if (has_ret) {
      unsigned bits = air_nir_bit_size(c, rt);
      unsigned comps = air_nir_components(rt);
      ret_glsl = glsl_vector_type(bits == 16 ? GLSL_TYPE_FLOAT16 :
                                  bits == 64 ? GLSL_TYPE_UINT64 : GLSL_TYPE_UINT, comps);
      fn->params[idx++] = (nir_parameter){
         .num_components = 1, .bit_size = 32,
         .is_return = true,
         .mode = nir_var_function_temp,
         .type = ret_glsl,
      };
   }
   for (unsigned i = 0; i < f->num_args; i++) {
      if (!is_param[i])
         continue;
      const struct air_type *at = f->args[i].type;
      if (type_is_aggregate(at)) {
         aggregate_declare_params(c, at, fn, &idx);
         continue;
      }
      fn->params[idx++] = (nir_parameter){
         .num_components = (uint8_t)air_nir_components(at),
         .bit_size = (uint8_t)air_nir_bit_size(c, at),
      };
   }
   for (unsigned i = 0; i < f->num_args; i++) {
      if (!is_param[i] && offset_passed(argv[i])) {
         fn->params[idx++] = (nir_parameter){
            .num_components = 1,
            .bit_size = (uint8_t)argv[i]->ptr.offset->bit_size,
         };
      }
   }
   fn->num_params = idx;   /* the pointers took no parameter (their run-time offsets one each) */

   /* cached before the body is translated, so that a callee which calls itself
    * meets its own declaration rather than being translated for ever */
   if (!specialised)
      _mesa_hash_table_insert(c->fns, f, fn);
   else
      _mesa_hash_table_insert(c->fns, fn, fn);

   nir_builder saved_b = c->b;
   const struct air_function *saved_fn = c->fn;
   struct hash_table *sv = c->values, *sb = c->blocks,
                     *sp = c->phi_cursors, *sr = c->phi_regs, *sbr = c->phi_bind_regs;
   nir_deref_instr *saved_ret = c->ret_deref;
   nir_deref_instr **saved_rets = c->ret_derefs;
   unsigned saved_num_rets = c->num_ret_derefs;
   const struct air_type *saved_ret_type = c->ret_type;

   c->fn = f;
   c->values = _mesa_pointer_hash_table_create(c->mem_ctx);
   c->blocks = _mesa_pointer_hash_table_create(c->mem_ctx);
   c->phi_cursors = _mesa_pointer_hash_table_create(c->mem_ctx);
   c->phi_regs = _mesa_pointer_hash_table_create(c->mem_ctx);
   c->phi_bind_regs = _mesa_pointer_hash_table_create(c->mem_ctx);
   c->b = nir_builder_at(nir_before_impl(nir_function_impl_create(fn)));
   c->b.constant_fold_alu = true;   /* (see main's builder) */
   /* as for the entry point: this translator emits an unstructured CFG with
    * nir_goto, and nir_goto_if asserts on a structured impl. A new impl is
    * structured by default, so a callee crashed on its first branch -- which
    * is what the corpus's reader count caught, 456 falling to 454. */
   c->b.impl->structured = false;
   c->ret_deref = NULL;
   c->ret_derefs = NULL;
   c->num_ret_derefs = 0;
   c->ret_type = NULL;

   /* this function's own globals, built in its own builder */
   materialise_globals(c);

   bool ok = true;
   if (agg_ret) {
      c->ret_derefs = rzalloc_array(c->mem_ctx, nir_deref_instr *, ret_leaves);
      c->num_ret_derefs = ret_leaves;
      c->ret_type = rt;
      for (unsigned k = 0; k < ret_leaves; k++)
         c->ret_derefs[k] = nir_build_deref_cast(&c->b, nir_load_param(&c->b, k), nir_var_function_temp,
                                                 return_glsl_type(c, ret_leaf[k]), 0);
      c->ret_deref = c->ret_derefs[0];
   } else if (has_ret) {
      /* The caller passes the deref's own def and the callee casts it back,
       * which is the convention nir_functions.c documents: load_param is
       * replaced by the call's source at inline time and nir_opt_deref removes
       * the cast afterwards.
       */
      c->ret_deref = nir_build_deref_cast(&c->b, nir_load_param(&c->b, 0),
                                          nir_var_function_temp, ret_glsl, 0);
   }
   unsigned pidx = agg_ret ? ret_leaves : has_ret ? 1 : 0;
   for (unsigned i = 0; i < f->num_args && ok; i++) {
      if (!is_param[i]) {
         struct air_val *v = argv[i];
         if (offset_passed(v))
            continue;                   /* bound after the ordinary parameters, below */
         if (v->kind == AIR_VAL_PTR && v->ptr.offset) {
            /* the constant offset as a def of this function's impl (the caller's is not usable here) */
            struct air_ptr p = v->ptr;
            p.offset = nir_imm_int(&c->b, (int)nir_src_as_uint(nir_src_for_ssa(v->ptr.offset)));
            v = air_val_ptr(c, p);
         }
         air_set(c, &f->args[i], v);   /* the caller's own resolution */
         continue;
      }
      if (type_is_aggregate(f->args[i].type)) {
         air_set(c, &f->args[i], aggregate_from_params(c, f->args[i].type, &pidx));
         continue;
      }
      air_set(c, &f->args[i], air_val_ssa(c, nir_load_param(&c->b, pidx++)));
   }
   for (unsigned i = 0; i < f->num_args && ok; i++) {
      if (!is_param[i] && offset_passed(argv[i])) {
         struct air_ptr p = argv[i]->ptr;
         p.offset = nir_load_param(&c->b, pidx++);
         air_set(c, &f->args[i], air_val_ptr(c, p));
      }
   }
   if (ok)
      ok = translate_body(c);

   c->b = saved_b;
   c->fn = saved_fn;
   c->values = sv;
   c->blocks = sb;
   c->phi_cursors = sp;
   c->phi_regs = sr;
   c->phi_bind_regs = sbr;
   c->ret_deref = saved_ret;
   c->ret_derefs = saved_rets;
   c->num_ret_derefs = saved_num_rets;
   c->ret_type = saved_ret_type;
   return ok ? fn : NULL;
}

/* A call of a function this module defines. The callee is translated as a
 * nir_function (above) and the call becomes a nir_call, which
 * nir_inline_functions folds in after the whole shader is translated.
 *
 * This used to be refused outright, with a comment saying no system shader
 * needed it because Apple's own inliner had already flattened every one. That
 * was not true: six entry points of the corpus call a function in their module
 * and five of them are compute kernels, which is how the claim went unnoticed
 * -- nothing dispatched a kernel until there was something to dispatch it with.
 */
bool
air_call_in_module(struct air_ctx *c, const struct air_inst *in, const char *name)
{
   const struct air_function *f = NULL;
   for (unsigned i = 0; i < c->m->num_functions; i++) {
      const char *fname = c->m->functions[i]->base.name;
      if (fname && !strcmp(fname, name)) {
         f = c->m->functions[i];
         break;
      }
   }
   if (!f)
      return air_error(c, "intrinsic: a call of %s, which is not a function of this "
                          "module", name);

   if (in->num_operands - 1 != f->num_args)
      return air_error(c, "intrinsic: a call of %s with %u arguments, wanting %u", name,
                       in->num_operands - 1, f->num_args);

   /* resolved in the CALLER's value map, which is the live one here */
   struct air_val **argv = rzalloc_array(c->mem_ctx, struct air_val *,
                                         MAX2(f->num_args, 1));
   bool *is_param = rzalloc_array(c->mem_ctx, bool, MAX2(f->num_args, 1));
   for (unsigned i = 0; i < f->num_args; i++) {
      argv[i] = air_get(c, in->operands[i + 1]);
      if (!argv[i])
         return false;
   }

   nir_function *fn = translate_callee(c, f, argv, is_param);
   if (!fn)
      return false;   /* translate_callee has said why */

   const struct air_type *rt = f->fn_type->elem;   /* FUNCTION: the return type */
   bool has_ret = rt && rt->kind != AIR_TYPE_VOID;

   nir_variable *tmp = NULL;
   nir_call_instr *call = nir_call_instr_create(c->nir, fn);
   unsigned idx = 0;
   const bool agg_ret = has_ret && type_is_aggregate(rt);
   nir_variable *ret_vars[AIR_MAX_RETURN_LEAVES];
   if (agg_ret) {
      const struct air_type *leaf[AIR_MAX_RETURN_LEAVES];
      unsigned leaves = 0;
      aggregate_leaf_types(rt, leaf, &leaves, AIR_MAX_RETURN_LEAVES);
      for (unsigned k = 0; k < leaves && k < AIR_MAX_RETURN_LEAVES; k++) {
         ret_vars[k] = nir_local_variable_create(c->b.impl, return_glsl_type(c, leaf[k]), "callret");
         call->params[idx++] = nir_src_for_ssa(&nir_build_deref_var(&c->b, ret_vars[k])->def);
      }
   } else if (has_ret) {
      unsigned bits = air_nir_bit_size(c, rt);
      unsigned comps = air_nir_components(rt);
      const struct glsl_type *t =
         glsl_vector_type(bits == 16 ? GLSL_TYPE_FLOAT16 :
                          bits == 64 ? GLSL_TYPE_UINT64 : GLSL_TYPE_UINT, comps);
      tmp = nir_local_variable_create(c->b.impl, t, "callret");
      call->params[idx++] = nir_src_for_ssa(&nir_build_deref_var(&c->b, tmp)->def);
   }
   /* operand 0 is the callee; the arguments follow it */
   for (unsigned i = 0; i < f->num_args; i++) {
      if (!is_param[i])
         continue;
      if (type_is_aggregate(f->args[i].type)) {
         if (aggregate_holds_pointer(argv[i]))
            return air_error(c, "intrinsic: argument %u of %s is an aggregate holding a pointer (a texture, a "
                                "sampler, a buffer)", i, name);
         aggregate_pass_params(c, argv[i], f->args[i].type, call, &idx);
         continue;
      }
      nir_def *a = air_get_ssa(c, in->operands[i + 1]);
      if (!a)
         return air_error(c, "intrinsic: argument %u of %s is not a value this can pass",
                          i, name);
      call->params[idx++] = nir_src_for_ssa(a);
   }
   for (unsigned i = 0; i < f->num_args; i++) {
      if (!is_param[i] && offset_passed(argv[i]))
         call->params[idx++] = nir_src_for_ssa(argv[i]->ptr.offset);
   }
   nir_builder_instr_insert(&c->b, &call->instr);

   if (agg_ret) {
      unsigned k = 0;
      air_set(c, &in->base, aggregate_load_return(c, rt, ret_vars, &k));
   } else if (has_ret)
      air_set(c, &in->base, air_val_ssa(c, nir_load_var(&c->b, tmp)));
   return true;
}

/* An air.visible library function, inside an entry point we generate for it.
 *
 * A library function is not a shader. Metal links one into a caller at
 * pipeline time through the ABI its own metadata describes -- air.stitching_info
 * names the four functions that do it (load_argument, copy_from_buffer,
 * store_return_value, destroy) and gives each argument's size and alignment --
 * and none of that is implemented here: it belongs to the runtime, not to the
 * compiler. So 135 modules of the corpus, a third of it, had nothing that
 * could be dispatched and were counted only as refused.
 *
 * This wraps one instead: argument i is read from device buffer 0 at byte
 * 16*i, the function is called, and the return value is written to device
 * buffer 1 at offset 0, by a single thread. The 16-byte slot is Apple's own
 * size and alignment for a float4 (the air.vector_type node beside each
 * stitching argument says 16 and 16), and a fixed slot makes the layout
 * derivable from the signature alone -- which is what lets a harness on this
 * machine fill the same bytes and compare the result.
 *
 * What that measures is the body: its arithmetic, its control flow, its calls.
 * What it does not touch is the stitching ABI, so a wrapped function is
 * reported in a row of its own and never counted among the corpus's entry
 * points.
 */
static bool
wrap_library_function(struct air_ctx *c, const struct air_md *out_md)
{
   const struct air_function *f = c->fn;

   /* The result buffer carries the function's own air.arg_type_name ("float4"
    * and the like, from air.visible_output): a harness that fills the argument
    * buffer has to know how to read the bytes that come back, and Apple's name
    * for the type is a better source for that than our reading of the LLVM
    * type -- the two can then be checked against each other.
    */
   add_resource(c, AIR_RESOURCE_DEVICE_BUFFER, 0, 16 * MAX2(f->num_args, 1),
                true, false, ~0u, NULL);
   add_resource(c, AIR_RESOURCE_DEVICE_BUFFER, 1, 16, false, true, ~0u, out_md);

   struct air_val **argv = rzalloc_array(c->mem_ctx, struct air_val *,
                                         MAX2(f->num_args, 1));
   bool *is_param = rzalloc_array(c->mem_ctx, bool, MAX2(f->num_args, 1));

   for (unsigned i = 0; i < f->num_args; i++) {
      const struct air_type *t = f->args[i].type;
      const struct air_type *e = t->kind == AIR_TYPE_VECTOR ? t->elem : t;
      bool numeric = e && (e->kind == AIR_TYPE_FLOAT || e->kind == AIR_TYPE_HALF ||
                           e->kind == AIR_TYPE_INT);
      if (!numeric || (t->kind != AIR_TYPE_VECTOR && t->kind != AIR_TYPE_FLOAT &&
                       t->kind != AIR_TYPE_HALF && t->kind != AIR_TYPE_INT))
         return air_error(c, "library: argument %u is neither a scalar nor a vector, "
                             "and this wrapper passes each argument in a 16-byte slot "
                             "of one buffer", i);
      if (air_nir_components(t) * air_nir_bit_size(c, t) > 16 * 8)
         return air_error(c, "library: argument %u does not fit the wrapper's 16-byte "
                             "slot", i);

      struct air_ptr p = { .class = AIR_PTR_SSBO, .binding = 0, .pointee = t };
      nir_def *v = air_load_ptr(c, air_ptr_add(c, p, nir_imm_int(&c->b, 16 * i)), t, 16);
      if (!v)
         return false;
      argv[i] = air_val_ssa(c, v);
      is_param[i] = true;
   }

   const struct air_type *rt = f->fn_type->elem;
   const struct air_type *re = rt && rt->kind == AIR_TYPE_VECTOR ? rt->elem : rt;
   if (!rt || rt->kind == AIR_TYPE_VOID)
      return air_error(c, "library: the function returns nothing, so a wrapper would "
                          "have nothing to write");
   if (!re || (re->kind != AIR_TYPE_FLOAT && re->kind != AIR_TYPE_HALF &&
               re->kind != AIR_TYPE_INT))
      return air_error(c, "library: the return value is neither a scalar nor a vector");

   nir_function *fn = translate_callee(c, f, argv, is_param);
   if (!fn)
      return false;   /* translate_callee has said why */

   unsigned bits = air_nir_bit_size(c, rt);
   unsigned comps = air_nir_components(rt);
   const struct glsl_type *gt =
      glsl_vector_type(bits == 16 ? GLSL_TYPE_FLOAT16 :
                       bits == 64 ? GLSL_TYPE_UINT64 : GLSL_TYPE_UINT, comps);
   nir_variable *tmp = nir_local_variable_create(c->b.impl, gt, "libret");

   nir_call_instr *call = nir_call_instr_create(c->nir, fn);
   unsigned idx = 0;
   call->params[idx++] = nir_src_for_ssa(&nir_build_deref_var(&c->b, tmp)->def);
   for (unsigned i = 0; i < f->num_args; i++)
      call->params[idx++] = nir_src_for_ssa(argv[i]->def);
   nir_builder_instr_insert(&c->b, &call->instr);

   struct air_ptr out = { .class = AIR_PTR_SSBO, .binding = 1, .pointee = rt };
   if (!air_store_ptr(c, out, nir_load_var(&c->b, tmp), 16))
      return false;

   /* the generated block ends the way a translated one does: this impl is
    * unstructured until nir_lower_goto_ifs runs over it
    */
   nir_goto(&c->b, c->b.impl->end_block);
   return true;
}

/* Can the edge FROM -> TO be taken? FROM reachable, and its terminator going
 * to TO: always for an unconditional branch or one decided while the shader
 * runs; for one on a known value, only where that value says. (FROM's
 * terminator is translated before TO's phis: reverse post-order.)
 */
static struct air_val *air_peek(struct air_ctx *c, const struct air_value *v);

/* (11-12: its looks at a condition are air_peek's -- a back edge's condition is defined further down, and air_get's
 * "used before it is defined" stayed in the error text though the edge was simply taken as live: a later refusal
 * that said nothing of its own was reported with that stale reason) */
static bool
edge_live(struct air_ctx *c, const struct air_block *from, const struct air_block *to)
{
   if (!c->reachable || !_mesa_set_search(c->reachable, from) || !from->num_insts)
      return false;
   const struct air_inst *term = from->insts[from->num_insts - 1];
   if (term->op == AIR_OP_BR && term->num_targets == 2) {
      struct air_val *cv = air_peek(c, term->operands[0]);
      nir_def *cond = cv && cv->kind == AIR_VAL_SSA ? cv->def : NULL;
      if (cond && cond->num_components == 1 && nir_def_is_const(cond)) {
         bool taken = nir_const_value_as_bool(nir_def_as_load_const(cond)->value[0], cond->bit_size);
         return term->targets[taken ? 0 : 1] == to;
      }
      return term->targets[0] == to || term->targets[1] == to;
   }
   if (term->op == AIR_OP_SWITCH) {
      struct air_val *cv = air_peek(c, term->operands[0]);
      nir_def *cond = cv && cv->kind == AIR_VAL_SSA ? cv->def : NULL;
      if (cond && cond->num_components == 1 && nir_def_is_const(cond)) {
         uint64_t sel = nir_const_value_as_uint(nir_def_as_load_const(cond)->value[0], cond->bit_size);
         const struct air_block *dest = term->targets[0];
         for (unsigned k = 0; k + 1 < term->num_targets; k++) {
            struct air_val *kv = air_peek(c, term->operands[1 + k]);
            nir_def *kd = kv && kv->kind == AIR_VAL_SSA ? kv->def : NULL;
            if (kd && nir_def_is_const(kd) &&
                nir_const_value_as_uint(nir_def_as_load_const(kd)->value[0], kd->bit_size) == sel) {
               dest = term->targets[1 + k];
               break;
            }
         }
         return dest == to;
      }
   }
   for (unsigned t = 0; t < term->num_targets; t++)
      if (term->targets[t] == to)
         return true;
   return false;
}

/* more about the error already said: where it was met (false, for the caller to return) */
static bool
air_error_more(struct air_ctx *c, const char *format, ...)
{
   char more[96];
   va_list ap;

   va_start(ap, format);
   vsnprintf(more, sizeof(more), format, ap);
   va_end(ap);
   if (strlen(c->error) + strlen(more) < sizeof(c->error))
      strcat(c->error, more);
   return false;
}

/* a pointer into a Metal buffer, constant or device: one kind to the back end (an address in the push constants) */
static bool
ptr_is_buffer(const struct air_ptr *p)
{
   return (p->class == AIR_PTR_SSBO || p->class == AIR_PTR_UBO) && !p->is_image;
}

/* V's value if it has one by now (a constant always has), without an error if not: a phi's operand that comes round
 * a loop is defined further down */
static struct air_val *
air_peek(struct air_ctx *c, const struct air_value *v)
{
   struct hash_entry *e = _mesa_hash_table_search(c->values, v);
   if (e)
      return e->data;
   return air_value_as_constant(v) ? air_get(c, v) : NULL;
}

/* two pointers that are one: the same value, or the same handle */
static bool
same_pointer(const struct air_val *a, const struct air_val *b)
{
   if (a == b)
      return true;
   if (a->kind != AIR_VAL_PTR || b->kind != AIR_VAL_PTR)
      return false;
   return a->ptr.class == b->ptr.class && a->ptr.binding == b->ptr.binding &&
          a->ptr.binding_def == b->ptr.binding_def && a->ptr.offset == b->ptr.offset;
}

static bool translate_body_blocks(struct air_ctx *c);

/* (each function's own reachable set: a callee is translated in the middle of
 * its caller's walk, and the caller's set must survive it)
 */
static bool
translate_body(struct air_ctx *c)
{
   struct set *outer = c->reachable;
   bool ok = translate_body_blocks(c);
   c->reachable = outer;
   return ok;
}

static bool
translate_body_blocks(struct air_ctx *c)
{
   const struct air_function *f = c->fn;

   /* phis first: each becomes a register, loaded where the phi is and stored
    * at the end of every predecessor (docs/air.md: NIR's own
    * nir_lower_goto_ifs does the same to make an unstructured CFG safe)
    */
   nir_builder entry = nir_builder_at(nir_before_impl(c->b.impl));
   for (unsigned bi = 0; bi < f->num_blocks; bi++) {
      const struct air_block *blk = &f->blocks[bi];
      for (unsigned i = 0; i < blk->num_insts; i++) {
         const struct air_inst *in = blk->insts[i];
         if (in->op != AIR_OP_PHI)
            continue;
         if (in->base.type->kind == AIR_TYPE_POINTER)
            continue;   /* a phi between pointers: chosen at translation (below) */
         nir_def *reg = nir_decl_reg(&entry, air_nir_components(in->base.type),
                                     air_nir_bit_size(c, in->base.type), 0);
         _mesa_hash_table_insert(c->phi_regs, in, reg);
      }
   }

   /* NIR's first block is the function's first; the rest are made as they
    * are reached, and nir_lower_goto_ifs puts them in order afterwards
    */
   _mesa_hash_table_insert(c->blocks, &f->blocks[0], nir_start_block(c->b.impl));
   c->reachable = _mesa_pointer_set_create(c->mem_ctx);
   _mesa_set_add(c->reachable, &f->blocks[0]);

   unsigned *order = rzalloc_array(c->mem_ctx, unsigned, MAX2(f->num_blocks, 1));
   unsigned num_order = reverse_post_order(c, f, order);

   for (unsigned oi = 0; oi < num_order; oi++) {
      unsigned bi = order[oi];
      const struct air_block *blk = &f->blocks[bi];
      /* a block that branches on known values never reach is not translated:
       * nothing jumps to it (a known condition jumps to its one target), and
       * what it would compute is used nowhere that runs -- the generic uber
       * shaders' features a pipeline did not ask for, with the null pointers
       * they were handed
       */
      if (!_mesa_set_search(c->reachable, blk))
         continue;
      nir_block *nb = nir_block_of(c, blk);
      c->b.cursor = nir_after_block(nb);

      /* the phis of this block are its first instructions */
      for (unsigned i = 0; i < blk->num_insts; i++) {
         const struct air_inst *in = blk->insts[i];
         if (in->op != AIR_OP_PHI)
            continue;
         if (in->base.type->kind == AIR_TYPE_POINTER) {
            /* the pointer of the one edge that can be taken (or of several
             * that bring the same one)
             */
            struct air_val *chosen = NULL;
            bool by_register = false, by_buffer = false, mixed = false;
            struct air_val *null_edge = NULL;
            for (unsigned k = 0; k < in->num_operands && k < in->num_targets; k++) {
               if (!edge_live(c, in->targets[k], blk))
                  continue;
               struct air_val *v = air_peek(c, in->operands[k]);
               if (!v) {
                  /* a pointer that comes round a loop (p = phi [start], [p + n]): defined further down -- through
                   * the registers (which buffer it stands in is not known yet: that too) */
                  by_register = true;
                  by_buffer = true;
                  continue;
               }
               if (v->kind == AIR_VAL_PTR && v->ptr.class == AIR_PTR_NONE) {
                  /* A NULL POINTER on this edge (MPS's cnnConvArray_8xIn: threadgroup memory or null): nothing may
                   * be read through it on that road, so the pointer is the one the other edges bring */
                  null_edge = v;
                  continue;
               }
               if (chosen && !same_pointer(chosen, v)) {
                  /* PLACES IN ONE BUFFER, chosen by the road taken: the offset goes through a register as any
                   * other phi's value (stored at the end of every predecessor, below); pointers into different
                   * buffers are still refused (MPS's kernelComputeStage1) */
                  if (chosen->kind != AIR_VAL_PTR || v->kind != AIR_VAL_PTR || chosen->ptr.is_image != v->ptr.is_image ||
                      (chosen->ptr.class != v->ptr.class && !(ptr_is_buffer(&chosen->ptr) && ptr_is_buffer(&v->ptr))))
                     return air_error(c, "instruction: a phi between pointers of different kinds (%d and %d) chosen while "
                                         "the shader runs (block %u)", chosen->kind == AIR_VAL_PTR ? (int)chosen->ptr.class : -1,
                                      v->kind == AIR_VAL_PTR ? (int)v->ptr.class : -1, bi);
                  if (chosen->ptr.binding != v->ptr.binding || chosen->ptr.binding_def != v->ptr.binding_def ||
                      chosen->ptr.class != v->ptr.class)
                     by_buffer = true;       /* TWO BUFFERS (MPS's cnnConvArray_8xIn): the index through a register too */
                  if (chosen->ptr.class != v->ptr.class)
                     mixed = true;           /* a constant buffer and a device buffer: read as a device buffer */
                  by_register = true;
               }
               chosen = v;
            }
            if (!chosen && !by_register && null_edge)
               chosen = null_edge;          /* (null on every edge) */
            if (by_register && (!chosen || chosen->kind != AIR_VAL_PTR))
               return air_error(c, "instruction: a phi between pointers none of which is known where it stands "
                                   "(block %u)", bi);
            if (by_register) {
               nir_def *reg = nir_decl_reg(&entry, 1, 32, 0);
               struct air_ptr p = chosen->ptr;

               _mesa_hash_table_insert(c->phi_regs, in, reg);
               p.offset = nir_load_reg(&c->b, reg);
               if (mixed)
                  p.class = AIR_PTR_SSBO;
               if (by_buffer && (p.class == AIR_PTR_SSBO || p.class == AIR_PTR_UBO) && !p.is_image) {
                  nir_def *breg = nir_decl_reg(&entry, 1, 32, 0);

                  _mesa_hash_table_insert(c->phi_bind_regs, in, breg);
                  p.binding_def = nir_load_reg(&c->b, breg);
               } else if (by_buffer && chosen->ptr.binding_def) {
                  return air_error(c, "instruction: a phi between pointers of this kind chosen while the shader runs "
                                      "(block %u)", bi);
               }
               air_set(c, &in->base, air_val_ptr(c, p));
               continue;
            }
            if (!chosen) {
               /* a block nothing reaches: any value will do, it never runs */
               chosen = in->num_operands ? air_get(c, in->operands[0]) : NULL;
               if (!chosen)
                  return false;
            }
            air_set(c, &in->base, chosen);
            continue;
         }
         struct hash_entry *e = _mesa_hash_table_search(c->phi_regs, in);
         air_set(c, &in->base,
                 air_val_ssa(c, nir_load_reg(&c->b, e->data)));
      }

      for (unsigned i = 0; i < blk->num_insts; i++) {
         const struct air_inst *in = blk->insts[i];
         if (in->op == AIR_OP_PHI)
            continue;
         if (!translate_instruction(c, in)) {
            /* say where, so the corpus report points at a line of the module */
            char where[64];
            snprintf(where, sizeof(where), " (block %u, instruction %u, opcode %u)",
                     bi, i, in->op);
            if (strlen(c->error) + strlen(where) < sizeof(c->error))
               strcat(c->error, where);
            return false;
         }
      }

      /* Where the phi stores of this block's successors go. A nop marks the
       * spot: without it, a block whose only instruction is its terminator
       * leaves the cursor at the end of the block, and a store put there
       * lands after the jump -- which NIR asserts on. vtn marks its blocks
       * the same way (vtn_cfg.c's end_nop).
       */
      nir_instr *end = &nir_nop(&c->b)->instr;
      nir_cursor at_end = nir_after_instr(end);
      _mesa_hash_table_insert(c->phi_cursors, blk,
                              ralloc_memdup(c->mem_ctx, &at_end, sizeof(at_end)));

      const struct air_inst *term = blk->insts[blk->num_insts - 1];
      if (_mesa_set_search(c->reachable, blk)) {
         for (unsigned t = 0; t < term->num_targets; t++)
            if (edge_live(c, blk, term->targets[t]))
               _mesa_set_add(c->reachable, term->targets[t]);
      }
      switch (term->op) {
      case AIR_OP_RET:
         if (!emit_return(c, term))
            return air_error_more(c, " (the return of block %u)", bi);
         nir_goto(&c->b, c->b.impl->end_block);
         break;
      case AIR_OP_BR:
         if (term->num_targets == 1) {
            nir_goto(&c->b, nir_block_of(c, term->targets[0]));
         } else {
            nir_def *cond = air_get_ssa(c, term->operands[0]);
            if (!cond)
               return air_error_more(c, " (the branch of block %u)", bi);
            if (term->targets[0] == term->targets[1])
               nir_goto(&c->b, nir_block_of(c, term->targets[0]));
            else if (cond->num_components == 1 && nir_def_is_const(cond))
               nir_goto(&c->b, nir_block_of(c, term->targets[nir_const_value_as_bool(
                  nir_def_as_load_const(cond)->value[0], cond->bit_size) ? 0 : 1]));
            else
               nir_goto_if(&c->b, nir_block_of(c, term->targets[0]), cond,
                           nir_block_of(c, term->targets[1]));
         }
         break;
      case AIR_OP_SWITCH: {
         /* (from upstream's tools/mesa/air) NIR has no switch: the cases become a chain of nir_goto_if, each test in
          * a block of its own, the last falling to the default; nir_lower_goto_ifs structures it as every other
          * branch here. Safe for the phis for the reason the conditional br is: the successors' phi stores all sit
          * at phi_cursors[blk], above the chain, in registers of their own. A selector that is a constant here takes
          * its one case (a function constant: CoreAnimation's uber shaders dispatch on them), and the blocks of the
          * other cases are left as unreachable as they are.
          */
         nir_def *cond = air_get_ssa(c, term->operands[0]);
         if (!cond)
            return false;
         nir_block *deflt = nir_block_of(c, term->targets[0]);
         unsigned n = term->num_targets - 1;
         if (nir_def_is_const(cond)) {
            uint64_t sel = nir_const_value_as_uint(nir_def_as_load_const(cond)->value[0], cond->bit_size);
            nir_block *to = deflt;
            for (unsigned k = 0; k < n; k++) {
               nir_def *cv = air_get_ssa(c, term->operands[1 + k]);
               if (!cv)
                  return false;
               if (nir_def_is_const(cv) &&
                   nir_const_value_as_uint(nir_def_as_load_const(cv)->value[0], cv->bit_size) == sel) {
                  to = nir_block_of(c, term->targets[1 + k]);
                  break;
               }
            }
            nir_goto(&c->b, to);
            break;
         }
         for (unsigned k = 0; k < n; k++) {
            nir_def *val = air_get_ssa(c, term->operands[1 + k]);
            if (!val)
               return false;
            nir_block *dest = nir_block_of(c, term->targets[1 + k]);
            bool last = k + 1 == n;
            nir_block *next = deflt;
            if (!last) {
               next = AIR_BLOCK_CREATE(c->nir, c->b.impl);
               exec_list_push_tail(&c->b.impl->body, &next->cf_node.node);
               next->cf_node.parent = &c->b.impl->cf_node;
            }
            /* a last case that goes where the default goes is no branch at all */
            if (dest == next)
               nir_goto(&c->b, dest);
            else
               nir_goto_if(&c->b, dest, nir_ieq(&c->b, cond, val), next);
            if (!last)
               c->b.cursor = nir_after_block(next);
         }
         if (n == 0)
            nir_goto(&c->b, deflt);
         break;
      }
      case AIR_OP_UNREACHABLE:
         nir_goto(&c->b, c->b.impl->end_block);
         break;
      default:
         return air_error(c, "instruction: block %u ends with opcode %u", bi, term->op);
      }
   }

   /* the phi stores, now that every value exists */
   for (unsigned bi = 0; bi < f->num_blocks; bi++) {
      const struct air_block *blk = &f->blocks[bi];
      for (unsigned i = 0; i < blk->num_insts; i++) {
         const struct air_inst *in = blk->insts[i];
         if (in->op != AIR_OP_PHI)
            continue;
         struct hash_entry *re = _mesa_hash_table_search(c->phi_regs, in);
         if (!re)
            continue; /* a phi in an unreachable block */
         for (unsigned k = 0; k < in->num_operands; k++) {
            struct hash_entry *ce = _mesa_hash_table_search(c->phi_cursors, in->targets[k]);
            if (!ce)
               continue; /* an edge from a block the entry cannot reach */
            c->b.cursor = *(nir_cursor *)ce->data;
            nir_def *val;
            if (in->base.type->kind == AIR_TYPE_POINTER) {
               /* a pointer phi's register holds the offset in its one buffer */
               struct air_val *pv = air_get(c, in->operands[k]);
               if (!pv)
                  return air_error_more(c, " (a pointer phi's operand %u, block %u)", k, bi);
               struct air_val *own = air_peek(c, &in->base);
               struct hash_entry *be = _mesa_hash_table_search(c->phi_bind_regs, in);
               if (pv->kind == AIR_VAL_PTR && pv->ptr.class == AIR_PTR_NONE && own && own->kind == AIR_VAL_PTR) {
                  /* (a null pointer's edge: the register's value is never used on that road) */
                  if (be)
                     nir_store_reg(&c->b, nir_imm_int(&c->b, own->ptr.binding), be->data);
                  nir_store_reg(&c->b, nir_imm_int(&c->b, 0), re->data);
                  *(nir_cursor *)ce->data = c->b.cursor;
                  continue;
               }
               if (pv->kind != AIR_VAL_PTR || !own || own->kind != AIR_VAL_PTR ||
                   (pv->ptr.class != own->ptr.class && !(be && ptr_is_buffer(&pv->ptr) && ptr_is_buffer(&own->ptr))) ||
                   (!be && (pv->ptr.binding != own->ptr.binding || pv->ptr.binding_def)))
                  return air_error(c, "instruction: a phi between pointers into different buffers chosen while the "
                                      "shader runs");
               if (be)
                  nir_store_reg(&c->b, ptr_binding(c, pv->ptr), be->data);
               val = pv->ptr.offset ? pv->ptr.offset : nir_imm_int(&c->b, 0);
            } else {
               val = air_get_ssa(c, in->operands[k]);
            }
            if (!val)
               return air_error_more(c, " (a phi's operand %u of %u, block %u, instruction %u)", k, in->num_operands, bi, i);
            nir_store_reg(&c->b, val, re->data);
            /* later stores into the same block go after this one */
            *(nir_cursor *)ce->data = c->b.cursor;
         }
      }
   }
   return true;
}

/* -- module-scope variables --------------------------------------------- */

/* the index !air.function_constants gives this global, or -1 */
static int
function_constant_index(struct air_ctx *c, const struct air_global *g)
{
   const struct air_named_md *n = named_md(c->m, "air.function_constants");
   for (unsigned i = 0; i < (n ? n->num_ops : 0); i++) {
      const struct air_md *node = n->ops[i];
      if (!node || node->kind != AIR_MD_NODE || !node->num_ops)
         continue;
      if (!node->ops[0] || node->ops[0]->kind != AIR_MD_VALUE ||
          node->ops[0]->value != &g->base)
         continue;
      /* !{ptr @g, !"uint", !"name", i32 index, i1 true} */
      for (unsigned k = 1; k < node->num_ops; k++) {
         uint64_t idx;
         if (md_int(node, k, &idx))
            return (int)idx;
      }
      return -1;
   }
   return -1;
}

/* Decide what every global is: once per shader, because this appends inline
 * samplers to info->resources and lays out the shared and scratch memory. It
 * records rather than builds -- making the value needs a builder, and which
 * function's builder matters (see materialise_globals).
 */
/* Does constant V mention TARGET (a global), through aggregates and constant
 * expressions? (from tools/mesa/air: on the kind, DATA has no elems)
 */
static bool
const_mentions(const struct air_value *v, const struct air_value *target, unsigned depth)
{
   if (!v || depth == 0)
      return false;
   if (v == target)
      return true;
   const struct air_constant *k = air_value_as_constant(v);
   if (!k)
      return false;
   switch (k->ckind) {
   case AIR_CONST_AGGREGATE:
      for (unsigned i = 0; i < k->num_elems; i++)
         if (const_mentions(k->elems[i], target, depth - 1))
            return true;
      return false;
   case AIR_CONST_EXPR:
      for (unsigned i = 0; i < k->num_operands; i++)
         if (const_mentions(k->operands[i], target, depth - 1))
            return true;
      return false;
   default:
      return false;
   }
}

/* Which program-scope variables are a VALUE rather than memory (tools/mesa/air's
 * census, ported): a scalar integer of 8/16/32/64 bits; every instruction
 * naming it names it as the whole pointer of a load or a store of its own
 * type; no constant and no other global's initialiser mentions it; exactly one
 * store, in an air.static_init function (run_static_initializers inlines those
 * at the top of the entry point, before the body). Whether its value is a
 * constant is known only when that store is translated (`known`).
 */
static void
plan_global_folding(struct air_ctx *c)
{
   for (unsigned i = 0; i < c->m->num_globals; i++) {
      struct air_global_plan *pl = &c->global_plan[i];
      const struct air_global *g = c->m->globals[i];
      if (pl->skip || pl->class != AIR_PTR_SCRATCH)
         continue;
      const struct air_type *t = g->value_type;
      if (!t || t->kind != AIR_TYPE_INT ||
          (t->bits != 8 && t->bits != 16 && t->bits != 32 && t->bits != 64))
         continue;
      bool ok = true;
      unsigned stores = 0;
      for (unsigned k = 0; ok && k < c->m->num_globals; k++)
         if (k != i && const_mentions(c->m->globals[k]->init, &g->base, 8))
            ok = false;
      for (unsigned fi = 0; ok && fi < c->m->num_functions; fi++) {
         const struct air_function *f = c->m->functions[fi];
         bool is_init = f->section && !strcmp(f->section, "air.static_init");
         for (unsigned bi = 0; ok && bi < f->num_blocks; bi++) {
            const struct air_block *blk = &f->blocks[bi];
            for (unsigned ii = 0; ok && ii < blk->num_insts; ii++) {
               const struct air_inst *in = blk->insts[ii];
               for (unsigned oi = 0; ok && oi < in->num_operands; oi++) {
                  const struct air_value *v = in->operands[oi];
                  if (v == &g->base) {
                     if (in->op == AIR_OP_LOAD && oi == 0 && in->base.type == t)
                        continue;
                     if (in->op == AIR_OP_STORE && oi == 1 && in->operands[0]->type == t && is_init) {
                        stores++;
                        continue;
                     }
                     ok = false;
                  } else if (const_mentions(v, &g->base, 8)) {
                     ok = false;
                  }
               }
            }
         }
      }
      if (ok && stores == 1)
         pl->fold = true;
   }
}

/* the globals a store's pointer can be rooted in, followed through
 * getelementptr, casts, selects and phis (instructions and constant
 * expressions); each one's scratch range marked as written outside the
 * static initializers
 */
static void
mark_store_roots(struct air_ctx *c, const struct air_value *v, unsigned depth)
{
   if (!v || depth == 0)
      return;
   if (v->kind == AIR_VALUE_GLOBAL) {
      for (unsigned i = 0; i < c->m->num_globals; i++) {
         if (&c->m->globals[i]->base != v)
            continue;
         const struct air_global_plan *pl = &c->global_plan[i];
         if (pl->class != AIR_PTR_SCRATCH || !pl->has_offset)
            return;
         if (c->num_scratch_dirty == ARRAY_SIZE(c->scratch_dirty)) {
            c->scratch_dirty_all = true;
            return;
         }
         c->scratch_dirty[c->num_scratch_dirty].lo = pl->offset;
         c->scratch_dirty[c->num_scratch_dirty].hi =
            pl->offset + air_type_alloc_size(c->m, c->m->globals[i]->value_type);
         c->num_scratch_dirty++;
         return;
      }
      return;
   }
   const struct air_constant *k = air_value_as_constant(v);
   if (k) {
      if (k->ckind == AIR_CONST_EXPR)
         for (unsigned i = 0; i < k->num_operands; i++)
            mark_store_roots(c, k->operands[i], depth - 1);
      return;
   }
   if (v->kind != AIR_VALUE_INSTRUCTION)
      return;
   const struct air_inst *in = (const struct air_inst *)v;
   switch (in->op) {
   case AIR_OP_GEP:
   case AIR_OP_CAST:
      mark_store_roots(c, in->operands[0], depth - 1);
      break;
   case AIR_OP_SELECT:
      mark_store_roots(c, in->operands[1], depth - 1);
      mark_store_roots(c, in->operands[2], depth - 1);
      break;
   case AIR_OP_PHI:
      for (unsigned i = 0; i < in->num_operands; i++)
         mark_store_roots(c, in->operands[i], depth - 1);
      break;
   default:
      break;
   }
}

static void
plan_scratch_dirty(struct air_ctx *c)
{
   for (unsigned fi = 0; fi < c->m->num_functions; fi++) {
      const struct air_function *f = c->m->functions[fi];
      if (f->section && !strcmp(f->section, "air.static_init"))
         continue;
      for (unsigned bi = 0; bi < f->num_blocks; bi++)
         for (unsigned ii = 0; ii < f->blocks[bi].num_insts; ii++) {
            const struct air_inst *in = f->blocks[bi].insts[ii];
            if (in->op == AIR_OP_STORE)
               mark_store_roots(c, in->operands[1], 16);
            else if (in->op == AIR_OP_ATOMICRMW || in->op == AIR_OP_CMPXCHG)
               mark_store_roots(c, in->operands[0], 16);
         }
   }
}

/* the plan of the global V is, when it is a foldable one */
static struct air_global_plan *
foldable_global_plan(struct air_ctx *c, const struct air_value *v)
{
   if (!c->global_plan || !v || v->kind != AIR_VALUE_GLOBAL)
      return NULL;
   for (unsigned i = 0; i < c->m->num_globals; i++)
      if (&c->m->globals[i]->base == v)
         return c->global_plan[i].fold ? &c->global_plan[i] : NULL;
   return NULL;
}

static void
plan_globals(struct air_ctx *c)
{
   c->global_plan = rzalloc_array(c->mem_ctx, struct air_global_plan,
                                  MAX2(c->m->num_globals, 1));
   for (unsigned i = 0; i < c->m->num_globals; i++) {
      struct air_global_plan *pl = &c->global_plan[i];
      const struct air_global *g = c->m->globals[i];
      const char *name = g->base.name ? g->base.name : "";

      /* llvm.compiler.used and llvm.global_ctors are lists for the linker,
       * not memory the shader reads
       */
      if (!strncmp(name, "llvm.", 5)) {
         pl->skip = true;
         continue;
      }

      if (g->externally_initialized) {
         int idx = function_constant_index(c, g);
         if (idx < 0) {
            air_error(c, "interface: an externally initialized variable that is not a "
                         "function constant");
            return;
         }
         pl->class = AIR_PTR_FUNCTION_CONSTANT;
         pl->binding = idx;
      } else if (!strncmp(name, "__air_sampler_state", 19)) {
         /* A sampler the shader declares itself. It needs a real index, not a
          * sentinel: the Adreno's texture instruction holds the sampler in a
          * four-bit field, and ~0u there asserts in ir3's encoder. The index
          * goes after the API's samplers, and the runtime creates the sampler
          * from the state word (docs/air.md section 7.7).
          */
         unsigned next = 0;
         for (unsigned k = 0; k < c->info->num_resources; k++)
            if (c->info->resources[k].kind == AIR_RESOURCE_SAMPLER)
               next = MAX2(next, c->info->resources[k].index + 1);
         pl->class = AIR_PTR_SAMPLER;
         pl->binding = next;

         uint64_t word = 0;
         if (g->init && !air_const_uint(g->init, &word)) {
            air_error(c, "interface: an inline sampler whose state is not a constant");
            return;
         }
         struct air_shader_info *info = c->info;
         info->resources = reralloc(c->mem_ctx, info->resources, struct air_resource,
                                    info->num_resources + 1);
         info->resources[info->num_resources++] = (struct air_resource) {
            .kind = AIR_RESOURCE_SAMPLER, .index = next, .read = true,
            .param = ~0u, .name = name,
            .inline_sampler = true, .sampler_state = word,
         };
      } else if (g->addrspace == 3) {
         /* a static threadgroup array */
         unsigned align = MAX2(g->align, air_type_abi_align(c->m, g->value_type));
         c->shared_size = ALIGN_POT(c->shared_size, MAX2(align, 1));
         pl->class = AIR_PTR_SHARED;
         pl->offset = c->shared_size;
         pl->has_offset = true;
         c->shared_size += air_type_alloc_size(c->m, g->value_type);
      } else {
         /* a program-scope variable: private to the invocation, written by
          * the static initializers below
          */
         unsigned align = MAX2(g->align, air_type_abi_align(c->m, g->value_type));
         c->scratch_size = ALIGN_POT(c->scratch_size, MAX2(align, 1));
         pl->class = AIR_PTR_SCRATCH;
         pl->offset = c->scratch_size;
         pl->has_offset = true;
         c->scratch_size += air_type_alloc_size(c->m, g->value_type);
      }
   }
}

/* Build the globals as values of the function being translated now. Called for
 * the entry point and again for every callee, because the byte offset is an
 * nir_def and a def belongs to one impl: handing a callee the entry point's
 * defs is what crashed nir_repair_ssa. Module-level constants need no such
 * care -- air_get materialises one on demand in whatever builder asks.
 */
static void
materialise_globals(struct air_ctx *c)
{
   for (unsigned i = 0; i < c->m->num_globals; i++) {
      const struct air_global_plan *pl = &c->global_plan[i];
      if (pl->skip)
         continue;
      const struct air_global *g = c->m->globals[i];
      struct air_ptr p = {
         .pointee = g->value_type,
         .class = pl->class,
         .binding = pl->binding,
         .offset = pl->has_offset ? nir_imm_int(&c->b, pl->offset) : NULL,
      };
      air_set(c, &g->base, air_val_ptr(c, p));
      /* a program-scope constant holding POINTERS (CoreAnimation's
       * unused_sampler: { ptr @__air_sampler_state }): memory cannot hold
       * them, so they go to the local-pointer table at their offsets, where a
       * load of a pointer from it finds them
       */
      if (pl->class == AIR_PTR_SCRATCH && pl->has_offset && g->init && g->value_type) {
         const struct air_constant *k = air_value_as_constant(g->init);
         const struct air_type *t = g->value_type;
         if (k && k->ckind == AIR_CONST_AGGREGATE && t->kind == AIR_TYPE_STRUCT) {
            for (unsigned m = 0; m < k->num_elems && m < t->num_members; m++) {
               if (t->members[m]->kind != AIR_TYPE_POINTER)
                  continue;
               struct air_val *pv = air_get(c, k->elems[m]);
               if (!pv)
                  return;
               uint64_t off = pl->offset + air_struct_member_offset(c->m, t, m);
               unsigned n = 0;
               while (n < c->num_scratch_ptrs && c->scratch_ptrs[n].offset != off)
                  n++;
               if (n == ARRAY_SIZE(c->scratch_ptrs))
                  return (void)air_error(c, "interface: more than %u pointers in program-scope constants",
                                         (unsigned)ARRAY_SIZE(c->scratch_ptrs));
               if (n == c->num_scratch_ptrs)
                  c->num_scratch_ptrs++;
               c->scratch_ptrs[n].offset = off;
               c->scratch_ptrs[n].val = pv;
            }
         }
      }
   }
}

/* The bytes of the constant V of type T, as memory holds them (little-endian), into DST; false for what is not plain
 * data -- a constant expression (a pointer's value), a pointer, a type with no size. (From tools/mesa/air.)
 */
static bool
write_constant(struct air_ctx *c, const struct air_type *t, const struct air_value *v, uint8_t *dst)
{
   const struct air_constant *k = air_value_as_constant(v);
   if (!k || !t)
      return false;
   const uint64_t size = air_type_alloc_size(c->m, t);
   switch (k->ckind) {
   case AIR_CONST_NULL:
   case AIR_CONST_UNDEF:
   case AIR_CONST_POISON:
      if (t->kind == AIR_TYPE_POINTER)
         return false;
      memset(dst, 0, size);
      return true;
   case AIR_CONST_INT:
   case AIR_CONST_FLOAT: {
      const unsigned bytes = t->kind == AIR_TYPE_INT ? (t->bits + 7) / 8 : t->kind == AIR_TYPE_HALF ? 2 :
                             t->kind == AIR_TYPE_FLOAT ? 4 : t->kind == AIR_TYPE_DOUBLE ? 8 : 0;
      if (bytes == 0 || bytes > 8)
         return false;
      memcpy(dst, &k->u, bytes);
      return true;
   }
   case AIR_CONST_DATA:
   case AIR_CONST_AGGREGATE: {
      if (t->kind != AIR_TYPE_ARRAY && t->kind != AIR_TYPE_VECTOR && t->kind != AIR_TYPE_STRUCT)
         return false;
      memset(dst, 0, size);
      for (unsigned i = 0; i < k->num_elems; i++) {
         if (t->kind == AIR_TYPE_STRUCT && i >= t->num_members)
            return false;
         const struct air_type *et = t->kind == AIR_TYPE_STRUCT ? t->members[i] : t->elem;
         const uint64_t eb = air_type_alloc_size(c->m, et);
         const uint64_t at = t->kind == AIR_TYPE_STRUCT ? air_struct_member_offset(c->m, t, i) : i * eb;
         if (at + eb > size)
            return false;
         if (k->ckind == AIR_CONST_DATA) {
            if (eb > 8)
               return false;
            memcpy(dst + at, &k->data[i], eb);
         } else if (!write_constant(c, et, k->elems[i], dst + at)) {
            return false;
         }
      }
      return true;
   }
   default:
      return false;
   }
}

/* A GLOBAL WITH ITS DATA -- a table, `static const float2 corners[3] = {...}`, or a program-scope variable with a
 * first value -- is in scratch memory, which nothing had written the initialiser to (11-13): a load at an index known
 * only while the shader runs read whatever was there. WebKit's WebGL (ANGLE's blit: its triangle's corners from such
 * a table by the vertex number) drew nothing on the Pixel 8a, every corner at 0,0. The initialiser's bytes are stored
 * at the entry point's start, a word at a time, before the static initialisers run (their stores come after and
 * stand). tools/mesa/air puts a read-only one in nir->constant_data instead; Mesa's Mali compiler has no
 * load_constant, and scratch is where every pointer into the global already points.
 */
static void
store_global_initialisers(struct air_ctx *c)
{
   for (unsigned i = 0; i < c->m->num_globals && !c->error[0]; i++) {
      const struct air_global_plan *pl = &c->global_plan[i];
      const struct air_global *g = c->m->globals[i];
      if (pl->skip || pl->class != AIR_PTR_SCRATCH || !pl->has_offset || !g->init || !g->value_type)
         continue;
      const uint64_t size = air_type_alloc_size(c->m, g->value_type);
      if (size == 0)
         continue;
      if (size > 16384) {
         air_error(c, "interface: a global of %llu bytes with an initialiser (16384 at most are written at the start)",
                   (unsigned long long)size);
         return;
      }
      uint8_t *bytes = rzalloc_size(c->mem_ctx, size + 4);
      if (!write_constant(c, g->value_type, g->init, bytes))
         continue;         /* (pointers and constant expressions: materialise_globals' table, or nothing) */
      uint64_t at = 0;
      for (; at + 4 <= size; at += 4) {
         uint32_t word;
         memcpy(&word, bytes + at, 4);
         nir_store_scratch(&c->b, nir_imm_int(&c->b, (int32_t)word), nir_imm_int(&c->b, (int32_t)(pl->offset + at)),
                           .align_mul = 4, .write_mask = 1);
      }
      for (; at < size; at++)
         nir_store_scratch(&c->b, nir_imm_intN_t(&c->b, bytes[at], 8), nir_imm_int(&c->b, (int32_t)(pl->offset + at)),
                           .align_mul = 1, .write_mask = 1);
   }
}

/* A static initializer with branches, run at translation: every condition in it
 * is decided by function constants (CoreAnimation's generic fixed_frag_lph_cph
 * and _cpf: short-circuit tests of which constants are defined), so its blocks
 * are walked along the path actually taken -- each instruction translated in
 * place at the top of the entry point, as a straight one's are; a phi takes
 * the value from the block the walk came from; a branch goes where its
 * (constant) condition says. Its stores then carry constants, which is what
 * plan_global_folding's loads answer with. A condition known only while the
 * shader runs is refused by name.
 */
static bool
walk_static_initializer(struct air_ctx *c, const struct air_function *f)
{
   const struct air_block *blk = &f->blocks[0], *from = NULL;
   for (unsigned steps = 0; steps < 4096; steps++) {
      const struct air_inst *term = NULL;
      for (unsigned k = 0; k < blk->num_insts; k++) {
         const struct air_inst *in = blk->insts[k];
         if (in->op == AIR_OP_PHI) {
            struct air_val *v = NULL;
            for (unsigned t = 0; t < in->num_targets && t < in->num_operands; t++)
               if (in->targets[t] == from)
                  v = air_get(c, in->operands[t]);
            if (!v)
               return air_error(c, "interface: a static initializer's phi with no value from block %u",
                                from ? from->index : 0), false;
            air_set(c, &in->base, v);
            continue;
         }
         if (in->op == AIR_OP_RET || in->op == AIR_OP_BR || in->op == AIR_OP_SWITCH ||
             in->op == AIR_OP_UNREACHABLE) {
            term = in;
            break;
         }
         if (!translate_instruction(c, in))
            return air_error_more(c, " (the static initializer %s: block %u, instruction %u, opcode %u)",
                                  f->base.name ? f->base.name : "?", blk->index, k, in->op);
      }
      if (!term || term->op == AIR_OP_RET)
         return true;
      if (term->op == AIR_OP_UNREACHABLE)
         return air_error(c, "interface: a static initializer reached unreachable"), false;
      const struct air_block *to = NULL;
      if (term->op == AIR_OP_BR && term->num_targets == 1) {
         to = term->targets[0];
      } else {
         nir_def *cond = air_get_ssa(c, term->operands[0]);
         if (!cond)
            return false;
         if (cond->num_components != 1 || !nir_def_is_const(cond))
            return air_error(c, "interface: a static initializer branching on a value known only while the "
                                "shader runs"), false;
         uint64_t sel = nir_const_value_as_uint(nir_def_as_load_const(cond)->value[0], cond->bit_size);
         if (term->op == AIR_OP_BR) {
            to = sel ? term->targets[0] : term->targets[1];
         } else {
            to = term->targets[0];
            for (unsigned k = 0; k + 1 < term->num_targets; k++) {
               nir_def *cv = air_get_ssa(c, term->operands[1 + k]);
               if (!cv)
                  return false;
               if (nir_def_is_const(cv) &&
                   nir_const_value_as_uint(nir_def_as_load_const(cv)->value[0], cv->bit_size) == sel) {
                  to = term->targets[1 + k];
                  break;
               }
            }
         }
      }
      from = blk;
      blk = to;
   }
   return air_error(c, "interface: a static initializer that does not end"), false;
}

/* The functions of @llvm.global_ctors, inlined at the top of the entry point:
 * this is where a function constant turns into the program-scope variable the
 * body reads (AV1SW computes kPixelShift from kBitDepth that way).
 */
static void
run_static_initializers(struct air_ctx *c)
{
   for (unsigned i = 0; i < c->m->num_functions; i++) {
      const struct air_function *f = c->m->functions[i];
      if (f->is_declaration || !f->section || strcmp(f->section, "air.static_init"))
         continue;
      if (f->num_args) {
         air_error(c, "interface: a static initializer with arguments");
         return;
      }
      if (f->num_blocks != 1) {
         c->in_static_init = true;
         bool ok = walk_static_initializer(c, f);
         c->in_static_init = false;
         if (!ok)
            return;
         continue;
      }
      /* its instructions, in place, without its ret */
      const struct air_block *blk = &f->blocks[0];
      c->in_static_init = true;
      for (unsigned k = 0; k < blk->num_insts; k++) {
         const struct air_inst *in = blk->insts[k];
         if (in->op == AIR_OP_RET)
            continue;
         if (!translate_instruction(c, in)) {
            c->in_static_init = false;
            return;
         }
      }
      c->in_static_init = false;
   }
}

/* -- the entry point ---------------------------------------------------- */

nir_shader *
air_to_nir(void *mem_ctx, const void *bitcode, size_t size, const char *entry_point,
           const struct air_to_nir_options *options,
           const nir_shader_compiler_options *nir_options,
           struct air_shader_info *info, const char **error)
{
   void *ctx = ralloc_context(mem_ctx);
   struct air_shader_info local_info = { 0 };
   struct air_ctx c = {
      .mem_ctx = ctx,
      .options = options,
      .info = info ? info : &local_info,
   };
   memset(c.info, 0, sizeof(*c.info));

   const char *read_error = NULL;
   c.m = air_module_read(ctx, bitcode, size, &read_error);
   if (!c.m) {
      if (error)
         *error = ralloc_asprintf(mem_ctx, "reader: %s", read_error);
      ralloc_free(ctx);
      return NULL;
   }

   const struct air_md *entry = NULL;
   int stage = module_stage(c.m, &entry);
   if (stage == -2) {
      if (error)
         *error = ralloc_strdup(mem_ctx, "stage: no entry point metadata");
      ralloc_free(ctx);
      return NULL;
   }
   /* air.visible: a library function, linked into its caller at pipeline time
    * (docs/air.md section 7.5). Not an entry point, so not a shader -- unless
    * the caller asks for the generated wrapper, which makes one compute shader
    * whose only job is to call it (wrap_library_function).
    */
   bool wrap = false;
   if (stage == MESA_SHADER_NONE) {
      if (!options || !options->library_wrapper) {
         if (error)
            *error = ralloc_strdup(mem_ctx,
                                   "stage: air.visible, a library function (not yet)");
         ralloc_free(ctx);
         return NULL;
      }
      stage = MESA_SHADER_COMPUTE;
      wrap = true;
   }

   if (!entry->num_ops || !entry->ops[0] || entry->ops[0]->kind != AIR_MD_VALUE ||
       entry->ops[0]->value->kind != AIR_VALUE_FUNCTION) {
      if (error)
         *error = ralloc_strdup(mem_ctx, "stage: entry point metadata without a function");
      ralloc_free(ctx);
      return NULL;
   }
   const struct air_function *fn = (const struct air_function *)entry->ops[0]->value;
   if (entry_point && strcmp(entry_point, fn->base.name ? fn->base.name : "")) {
      if (error)
         *error = ralloc_asprintf(mem_ctx, "stage: this module's entry point is %s",
                                  fn->base.name);
      ralloc_free(ctx);
      return NULL;
   }
   if (fn->is_declaration) {
      if (error)
         *error = ralloc_strdup(mem_ctx, "stage: the entry point has no body");
      ralloc_free(ctx);
      return NULL;
   }
   c.fn = fn;

   c.nir = AIR_SHADER_CREATE(mem_ctx, stage, nir_options);
   c.nir->info.name = ralloc_strdup(c.nir, fn->base.name ? fn->base.name : "air");
   c.nir->info.internal = false;
   c.fns = _mesa_pointer_hash_table_create(ctx);
   nir_function *main = nir_function_create(c.nir, "main");
   main->is_entrypoint = true;
   c.b = nir_builder_at(nir_before_impl(nir_function_impl_create(main)));
   /* operations on values already known fold as they are built: a function
    * constant compared with zero is a constant, and so is the select or the
    * branch it decides (CoreAnimation's generic uber shaders choose a module
    * constant over a buffer so, and dispatch on dozens of such tests)
    */
   c.b.constant_fold_alu = true;
   c.b.impl->structured = false;

   c.values = _mesa_pointer_hash_table_create(ctx);
   c.globals = _mesa_pointer_hash_table_create(ctx);
   c.blocks = _mesa_pointer_hash_table_create(ctx);
   c.phi_cursors = _mesa_pointer_hash_table_create(ctx);
   c.phi_regs = _mesa_pointer_hash_table_create(ctx);
   c.phi_bind_regs = _mesa_pointer_hash_table_create(ctx);
   c.num_params = fn->num_args;
   c.params = rzalloc_array(ctx, struct air_val *, MAX2(c.num_params, 1));

   c.info->stage = stage;
   c.info->entry_point = fn->base.name;

   /* the outputs: fields of the returned struct, in order. A wrapped library
    * function has none of this: its result is a value the wrapper stores to a
    * buffer, not a shader output, and its own metadata uses air.visible_input
    * and air.visible_output rather than the keys translate_input reads.
    */
   const struct air_md *out_node = (!wrap && entry->num_ops > 1) ? entry->ops[1] : NULL;
   const struct air_type *rt = fn->fn_type->elem;
   unsigned num_fields = wrap ? 0 :
                         rt->kind == AIR_TYPE_STRUCT ? rt->num_members :
                         rt->kind == AIR_TYPE_VOID ? 0 : 1;
   c.num_outputs = num_fields;
   c.outputs = rzalloc_array(ctx, __typeof__(*c.outputs), MAX2(num_fields, 1));

   unsigned next_out_varying = 0, next_in_varying = 0;
   if (out_node && out_node->num_ops != num_fields && num_fields) {
      air_error(&c, "interface: %u outputs for %u result fields", out_node->num_ops,
                num_fields);
   } else {
      for (unsigned i = 0; i < num_fields && !c.error[0]; i++) {
         const struct air_type *ft = rt->kind == AIR_TYPE_STRUCT ? rt->members[i] : rt;
         if (!out_node || i >= out_node->num_ops || !out_node->ops[i]) {
            air_error(&c, "interface: result field %u has no metadata", i);
            break;
         }
         translate_output(&c, out_node->ops[i], i, ft, &next_out_varying);
      }
   }

   /* the inputs */
   const struct air_md *in_node = (!wrap && entry->num_ops > 2) ? entry->ops[2] : NULL;
   for (unsigned i = 0; i < (in_node ? in_node->num_ops : 0) && !c.error[0]; i++)
      translate_input(&c, in_node->ops[i], &next_in_varying);

   /* the entry point's parameters, as the body will read them */
   for (unsigned i = 0; i < (wrap ? 0 : c.num_params) && !c.error[0]; i++) {
      if (c.params[i])
         air_set(&c, &fn->args[i], c.params[i]);
      else
         air_error(&c, "interface: parameter %u has no metadata", i);
   }

   if (!c.error[0]) {
      plan_globals(&c);
      plan_global_folding(&c);
      plan_scratch_dirty(&c);
   }
   if (!c.error[0] && c.tg_arg_param_plus1 && c.params[c.tg_arg_param_plus1 - 1]) {
      /* the threadgroup argument past the static arrays (at 0 the two would
       * share their first bytes; hakios-metal's tg_arg is the test) */
      unsigned base = ALIGN_POT(c.shared_size, 16);
      c.params[c.tg_arg_param_plus1 - 1]->ptr.offset = base ? nir_imm_int(&c.b, base) : NULL;
      c.info->has_threadgroup_arg = true;
      c.info->threadgroup_arg_base = base;
   }
   if (!c.error[0])
      materialise_globals(&c);
   if (!c.error[0])
      store_global_initialisers(&c);
   if (!c.error[0])
      run_static_initializers(&c);

   if (!c.error[0]) {
      if (wrap) {
         /* air.visible's second operand is the result node, whose one entry
          * carries air.visible_output and the type's name
          */
         const struct air_md *rn = entry->num_ops > 1 ? entry->ops[1] : NULL;
         wrap_library_function(&c, rn && rn->num_ops ? rn->ops[0] : NULL);
      }
      else
         translate_body(&c);
   }

   if (c.error[0]) {
      if (error)
         *error = ralloc_strdup(mem_ctx, c.error);
      ralloc_free(c.nir);
      ralloc_free(ctx);
      return NULL;
   }

   /* What the shader expects to be bound, as NIR counts it: Metal's indices
    * are the binding numbers, so a shader using [[buffer(14)]] alone still
    * declares fifteen UBOs. ir3 and NIR's own passes read these.
    */
   for (unsigned i = 0; i < c.info->num_resources; i++) {
      const struct air_resource *r = &c.info->resources[i];
      switch (r->kind) {
      case AIR_RESOURCE_CONSTANT_BUFFER:
         c.nir->info.num_ubos = MAX2(c.nir->info.num_ubos, r->index + 1);
         break;
      case AIR_RESOURCE_DEVICE_BUFFER:
         c.nir->info.num_ssbos = MAX2(c.nir->info.num_ssbos, r->index + 1);
         break;
      case AIR_RESOURCE_TEXTURE:
         c.nir->info.num_textures = MAX2(c.nir->info.num_textures, r->index + 1);
         BITSET_SET(c.nir->info.textures_used, r->index);
         break;
      case AIR_RESOURCE_IMAGE:
         c.nir->info.num_images = MAX2(c.nir->info.num_images, r->index + 1);
         BITSET_SET(c.nir->info.images_used, r->index);
         break;
      default:
         break;
      }
   }

   c.nir->scratch_size = c.scratch_size;
   c.nir->info.shared_size = c.shared_size;
   c.info->threadgroup_bytes = c.shared_size;
   c.nir->info.io_lowered = false;
   if (stage == MESA_SHADER_COMPUTE) {
      /* the threadgroup size is a dispatch argument, not a shader constant */
      c.nir->info.workgroup_size_variable = true;
   }

   nir_lower_goto_ifs(c.nir);

   /* Fold in any function of the module the shader called, and then drop the
    * callees. This has to come after nir_lower_goto_ifs and not before: this
    * translator emits an unstructured CFG, and nir_inline_functions walks the
    * control-flow tree, which asserts on one. The pass order inside is the one
    * nir_functions.c prescribes -- inline, then nir_opt_deref to undo the cast
    * a pointer parameter leaves behind, then delete everything but the entry
    * point, which every consumer of this NIR assumes.
    */
   if (_mesa_hash_table_num_entries(c.fns)) {
      NIR_PASS(_, c.nir, nir_inline_functions);
      NIR_PASS(_, c.nir, nir_opt_deref);
      nir_remove_non_entrypoints(c.nir);
      NIR_PASS(_, c.nir, nir_lower_variable_initializers, nir_var_function_temp);
      NIR_PASS(_, c.nir, nir_split_var_copies);
      NIR_PASS(_, c.nir, nir_lower_var_copies);
      NIR_PASS(_, c.nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, c.nir, nir_opt_dce);
   }

   nir_index_ssa_defs(nir_shader_get_entrypoint(c.nir));

   if (info) {
      /* the info's arrays live under mem_ctx, not under the scratch context */
      if (info->num_resources)
         info->resources = ralloc_memdup(mem_ctx, info->resources,
                                         info->num_resources * sizeof(*info->resources));
      if (info->num_varyings)
         info->varyings = ralloc_memdup(mem_ctx, info->varyings,
                                        info->num_varyings * sizeof(*info->varyings));
      info->entry_point = ralloc_strdup(mem_ctx, info->entry_point);
   }
   ralloc_free(ctx);
   if (error)
      *error = NULL;
   return c.nir;
}
