/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * The air.* functions. There are 125 distinct declarations in the corpus but
 * their names are <operation>.<type suffix>, so this is a table of about 50
 * operations in four families (docs/metal.md 9.2). The suffix is not trusted
 * for anything: the types come from the call's own LLVM types, and the suffix
 * is only what picks the handler.
 *
 * What each name means is not guessed: it is read off Apple's own compiler.
 * AGXCompilerCore carries the whole vocabulary as a table of names at
 * 0x16d082..0x16d400 of the library in this device's shared cache (the list is
 * in docs/air.md section 3.2), and that table distinguishes exactly the
 * functions the Metal Shading Language Specification defines -- round from
 * rint, exp from exp2, fmod from mod, pow from powr. Two of this file's
 * handlers were guesses before that table was read; one of them was wrong.
 */

#include "air_private.h"

#include <string.h>

#include "nir_builtin_builder.h"
#include "util/u_math.h"

/* the dot-separated parts of an air.* name, after "air." */
#define AIR_MAX_PARTS 8
struct air_name {
   unsigned num;
   char part[AIR_MAX_PARTS][48];   /* (normalize_function_constant_predicate: 37) */
};

static bool
split_name(const char *name, struct air_name *out)
{
   out->num = 0;
   const char *p = name + 4; /* past "air." */
   while (*p && out->num < AIR_MAX_PARTS) {
      const char *dot = strchr(p, '.');
      size_t len = dot ? (size_t)(dot - p) : strlen(p);
      if (len >= sizeof(out->part[0]))
         return false;
      memcpy(out->part[out->num], p, len);
      out->part[out->num][len] = 0;
      out->num++;
      if (!dot)
         return true;
      p = dot + 1;
   }
   return out->num < AIR_MAX_PARTS;
}

static bool
is_signed_part(const char *s)
{
   return !strcmp(s, "s");
}

/* -- the families ------------------------------------------------------- */

static bool
math_op(struct air_ctx *c, const struct air_inst *in, const struct air_name *n, bool *handled)
{
   const char *op = n->part[0];
   unsigned argc = in->num_operands - 1;

   /* the families this file does not handle are named before the arguments
    * are touched, so that their error says so rather than complaining about
    * a texture handle used as a number
    */
   if (strstr(op, "texture") || strstr(op, "_depth_") || !strncmp(op, "sample", 6) ||
       !strncmp(op, "gather", 6) || !strncmp(op, "read_", 5) || !strncmp(op, "write_", 6) ||
       strstr(op, "_lod_")) {
      *handled = false;
      return true;
   }

   nir_def *a[4] = { 0 };
   for (unsigned i = 0; i < argc && i < 4; i++)
      if (!(a[i] = air_get_ssa(c, in->operands[1 + i])))
         return false;

   bool is_float = air_type_is_float(air_type_scalar(in->base.type));
   /* the sign of an integer operation is the last-but-one part: air.min.s.i32 */
   bool is_signed = n->num >= 3 ? is_signed_part(n->part[1]) : true;
   nir_def *r = NULL;
   *handled = true;

   /* fast math is what every one of these means: Metal's "fast" library */
   bool was_exact = AIR_B_EXACT(c->b);
   AIR_B_SET_EXACT(c->b, false);

   if (!strcmp(op, "fast_fabs") || !strcmp(op, "fabs")) {
      r = nir_fabs(&c->b, a[0]);
   } else if (!strcmp(op, "fast_sqrt") || !strcmp(op, "sqrt")) {
      r = nir_fsqrt(&c->b, a[0]);
   } else if (!strcmp(op, "fast_rsqrt") || !strcmp(op, "rsqrt")) {
      r = nir_frsq(&c->b, a[0]);
   } else if (!strcmp(op, "fast_floor") || !strcmp(op, "floor")) {
      r = nir_ffloor(&c->b, a[0]);
   } else if (!strcmp(op, "fast_ceil") || !strcmp(op, "ceil")) {
      r = nir_fceil(&c->b, a[0]);
   } else if (!strcmp(op, "fast_round") || !strcmp(op, "round")) {
      /* round() takes a half away from zero; rint() takes it to even, and
       * Apple's compiler has both air.fast_round and air.fast_rint, so this
       * one is not the even one. (It was nir_fround_even here until the
       * vocabulary table said otherwise.)
       *
       * Truncate, then step one away from zero if the remainder reaches a
       * half. Not trunc(x + copysign(0.5, x)): that addition rounds for an x
       * just under a half and answers 1 where the answer is 0. x - trunc(x)
       * is exact for every float, so this needs no magnitude guard, and it
       * keeps the sign of a negative zero. Checked bit-exact against C's
       * roundf over the halves, the sub-halves, both zeros and 2^24.
       */
      nir_def *t = nir_ftrunc(&c->b, a[0]);
      nir_def *d = nir_fabs(&c->b, nir_fsub(&c->b, a[0], t));
      r = nir_bcsel(&c->b, nir_fge_imm(&c->b, d, 0.5),
                    nir_fadd(&c->b, t, nir_fsign(&c->b, a[0])), t);
   } else if (!strcmp(op, "fast_rint") || !strcmp(op, "rint")) {
      r = nir_fround_even(&c->b, a[0]);
   } else if (!strcmp(op, "fast_trunc") || !strcmp(op, "trunc")) {
      r = nir_ftrunc(&c->b, a[0]);
   } else if (!strcmp(op, "fast_fract") || !strcmp(op, "fract")) {
      r = nir_ffract(&c->b, a[0]);
   } else if (!strcmp(op, "fast_sin") || !strcmp(op, "sin")) {
      r = nir_fsin(&c->b, a[0]);
   } else if (!strcmp(op, "fast_cos") || !strcmp(op, "cos")) {
      r = nir_fcos(&c->b, a[0]);
   } else if (!strcmp(op, "fast_sinpi") || !strcmp(op, "sinpi")) {
      /* sin(pi x), cos(pi x), tan(pi x): the argument brought into one period first (x - 2 round(x / 2), in
       * [-1, 1]) so that large x keep their phase (mediaserverd's sparserendering_precalcLUTs: air.cospi.f16) */
      nir_def *t = nir_fsub(&c->b, a[0], nir_fmul_imm(&c->b, nir_fround_even(&c->b, nir_fmul_imm(&c->b, a[0], 0.5)), 2.0));
      r = nir_fsin(&c->b, nir_fmul_imm(&c->b, t, M_PI));
   } else if (!strcmp(op, "fast_cospi") || !strcmp(op, "cospi")) {
      nir_def *t = nir_fsub(&c->b, a[0], nir_fmul_imm(&c->b, nir_fround_even(&c->b, nir_fmul_imm(&c->b, a[0], 0.5)), 2.0));
      r = nir_fcos(&c->b, nir_fmul_imm(&c->b, t, M_PI));
   } else if (!strcmp(op, "fast_tanpi") || !strcmp(op, "tanpi")) {
      nir_def *t = nir_fmul_imm(&c->b, nir_fsub(&c->b, a[0], nir_fround_even(&c->b, a[0])), M_PI);
      r = nir_fdiv(&c->b, nir_fsin(&c->b, t), nir_fcos(&c->b, t));
   } else if (!strcmp(op, "fast_tan") || !strcmp(op, "tan")) {
      r = nir_fdiv(&c->b, nir_fsin(&c->b, a[0]), nir_fcos(&c->b, a[0]));
   } else if (!strcmp(op, "fast_atan2") || !strcmp(op, "atan2")) {
      /* Mesa's nir_atan2 is right everywhere but the origin, where its own
       * comment says it deliberately pretends 0/0 = 1 because GLSL licenses an
       * implementation to deviate from IEEE at (0, 0). That gives
       * atan2(0, 0) = 3*pi/4. Metal's atan2 is OpenCL's, which keeps the IEEE
       * values:
       *
       *    atan2(+-0, x > 0) = +-0        atan2(+-0, +0) = +-0
       *    atan2(+-0, x < 0) = +-pi       atan2(+-0, -0) = +-pi
       *
       * so a shader taking the angle of a vector that happens to be zero gets
       * 135 degrees instead of 0. That is not a precision deviation of the
       * kind the fast_ prefix allows, so the origin is handled here and the
       * rest left to Mesa.
       *
       * The sign of a zero cannot be had from a float comparison -- -0 == 0 --
       * so the sign bit is read as an integer. A NIR value is a bit vector and
       * ilt reinterprets it; nir_atan2 avoids this only to match the GLSL
       * front-end, which is not a constraint here.
       */
      nir_def *y = a[0], *x = a[1];
      nir_def *origin = nir_iand(&c->b, nir_feq_imm(&c->b, y, 0.0),
                                 nir_feq_imm(&c->b, x, 0.0));
      nir_def *mag = nir_fmul_imm(&c->b,
                                  nir_b2fN(&c->b, nir_ilt_imm(&c->b, x, 0),
                                           y->bit_size),
                                  M_PI);
      nir_def *at_origin =
         nir_bcsel(&c->b, nir_ilt_imm(&c->b, y, 0), nir_fneg(&c->b, mag), mag);
      r = nir_bcsel(&c->b, origin, at_origin, nir_atan2(&c->b, y, x));
   } else if (!strcmp(op, "fast_tanh") || !strcmp(op, "tanh") ||
              !strcmp(op, "fast_sinh") || !strcmp(op, "sinh") ||
              !strcmp(op, "fast_cosh") || !strcmp(op, "cosh")) {
      /* the hyperbolic ones from the exponential (MPS's batch normalisation with a tanh activation,
       * TARR_batchnorm_inference_kernel, was refused for air.fast_tanh.v4f32: mediaanalysisd on the Pixel 8a,
       * 11-13). tanh = (e^2x - 1) / (e^2x + 1) with x held to +-10, where a float's tanh is 1 already and e^2x
       * still far from infinity (and from a half's: 4.8e8 does not fit one, so a half is held to +-5) */
      if (strstr(op, "tanh")) {
         const double lim = a[0]->bit_size == 16 ? 5.0 : 10.0;
         nir_def *x = nir_fmin(&c->b, nir_fmax(&c->b, a[0], nir_imm_floatN_t(&c->b, -lim, a[0]->bit_size)),
                               nir_imm_floatN_t(&c->b, lim, a[0]->bit_size));
         nir_def *e = nir_fexp2(&c->b, nir_fmul_imm(&c->b, x, 2.0 * M_LOG2E));
         r = nir_fdiv(&c->b, nir_fadd_imm(&c->b, e, -1.0), nir_fadd_imm(&c->b, e, 1.0));
      } else {
         nir_def *e = nir_fexp2(&c->b, nir_fmul_imm(&c->b, a[0], M_LOG2E));
         nir_def *n = nir_fexp2(&c->b, nir_fmul_imm(&c->b, a[0], -M_LOG2E));
         r = nir_fmul_imm(&c->b, strstr(op, "sinh") ? nir_fsub(&c->b, e, n) : nir_fadd(&c->b, e, n), 0.5);
      }
   } else if (!strcmp(op, "fast_asin") || !strcmp(op, "asin")) {
      r = nir_asin(&c->b, a[0]);
   } else if (!strcmp(op, "fast_acos") || !strcmp(op, "acos")) {
      r = nir_acos(&c->b, a[0]);
   } else if (!strcmp(op, "fast_atan") || !strcmp(op, "atan")) {
      r = nir_atan(&c->b, a[0]);
   } else if (!strcmp(op, "fast_exp") || !strcmp(op, "exp")) {
      /* the natural one: Apple's compiler names exp and exp2 separately */
      r = nir_fexp2(&c->b, nir_fmul_imm(&c->b, a[0], M_LOG2E));
   } else if (!strcmp(op, "fast_exp2") || !strcmp(op, "exp2")) {
      r = nir_fexp2(&c->b, a[0]);
   } else if (!strcmp(op, "fast_exp10") || !strcmp(op, "exp10")) {
      r = nir_fexp2(&c->b, nir_fmul_imm(&c->b, a[0], 3.321928094887362)); /* log2(10) */
   } else if (!strcmp(op, "fast_log") || !strcmp(op, "log")) {
      r = nir_fmul_imm(&c->b, nir_flog2(&c->b, a[0]), 1.0 / M_LOG2E);
   } else if (!strcmp(op, "fast_log2") || !strcmp(op, "log2")) {
      r = nir_flog2(&c->b, a[0]);
   } else if (!strcmp(op, "fast_log10") || !strcmp(op, "log10")) {
      r = nir_fmul_imm(&c->b, nir_flog2(&c->b, a[0]), 0.30102999566398120); /* 1/log2(10) */
   } else if (!strcmp(op, "fast_pow") || !strcmp(op, "pow") ||
              !strcmp(op, "fast_powr") || !strcmp(op, "powr")) {
      /* powr is pow with x >= 0 promised, which is what nir_fpow computes */
      r = nir_fpow(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_fmod") || !strcmp(op, "fmod") ||
              !strcmp(op, "fast_mod") || !strcmp(op, "mod")) {
      r = nir_frem(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_divide") || !strcmp(op, "divide")) {
      r = nir_fdiv(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "saturate") || !strcmp(op, "fast_saturate")) {
      r = nir_fsat(&c->b, a[0]);
   } else if (!strcmp(op, "sign") && is_float) {
      /* Metal's sign is OpenCL's: sign(-0) is -0, sign(+0) is +0, sign(NaN)
       * is +0. nir_fsign will not promise that -- its own documentation says
       * that for a zero source "there is a preference for the result to have
       * the same sign, but this is not required (it is required by OpenCL)",
       * and likewise for NaN. A backend is therefore licensed to lose it, and
       * softpipe does: it answered +0 for -0 where NIR's own constant folder
       * answered -0, so folding the shader could never have shown this.
       *
       * Built from comparisons instead, which promise what Metal needs:
       * +1 where positive, -1 where negative, 0 for NaN because neither
       * comparison holds, and a zero handed back as it came so it keeps its
       * sign. No immediate is replicated, so this is the same for a vector.
       */
      nir_def *x = a[0];
      nir_def *pos = nir_b2fN(&c->b, nir_fgt_imm(&c->b, x, 0.0), x->bit_size);
      nir_def *neg = nir_b2fN(&c->b, nir_flt_imm(&c->b, x, 0.0), x->bit_size);
      r = nir_bcsel(&c->b, nir_feq_imm(&c->b, x, 0.0), x,
                    nir_fsub(&c->b, pos, neg));
   } else if (!strcmp(op, "sign")) {
      r = nir_isign(&c->b, a[0]);   /* the float case is above */
   } else if (!strcmp(op, "mix")) {
      /* mix(x, y, a) = x + a * (y - x), and that is what Metal defines */
      r = nir_fadd(&c->b, a[0],
                   nir_fmul(&c->b, a[2], nir_fsub(&c->b, a[1], a[0])));
   } else if (!strcmp(op, "dot")) {
      r = nir_fdot(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_fmin") || !strcmp(op, "fmin") || !strcmp(op, "min")) {
      r = is_float ? nir_fmin(&c->b, a[0], a[1])
                   : (is_signed ? nir_imin(&c->b, a[0], a[1]) : nir_umin(&c->b, a[0], a[1]));
   } else if (!strcmp(op, "fast_fmax") || !strcmp(op, "fmax") || !strcmp(op, "max")) {
      r = is_float ? nir_fmax(&c->b, a[0], a[1])
                   : (is_signed ? nir_imax(&c->b, a[0], a[1]) : nir_umax(&c->b, a[0], a[1]));
   } else if (!strcmp(op, "fast_fmin3") || !strcmp(op, "fmin3") || !strcmp(op, "min3") ||
              !strcmp(op, "fast_fmax3") || !strcmp(op, "fmax3") || !strcmp(op, "max3") ||
              !strcmp(op, "fast_fmedian3") || !strcmp(op, "fmedian3") || !strcmp(op, "median3")) {
      /* of three (11-12: the widgets' uber_fragment, air.fmax3.f16): the two-operand ones of this build composed */
      const bool lo = strstr(op, "min3") != NULL, mid = strstr(op, "median3") != NULL;
#define AIR_MIN2(x, y) (is_float ? nir_fmin(&c->b, x, y) : is_signed ? nir_imin(&c->b, x, y) : nir_umin(&c->b, x, y))
#define AIR_MAX2(x, y) (is_float ? nir_fmax(&c->b, x, y) : is_signed ? nir_imax(&c->b, x, y) : nir_umax(&c->b, x, y))
      if (!a[2]) {
         AIR_B_SET_EXACT(c->b, was_exact);
         return air_error(c, "intrinsic: %s without three operands", op);
      }
      if (mid)
         r = AIR_MAX2(AIR_MIN2(a[0], a[1]), AIR_MIN2(AIR_MAX2(a[0], a[1]), a[2]));
      else if (lo)
         r = AIR_MIN2(AIR_MIN2(a[0], a[1]), a[2]);
      else
         r = AIR_MAX2(AIR_MAX2(a[0], a[1]), a[2]);
#undef AIR_MIN2
#undef AIR_MAX2
   } else if (!strcmp(op, "fast_clamp") || !strcmp(op, "clamp")) {
      if (is_float)
         r = nir_fmin(&c->b, nir_fmax(&c->b, a[0], a[1]), a[2]);
      else if (is_signed)
         r = nir_imin(&c->b, nir_imax(&c->b, a[0], a[1]), a[2]);
      else
         r = nir_umin(&c->b, nir_umax(&c->b, a[0], a[1]), a[2]);
   } else if (!strcmp(op, "fast_ldexp") || !strcmp(op, "ldexp")) {
      r = nir_ldexp(&c->b, a[0], nir_i2i32(&c->b, a[1]));
   } else if (!strcmp(op, "fast_fma") || !strcmp(op, "fma") || !strcmp(op, "mad")) {
      r = nir_ffma(&c->b, a[0], a[1], a[2]);
   } else if (!strcmp(op, "normalize_function_constant_predicate")) {
      /* Metal's compiler keeps whether a dependent function constant applies
       * in __metal_implicit_fc_pred_N bytes, each put through this: a byte is
       * a predicate when it is not zero (tools/mesa/air's reading, ported)
       */
      r = nir_b2iN(&c->b, nir_ine_imm(&c->b, a[0], 0), air_nir_bit_size(c, in->base.type));
   } else if (!strcmp(op, "abs")) {
      r = is_float ? nir_fabs(&c->b, a[0]) : nir_iabs(&c->b, a[0]);
   } else if (!strcmp(op, "abs_diff")) {
      r = is_signed ? nir_iabs(&c->b, nir_isub(&c->b, a[0], a[1]))
                    : nir_isub(&c->b, nir_umax(&c->b, a[0], a[1]),
                               nir_umin(&c->b, a[0], a[1]));
   } else if (!strcmp(op, "add_sat")) {
      r = is_signed ? nir_iadd_sat(&c->b, a[0], a[1]) : nir_uadd_sat(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "sub_sat")) {
      r = is_signed ? nir_isub_sat(&c->b, a[0], a[1]) : nir_usub_sat(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "mul_hi")) {
      r = is_signed ? nir_imul_high(&c->b, a[0], a[1]) : nir_umul_high(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "popcount")) {
      r = nir_bit_count(&c->b, a[0]);
   } else if (!strcmp(op, "clz")) {
      r = nir_uclz(&c->b, a[0]);
   } else if (!strcmp(op, "ctz")) {
      /* Metal's ctz returns the type's width when the value is zero */
      nir_def *lsb = nir_find_lsb(&c->b, a[0]);
      r = nir_bcsel(&c->b, nir_ieq_imm(&c->b, a[0], 0),
                    nir_imm_intN_t(&c->b, a[0]->bit_size, a[0]->bit_size), lsb);
   } else if (!strcmp(op, "reverse_bits")) {
      r = nir_bitfield_reverse(&c->b, a[0]);
   } else if (!strcmp(op, "rotate")) {
      r = nir_urol(&c->b, a[0], nir_u2u32(&c->b, a[1]));
   } else if (!strcmp(op, "extract_bits")) {
      r = is_signed ? nir_ibitfield_extract(&c->b, a[0], nir_u2u32(&c->b, a[1]),
                                            nir_u2u32(&c->b, a[2]))
                    : nir_ubitfield_extract(&c->b, a[0], nir_u2u32(&c->b, a[1]),
                                            nir_u2u32(&c->b, a[2]));
   } else if (!strcmp(op, "insert_bits")) {
      r = nir_bitfield_insert(&c->b, a[0], a[1], nir_u2u32(&c->b, a[2]),
                              nir_u2u32(&c->b, a[3]));
   } else if (!strcmp(op, "dfdx")) {
      r = nir_ddx(&c->b, a[0]);
   } else if (!strcmp(op, "dfdy")) {
      r = nir_ddy(&c->b, a[0]);
   } else if (!strcmp(op, "fwidth")) {
      r = nir_fadd(&c->b, nir_fabs(&c->b, nir_ddx(&c->b, a[0])),
                   nir_fabs(&c->b, nir_ddy(&c->b, a[0])));
   } else if (!strcmp(op, "get_simd_width") || !strcmp(op, "get_thread_execution_width")) {
      r = nir_load_subgroup_size(&c->b);
   } else if (!strcmp(op, "is_uniform")) {
      /* air.is_uniform.<type>(x): asks whether x is the same in every thread of the SIMD group, so that a kernel can
       * take a shorter road (MetalPerformanceShaders' cnnConvArray_*, which mediaserverd asserts on when it does
       * not compile). "No" is always a right answer -- the general road is taken. (11-12: read from its use, not
       * from a specification) */
      r = nir_imm_zero(&c->b, air_nir_components(in->base.type), air_nir_bit_size(c, in->base.type));
   } else if (!strcmp(op, "all")) {
      r = nir_ball(&c->b, a[0]);
   } else if (!strcmp(op, "any")) {
      r = nir_bany(&c->b, a[0]);
   } else if (!strcmp(op, "convert")) {
      /* air.convert.<dst sign>.<dst type>.<src sign>.<src type> */
      if (n->num < 5) {
         AIR_B_SET_EXACT(c->b, was_exact);
         return air_error(c, "intrinsic: %s (a convert without both types)", op);
      }
      const struct air_type *dt = in->base.type;
      const struct air_type *st = in->operands[1]->type;
      unsigned bits = air_nir_bit_size(c, dt);
      bool dst_float = n->part[1][0] == 'f';
      bool src_float = n->part[3][0] == 'f';
      bool src_signed = n->part[3][0] == 's';
      nir_def *x = a[0];
      if (!src_float && air_type_bit_size(air_type_scalar(st)) == 1)
         x = src_signed ? nir_ineg(&c->b, nir_b2i32(&c->b, x)) : nir_b2i32(&c->b, x);
      if (!dst_float && bits == 1)
         /* to bool (air.convert.u.v2i1.s.v2i32: Metal's bool2(int2)): not zero. It was i2i1 / f2u1, conversions NIR has
          * no instruction for on the Mali -- "Unhandled ALU op i2i1", CoreAnimation's uber_fragment (11-13) */
         r = src_float ? nir_fneu_imm(&c->b, x, 0.0) : nir_ine_imm(&c->b, x, 0);
      else if (dst_float && src_float)
         r = nir_f2fN(&c->b, x, bits);
      else if (dst_float)
         r = src_signed ? nir_i2fN(&c->b, x, bits) : nir_u2fN(&c->b, x, bits);
      else if (src_float)
         r = n->part[1][0] == 's' ? nir_f2iN(&c->b, x, bits) : nir_f2uN(&c->b, x, bits);
      else
         r = src_signed ? nir_i2iN(&c->b, x, bits) : nir_u2uN(&c->b, x, bits);
   } else {
      *handled = false;
   }

   AIR_B_SET_EXACT(c->b, was_exact);
   if (!*handled)
      return true;
   if (!r)
      return false;
   air_set(c, &in->base, air_val_ssa(c, r));
   return true;
}

/* -- atomics ------------------------------------------------------------ */

static bool
atomic_op(struct air_ctx *c, const struct air_inst *in, const struct air_name *n)
{
   /* air.atomic.{global,local}.<op>[.<sign>].<type>(ptr, value?, order,
    * scope, volatile)
    */
   if (n->num < 3)
      return air_error(c, "intrinsic: an atomic without a name");
   const char *where = n->part[1];
   const char *op = n->part[2];
   bool is_signed = n->num >= 5 ? is_signed_part(n->part[3]) : true;

   struct air_val *pv = air_get(c, in->operands[1]);
   if (!pv)
      return false;
   if (pv->kind != AIR_VAL_PTR)
      return air_error(c, "intrinsic: an atomic on something that is not a pointer");
   struct air_ptr p = pv->ptr;

   bool shared = !strcmp(where, "local");
   if (shared != (p.class == AIR_PTR_SHARED))
      return air_error(c, "intrinsic: air.atomic.%s on %s memory", where,
                       p.class == AIR_PTR_SHARED ? "threadgroup" : "device");

   nir_def *off = p.offset ? p.offset : nir_imm_int(&c->b, 0);

   /* The memory order, which this lowering is right for exactly one value of.
    * That argument is Metal's memory_order, and every atomic in the corpus
    * passes 0 -- memory_order_relaxed. A relaxed load or store of an aligned
    * 32-bit word is a plain load or store, which is what happens below, and a
    * relaxed read-modify-write needs no barrier either. A stronger order would
    * need one that is not emitted, so rather than ignore the argument and be
    * quietly wrong on a shader that asks for more, refuse it by name. Read
    * through air_const_uint because a zero arrives as CST_CODE_NULL about as
    * often as as an integer.
    */
   unsigned iorder = !strcmp(op, "load") ? 2 : 3;
   uint64_t order;
   if (in->num_operands <= iorder ||
       !air_const_uint(in->operands[iorder], &order))
      return air_error(c, "intrinsic: air.atomic.%s.%s with a memory order that is "
                          "not a constant", where, op);
   if (order != 0)
      return air_error(c, "intrinsic: air.atomic.%s.%s with memory_order %u; only "
                          "relaxed (0) is translated, and a stronger one needs a "
                          "barrier this does not emit", where, op, (unsigned)order);

   if (!strcmp(op, "load")) {
      nir_def *r = air_load_ptr(c, p, in->base.type, 0);
      if (!r)
         return false;
      air_set(c, &in->base, air_val_ssa(c, r));
      return true;
   }
   if (!strcmp(op, "store")) {
      nir_def *v = air_get_ssa(c, in->operands[2]);
      return v && air_store_ptr(c, p, v, 0);
   }

   nir_atomic_op aop;
   if (!strcmp(op, "add")) aop = nir_atomic_op_iadd;
   else if (!strcmp(op, "min")) aop = is_signed ? nir_atomic_op_imin : nir_atomic_op_umin;
   else if (!strcmp(op, "max")) aop = is_signed ? nir_atomic_op_imax : nir_atomic_op_umax;
   else if (!strcmp(op, "and")) aop = nir_atomic_op_iand;
   else if (!strcmp(op, "or")) aop = nir_atomic_op_ior;
   else if (!strcmp(op, "xor")) aop = nir_atomic_op_ixor;
   else if (!strcmp(op, "xchg")) aop = nir_atomic_op_xchg;
   else return air_error(c, "intrinsic: atomic %s", op);

   nir_def *v = air_get_ssa(c, in->operands[2]);
   if (!v)
      return false;
   nir_def *r;
   if (shared) {
      r = nir_shared_atomic(&c->b, v->bit_size, off, v, .atomic_op = aop);
   } else if (p.class == AIR_PTR_SSBO) {
      r = nir_ssbo_atomic(&c->b, v->bit_size, ptr_binding(c, p), off, v,
                          .atomic_op = aop);
   } else {
      return air_error(c, "intrinsic: an atomic on a pointer that is not a buffer");
   }
   if (in->base.type && in->base.type->kind != AIR_TYPE_VOID)
      air_set(c, &in->base, air_val_ssa(c, r));
   return true;
}

/* -- barriers ----------------------------------------------------------- */

static bool
barrier_op(struct air_ctx *c, const struct air_inst *in)
{
   /* air.wg.barrier(i32 flags, i32 scope): the flags say which memory is
    * ordered. 2 is threadgroup memory, 1 device, 4 texture; the corpus only
    * ever passes 2 with scope 1 (the threadgroup).
    */
   uint64_t flags = 2;
   if (in->num_operands > 1)
      air_const_uint(in->operands[1], &flags); /* left alone when not readable */

   nir_variable_mode modes = 0;
   if (flags & 1)
      modes |= nir_var_mem_ssbo | nir_var_mem_global;
   if (flags & 2)
      modes |= nir_var_mem_shared;
   if (flags & 4)
      modes |= nir_var_image;
   if (!modes)
      modes = nir_var_mem_shared;

   nir_barrier(&c->b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = modes);
   return true;
}

/* -- llvm.* ------------------------------------------------------------- */

static bool
llvm_op(struct air_ctx *c, const struct air_inst *in, const char *name)
{
   if (!strncmp(name, "llvm.lifetime.", 14))
      return true; /* scope markers: nothing to emit */
   if (!strcmp(name, "llvm.assume") || !strncmp(name, "llvm.assume.", 12))
      return true; /* the optimiser's promise about a value: nothing to emit (MPS's cnnConvArray kernels) */
   if (!strncmp(name, "llvm.fmuladd.", 13) || !strncmp(name, "llvm.fma.", 9)) {
      /* a * b + c, fused or not as the target likes (widgets' primitive_gradient_fragment: llvm.fmuladd.f16) */
      nir_def *x = air_get_ssa(c, in->operands[1]);
      nir_def *y = air_get_ssa(c, in->operands[2]);
      nir_def *z = air_get_ssa(c, in->operands[3]);
      if (!x || !y || !z)
         return false;
      air_set(c, &in->base, air_val_ssa(c, nir_fadd(&c->b, nir_fmul(&c->b, x, y), z)));
      return true;
   }
   if (!strncmp(name, "llvm.memcpy.", 12)) {
      /* a copy between two of our pointers, of a constant length */
      struct air_val *dst = air_get(c, in->operands[1]);
      struct air_val *src = air_get(c, in->operands[2]);
      if (!dst || !src)
         return false;
      if (dst->kind != AIR_VAL_PTR || src->kind != AIR_VAL_PTR)
         return air_error(c, "intrinsic: memcpy that is not between pointers");
      uint64_t bytes;
      if (!air_const_uint(in->operands[3], &bytes))
         return air_error(c, "intrinsic: memcpy of a run-time length");
      /* in 4-byte pieces, which is what every memcpy in the corpus is a
       * multiple of (they copy float vectors out of constant buffers)
       */
      for (uint64_t off = bytes & ~3ull; off < bytes; off++) {
         /* (the last bytes of a length that is not whole words, one at a time) */
         struct air_ptr s = air_ptr_add(c, src->ptr, nir_imm_int(&c->b, off));
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         static const struct air_type u8 = { .kind = AIR_TYPE_INT, .bits = 8 };
         nir_def *v = air_load_ptr(c, s, &u8, 1);
         if (!v || !air_store_ptr(c, d, v, 1))
            return false;
      }
      bytes &= ~3ull;
      for (uint64_t off = 0; off < bytes; off += 4) {
         struct air_ptr s = air_ptr_add(c, src->ptr, nir_imm_int(&c->b, off));
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         static const struct air_type u32 = { .kind = AIR_TYPE_INT, .bits = 32 };
         nir_def *v = air_load_ptr(c, s, &u32, 4);
         if (!v || !air_store_ptr(c, d, v, 4))
            return false;
      }
      return true;
   }
   if (!strncmp(name, "llvm.memset.", 12)) {
      struct air_val *dst = air_get(c, in->operands[1]);
      nir_def *byte = air_get_ssa(c, in->operands[2]);
      if (!dst || !byte)
         return false;
      if (dst->kind != AIR_VAL_PTR)
         return air_error(c, "intrinsic: memset of something that is not a pointer");
      uint64_t bytes;
      if (!air_const_uint(in->operands[3], &bytes))
         return air_error(c, "intrinsic: memset of a run-time length");
      if (!nir_src_is_const(nir_src_for_ssa(byte)))
         return air_error(c, "intrinsic: memset of a run-time byte");
      uint32_t b = nir_src_as_uint(nir_src_for_ssa(byte)) & 0xff;
      uint32_t word = b | b << 8 | b << 16 | b << 24;
      for (uint64_t off = bytes & ~3ull; off < bytes; off++) {
         /* (the last bytes of a length that is not whole words: MPS's kernelComputeStage1) */
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         if (!air_store_ptr(c, d, nir_imm_intN_t(&c->b, b, 8), 1))
            return false;
      }
      bytes &= ~3ull;
      for (uint64_t off = 0; off < bytes; off += 4) {
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         if (!air_store_ptr(c, d, nir_imm_int(&c->b, word), 4))
            return false;
      }
      return true;
   }
   if (!strcmp(name, "llvm.trap"))
      return air_error(c, "intrinsic: llvm.trap");
   return air_error(c, "intrinsic: %s", name);
}

/* -- textures ------------------------------------------------------------
 *
 * The largest family and the one with the long tail. Its vocabulary is
 * Apple's, read off AGXCompilerCore the same way the arithmetic was
 * (docs/air.md section 3.2): the names there end in an underscore --
 * air.sample_texture_, air.gather_compare_depth_, air.get_num_mip_levels_texture_
 * -- because the dimensionality is appended to them. That is why the earlier
 * survey counted "about 117 get_*_texture_* variants": it is some twenty
 * operations crossed with nine dimensionalities, not a hundred separate
 * things.
 *
 * The signatures are measured from the corpus, not guessed:
 *
 *   sample_<kind>_<dim> (tex, samp, [plane,] coord, i1 has_offset, offset,
 *                        i1 lod_is_level, float lod_or_bias, float min_lod,
 *                        i32 flags)                  -> { <4 x T>, i8 }
 *   gather_<kind>_<dim> (tex, samp, coord, i1 has_offset, offset,
 *                        i32 component, i32 flags)   -> { <4 x T>, i8 }
 *   read_<kind>_<dim>   (tex, [plane,] coord, i32 lod_or_sample, i32 flags)
 *   write_<kind>_<dim>  (tex, coord, value, i32 lod, i32 flags)
 *   get_<what>_<kind>_<dim> (tex[, i32 lod])         -> i32
 *
 * The i1 before the level is read as "the float is a level, not a bias", so
 * (false, 0.0) is a plain sample -- which is what every one of the corpus's
 * sample calls passes. The level and bias forms are written here but no
 * module exercises them.
 *
 * The second field of the returned struct is Metal's sparse-residency byte.
 * No module in the corpus reads it (every extractvalue of a texture result
 * takes field 0), so it is produced as zero rather than invented.
 */

/* COORD with LAYER as one more component */
static nir_def *
nir_vec_with_layer(nir_builder *b, nir_def *coord, nir_def *layer)
{
   nir_def *comps[4];
   unsigned n = coord->num_components;

   for (unsigned i = 0; i < n; i++)
      comps[i] = nir_channel(b, coord, i);
   comps[n] = layer;
   return nir_vec(b, comps, n + 1);
}

struct air_tex_name {
   const char *op;        /* "sample", "gather", "read", "write", "get", "lod" */
   const char *query;     /* for "get": width, height, depth, array_size, ... */
   bool compare;          /* sample_compare / gather_compare */
   bool depth;            /* a depth texture rather than a colour one */
   enum glsl_sampler_dim dim;
   bool is_array, is_ms, is_buffer;
};

static bool
eat(const char **s, const char *prefix)
{
   size_t n = strlen(prefix);
   if (strncmp(*s, prefix, n))
      return false;
   *s += n;
   return true;
}

/* air.<op>_<kind>_<dim> -> the pieces, or false when it is not a texture name */
static bool
parse_tex_name(const char *name, struct air_tex_name *t)
{
   memset(t, 0, sizeof(*t));
   const char *s = name;

   if (eat(&s, "sample_compare_")) {
      t->op = "sample";
      t->compare = true;
   } else if (eat(&s, "sample_")) {
      t->op = "sample";
   } else if (eat(&s, "gather_compare_")) {
      t->op = "gather";
      t->compare = true;
   } else if (eat(&s, "gather_")) {
      t->op = "gather";
   } else if (eat(&s, "read_")) {
      t->op = "read";
   } else if (eat(&s, "write_")) {
      t->op = "write";
   } else if (eat(&s, "calculate_clamped_lod_") || eat(&s, "calculate_unclamped_lod_")) {
      t->op = "lod";
   } else if (eat(&s, "get_")) {
      t->op = "get";
      static const char *queries[] = {
         "num_mip_levels_", "num_samples_", "array_size_", "width_", "height_", "depth_",
      };
      static const char *names[] = {
         "num_mip_levels", "num_samples", "array_size", "width", "height", "depth",
      };
      for (unsigned i = 0; i < ARRAY_SIZE(queries); i++) {
         if (eat(&s, queries[i])) {
            t->query = names[i];
            break;
         }
      }
      if (!t->query)
         return false;
   } else {
      return false;
   }

   if (eat(&s, "texture_buffer_1d")) {
      t->dim = GLSL_SAMPLER_DIM_BUF;
      t->is_buffer = true;
      return *s == 0;
   }
   if (eat(&s, "depth_"))
      t->depth = true;
   else if (!eat(&s, "texture_"))
      return false;

   /* the dimensionality, longest first so that 2d_ms_array beats 2d */
   static const struct {
      const char *suffix;
      enum glsl_sampler_dim dim;
      bool array, ms;
   } dims[] = {
      { "cube_array", GLSL_SAMPLER_DIM_CUBE, true, false },
      { "2d_ms_array", GLSL_SAMPLER_DIM_MS, true, true },
      { "2d_array", GLSL_SAMPLER_DIM_2D, true, false },
      { "1d_array", GLSL_SAMPLER_DIM_1D, true, false },
      { "2d_ms", GLSL_SAMPLER_DIM_MS, false, true },
      { "cube", GLSL_SAMPLER_DIM_CUBE, false, false },
      { "1d", GLSL_SAMPLER_DIM_1D, false, false },
      { "2d", GLSL_SAMPLER_DIM_2D, false, false },
      { "3d", GLSL_SAMPLER_DIM_3D, false, false },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(dims); i++) {
      if (!strcmp(s, dims[i].suffix)) {
         t->dim = dims[i].dim;
         t->is_array = dims[i].array;
         t->is_ms = dims[i].ms;
         return true;
      }
   }
   return false;
}

static unsigned
coord_components_of(const struct air_tex_name *t)
{
   unsigned n = t->dim == GLSL_SAMPLER_DIM_1D ? 1 :
                t->dim == GLSL_SAMPLER_DIM_3D ? 3 :
                t->dim == GLSL_SAMPLER_DIM_CUBE ? 3 :
                t->dim == GLSL_SAMPLER_DIM_BUF ? 1 : 2;
   return n + (t->is_array ? 1 : 0);
}

/* the result type of a texture operation, from the LLVM type it returns */
static nir_alu_type
tex_dest_type(struct air_ctx *c, const struct air_type *ret, const struct air_name *n)
{
   const struct air_type *t = ret;
   if (t->kind == AIR_TYPE_STRUCT && t->num_members)
      t = t->members[0];
   const struct air_type *scalar = air_type_scalar(t);
   unsigned bits = air_type_bit_size(scalar);
   if (air_type_is_float(scalar))
      return nir_type_float | bits;
   /* the signedness is in the name, not in the type: .s or .u */
   for (unsigned i = 1; i < n->num; i++) {
      if (!strcmp(n->part[i], "s"))
         return nir_type_int | bits;
      if (!strcmp(n->part[i], "u"))
         return nir_type_uint | bits;
   }
   return nir_type_uint | bits;
}

/* The pointer a texture or sampler argument carries. */
static bool
tex_handle(struct air_ctx *c, const struct air_inst *in, unsigned operand,
           enum air_ptr_class want, struct air_ptr *out)
{
   struct air_val *v = air_get(c, in->operands[operand]);
   if (!v)
      return false;
   if (v->kind != AIR_VAL_PTR || v->ptr.class != want)
      return air_error(c, "intrinsic: argument %u is not a %s", operand - 1,
                       want == AIR_PTR_TEXTURE ? "texture" : "sampler");
   *out = v->ptr;
   return true;
}

/* a constant i1/i32 argument, which these intrinsics use as flags */
static bool
tex_flag(struct air_ctx *c, const struct air_inst *in, unsigned operand, uint64_t *out)
{
   if (!air_const_uint(in->operands[operand], out))
      return air_error(c, "intrinsic: argument %u is not a constant", operand - 1);
   return true;
}

/* the { value, residency } struct these return */
static void
set_tex_result(struct air_ctx *c, const struct air_inst *in, nir_def *value)
{
   const struct air_type *ret = in->base.type;
   if (ret->kind != AIR_TYPE_STRUCT) {
      air_set(c, &in->base, air_val_ssa(c, value));
      return;
    }
   struct air_val *agg = rzalloc(c->mem_ctx, struct air_val);
   agg->kind = AIR_VAL_AGG;
   agg->num_elems = ret->num_members;
   agg->elems = rzalloc_array(c->mem_ctx, struct air_val *, MAX2(ret->num_members, 1));
   agg->elems[0] = air_val_ssa(c, value);
   for (unsigned i = 1; i < ret->num_members; i++) {
      /* the sparse-residency byte: nothing in the corpus reads it */
      unsigned bits = air_nir_bit_size(c, ret->members[i]);
      agg->elems[i] = air_val_ssa(c, nir_imm_intN_t(&c->b, 0, bits));
   }
   air_set(c, &in->base, agg);
}


/* A query: a size, a level count or a sample count, of a texture or an image. */
static nir_def *
query_texture(struct air_ctx *c, const struct air_ptr *tex, const struct air_tex_name *t,
              nir_texop op, nir_def *lod, unsigned comps)
{
   if (tex->is_image) {
      nir_def *index = nir_imm_int(&c->b, tex->binding);
      switch (op) {
      case nir_texop_txs: {
         /* an image size has as many components as the image has dimensions;
          * the caller passes 0 for "as many as there are"
          */
         unsigned n = comps ? comps :
            (t->dim == GLSL_SAMPLER_DIM_1D || t->dim == GLSL_SAMPLER_DIM_BUF ? 1 :
             t->dim == GLSL_SAMPLER_DIM_3D ? 3 : 2) + (t->is_array ? 1 : 0);
         return nir_image_size(&c->b, n, 32, index, lod ? lod : nir_imm_int(&c->b, 0),
                               .image_dim = t->dim, .image_array = t->is_array);
      }
      case nir_texop_query_levels:
         return nir_image_levels(&c->b, 32, index, .image_dim = t->dim,
                                 .image_array = t->is_array);
      default:
         return nir_image_samples(&c->b, 32, index, .image_dim = t->dim,
                                  .image_array = t->is_array);
      }
   }

   nir_tex_instr *q = nir_tex_instr_create(c->nir, lod ? 1 : 0);
   q->op = op;
   q->sampler_dim = t->dim;
   q->is_array = t->is_array;
   q->dest_type = nir_type_int32;
   q->texture_index = tex->binding;
   q->sampler_index = 0;
   if (lod)
      q->src[0] = (nir_tex_src){ .src_type = nir_tex_src_lod, .src = nir_src_for_ssa(lod) };
   /* NIR fixes the size of a query's result itself -- txs is two components
    * for a 2D texture whatever the caller asks one of them for -- so it is
    * asked here and the component picked afterwards.
    */
   if (comps == 0)
      comps = nir_tex_instr_dest_size(q);
   nir_def_init(&q->instr, &q->def, comps, 32);
   nir_builder_instr_insert(&c->b, &q->instr);
   return &q->def;
}

/* is every component of this value a constant zero? */
static bool
all_zero(nir_def *def)
{
   if (!nir_src_is_const(nir_src_for_ssa(def)))
      return false;
   for (unsigned i = 0; i < def->num_components; i++)
      if (nir_src_comp_as_int(nir_src_for_ssa(def), i) != 0)
         return false;
   return true;
}

/* air.get_null_texture_*: the handle of no texture, a binding no [[texture(n)]]
 * has. Metal reads and samples of a null texture give zero, so every use of it
 * folds to zeros here and no slot is bound (tools/mesa/air binds a 1x1 zero
 * texture at a reserved index instead). CoreAnimation's generic uber shaders
 * pick it, on a function constant, for the textures a pipeline does not use.
 */
#define AIR_NULL_TEXTURE 0xfffeu

/* texture_op's "is the number a level" flag, forced (0, 1) while it translates one sample both ways; -1: as the call says */
static __thread int tex_level_forced = -1;

static bool
texture_op(struct air_ctx *c, const struct air_inst *in, const struct air_name *n)
{
   if (!strncmp(n->part[0], "get_null_texture", 16)) {
      struct air_ptr p = { .class = AIR_PTR_TEXTURE, .binding = AIR_NULL_TEXTURE };
      air_set(c, &in->base, air_val_ptr(c, p));
      return true;
   }
   if (!strncmp(n->part[0], "is_null_texture", 15)) {
      struct air_val *tv = air_get(c, in->operands[1]);
      if (!tv)
         return false;
      if (tv->kind != AIR_VAL_PTR || tv->ptr.class != AIR_PTR_TEXTURE)
         return air_error(c, "intrinsic: air.%s of something that is not a texture", n->part[0]);
      air_set(c, &in->base, air_val_ssa(c, nir_imm_bool(&c->b, tv->ptr.binding == AIR_NULL_TEXTURE)));
      return true;
   }
   struct air_tex_name t;
   if (!parse_tex_name(n->part[0], &t))
      return air_error(c, "intrinsic: air.%s (a texture name this cannot take apart)",
                       n->part[0]);
   if (t.is_buffer)
      return air_error(c, "intrinsic: air.%s (texture buffers, not yet)", n->part[0]);
   if (t.compare)
      return air_error(c, "intrinsic: air.%s (compare sampling, not yet)", n->part[0]);

   struct air_ptr tex;
   if (!tex_handle(c, in, 1, AIR_PTR_TEXTURE, &tex))
      return false;
   if (tex.binding == AIR_NULL_TEXTURE) {
      const struct air_type *ret = in->base.type;
      if (!ret || ret->kind == AIR_TYPE_VOID)
         return true;    /* a write to no texture */
      const struct air_type *vt = ret->kind == AIR_TYPE_STRUCT ? ret->members[0] : ret;
      set_tex_result(c, in, nir_imm_zero(&c->b, air_nir_components(vt), air_nir_bit_size(c, vt)));
      return true;
   }

   unsigned arg = 2;   /* operands[0] is the callee, [1] the texture */
   struct air_ptr samp = { 0 };
   bool have_sampler = !strcmp(t.op, "sample") || !strcmp(t.op, "gather") ||
                       !strcmp(t.op, "lod");
   if (have_sampler) {
      if (!tex_handle(c, in, arg, AIR_PTR_SAMPLER, &samp))
         return false;
      arg++;
   }

   /* A depth texture takes a plane or aspect selector before the coordinate,
    * and every call in the corpus passes 1. What else it can be is not known
    * from the data, so anything else is refused rather than assumed.
    */
   if (t.depth) {
      uint64_t plane;
      if (!tex_flag(c, in, arg, &plane))
         return false;
      if (plane != 1)
         return air_error(c, "intrinsic: air.%s with aspect %u (only 1 is understood)",
                          n->part[0], (unsigned)plane);
      arg++;
   }

   unsigned ncoord = coord_components_of(&t);
   nir_alu_type dest_type = tex_dest_type(c, in->base.type, n);
   unsigned dest_bits = nir_alu_type_get_type_size(dest_type);

   if (!strcmp(t.op, "get")) {
      /* the queries: a size, a level count or a sample count */
      nir_def *res;
      if (!strcmp(t.query, "num_samples")) {
         res = query_texture(c, &tex, &t, nir_texop_texture_samples, NULL, 1);
      } else if (!strcmp(t.query, "num_mip_levels")) {
         res = query_texture(c, &tex, &t, nir_texop_query_levels, NULL, 1);
      } else {
         nir_def *lod = nir_imm_int(&c->b, 0);
         if (in->num_operands > arg) {
            nir_def *given = air_get_ssa(c, in->operands[arg]);
            if (!given)
               return false;
            lod = nir_u2u32(&c->b, given);
         }
         nir_def *size = query_texture(c, &tex, &t, nir_texop_txs, lod, 0);
         unsigned comp = !strcmp(t.query, "width") ? 0 :
                         !strcmp(t.query, "height") ? 1 :
                         !strcmp(t.query, "depth") ? 2 : size->num_components - 1;
         if (!strcmp(t.query, "array_size") && !t.is_array)
            return air_error(c, "intrinsic: air.%s on a texture that is not an array",
                             n->part[0]);
         if (comp >= size->num_components)
            return air_error(c, "intrinsic: air.%s asks for component %u of %u",
                             n->part[0], comp, size->num_components);
         res = nir_channel(&c->b, size, comp);
      }
      air_set(c, &in->base, air_val_ssa(c, nir_u2uN(&c->b, res, dest_bits)));
      return true;
   }

   if (!strcmp(t.op, "write")) {
      nir_def *coord = air_get_ssa(c, in->operands[arg++]);
      if (!coord)
         return false;
      if (t.is_array && coord->num_components + 1 == ncoord) {
         /* an array's layer is an operand of its own after the coordinate (below) */
         nir_def *layer = air_get_ssa(c, in->operands[arg++]);
         if (!layer)
            return false;
         coord = nir_vec_with_layer(&c->b, nir_u2u32(&c->b, coord), nir_u2u32(&c->b, layer));
      }
      nir_def *value = air_get_ssa(c, in->operands[arg++]);
      if (!value)
         return false;
      uint64_t lod;
      if (!tex_flag(c, in, arg, &lod))
         return false;
      if (lod != 0)
         return air_error(c, "intrinsic: air.%s to mip level %u (only 0 is handled)",
                          n->part[0], (unsigned)lod);
      if (!tex.is_image)
         return air_error(c, "intrinsic: air.%s to a texture the interface says is not "
                             "written", n->part[0]);
      nir_image_store(&c->b, nir_imm_int(&c->b, tex.binding),
                      nir_pad_vector(&c->b, nir_u2u32(&c->b, coord), 4),
                      nir_imm_int(&c->b, 0), nir_pad_vector(&c->b, value, 4),
                      nir_imm_int(&c->b, 0),
                      .image_dim = t.dim, .image_array = t.is_array,
                      .src_type = dest_type);
      return true;
   }

   /* sample, gather, read: a texture instruction, or an image load */
   nir_def *coord = air_get_ssa(c, in->operands[arg++]);
   if (!coord)
      return false;
   bool coord_is_float = air_type_is_float(air_type_scalar(in->operands[arg - 1]->type));
   if (t.is_array && coord->num_components + 1 == ncoord) {
      /* AN ARRAY'S LAYER: Metal's compiler passes it as an i32 operand of its own after the coordinate
       * (air.sample_texture_2d_array(texture, sampler, <2 x float>, i32 layer, ...), read_ and write_ alike); NIR
       * has it as the coordinate's last component, a float beside float coordinates (mediaserverd's
       * colorCubeFragment2D: "a 2-component coordinate, wanting 3") */
      nir_def *layer = air_get_ssa(c, in->operands[arg++]);
      if (!layer)
         return false;
      layer = nir_u2u32(&c->b, layer);
      if (coord_is_float)
         layer = nir_u2fN(&c->b, layer, coord->bit_size);
      else
         coord = nir_u2u32(&c->b, coord);
      coord = nir_vec_with_layer(&c->b, coord, layer);
   }
   if (coord->num_components != ncoord)
      return air_error(c, "intrinsic: air.%s with a %u-component coordinate, wanting %u",
                       n->part[0], coord->num_components, ncoord);

   if (!strcmp(t.op, "read") && tex.is_image) {
      uint64_t extra;
      if (!tex_flag(c, in, arg, &extra))
         return false;
      nir_def *res = nir_image_load(&c->b, 4, dest_bits, nir_imm_int(&c->b, tex.binding),
                                    nir_pad_vector(&c->b, nir_u2u32(&c->b, coord), 4),
                                    nir_imm_int(&c->b, t.is_ms ? extra : 0),
                                    nir_imm_int(&c->b, t.is_ms ? 0 : extra),
                                    .image_dim = t.dim, .image_array = t.is_array,
                                    .dest_type = dest_type);
      set_tex_result(c, in, res);
      return true;
   }

   nir_tex_src srcs[6];
   unsigned nsrc = 0;
   srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_coord,
                                 .src = nir_src_for_ssa(coord_is_float ? coord : nir_u2u32(&c->b, coord)) };

   nir_texop op;
   if (!strcmp(t.op, "sample")) {
      uint64_t has_offset = 0, is_level;
      /* (a cube's sample has no offset: air.sample_texture_cube(texture, sampler, float3, i1 level?, float, float,
       * i32) -- read as a 2D one's, its level took the offset's place and "argument 5 is not a constant" refused the
       * shader: the WebGL aquarium's sky reflections, 11-13. Told by the operand itself: an offset is a vector of
       * integers after an i1.) */
      const struct air_type *after = arg + 1 < in->num_operands ? in->operands[arg + 1]->type : NULL;
      const bool offset_there = after != NULL && (after->kind == AIR_TYPE_VECTOR ? after->elem->kind == AIR_TYPE_INT
                                                                                  : after->kind == AIR_TYPE_INT && after->bits != 1);
      if (offset_there) {
         if (!tex_flag(c, in, arg, &has_offset))
            return false;
         arg++;
         nir_def *offset = air_get_ssa(c, in->operands[arg++]);
         if (!offset)
            return false;
         if (has_offset && !all_zero(offset))
            srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_offset,
                                          .src = nir_src_for_ssa(nir_u2u32(&c->b, offset)) };
      }
      if (tex_level_forced >= 0) {
         is_level = (uint64_t)tex_level_forced;       /* (the second and third time through: see below) */
      } else if (!air_const_uint(in->operands[arg], &is_level)) {
         /* (11-13) WHETHER THE NUMBER IS A LEVEL OR A BIAS, KNOWN ONLY WHILE THE SHADER RUNS -- LLVM folds two sample
          * calls of a GL-on-Metal layer's texture helper into one with the flag a value (WebKit's WebGL through ANGLE:
          * the aquarium's ground and its inner shader were refused, "argument 5 is not a constant"). Sampled both
          * ways, the flag choosing between the two results. */
         nir_def *flag = air_get_ssa(c, in->operands[arg]);
         nir_def *both[2];
         if (!flag)
            return false;
         for (int way = 1; way >= 0; way--) {
            tex_level_forced = way;
            const bool ok = texture_op(c, in, n);
            tex_level_forced = -1;
            struct air_val *v = ok ? air_get(c, &in->base) : NULL;
            if (!v)
               return false;
            both[way] = v->kind == AIR_VAL_AGG ? v->elems[0]->def : v->def;
         }
         set_tex_result(c, in, nir_bcsel(&c->b, flag->bit_size == 1 ? flag : nir_ine_imm(&c->b, flag, 0), both[1], both[0]));
         return true;
      }
      arg++;
      nir_def *level = air_get_ssa(c, in->operands[arg++]);
      nir_def *min_lod = air_get_ssa(c, in->operands[arg++]);
      if (!level || !min_lod)
         return false;
      bool level_is_zero = nir_src_is_const(nir_src_for_ssa(level)) &&
                           nir_src_as_float(nir_src_for_ssa(level)) == 0.0;
      if (is_level) {
         op = nir_texop_txl;
         srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_lod,
                                       .src = nir_src_for_ssa(level) };
      } else if (!level_is_zero) {
         op = nir_texop_txb;
         srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_bias,
                                       .src = nir_src_for_ssa(level) };
      } else {
         op = nir_texop_tex;
      }
      if (!(nir_src_is_const(nir_src_for_ssa(min_lod)) &&
            nir_src_as_float(nir_src_for_ssa(min_lod)) == 0.0))
         srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_min_lod,
                                       .src = nir_src_for_ssa(min_lod) };
   } else if (!strcmp(t.op, "gather")) {
      uint64_t has_offset;
      if (!tex_flag(c, in, arg, &has_offset))
         return false;
      arg++;
      nir_def *offset = air_get_ssa(c, in->operands[arg++]);
      if (!offset)
         return false;
      if (has_offset && !all_zero(offset))
         srcs[nsrc++] = (nir_tex_src){ .src_type = nir_tex_src_offset,
                                       .src = nir_src_for_ssa(nir_u2u32(&c->b, offset)) };
      op = nir_texop_tg4;
   } else {
      /* read: a fetch by integer coordinate, with a level or a sample. Neither
       * has to be a constant -- read_texture_2d_ms is called with a computed
       * sample index throughout AppleMetalGLRenderer.
       */
      nir_def *which = air_get_ssa(c, in->operands[arg++]);
      if (!which)
         return false;
      which = nir_u2u32(&c->b, which);
      op = t.is_ms ? nir_texop_txf_ms : nir_texop_txf;
      srcs[nsrc++] = (nir_tex_src){
         .src_type = t.is_ms ? nir_tex_src_ms_index : nir_tex_src_lod,
         .src = nir_src_for_ssa(which),
      };
   }

   nir_tex_instr *tex_instr = nir_tex_instr_create(c->nir, nsrc);
   tex_instr->op = op;
   tex_instr->sampler_dim = t.dim;
   tex_instr->is_array = t.is_array;
   tex_instr->is_shadow = false;
   tex_instr->dest_type = dest_type;
   tex_instr->coord_components = ncoord;
   tex_instr->texture_index = tex.binding;
   tex_instr->sampler_index = have_sampler ? samp.binding : 0;
   if (op == nir_texop_tg4) {
      uint64_t component;
      if (!tex_flag(c, in, arg, &component))
         return false;
      tex_instr->component = component;
   }
   for (unsigned i = 0; i < nsrc; i++)
      tex_instr->src[i] = srcs[i];

   const struct air_type *value_type = in->base.type;
   if (value_type->kind == AIR_TYPE_STRUCT && value_type->num_members)
      value_type = value_type->members[0];

   /* The width of the result is NIR's to decide: a sample is four components
    * whatever the shader does with them. Metal's depth2d.sample() returns one
    * float, which is channel 0 of those four -- the same thing GLSL gets from
    * a depth texture read without a comparison.
    */
   unsigned nir_comps = nir_tex_instr_dest_size(tex_instr);
   nir_def_init(&tex_instr->instr, &tex_instr->def, nir_comps, dest_bits);
   nir_builder_instr_insert(&c->b, &tex_instr->instr);

   nir_def *res = &tex_instr->def;
   unsigned want = air_nir_components(value_type);
   if (want < nir_comps)
      res = nir_trim_vector(&c->b, res, want);
   else if (want > nir_comps)
      return air_error(c, "intrinsic: air.%s gives %u components, the shader wants %u",
                       n->part[0], nir_comps, want);
   set_tex_result(c, in, res);
   return true;
}

/* -- function constants ------------------------------------------------- */

/* air.is_function_constant_defined(ptr to the constant's global): whether the
 * pipeline supplied a value for it -- known at translation time (it is in
 * air_to_nir_options.constants or not), so it folds to a constant and the
 * branch it guards with it. Handled before the math ops: its argument is a
 * pointer. (Ported from tools/mesa/air; CoreAnimation's generic uber shaders
 * ask it of every constant, 11-09.)
 */
static bool
fc_defined_op(struct air_ctx *c, const struct air_inst *in)
{
   if (in->num_operands != 2)
      return air_error(c, "intrinsic: air.is_function_constant_defined of %u arguments",
                       in->num_operands - 1);
   struct air_val *p = air_get(c, in->operands[1]);
   if (!p)
      return false;
   if (p->kind != AIR_VAL_PTR || p->ptr.class != AIR_PTR_FUNCTION_CONSTANT)
      return air_error(c, "intrinsic: air.is_function_constant_defined of something "
                          "that is not a function constant");
   bool defined = false;
   for (unsigned i = 0; c->options && i < c->options->num_constants; i++)
      if (c->options->constants[i].index == p->ptr.binding)
         defined = true;
   air_set(c, &in->base, air_val_ssa(c, nir_imm_bool(&c->b, defined)));
   return true;
}

/* -- the dispatcher ----------------------------------------------------- */

bool
air_translate_call(struct air_ctx *c, const struct air_inst *in, const char *name)
{
   if (!strncmp(name, "llvm.", 5))
      return llvm_op(c, in, name);

   if (strncmp(name, "air.", 4))
      return air_call_in_module(c, in, name);

   struct air_name n;
   if (!split_name(name, &n))
      return air_error(c, "intrinsic: %s (a name this cannot take apart)", name);

   if (!strcmp(n.part[0], "is_function_constant_defined"))
      return fc_defined_op(c, in);
   if (n.num >= 2 && !strcmp(n.part[0], "wg") && !strcmp(n.part[1], "barrier"))
      return barrier_op(c, in);
   /* air.simdgroup.barrier(flags, scope): the threads of one SIMD group wait for each other. The threadgroup's
    * barrier makes every thread of the threadgroup wait, the SIMD group's among them -- stronger, and right as long
    * as every thread of the threadgroup comes to it (MPS's cnnConvArray kernels: straight-line, or loops of one
    * count for all). (11-12: not a narrower barrier of the Mali's; none is known) */
   if (n.num >= 2 && !strcmp(n.part[0], "simdgroup") && !strcmp(n.part[1], "barrier"))
      return barrier_op(c, in);
   if (n.num >= 1 && !strcmp(n.part[0], "discard_fragment")) {
      /* Metal's discard_fragment(): the fragment writes nothing from here on, the function runs on (its
       * derivatives stay defined) -- NIR's demote (the widgets' mask_color_fragment and mask_accumulate_fragment) */
      if (c->nir->info.stage != MESA_SHADER_FRAGMENT)
         return air_error(c, "intrinsic: air.discard_fragment outside a fragment function");
      nir_demote(&c->b);
      return true;
   }
   if (!strcmp(n.part[0], "atomic"))
      return atomic_op(c, in, &n);

   bool handled = false;
   if (!math_op(c, in, &n, &handled))
      return false;
   if (handled)
      return true;

   const char *op = n.part[0];
   if (strstr(op, "texture") || strstr(op, "_depth_") || !strncmp(op, "sample", 6) ||
       !strncmp(op, "gather", 6) || !strncmp(op, "read_", 5) || !strncmp(op, "write_", 6) ||
       strstr(op, "_lod_"))
      return texture_op(c, in, &n);

   /* (from upstream's tools/mesa/air) air.pack.{u,s}norm{4x8,2x16}.v<N>f{16,32} and air.unpack the other way: Metal's
    * pack_float_to_unorm4x8 and its kin, which are GLSL's packUnorm4x8 and its kin -- clamp, scale, round to nearest,
    * component 0 in the low bits -- and so NIR's pack_unorm_4x8 and the rest, over 32-bit floats (a half vector is
    * widened first, exactly). CoreAnimation's fixed_frag_lph_cpf unpacks its colours with the half variant.
    */
   if ((!strcmp(op, "pack") || !strcmp(op, "unpack")) && n.num >= 2) {
      const char *f = n.part[1];
      const bool un = !strcmp(f, "unorm4x8") || !strcmp(f, "unorm2x16");
      const bool sn = !strcmp(f, "snorm4x8") || !strcmp(f, "snorm2x16");
      if (un || sn) {
         const bool x8 = strstr(f, "4x8") != NULL;
         if (!strcmp(op, "pack")) {
            nir_def *v = in->num_operands > 1 ? air_get_ssa(c, in->operands[1]) : NULL;
            if (!v)
               return false;
            if (v->bit_size != 32)
               v = nir_f2f32(&c->b, v);
            nir_def *r = x8 ? (un ? nir_pack_unorm_4x8(&c->b, v) : nir_pack_snorm_4x8(&c->b, v))
                            : (un ? nir_pack_unorm_2x16(&c->b, v) : nir_pack_snorm_2x16(&c->b, v));
            air_set(c, &in->base, air_val_ssa(c, r));
            return true;
         }
         nir_def *w = in->num_operands > 1 ? air_get_ssa(c, in->operands[1]) : NULL;
         if (!w)
            return false;
         nir_def *r = x8 ? (un ? nir_unpack_unorm_4x8(&c->b, w) : nir_unpack_snorm_4x8(&c->b, w))
                         : (un ? nir_unpack_unorm_2x16(&c->b, w) : nir_unpack_snorm_2x16(&c->b, w));
         if (air_nir_bit_size(c, in->base.type) == 16)
            r = nir_f2f16(&c->b, r);
         air_set(c, &in->base, air_val_ssa(c, r));
         return true;
      }
   }

   return air_error(c, "intrinsic: %s", name);
}
