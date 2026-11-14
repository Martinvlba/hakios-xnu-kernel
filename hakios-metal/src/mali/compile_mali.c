/*
 * The Mali build's shader compiler: a Metal compute function's AIR compiled for the Pixel 8a's Mali-G715 (see
 * compile_mali.h for what the code expects of the dispatch). hakios-kernel's tools/mesa/pan/air_pan.c as a library:
 *
 *   AIR -> air_to_nir (tools/mesa/air, the function constants applied)
 *       -> Metal's buffers made addresses: load_ubo/load_ssbo/store_ssbo/ssbo_atomic* become global accesses at
 *          base(n) + offset, base(n) the push-constant words 2n, 2n+1
 *       -> a thread past the grid returns at once (dispatchThreads: with a last threadgroup that is not full: the
 *          hardware runs whole threadgroups, Metal runs only the grid's threads)
 *       -> pan_preprocess_nir, pan_postprocess_nir
 *       -> threads_per_threadgroup and threadgroups_per_grid from push constants (as panvk's sysvals)
 *       -> pan_shader_compile, GPU_ID 0xb8a21020, the first MALI_KERNEL_RESERVED push-constant words the driver's
 *
 * Threadgroup memory (wls_size, the static arrays; a [[threadgroup(0)]] argument lies past them, sized by the encoder)
 * and spilled registers (tls_size, per thread) are the dispatch's to provide: c_api_mali.cpp's local storage descriptor.
 *
 * Built with Mesa's own flags for its iOS build (hakios-kernel's xnu-deps/build/mesa26/out-ios/cflags, written by
 * tools/mesa/pan/mk-pan-ios.sh) plus -DAIR_MESA_NEW_NIR; `make ios-mali` reads them.
 */
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_lower_blend.h"
#include <dlfcn.h>
#include <sys/ucontext.h>
#include "compiler/glsl_types.h"
#include "compiler/air/nir_air.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/compiler/pan_nir.h"
#include "panfrost/model/pan_model.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"

#include "compile_mali.h"

#define GPU_ID          0xb8a21020ULL

static void
say(char *error, size_t error_size, const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	vsnprintf(error, error_size, format, ap);
	va_end(ap);
}

/* the colour attachments' formats for this thread's fragment compile (mali_fragment_compile's blend; NULL: RGBA8) */
static __thread const struct mali_shader_blend *blend_formats;

static nir_def *
push_words(nir_builder *b, unsigned word, unsigned count, unsigned bit_size)
{
	nir_def *c[3];

	for (unsigned i = 0; i < count; i++) {
		c[i] = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 4 * (word + i)), .base = 0);
	}
	nir_def *v = nir_vec(b, c, count);
	return bit_size == 32 ? v : nir_u2uN(b, v, bit_size);
}

/* [[buffer(n)]]'s slot in the push constants, for this thread's compile (compile_stage sets it; the driver's own
 * shaders use indices under MALI_KERNEL_BUFFERS, which stand for themselves) */
static __thread uint8_t slot_of[32];
static __thread bool slots_set;
static __thread uint32_t buffers_used;         /* bit n: [[buffer(n)]] is one of the function's */

static nir_def *
buffer_address(nir_builder *b, nir_def *index, nir_def *offset)
{
	if (slots_set && nir_src_is_const(nir_src_for_ssa(index)) && nir_src_as_uint(nir_src_for_ssa(index)) < 32) {
		index = nir_imm_int(b, slot_of[nir_src_as_uint(nir_src_for_ssa(index))]);
	}
	nir_def *base;

	if (slots_set && !nir_src_is_const(nir_src_for_ssa(index))) {
		/* A BUFFER CHOSEN WHILE THE SHADER RUNS (the front end's binding_def: a phi or select between two
		 * buffers, MPS's cnnConvArray kernels): its address chosen among the function's buffers' */
		base = nir_imm_int64(b, 0);
		for (unsigned n = 0; n < 31; n++) {
			if (buffers_used & 1u << n) {
				base = nir_bcsel(b, nir_ieq_imm(b, index, n),
				    nir_load_push_constant(b, 1, 64, nir_imm_int(b, slot_of[n] * 8), .base = 0), base);
			}
		}
	} else {
		base = nir_load_push_constant(b, 1, 64, nir_imul_imm(b, index, 8), .base = 0);
	}
	return nir_iadd(b, base, nir_u2u64(b, offset));
}

/* Metal's buffers -> global accesses at the buffer's address */
static bool
lower_buffers(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	nir_def *address, *value;

	(void)data;
	b->cursor = nir_before_instr(&in->instr);
	switch (in->intrinsic) {
	case nir_intrinsic_load_ubo:
	case nir_intrinsic_load_ssbo:
		address = buffer_address(b, in->src[0].ssa, in->src[1].ssa);
		value = nir_load_global(b, in->def.num_components, in->def.bit_size, address,
		    .align_mul = nir_intrinsic_align_mul(in), .align_offset = nir_intrinsic_align_offset(in));
		break;
	case nir_intrinsic_store_ssbo:
		address = buffer_address(b, in->src[1].ssa, in->src[2].ssa);
		nir_store_global(b, in->src[0].ssa, address, .write_mask = nir_intrinsic_write_mask(in),
		    .align_mul = nir_intrinsic_align_mul(in), .align_offset = nir_intrinsic_align_offset(in));
		nir_instr_remove(&in->instr);
		return true;
	case nir_intrinsic_ssbo_atomic:
		address = buffer_address(b, in->src[0].ssa, in->src[1].ssa);
		value = nir_global_atomic(b, in->def.bit_size, address, in->src[2].ssa,
		    .atomic_op = nir_intrinsic_atomic_op(in));
		break;
	case nir_intrinsic_ssbo_atomic_swap:
		address = buffer_address(b, in->src[0].ssa, in->src[1].ssa);
		value = nir_global_atomic_swap(b, in->def.bit_size, address, in->src[2].ssa, in->src[3].ssa,
		    .atomic_op = nir_intrinsic_atomic_op(in));
		break;
	default:
		return false;
	}
	nir_def_rewrite_uses(&in->def, value);
	nir_instr_remove(&in->instr);
	return true;
}

/* the threadgroup's size and the grid's, as panvk's sysvals: from the push constants the dispatch writes */
static bool
lower_sysvals(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	nir_def *value;

	(void)data;
	b->cursor = nir_before_instr(&in->instr);
	switch (in->intrinsic) {
	case nir_intrinsic_load_workgroup_size:
		value = push_words(b, MALI_KERNEL_WORKGROUP_WORD, 3, in->def.bit_size);
		break;
	case nir_intrinsic_load_num_workgroups:
		value = push_words(b, MALI_KERNEL_GROUPS_WORD, 3, in->def.bit_size);
		break;
	case nir_intrinsic_load_base_workgroup_id:
		value = nir_imm_zero(b, 3, in->def.bit_size);
		break;
	default:
		return false;
	}
	nir_def_rewrite_uses(&in->def, value);
	nir_instr_remove(&in->instr);
	return true;
}

/*
 * Textures, samplers and images as Valhall addresses them: a handle `table << 24 | index` into the resource table the
 * dispatch gives in r0 -- gallium panfrost's pan_nir_lower_res_indices.c for arch >= 9 (MIT), with its numbering of the
 * tables. A texel fetch needs no sampler, but the hardware wants a valid one: sampler 0 (the dispatch binds one).
 */
static bool
lower_res_indices(nir_builder *b, nir_instr *instr, void *data)
{
	(void)data;
	b->cursor = nir_before_instr(instr);
	if (instr->type == nir_instr_type_tex) {
		nir_tex_instr *tex = nir_instr_as_tex(instr);
		nir_def *tex_offset = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
		nir_def *samp_offset = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);

		if (tex_offset != NULL) {
			nir_tex_instr_add_src(tex, nir_tex_src_texture_offset,
			    pan_nir_res_handle(b, MALI_TABLE_TEXTURE, tex->texture_index, tex_offset));
		} else {
			tex->texture_index = pan_res_handle(MALI_TABLE_TEXTURE, tex->texture_index);
		}
		if (!nir_tex_instr_need_sampler(tex)) {
			tex->sampler_index = pan_res_handle(MALI_TABLE_SAMPLER, 0);
		} else if (samp_offset != NULL) {
			nir_tex_instr_add_src(tex, nir_tex_src_sampler_offset,
			    pan_nir_res_handle(b, MALI_TABLE_SAMPLER, tex->sampler_index, samp_offset));
		} else {
			tex->sampler_index = pan_res_handle(MALI_TABLE_SAMPLER, tex->sampler_index);
		}
		return true;
	}
	if (instr->type != nir_instr_type_intrinsic) {
		return false;
	}
	nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
	switch (in->intrinsic) {
	case nir_intrinsic_image_load:
	case nir_intrinsic_image_store:
	case nir_intrinsic_image_atomic:
	case nir_intrinsic_image_atomic_swap:
	case nir_intrinsic_image_size:
	case nir_intrinsic_image_samples:
		nir_src_rewrite(&in->src[0], pan_nir_res_handle(b, MALI_TABLE_IMAGE, 0, in->src[0].ssa));
		return true;
	default:
		return false;
	}
}

/*
 * Image stores as Mesa's Valhall lowering takes them: a 32-bit source type (the front end gives the base type alone, and
 * half-precision writes 16-bit values), the data widened to match.
 */
static bool
widen_image_stores(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	(void)data;
	if (in->intrinsic != nir_intrinsic_image_store) {
		return false;
	}
	nir_alu_type type = nir_intrinsic_src_type(in), base = nir_alu_type_get_base_type(type);
	nir_def *value = in->src[3].ssa;

	if (nir_alu_type_get_type_size(type) == 32 && value->bit_size == 32) {
		return false;
	}
	b->cursor = nir_before_instr(&in->instr);
	if (value->bit_size != 32) {
		value = base == nir_type_float ? nir_f2f32(b, value) : base == nir_type_int ? nir_i2i32(b, value)
		                                                                         : nir_u2u32(b, value);
		nir_src_rewrite(&in->src[3], value);
	}
	nir_intrinsic_set_src_type(in, base | 32);
	return true;
}

/* a draw's system values, from the push constants the draw writes (compile_mali.h: MALI_DRAW_*) */
static bool
lower_draw_sysvals(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	nir_def *value;

	(void)data;
	b->cursor = nir_before_instr(&in->instr);
	switch (in->intrinsic) {
	case nir_intrinsic_load_viewport_scale:
		/* (MALI_FIXED_VIEWPORT=N: a debugging aid, the viewport of an N x N target baked in, as the priming draw's) */
		if (getenv("MALI_FIXED_VIEWPORT") != NULL) {
			float n = (float)atoi(getenv("MALI_FIXED_VIEWPORT"));
			value = nir_imm_vec3(b, n / 2, -n / 2, 1.0f);
			break;
		}
		value = push_words(b, MALI_DRAW_VIEWPORT_SCALE, 3, 32);
		break;
	case nir_intrinsic_load_viewport_offset:
		if (getenv("MALI_FIXED_VIEWPORT") != NULL) {
			float n = (float)atoi(getenv("MALI_FIXED_VIEWPORT"));
			value = nir_imm_vec3(b, n / 2, n / 2, 0.0f);
			break;
		}
		value = push_words(b, MALI_DRAW_VIEWPORT_OFFSET, 3, 32);
		break;
	case nir_intrinsic_load_first_vertex:
		value = push_words(b, MALI_DRAW_FIRST_VERTEX, 1, 32);
		break;
	case nir_intrinsic_load_base_instance:
		value = push_words(b, MALI_DRAW_BASE_INSTANCE, 1, 32);
		break;
	case nir_intrinsic_load_blend_const_color_r_float:
	case nir_intrinsic_load_blend_const_color_g_float:
	case nir_intrinsic_load_blend_const_color_b_float:
	case nir_intrinsic_load_blend_const_color_a_float: {
		/* the blend colour (blending in the shader, Metal's BlendColor/BlendAlpha factors): four halves in words
		 * MALI_DRAW_BLEND_COLOR (r g) and + 1 (b a), from the draw's -setBlendColorRed:green:blue:alpha: */
		unsigned c = in->intrinsic == nir_intrinsic_load_blend_const_color_r_float ? 0 :
		    in->intrinsic == nir_intrinsic_load_blend_const_color_g_float ? 1 :
		    in->intrinsic == nir_intrinsic_load_blend_const_color_b_float ? 2 : 3;
		nir_def *w = push_words(b, MALI_DRAW_BLEND_COLOR + c / 2, 1, 32);
		value = nir_channel(b, nir_unpack_half_2x16(b, w), c & 1);
		break;
	}
	case nir_intrinsic_load_rt_conversion_pan: {
		/* a framebuffer fetch's (LD_TILE) conversion: the tile buffer's format, as the blend descriptor's word 3 that
		 * c_api_mali.cpp writes for every colour target it serves (R8G8B8A8 in the tile buffer; a float target's own
		 * format: rtBlend there) */
		unsigned rt = nir_intrinsic_base(in);
		unsigned format = blend_formats != NULL && rt < 4 ? blend_formats->format[rt] : 0;

		/* (a half-float format is told only for a target whose blend is computed in the shader: its tile holds
		 * the floats) */
		value = nir_imm_int(b, format == 55 ? (167 << 12 | 16) : format == 25 ? 199 << 12 : format == 65 ? 207 << 12 :
		    format == 115 ? 223 << 12 : format == 92 ? 89 << 12 : format == 71 || format == 81 ? (237 << 12 | 1 << 20) :
		    237 << 12);
		break;
	}
	default:
		return false;
	}
	if (in->def.bit_size != 32) {
		value = nir_u2uN(b, value, in->def.bit_size);
	}
	nir_def_rewrite_uses(&in->def, value);
	nir_instr_remove(&in->instr);
	return true;
}

/* if (any(global_id >= grid)) return; at the top of the entry point */
static void
bound_to_grid(nir_shader *s)
{
	nir_function_impl *impl = nir_shader_get_entrypoint(s);
	nir_builder b = nir_builder_at(nir_before_impl(impl));
	nir_def *id = nir_load_global_invocation_id(&b, 32);
	nir_def *grid = push_words(&b, MALI_KERNEL_GRID_WORD, 3, 32);

	nir_push_if(&b, nir_bany(&b, nir_uge(&b, id, grid)));
	nir_jump(&b, nir_jump_return);
	nir_pop_if(&b, NULL);
	nir_progress(true, impl, nir_metadata_none);
	NIR_PASS(_, s, nir_lower_returns);
}

/* what the back end cannot take (Mesa's asserts are compiled out: it would emit garbage), by name */
static const char *
refusal(nir_shader *s)
{
	nir_foreach_function_impl(impl, s) {
		nir_foreach_block(block, impl) {
			nir_foreach_instr(instr, block) {
				if (instr->type != nir_instr_type_intrinsic) {
					continue;
				}
				nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
				switch (in->intrinsic) {
				case nir_intrinsic_load_push_constant:
					if (!nir_src_is_const(in->src[0])) {
						return "a buffer chosen at run time (an argument buffer, or a buffer index computed by the kernel)";
					}
					break;
				/* (a load_ssbo here is Mesa's own: a descriptor read through table 62, the resource table
				 * itself, for an image's size; Metal's buffers were made addresses before Mesa's passes) */
				case nir_intrinsic_load_ubo:
				case nir_intrinsic_store_ssbo:
				case nir_intrinsic_ssbo_atomic:
				case nir_intrinsic_ssbo_atomic_swap:
				case nir_intrinsic_get_ssbo_size:
				case nir_intrinsic_get_ubo_size:
					return "a buffer access left as a descriptor (a buffer's length asked, or an unusual form)";
				case nir_intrinsic_image_deref_load:
				case nir_intrinsic_image_deref_store:
				case nir_intrinsic_image_deref_size:
				case nir_intrinsic_image_deref_atomic:
					return "an image left as a variable (not lowered to an index)";
				default:
					break;
				}
			}
		}
	}
	return NULL;
}

/* an I/O variable's size, in attribute slots (gallium panfrost's glsl_type_size) */
static unsigned
attribute_slots(const struct glsl_type *type, bool bindless)
{
	(void)bindless;
	return glsl_count_attribute_slots(type, false);
}

static pthread_once_t types_once = PTHREAD_ONCE_INIT;

static void
types_init(void)
{
	glsl_type_singleton_init_or_ref();
}

/*
 * A vertex function's [[stage_in]] attributes fetched in the shader, as Metal's vertex descriptor says: each read of an
 * input variable at VERT_ATTRIB_GENERIC0 + n becomes a global load at buffer(b)'s address + offset + stride * index --
 * the index the vertex's id, or the instance's (divided by the step rate, plus the base instance), or 0 -- converted
 * from the attribute's format, the missing components (0, 0, 0, 1). The buffer's address comes in the push constants as
 * any [[buffer(n)]]'s; the buffers read are added to the stage's resources so that the draw fills them in.
 */
struct fetch_state {
	const struct mali_vertex_fetch *fetch;
	const char *why;
	uint32_t buffers;       /* bit n: buffer n read */
};

static bool
lower_vertex_fetch(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	struct fetch_state *st = data;

	if (in->intrinsic != nir_intrinsic_load_deref) {
		return false;
	}
	nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(in->src[0]));
	if (var == NULL || var->data.mode != nir_var_shader_in || var->data.location < VERT_ATTRIB_GENERIC0) {
		return false;
	}
	unsigned slot = var->data.location - VERT_ATTRIB_GENERIC0;
	if (slot >= MALI_VERTEX_SLOTS || st->fetch->attributes[slot].format == 0) {
		st->why = "a [[stage_in]] attribute the vertex descriptor does not describe";
		return false;
	}
	unsigned format = st->fetch->attributes[slot].format, buffer = st->fetch->attributes[slot].buffer;
	if (buffer >= MALI_VERTEX_SLOTS) {
		st->why = "a vertex attribute from a buffer past 30";
		return false;
	}
	unsigned stride = st->fetch->layouts[buffer].stride, step = st->fetch->layouts[buffer].step_function;
	unsigned rate = st->fetch->layouts[buffer].step_rate ? st->fetch->layouts[buffer].step_rate : 1;

	b->cursor = nir_before_instr(&in->instr);
	nir_def *index = step == 1 ? nir_load_vertex_id(b) :
	    step == 2 ? nir_iadd(b, nir_udiv_imm(b, nir_load_instance_id(b), rate), nir_load_base_instance(b)) : nir_imm_int(b, 0);
	nir_def *addr = buffer_address(b, nir_imm_int(b, buffer),
	    nir_iadd_imm(b, nir_imul_imm(b, index, stride), st->fetch->attributes[slot].offset));
	nir_def *v;
	switch (format) {
	case 28: case 29: case 30: case 31:     /* Float .. Float4 */
		v = nir_load_global(b, format - 27, 32, addr, .align_mul = 4);
		break;
	case 53: case 25: case 26: case 27:     /* Half, Half2 .. Half4 */
		v = nir_f2f32(b, nir_load_global(b, format == 53 ? 1 : format - 23, 16, addr, .align_mul = 2));
		break;
	case 9: case 42:                        /* UChar4Normalized, UChar4Normalized_BGRA */
		v = nir_unpack_unorm_4x8(b, nir_load_global(b, 1, 32, addr, .align_mul = 4));
		if (format == 42) {
			v = nir_swizzle(b, v, (unsigned[]){ 2, 1, 0, 3 }, 4);
		}
		break;
	default: {
		/* THE REST OF MTLVertexFormat (11-12: VectorKit's map draws -- shorts, bytes, ints, 10:10:10:2): the
		 * components loaded at their size, then as the shader's variable wants them -- an integer variable takes
		 * them widened, a float one takes integers converted and normalized ones scaled (signed: -1 at the least
		 * two values, as Metal) */
		static const struct { uint8_t format, kind, bits, n; } kinds[] = {   /* kind: 0 uint, 1 int, 2 unorm, 3 snorm */
			{ 1, 0, 8, 2 }, { 2, 0, 8, 3 }, { 3, 0, 8, 4 }, { 4, 1, 8, 2 }, { 5, 1, 8, 3 }, { 6, 1, 8, 4 },
			{ 7, 2, 8, 2 }, { 8, 2, 8, 3 }, { 10, 3, 8, 2 }, { 11, 3, 8, 3 }, { 12, 3, 8, 4 },
			{ 13, 0, 16, 2 }, { 14, 0, 16, 3 }, { 15, 0, 16, 4 }, { 16, 1, 16, 2 }, { 17, 1, 16, 3 }, { 18, 1, 16, 4 },
			{ 19, 2, 16, 2 }, { 20, 2, 16, 3 }, { 21, 2, 16, 4 }, { 22, 3, 16, 2 }, { 23, 3, 16, 3 }, { 24, 3, 16, 4 },
			{ 32, 1, 32, 1 }, { 33, 1, 32, 2 }, { 34, 1, 32, 3 }, { 35, 1, 32, 4 },
			{ 36, 0, 32, 1 }, { 37, 0, 32, 2 }, { 38, 0, 32, 3 }, { 39, 0, 32, 4 },
			{ 45, 0, 8, 1 }, { 46, 1, 8, 1 }, { 47, 2, 8, 1 }, { 48, 3, 8, 1 },
			{ 49, 0, 16, 1 }, { 50, 1, 16, 1 }, { 51, 2, 16, 1 }, { 52, 3, 16, 1 },
		};
		const bool to_int = glsl_base_type_is_integer(glsl_get_base_type(glsl_without_array(var->type)));
		unsigned kind = ~0u, vbits = 0, n = 0;

		for (unsigned i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++) {
			if (kinds[i].format == format) {
				kind = kinds[i].kind;
				vbits = kinds[i].bits;
				n = kinds[i].n;
			}
		}
		if (format == 40 || format == 41) {
			/* Int1010102Normalized, UInt1010102Normalized: x in the low ten bits, w the top two */
			nir_def *word = nir_load_global(b, 1, 32, addr, .align_mul = 4), *p[4];

			for (unsigned i = 0; i < 4; i++) {
				unsigned width = i == 3 ? 2 : 10;
				nir_def *field = format == 41 ? nir_ubitfield_extract_imm(b, word, i * 10, width) :
				    nir_ibitfield_extract(b, word, nir_imm_int(b, i * 10), nir_imm_int(b, width));

				p[i] = to_int ? field : format == 41 ? nir_fdiv_imm(b, nir_u2f32(b, field), (double)((1u << width) - 1)) :
				    nir_fmax(b, nir_fdiv_imm(b, nir_i2f32(b, field), (double)((1u << (width - 1)) - 1)), nir_imm_float(b, -1.0f));
			}
			v = nir_vec(b, p, 4);
		} else if (kind == ~0u) {
			st->why = "a vertex attribute of an MTLVertexFormat this build does not know";
			return false;
		} else {
			v = nir_load_global(b, n, vbits, addr, .align_mul = vbits / 8);
			if (to_int) {
				v = (kind & 1) ? nir_i2i32(b, v) : nir_u2u32(b, v);
			} else if (kind == 0) {
				v = nir_u2f32(b, v);
			} else if (kind == 1) {
				v = nir_i2f32(b, v);
			} else if (kind == 2) {
				v = nir_fdiv_imm(b, nir_u2f32(b, v), (double)((1ull << vbits) - 1));
			} else {
				v = nir_fmax(b, nir_fdiv_imm(b, nir_i2f32(b, v), (double)((1ull << (vbits - 1)) - 1)), nir_imm_floatN_t(b, -1.0, 32));
			}
		}
		if (to_int) {
			/* an integer variable: the missing components (0, 0, 0, 1) as integers, at the variable's size */
			unsigned want = in->def.num_components;
			nir_def *c[4];

			for (unsigned i = 0; i < want; i++) {
				c[i] = i < v->num_components ? nir_channel(b, v, i) : nir_imm_int(b, i == 3 ? 1 : 0);
			}
			v = nir_vec(b, c, want);
			if (in->def.bit_size != 32) {
				v = nir_u2uN(b, v, in->def.bit_size);
			}
			nir_def_rewrite_uses(&in->def, v);
			nir_instr_remove(&in->instr);
			st->buffers |= 1u << buffer;
			return true;
		}
		break;
	}
	}
	unsigned want = in->def.num_components, bits = in->def.bit_size;
	nir_def *c[4];
	for (unsigned i = 0; i < want; i++) {
		c[i] = i < v->num_components ? nir_channel(b, v, i) : nir_imm_float(b, i == 3 ? 1.0f : 0.0f);
	}
	v = nir_vec(b, c, want);
	if (bits != 32) {
		v = nir_f2fN(b, v, bits);
	}
	nir_def_rewrite_uses(&in->def, v);
	nir_instr_remove(&in->instr);
	st->buffers |= 1u << buffer;
	return true;
}

/*
 * From NIR on, for the AIR path and the driver's own shaders alike: Metal's buffers made addresses, panfrost's pre- and
 * post-processing, I/O variables to intrinsics, the resource tables, the draw's system values, pan_shader_compile; OUT's
 * stage, name and resources are the caller's. 0, or -1 with ERROR (OUT then partly filled: the caller frees it).
 */
/* f2u1/f2i1 -> x != 0.0: the Mali back end has no instruction for them (bi_emit_alu: "Unhandled ALU op f2u1", a trap);
 * the front end avoids them, but NIR's own passes still make some (variable_blur_frag_lph); 0 and 1 (-1) are the only
 * results in range, the rest are undefined */
static bool
lower_f2b1(nir_builder *b, nir_alu_instr *alu, void *data)
{
	(void)data;
	if ((alu->op != nir_op_f2u1 && alu->op != nir_op_f2i1)) {
		return false;
	}
	b->cursor = nir_before_instr(&alu->instr);
	nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
	nir_def_replace(&alu->def, nir_fneu_imm(b, x, 0.0));
	return true;
}

/* the blend mali_fragment_compile asks for (this thread's compile only) */
static __thread const struct mali_shader_blend *blend_now;

/* the buffer whose bytes this thread's fragment compiles fold in (mali_compile_known) */
static __thread const struct mali_known_buffer *known_now;

void
mali_compile_known(const struct mali_known_buffer *known)
{
	known_now = known;
}

/* a load of the known buffer at a constant offset, inside its bytes: those bytes */
static bool
fold_known_load(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	const struct mali_known_buffer *k = data;

	if ((in->intrinsic != nir_intrinsic_load_ubo && in->intrinsic != nir_intrinsic_load_ssbo) ||
	    !nir_src_is_const(in->src[0]) || nir_src_as_uint(in->src[0]) != k->binding || !nir_src_is_const(in->src[1])) {
		return false;
	}
	unsigned bits = in->def.bit_size, n = in->def.num_components;
	uint64_t at = nir_src_as_uint(in->src[1]);
	if ((bits != 8 && bits != 16 && bits != 32 && bits != 64) || at + (uint64_t)n * (bits / 8) > k->length) {
		return false;
	}
	nir_const_value v[NIR_MAX_VEC_COMPONENTS];
	memset(v, 0, sizeof(v));
	for (unsigned c = 0; c < n; c++) {
		uint64_t word = 0;

		memcpy(&word, k->bytes + at + c * (bits / 8), bits / 8);
		v[c] = nir_const_value_for_raw_uint(word, bits);
	}
	b->cursor = nir_before_instr(&in->instr);
	nir_def_replace(&in->def, nir_build_imm(b, n, bits, v));
	return true;
}

/* the known buffer's loads folded, and what they decide with them: branches taken or not, loops of a known count
 * unrolled (their loads then at constant offsets too), the rest dead -- until nothing changes */
static void
fold_known(nir_shader *s, const struct mali_known_buffer *k)
{
	bool progress;
	unsigned round = 0;

	do {
		progress = false;
		NIR_PASS(progress, s, nir_shader_intrinsics_pass, fold_known_load, nir_metadata_control_flow, (void *)k);
		NIR_PASS(progress, s, nir_lower_vars_to_ssa);
		NIR_PASS(progress, s, nir_opt_copy_prop);
		NIR_PASS(progress, s, nir_opt_constant_folding);
		NIR_PASS(progress, s, nir_opt_algebraic);
		NIR_PASS(progress, s, nir_opt_dead_cf);
		NIR_PASS(progress, s, nir_opt_if, nir_opt_if_optimize_phi_true_false);
		NIR_PASS(progress, s, nir_opt_loop_unroll);
		NIR_PASS(progress, s, nir_opt_cse);
		NIR_PASS(progress, s, nir_opt_dce);
	} while (progress && ++round < 32);
	nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
}

/* Metal's MTLBlendFactor as gallium's (nir_lower_blend's) */
static enum pipe_blendfactor
pipe_factor(unsigned metal)
{
	static const enum pipe_blendfactor table[19] = {
		PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_SRC_COLOR, PIPE_BLENDFACTOR_INV_SRC_COLOR,
		PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLENDFACTOR_DST_COLOR, PIPE_BLENDFACTOR_INV_DST_COLOR,
		PIPE_BLENDFACTOR_DST_ALPHA, PIPE_BLENDFACTOR_INV_DST_ALPHA, PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE,
		PIPE_BLENDFACTOR_CONST_COLOR, PIPE_BLENDFACTOR_INV_CONST_COLOR, PIPE_BLENDFACTOR_CONST_ALPHA,
		PIPE_BLENDFACTOR_INV_CONST_ALPHA, PIPE_BLENDFACTOR_SRC1_COLOR, PIPE_BLENDFACTOR_INV_SRC1_COLOR,
		PIPE_BLENDFACTOR_SRC1_ALPHA, PIPE_BLENDFACTOR_INV_SRC1_ALPHA,
	};
	return metal < 19 ? table[metal] : PIPE_BLENDFACTOR_ONE;
}

/* BLENDING IN THE SHADER (what fixed function cannot do: two different multipliers, min/max): each such target's output
 * replaced by blend(output, the target's pixel read from the tile buffer) -- Mesa's nir_lower_blend, as panfrost's blend
 * shaders use it; the tile buffer holds R8G8B8A8 (c_api_mali.cpp's internal format for every target it serves). The
 * hardware then writes the result as it is (c_api_mali.cpp: that target's equation a replace that reads the tile). */
static void
lower_shader_blend(nir_shader *s, const struct mali_shader_blend *blend, struct mali_kernel_binary *out)
{
	nir_lower_blend_options o;

	memset(&o, 0, sizeof(o));
	for (unsigned i = 0; i < 8; i++) {
		nir_lower_blend_rt *rt = &o.rt[i];
		const struct mali_blend_rt *m = i < 4 ? &blend->rt[i] : NULL;

		rt->format = blend->format[i < 4 ? i : 0] == 55 && i < 4 ? PIPE_FORMAT_R32_FLOAT :
		    i < 4 && blend->format[i] == 25 ? PIPE_FORMAT_R16_FLOAT :
		    i < 4 && blend->format[i] == 65 ? PIPE_FORMAT_R16G16_FLOAT :
		    i < 4 && blend->format[i] == 115 ? PIPE_FORMAT_R16G16B16A16_FLOAT :
		    i < 4 && blend->format[i] == 92 ? PIPE_FORMAT_R11G11B10_FLOAT : PIPE_FORMAT_R8G8B8A8_UNORM;
		rt->colormask = 0xf;
		rt->rgb = (nir_lower_blend_channel){ PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO };
		rt->alpha = rt->rgb;
		if (m == NULL || !(blend->mask & 1u << i)) {
			continue;               /* a replace: nothing done to that target */
		}
		/* (Metal's write mask: alpha 1, blue 2, green 4, red 8) */
		rt->colormask = (m->write_mask & 8 ? 1u : 0) | (m->write_mask & 4 ? 2u : 0) | (m->write_mask & 2 ? 4u : 0) |
		    (m->write_mask & 1 ? 8u : 0);
		rt->rgb = (nir_lower_blend_channel){ (enum pipe_blend_func)m->rgb_op, pipe_factor(m->src_rgb), pipe_factor(m->dst_rgb) };
		rt->alpha = (nir_lower_blend_channel){ (enum pipe_blend_func)m->alpha_op, pipe_factor(m->src_alpha),
		    pipe_factor(m->dst_alpha) };
	}
	o.scalar_blend_const = true;
	NIR_PASS(_, s, nir_lower_blend, &o);
	out->rt_read |= blend->mask;
}

/* a word in the boot-args (the phone's; the host's compile has none) */
static int
mali_boot_arg(const char *word)
{
#ifdef __APPLE__
	char args[1024] = { 0 };
	size_t size = sizeof(args) - 1;

	return sysctlbyname("kern.bootargs", args, &size, NULL, 0) == 0 && strstr(args, word) != NULL;
#else
	(void)word;
	return 0;               /* (the host's compile: tools/mesa/pan/mali_stage_host) */
#endif
}

/*
 * THE VARYINGS' NAMES, kept after the vertex stage's layout (11-13): a fragment function is LINKED to the vertex
 * function by each input's name ("user(locn2)", ANGLE's "generated(10_uv_normalDv3_f)"), and may take fewer than the
 * vertex function gives. Both sides numbered theirs in the order they come, and the fragment's third was the vertex's
 * third whatever it was called: the WebGL aquarium's inner globe took the tangent for its normal and the binormal for
 * its view vector (vertex: position, texCoord, tangent, binormal, normal, surfaceToLight, surfaceToView; fragment:
 * position, texCoord, normal, surfaceToView) -- its refraction in pieces, the tester's reference picture beside ours.
 */
struct varying_names {
	unsigned count;
	struct { char name[76]; unsigned slot; } v[32];
};

static struct varying_names *
names_of(const void *vs_varyings)
{
	return vs_varyings != NULL ? (struct varying_names *)((char *)vs_varyings + sizeof(struct pan_varying_layout)) : NULL;
}

/* "generated(<length><identifier><type>)": the identifier the same in both */
static bool
same_identifier(const char *a, const char *b)
{
	unsigned long la, lb;
	char *ea, *eb;

	if (strncmp(a, "generated(", 10) != 0 || strncmp(b, "generated(", 10) != 0) {
		return false;
	}
	la = strtoul(a + 10, &ea, 10);
	lb = strtoul(b + 10, &eb, 10);
	return la != 0 && la == lb && strlen(ea) >= la && strlen(eb) >= lb && strncmp(ea, eb, la) == 0;
}

/* the fragment function's inputs given the slots of the vertex outputs of their names; one the vertex function does
 * not have keeps a slot past every one of them */
static void
link_varyings(nir_shader *s, const struct air_shader_info *info, const void *vs_varyings)
{
	const struct varying_names *names = names_of(vs_varyings);
	unsigned to[64], spare = 0, moved = 0;

	if (names == NULL || names->count == 0 || info->num_varyings == 0) {
		return;
	}
	for (unsigned i = 0; i < names->count; i++) {
		spare = names->v[i].slot + 1 > spare ? names->v[i].slot + 1 : spare;
	}
	for (unsigned i = 0; i < 64; i++) {
		to[i] = ~0u;
	}
	for (unsigned i = 0; i < info->num_varyings; i++) {
		const struct air_varying *f = &info->varyings[i];
		unsigned slot = ~0u;

		if (f->slot < VARYING_SLOT_VAR0 || f->slot >= VARYING_SLOT_VAR0 + 64 || f->name == NULL) {
			continue;
		}
		for (unsigned k = 0; k < names->count; k++) {
			if (strcmp(names->v[k].name, f->name) == 0) {
				slot = names->v[k].slot;
				break;
			}
		}
		/* the same identifier of another type ("generated(5colorDv4_Dh)" against "generated(5colorDv4_f)":
		 * CoreAnimation's half-precision vertex functions with a float fragment function -- by the exact word alone
		 * WebKit's GPU process drew its layers in pieces and faulted, day13/aq17) */
		for (unsigned k = 0; slot == ~0u && k < names->count; k++) {
			if (same_identifier(names->v[k].name, f->name)) {
				slot = names->v[k].slot;
			}
		}
		/* none of its name: the vertex function's of the same place, as before; failing that, a slot of its own */
		if (slot == ~0u) {
			slot = i < names->count ? names->v[i].slot : spare++;
		}
		to[f->slot - VARYING_SLOT_VAR0] = slot;
		moved += slot != f->slot;
	}
	if (moved == 0) {
		return;
	}
	if (mali_boot_arg("mali_linksay=1")) {
		/* (mali_linksay=1: each relinking said -- the fragment's names and where they go, the vertex's names) */
		char line[900];
		int at = snprintf(line, sizeof(line), "mali link %s:", s->info.name ? s->info.name : "?");

		for (unsigned i = 0; i < info->num_varyings && at < (int)sizeof(line) - 110; i++) {
			const struct air_varying *f = &info->varyings[i];
			at += snprintf(line + at, sizeof(line) - (size_t)at, " %s %u->%u;", f->name ? f->name : "?", f->slot - VARYING_SLOT_VAR0,
			    f->slot >= VARYING_SLOT_VAR0 && f->slot < VARYING_SLOT_VAR0 + 64 ? to[f->slot - VARYING_SLOT_VAR0] - VARYING_SLOT_VAR0 : 99);
		}
		at += snprintf(line + at, sizeof(line) - (size_t)at, " | vertex:");
		for (unsigned k = 0; k < names->count && at < (int)sizeof(line) - 90; k++) {
			at += snprintf(line + at, sizeof(line) - (size_t)at, " %s=%u", names->v[k].name, names->v[k].slot - VARYING_SLOT_VAR0);
		}
		fprintf(stderr, "%s\n", line);
	}
	nir_foreach_variable_with_modes(var, s, nir_var_shader_in) {
		if (var->data.location >= VARYING_SLOT_VAR0 && var->data.location < VARYING_SLOT_VAR0 + 64 &&
		    to[var->data.location - VARYING_SLOT_VAR0] != ~0u) {
			var->data.location = (int)to[var->data.location - VARYING_SLOT_VAR0];
		}
	}
}

static int
compile_nir(nir_shader *s, mesa_shader_stage want, const void *vs_varyings, struct mali_kernel_binary *out, char *error,
    size_t error_size)
{
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID, .fau = { .reserved = MALI_KERNEL_RESERVED } };
	struct pan_varying_layout *varyings = NULL;
	struct pan_shader_info info;
	struct util_dynarray binary;
	const char *why = NULL;
	int result = -1;

	if (want == MESA_SHADER_FRAGMENT && known_now != NULL) {
		fold_known(s, known_now);
	}
	if (want == MESA_SHADER_FRAGMENT) {
		out->rt_read = (unsigned) ((s->info.outputs_read >> FRAG_RESULT_DATA0) & 0xff);
		nir_foreach_shader_out_variable(var, s) {
			if (var->data.location >= FRAG_RESULT_DATA0 && var->data.location < FRAG_RESULT_DATA0 + 8) {
				out->rt_written |= 1u << (var->data.location - FRAG_RESULT_DATA0);
			}
		}
	}
	NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_buffers, nir_metadata_control_flow, NULL);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, widen_image_stores, nir_metadata_control_flow, NULL);
	if (want == MESA_SHADER_COMPUTE) {
		bound_to_grid(s);
	}
	if (want == MESA_SHADER_FRAGMENT && !s->info.fs.pixel_center_integer) {
		/* Metal's [[position]] is the pixel's centre (x + 0.5, y + 0.5); panfrost's frag_coord is the pixel's corner
		 * (nir_lower_frag_coord_to_pixel_coord: "pixel-center lowering already applied", as a Vulkan driver does with
		 * this pass). Without it every position a shader read was half a pixel up and left. */
		NIR_PASS(_, s, nir_lower_wpos_center);
	}
	pan_preprocess_nir(s, GPU_ID);
	if (want != MESA_SHADER_COMPUTE) {
		/* the stage's inputs and outputs from variables to I/O intrinsics, as gallium panfrost does after
		 * pan_preprocess_nir (pan_shader.c): the front end leaves them as variables */
		nir_assign_io_var_locations(s, nir_var_shader_in);
		nir_assign_io_var_locations(s, nir_var_shader_out);
		NIR_PASS(_, s, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
		NIR_PASS(_, s, nir_lower_io, nir_var_shader_in | nir_var_shader_out, attribute_slots,
		    nir_lower_io_use_interpolated_input_intrinsics);
		if (want == MESA_SHADER_FRAGMENT && blend_now != NULL && blend_now->mask != 0) {
			lower_shader_blend(s, blend_now, out);
		}
		NIR_PASS(_, s, nir_opt_constant_folding);
	}
	NIR_PASS(_, s, nir_shader_instructions_pass, lower_res_indices, nir_metadata_control_flow, NULL);
	if (want == MESA_SHADER_VERTEX) {
		/* where the varyings go, compact, as panvk lays them out; the fragment stage is compiled against the same */
		varyings = calloc(1, sizeof(*varyings) + sizeof(struct varying_names));  /* (the names after it: compile_stage) */
		if (varyings == NULL) {
			say(error, error_size, "no memory");
			goto done;
		}
		/*
		 * (11-11) mali_mediump=0: varyings the shader marks low precision kept at 32 bits -- Control Center's tiles showed
		 * dashed inner outlines and diagonal seams from their corners, as a coordinate carried in 16 bits would draw
		 * (TODO 8h). mali_varysay=1: each vertex program's layout said (40 at most).
		 */
		static int mediump = -1, varysay = -1, said;
		if (mediump < 0) {
			mediump = !mali_boot_arg("mali_mediump=0");
			varysay = mali_boot_arg("mali_varysay=1");
		}
		pan_varying_collect_formats(varyings, s, GPU_ID, true, mediump != 0);
		pan_build_varying_layout_compact(varyings, s, GPU_ID);
		if (varysay && said < 40) {
			said++;
			char line[512];
			int at = snprintf(line, sizeof(line), "mali varyings of %s:", s->info.name ? s->info.name : "?");
			for (unsigned i = 0; i < varyings->count && at < (int)sizeof(line) - 40; i++) {
				const struct pan_varying_slot *v = &varyings->slots[i];
				at += snprintf(line + at, sizeof(line) - at, " [loc %u %s%u x%u]", (unsigned)v->location,
				    (v->alu_type & 0x80) ? "f" : "i", (unsigned)(v->alu_type & 0x7f),
				    (unsigned)v->ncomps);
			}
			fprintf(stderr, "%s (mediump lowering %s)\n", line, mediump ? "on" : "off");
		}
		inputs.varying_layout = varyings;
		inputs.trust_varying_flat_highp_types = true;
	} else if (want == MESA_SHADER_FRAGMENT) {
		inputs.varying_layout = vs_varyings;
		inputs.trust_varying_flat_highp_types = true;
	}
	if (want == MESA_SHADER_VERTEX) {
		/*
		 * (11-14) WHAT THE VERTEX FUNCTION WRITES, told to the compiler: the front end leaves info.outputs_written 0
		 * and nothing here gathered it, and the compiler goes by it -- a [[point_size]] was not written to the
		 * position FIFO at all (points one pixel large whatever the function said: MetalTest's page 17), and a
		 * [[render_target_array_index]] would not have been either. From the output variables, which are all there.
		 * Those two and the viewport's index only -- what the FIFO's extended entries carry: with the position's bit
		 * as well the compiler's pass for noperspective varyings ran and va_optimize crashed on every vertex function.
		 */
		nir_foreach_shader_out_variable(var, s) {
			if (var->data.location == VARYING_SLOT_PSIZ || var->data.location == VARYING_SLOT_LAYER ||
			    var->data.location == VARYING_SLOT_VIEWPORT) {
				s->info.outputs_written |= BITFIELD64_BIT(var->data.location);
			}
		}
	}
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(s, &inputs, &info);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, want == MESA_SHADER_COMPUTE ? lower_sysvals : lower_draw_sysvals,
	    nir_metadata_control_flow, NULL);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_algebraic);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_dce);
	NIR_PASS(_, s, nir_shader_alu_pass, lower_f2b1, nir_metadata_control_flow, NULL);
	if ((why = refusal(s)) != NULL) {
		say(error, error_size, "%s: %s", out->name, why);
		goto done;
	}
	if (getenv("MALI_COMPILE_NIR") != NULL) {
		nir_print_shader(s, stderr);            /* (host debugging: the NIR the back end is given) */
	}
	binary = UTIL_DYNARRAY_INIT;
	pan_shader_compile(s, &inputs, &binary, &info);
	if (binary.size == 0) {
		util_dynarray_fini(&binary);
		say(error, error_size, "%s: the panfrost compiler gave no code", out->name);
		goto done;
	}
	if (info.fau.count > MALI_KERNEL_MAX_FAU) {
		util_dynarray_fini(&binary);
		say(error, error_size, "%s: %u push-constant words", out->name, info.fau.count);
		goto done;
	}
	pan_fau_foreach_reloc(&info.fau, i) {
		util_dynarray_fini(&binary);
		say(error, error_size, "%s: a buffer pushed into the constants (not expected)", out->name);
		goto done;
	}
	out->code = malloc(binary.size);
	if (out->code == NULL) {
		util_dynarray_fini(&binary);
		say(error, error_size, "no memory");
		goto done;
	}
	memcpy(out->code, binary.data, binary.size);
	out->size = binary.size;
	util_dynarray_fini(&binary);
	out->work_regs = info.work_reg_count;
	out->preload = info.preload;
	/* (11-11) the compiler's "contains barrier": its helper threads must run to the end (the program descriptor's bit
	 * 28, as panvk sets it; c_api_mali.cpp sets that bit for every fragment shader now -- TODO 8h) */
	out->helpers = want == MESA_SHADER_FRAGMENT && info.contains_barrier;
	out->fau_words = info.fau.count > MALI_KERNEL_RESERVED ? info.fau.count : MALI_KERNEL_RESERVED;
	pan_fau_foreach_imm(&info.fau, i) {
		out->consts[i - MALI_KERNEL_RESERVED] = info.fau.words[i].constant;
	}
	out->tls_size = info.tls_size;
	out->wls_size = info.wls_size;
	if (want == MESA_SHADER_VERTEX) {
		out->idvs = info.vs.idvs;
		out->secondary_offset = info.vs.secondary_enable ? info.vs.secondary_offset : 0;
		out->secondary_work_regs = info.vs.secondary_work_reg_count;
		out->secondary_preload = info.vs.secondary_preload;
		out->writes_psiz = info.vs.writes_point_size;
		out->no_psiz_offset = info.vs.no_psiz_offset;
		out->extended_fifo = info.vs.needs_extended_fifo;
		out->writes_layer = (s->info.outputs_written & VARYING_BIT_LAYER) != 0;
		out->varying_bytes = varyings->generic_size_B;
		out->varyings = varyings;               /* (handed to the fragment stage's compile) */
		varyings = NULL;
		if (!out->idvs) {
			say(error, error_size, "%s: the compiler made no IDVS variant (the Mali build draws with IDVS only)", out->name);
			goto done;
		}
	}
	result = 0;
done:
	free(varyings);
	return result;
}


/*
 * A COMPILE THAT TRAPS IS A COMPILE THAT FAILED, not a dead window server: Mesa's back end ends a case it has no code
 * for in a trap or an abort (bi_emit_alu's "Unhandled ALU op f2u1" took backboardd down, and launchd started it again
 * for the same end). While a thread compiles, SIGTRAP, SIGABRT and SIGILL on that thread jump back out of the compile,
 * which then fails with the signal named (its memory is left behind); on any other thread they go to the handler there
 * was before (the window server's own crash line, inject.c), and then as they would have.
 */
static __thread sigjmp_buf *compile_guard;
static struct sigaction guard_before[3];
static const int guard_signals[3] = { SIGTRAP, SIGABRT, SIGILL };
static pthread_once_t guard_once = PTHREAD_ONCE_INIT;

/* where a compile stopped (11-13): the stopping thread's frames as image + offset, for the refusal's sentence */
static __thread char guard_where[420];

static void
guard_frames(void *context)
{
#if defined(__APPLE__) && defined(__aarch64__)
	const ucontext_t *uc = (const ucontext_t *)context;
	size_t at = 0;

	guard_where[0] = 0;
	if (uc == NULL || uc->uc_mcontext == NULL) {
		return;
	}
	uintptr_t pc = (uintptr_t)uc->uc_mcontext->__ss.__pc, fp = (uintptr_t)uc->uc_mcontext->__ss.__fp;
	const uintptr_t sp = (uintptr_t)uc->uc_mcontext->__ss.__sp;
	for (int i = 0; i < 14 && at + 40 < sizeof(guard_where); i++) {
		Dl_info where;

		if (pc != 0 && dladdr((void *)pc, &where) != 0 && where.dli_fname != NULL) {
			const char *base = strrchr(where.dli_fname, '/');

			at += (size_t)snprintf(guard_where + at, sizeof(guard_where) - at, " %s+%#lx", base != NULL ? base + 1 : where.dli_fname,
			    (unsigned long)(pc - (uintptr_t)where.dli_fbase));
		} else {
			at += (size_t)snprintf(guard_where + at, sizeof(guard_where) - at, " %#lx", (unsigned long)pc);
		}
		if (fp == 0 || (fp & 7) != 0 || fp < sp || fp - sp > (64u << 20)) {
			break;
		}
		pc = ((const uintptr_t *)fp)[1];
		const uintptr_t next = ((const uintptr_t *)fp)[0];
		if (next <= fp) {
			break;
		}
		fp = next;
	}
#else
	(void)context;
#endif
}

static void
guard_handler(int signal_number, siginfo_t *info, void *context)
{
	if (compile_guard != NULL) {
		guard_frames(context);
		siglongjmp(*compile_guard, signal_number);
	}
	for (int i = 0; i < 3; i++) {
		if (guard_signals[i] != signal_number) {
			continue;
		}
		const struct sigaction *b = &guard_before[i];
		if ((b->sa_flags & SA_SIGINFO) && b->sa_sigaction != NULL) {
			b->sa_sigaction(signal_number, info, context);
			return;
		}
		if (b->sa_handler != SIG_DFL && b->sa_handler != SIG_IGN && b->sa_handler != NULL) {
			b->sa_handler(signal_number);
			return;
		}
	}
	signal(signal_number, SIG_DFL);
	raise(signal_number);
}

static void
guard_install(void)
{
	struct sigaction action;

	memset(&action, 0, sizeof(action));
	action.sa_sigaction = guard_handler;
	action.sa_flags = SA_SIGINFO | SA_NODEFER;
	for (int i = 0; i < 3; i++) {
		sigaction(guard_signals[i], &action, &guard_before[i]);
	}
}

static int
compile_stage(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count, mesa_shader_stage want,
    const void *vs_varyings, const struct mali_vertex_fetch *fetch, struct mali_kernel_binary *out, char *error,
    size_t error_size)
{
	const nir_shader_compiler_options *options = pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), want, false);
	struct air_constant_value *values = NULL;
	struct air_to_nir_options air_options = { 0 };
	struct air_shader_info air_info;
	const char *why = NULL;
	void *mem = ralloc_context(NULL);
	nir_shader *s;
	int result = -1;

	memset(out, 0, sizeof(*out));
	pthread_once(&types_once, types_init);
	pthread_once(&guard_once, guard_install);
	sigjmp_buf jump;
	sigjmp_buf *volatile outer = compile_guard;
	int caught = sigsetjmp(jump, 1);
	if (caught != 0) {
		compile_guard = outer;
		say(error, error_size, "the compiler stopped (signal %d) on %s; at%s", caught, out->name[0] ? out->name : "a function",
		    guard_where[0] ? guard_where : " (no frames)");
		out->code = NULL;
		out->varyings = NULL;
		return -1;
	}
	compile_guard = &jump;
	if (count != 0) {
		values = ralloc_array(mem, struct air_constant_value, count);
		for (unsigned i = 0; i < count; i++) {
			values[i].index = constants[i].index;
			values[i].bits = constants[i].bits;
		}
		air_options.constants = values;
		air_options.num_constants = count;
	}
	s = air_to_nir(mem, bitcode, size, NULL, &air_options, options, &air_info, &why);
	if (s == NULL) {
		say(error, error_size, "the AIR front end: %s", why ? why : "refused");
		goto done;
	}
	if (air_info.stage != want) {
		say(error, error_size, "%s is not a %s function", air_info.entry_point,
		    want == MESA_SHADER_COMPUTE ? "kernel" : want == MESA_SHADER_VERTEX ? "vertex" : "fragment");
		goto done;
	}
	out->stage = want == MESA_SHADER_COMPUTE ? 0 : want == MESA_SHADER_VERTEX ? 1 : 2;
	snprintf(out->name, sizeof(out->name), "%s", air_info.entry_point);
	/* THE BUFFERS' SLOTS (11-12): the push constants hold MALI_KERNEL_BUFFERS addresses; Metal numbers buffers to 30
	 * (MPS's kernels use [[buffer(30)]], vertex buffers sit high by convention). An index under the count is its own
	 * slot; one past it takes a slot none of the function's buffers (nor its vertex descriptor's) has. */
	{
		uint32_t used = 0, low = 0;

		for (unsigned i = 0; i < air_info.num_resources; i++) {
			const struct air_resource *r = &air_info.resources[i];

			if ((r->kind == AIR_RESOURCE_DEVICE_BUFFER || r->kind == AIR_RESOURCE_CONSTANT_BUFFER) && r->index < 31) {
				used |= 1u << r->index;
			}
		}
		if (want == MESA_SHADER_VERTEX && fetch != NULL) {
			for (unsigned a = 0; a < MALI_VERTEX_SLOTS; a++) {
				if (fetch->attributes[a].format != 0 && fetch->attributes[a].buffer < 31) {
					used |= 1u << fetch->attributes[a].buffer;
				}
			}
		}
		low = used & ((1u << MALI_KERNEL_BUFFERS) - 1);
		for (unsigned n = 0; n < 32; n++) {
			slot_of[n] = n < MALI_KERNEL_BUFFERS ? n : 0xff;
		}
		for (unsigned n = MALI_KERNEL_BUFFERS; n < 31; n++) {
			if (!(used & 1u << n)) {
				continue;
			}
			for (int k = MALI_KERNEL_BUFFERS - 1; k >= 0; k--) {
				if (!(low & 1u << k)) {
					slot_of[n] = (uint8_t)k;
					low |= 1u << k;
					break;
				}
			}
			if (slot_of[n] == 0xff) {
				say(error, error_size, "%s: more than %u buffers (with [[buffer(%u)]])", out->name, MALI_KERNEL_BUFFERS, n);
				goto done;
			}
		}
		buffers_used = used;
		slots_set = true;
	}
	for (unsigned i = 0; i < air_info.num_resources; i++) {
		const struct air_resource *r = &air_info.resources[i];

		if (out->num_resources == MALI_KERNEL_MAX_RESOURCES) {
			say(error, error_size, "%s: more than %u resources", out->name, MALI_KERNEL_MAX_RESOURCES);
			goto done;
		}
		if ((r->kind == AIR_RESOURCE_TEXTURE || r->kind == AIR_RESOURCE_IMAGE || r->kind == AIR_RESOURCE_SAMPLER) &&
		    r->index >= MALI_KERNEL_TEXTURES) {
			say(error, error_size, "%s: a texture or sampler at index %u (the Mali build binds 0 to %u)", out->name,
			    r->index, MALI_KERNEL_TEXTURES - 1);
			goto done;
		}
		if ((r->kind == AIR_RESOURCE_DEVICE_BUFFER || r->kind == AIR_RESOURCE_CONSTANT_BUFFER) && r->index >= 31) {
			say(error, error_size, "%s: [[buffer(%u)]] (Metal's buffers are 0 to 30)", out->name, r->index);
			goto done;
		}
		out->resources[out->num_resources++] = (struct mali_kernel_resource){ r->kind, r->index, r->size, r->read, r->write,
		    r->inline_sampler ? 1u : 0u, r->sampler_state,
		    r->kind == AIR_RESOURCE_DEVICE_BUFFER || r->kind == AIR_RESOURCE_CONSTANT_BUFFER ? slot_of[r->index] : 0u };
	}
	out->threadgroup_bytes = air_info.threadgroup_bytes;
	out->threadgroup_arg_base = air_info.has_threadgroup_arg ? air_info.threadgroup_arg_base : ~0u;
	for (unsigned i = 0; i < 3; i++) {
		out->workgroup[i] = s->info.workgroup_size_variable ? 0 : s->info.workgroup_size[i];
	}

	if (want == MESA_SHADER_VERTEX && fetch != NULL) {
		struct fetch_state st = { fetch, NULL, 0 };

		NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_vertex_fetch, nir_metadata_control_flow, &st);
		if (st.why != NULL) {
			say(error, error_size, "%s: %s", out->name, st.why);
			goto done;
		}
		NIR_PASS(_, s, nir_remove_dead_variables, nir_var_shader_in, NULL);
		s->info.inputs_read = 0;
		for (unsigned n = 0; n < MALI_VERTEX_SLOTS; n++) {
			if (!(st.buffers & 1u << n)) {
				continue;
			}
			if (out->num_resources == MALI_KERNEL_MAX_RESOURCES) {
				say(error, error_size, "%s: more than %u resources", out->name, MALI_KERNEL_MAX_RESOURCES);
				goto done;
			}
			out->resources[out->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_DEVICE_BUFFER, n, 0, 1, 0, 0, 0,
			    slot_of[n] };
		}
	}
	if (want == MESA_SHADER_FRAGMENT) {
		link_varyings(s, &air_info, vs_varyings);
	}
	result = compile_nir(s, want, vs_varyings, out, error, error_size);
	if (result == 0 && want == MESA_SHADER_VERTEX && out->varyings != NULL) {
		struct varying_names *names = names_of(out->varyings);

		for (unsigned i = 0; i < air_info.num_varyings && names->count < 32; i++) {
			if (air_info.varyings[i].name != NULL) {
				snprintf(names->v[names->count].name, sizeof(names->v[0].name), "%s", air_info.varyings[i].name);
				names->v[names->count].slot = air_info.varyings[i].slot;
				names->count++;
			}
		}
	}
done:
	compile_guard = outer;
	ralloc_free(mem);
	if (result != 0) {
		mali_kernel_binary_free(out);
	}
	return result;
}

int
mali_kernel_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    struct mali_kernel_binary *out, char *error, size_t error_size)
{
	return compile_stage(bitcode, size, constants, count, MESA_SHADER_COMPUTE, NULL, NULL, out, error, error_size);
}

int
mali_stage_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count, int fragment,
    const struct mali_kernel_binary *vertex, struct mali_kernel_binary *out, char *error, size_t error_size)
{
	if (fragment && (vertex == NULL || vertex->varyings == NULL)) {
		say(error, error_size, "a fragment function compiled without its vertex function");
		return -1;
	}
	return compile_stage(bitcode, size, constants, count, fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
	    fragment ? vertex->varyings : NULL, NULL, out, error, error_size);
}

int
mali_fragment_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    const struct mali_kernel_binary *vertex, const struct mali_shader_blend *blend, struct mali_kernel_binary *out, char *error,
    size_t error_size)
{
	int r;

	blend_now = blend;
	blend_formats = blend;
	r = mali_stage_compile(bitcode, size, constants, count, 1, vertex, out, error, error_size);
	blend_now = NULL;
	blend_formats = NULL;
	return r;
}

int
mali_vertex_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    const struct mali_vertex_fetch *fetch, struct mali_kernel_binary *out, char *error, size_t error_size)
{
	return compile_stage(bitcode, size, constants, count, MESA_SHADER_VERTEX, NULL, fetch, out, error, error_size);
}

/* texture INDEX's texel at the fragment's pixel (plus OFFSET, an ivec2, or NULL), read with txf; of a multisample
 * texture (MS) sample SAMPLE's, with txf_ms. A vec4 of TYPE. */
static nir_def *
texel_here(nir_builder *b, unsigned index, nir_alu_type type, bool ms, nir_def *sample, nir_def *offset)
{
	nir_def *xy = nir_f2i32(b, nir_trim_vector(b, nir_load_frag_coord(b), 2));
	nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);

	if (offset != NULL) {
		xy = nir_iadd(b, xy, offset);
	}
	tex->op = ms ? nir_texop_txf_ms : nir_texop_txf;
	tex->sampler_dim = ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
	tex->dest_type = type;
	tex->coord_components = 2;
	tex->texture_index = index;
	tex->sampler_index = 0;
	tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, xy);
	tex->src[1] = ms ? nir_tex_src_for_ssa(nir_tex_src_ms_index, sample) : nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
	nir_def_init(&tex->instr, &tex->def, 4, 32);
	nir_builder_instr_insert(b, &tex->instr);
	return &tex->def;
}

/*
 * The tile buffer's preload, as a draw (c_api_mali.cpp: a pass that loads draws this first, in the same Mali pass): one
 * triangle over the whole target from [[vertex_id]] 0..2, its fragment the target's own texel at the fragment's
 * position -- texture 0, read with txf (no sampler, no filtering). What panfrost's frame shaders do with a pre-frame
 * DCD, through the ordinary draw path instead: on a tiler a tile reads its own pixels before it writes them back.
 */
int
mali_preload_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, unsigned targets, char *error,
    size_t error_size)
{
	return mali_preload_zs_compile(vs, fs, targets, 0, error, error_size);
}

/*
 * ... and the pre-frame shader that loads depth and stencil (ZS: 1 depth, 2 stencil) too -- the framebuffer
 * descriptor's Z/S preload bits load nothing on the Mali (Mesa never sets them: its frame shaders load Z/S): the depth
 * attachment as texture TARGETS (R32F, channel 0 to the depth output), the stencil plane as the next (R8UI, its low
 * byte to the stencil output), as panfrost's pan_fb_nir.c does it
 */
int
mali_preload_zs_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, unsigned targets, unsigned zs,
    char *error, size_t error_size)
{
	void *mem = ralloc_context(NULL);
	nir_builder b;
	nir_variable *var;
	int result = -1;

	memset(vs, 0, sizeof(*vs));
	memset(fs, 0, sizeof(*fs));
	pthread_once(&types_once, types_init);

	/* (-1,-1) (3,-1) (-1,3): the target and more */
	b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_VERTEX, false), "hakios_preload_vs");
	ralloc_steal(mem, b.shader);
	{
		nir_def *id = nir_load_vertex_id(&b);
		nir_def *x = nir_fadd_imm(&b, nir_u2f32(&b, nir_ishl_imm(&b, nir_iand_imm(&b, id, 1), 2)), -1.0f);
		nir_def *y = nir_fadd_imm(&b, nir_u2f32(&b, nir_ishl_imm(&b, nir_iand_imm(&b, id, 2), 1)), -1.0f);

		var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "position");
		var->data.location = VARYING_SLOT_POS;
		nir_store_var(&b, var, nir_vec4(&b, x, y, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)), 0xf);
	}
	vs->stage = 1;
	snprintf(vs->name, sizeof(vs->name), "hakios_preload_vs");
	if (compile_nir(b.shader, MESA_SHADER_VERTEX, NULL, vs, error, error_size) != 0) {
		goto done;
	}

	b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_FRAGMENT, false), "hakios_preload_fs");
	ralloc_steal(mem, b.shader);
	b.shader->info.fs.origin_upper_left = true;     /* as the front end's fragment functions (Metal's origin) */
	/* colour attachment n from texture n, for each of TARGETS (1 to 4) */
	fs->stage = 2;
	snprintf(fs->name, sizeof(fs->name), zs ? "hakios_preload_fs%u_zs%u" : "hakios_preload_fs%u", targets, zs);
	/* (ZS 4: the attachments multisample -- each sample its own, the shader run a sample: txf_ms at the sample id) */
	const bool ms = (zs & 4) != 0;
	nir_def *sample = ms ? nir_load_sample_id(&b) : NULL;
	for (unsigned t = 0; t < targets && t < 4; t++) {
		nir_def *v = texel_here(&b, t, nir_type_float32, ms, sample, NULL);

		var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
		var->data.location = FRAG_RESULT_DATA0 + t;
		nir_store_var(&b, var, v, 0xf);
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_TEXTURE, t, 0, 1, 0 };
	}
	for (unsigned k = 0, index = targets; k < 2; k++) {
		if (!(zs & (1u << k))) {
			continue;
		}
		nir_def *v = nir_channel(&b, texel_here(&b, index, k == 0 ? nir_type_float32 : nir_type_uint32, ms, sample, NULL), 0);
		var = nir_variable_create(b.shader, nir_var_shader_out, k == 0 ? glsl_float_type() : glsl_uint_type(),
		    k == 0 ? "depth" : "stencil");
		var->data.location = k == 0 ? FRAG_RESULT_DEPTH : FRAG_RESULT_STENCIL;
		nir_store_var(&b, var, k == 0 ? v : nir_iand_imm(&b, v, 0xff), 0x1);
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_TEXTURE, index, 0, 1, 0 };
		index++;
	}
	if (compile_nir(b.shader, MESA_SHADER_FRAGMENT, vs->varyings, fs, error, error_size) != 0) {
		goto done;
	}
	result = 0;
done:
	ralloc_free(mem);
	if (result != 0) {
		mali_kernel_binary_free(vs);
		mali_kernel_binary_free(fs);
	}
	return result;
}

/*
 * The driver's blit draw (c_api_mali.cpp: scaled texture copies, mipmap generation): the preload's triangle over the
 * viewport, its fragment texture 0 sampled with sampler 0 at level 0, at the normalized coordinate
 * u = x * w0 + y * w1 + w2, v = x * w3 + y * w4 + w5 of the fragment's position (x, y) -- push constant words 0 to 5,
 * which a draw with no buffers leaves free.
 */
int
mali_blit_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, char *error, size_t error_size)
{
	struct mali_kernel_binary pre_fs;
	void *mem = ralloc_context(NULL);
	nir_builder b;
	nir_variable *var;
	int result = -1;

	memset(fs, 0, sizeof(*fs));
	/* the vertex stage is the preload's */
	if (mali_preload_compile(vs, &pre_fs, 1, error, error_size) != 0) {
		ralloc_free(mem);
		return -1;
	}
	mali_kernel_binary_free(&pre_fs);
	snprintf(vs->name, sizeof(vs->name), "hakios_blit_vs");

	b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_FRAGMENT, false), "hakios_blit_fs");
	ralloc_steal(mem, b.shader);
	b.shader->info.fs.origin_upper_left = true;
	fs->stage = 2;
	snprintf(fs->name, sizeof(fs->name), "hakios_blit_fs");
	{
		nir_def *fc = nir_load_frag_coord(&b);
		nir_def *x = nir_channel(&b, fc, 0), *y = nir_channel(&b, fc, 1);
		nir_def *mu = push_words(&b, 0, 3, 32), *mv = push_words(&b, 3, 3, 32);
		nir_def *uv = nir_vec2(&b,
		    nir_ffma(&b, x, nir_channel(&b, mu, 0), nir_ffma(&b, y, nir_channel(&b, mu, 1), nir_channel(&b, mu, 2))),
		    nir_ffma(&b, x, nir_channel(&b, mv, 0), nir_ffma(&b, y, nir_channel(&b, mv, 1), nir_channel(&b, mv, 2))));
		nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);

		tex->op = nir_texop_txl;
		tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
		tex->dest_type = nir_type_float32;
		tex->coord_components = 2;
		tex->texture_index = 0;
		tex->sampler_index = 0;
		tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
		tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
		nir_def_init(&tex->instr, &tex->def, 4, 32);
		nir_builder_instr_insert(&b, &tex->instr);

		var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
		var->data.location = FRAG_RESULT_DATA0;
		nir_store_var(&b, var, &tex->def, 0xf);
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_TEXTURE, 0, 0, 1, 0 };
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_SAMPLER, 0, 0, 1, 0 };
	}
	if (compile_nir(b.shader, MESA_SHADER_FRAGMENT, vs->varyings, fs, error, error_size) != 0) {
		goto done;
	}
	result = 0;
done:
	ralloc_free(mem);
	if (result != 0) {
		mali_kernel_binary_free(vs);
		mali_kernel_binary_free(fs);
	}
	return result;
}

unsigned
mali_bitcode_constant_count(const void *bitcode, size_t size)
{
	void *mem = ralloc_context(NULL);
	struct air_constant_value *values = NULL;
	unsigned n = air_module_function_constants(mem, bitcode, size, &values);

	ralloc_free(mem);
	return n;
}

void
mali_kernel_binary_free(struct mali_kernel_binary *binary)
{
	free(binary->code);
	free(binary->varyings);
	binary->code = NULL;
	binary->varyings = NULL;
	binary->size = 0;
}

/*
 * The driver's resolve draw (c_api_mali.cpp: hakios_metal_cb_blit from a multisample texture): the preload's triangle,
 * its fragment the mean of texture 0's SAMPLES samples at the fragment's pixel plus (push words 0, 1) -- the copy's
 * source origin less its destination's.
 */
int
mali_resolve_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, unsigned samples, char *error,
    size_t error_size)
{
	struct mali_kernel_binary pre_fs;
	void *mem = ralloc_context(NULL);
	nir_builder b;
	nir_variable *var;
	int result = -1;

	if (mali_preload_compile(vs, &pre_fs, 1, error, error_size) != 0) {
		ralloc_free(mem);
		return -1;
	}
	mali_kernel_binary_free(&pre_fs);
	snprintf(vs->name, sizeof(vs->name), "hakios_resolve_vs");
	b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_FRAGMENT, false), "hakios_resolve_fs");
	ralloc_steal(mem, b.shader);
	b.shader->info.fs.origin_upper_left = true;
	fs->stage = 2;
	snprintf(fs->name, sizeof(fs->name), "hakios_resolve_fs%u", samples);
	{
		nir_def *offset = push_words(&b, 0, 2, 32), *sum = NULL;
		for (unsigned i = 0; i < samples; i++) {
			nir_def *v = texel_here(&b, 0, nir_type_float32, true, nir_imm_int(&b, (int)i), offset);
			sum = sum ? nir_fadd(&b, sum, v) : v;
		}
		var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
		var->data.location = FRAG_RESULT_DATA0;
		nir_store_var(&b, var, nir_fmul_imm(&b, sum, 1.0 / samples), 0xf);
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_TEXTURE, 0, 0, 1, 0 };
		fs->resources[fs->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_SAMPLER, 0, 0, 1, 0 };
	}
	if (compile_nir(b.shader, MESA_SHADER_FRAGMENT, vs->varyings, fs, error, error_size) != 0) {
		goto done;
	}
	result = 0;
done:
	ralloc_free(mem);
	if (result != 0) {
		mali_kernel_binary_free(vs);
		mali_kernel_binary_free(fs);
	}
	return result;
}
