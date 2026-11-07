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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
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

static nir_def *
buffer_address(nir_builder *b, nir_def *index, nir_def *offset)
{
	nir_def *base = nir_load_push_constant(b, 1, 64, nir_imul_imm(b, index, 8), .base = 0);

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
	if (buffer >= MALI_KERNEL_BUFFERS || buffer >= MALI_VERTEX_SLOTS) {
		st->why = "a vertex attribute from a buffer past 15";
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
	default:
		st->why = "a vertex attribute format other than float, half and normalized uchar4 (not on the Mali yet)";
		return false;
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

	if (want == MESA_SHADER_FRAGMENT) {
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
	pan_preprocess_nir(s, GPU_ID);
	if (want != MESA_SHADER_COMPUTE) {
		/* the stage's inputs and outputs from variables to I/O intrinsics, as gallium panfrost does after
		 * pan_preprocess_nir (pan_shader.c): the front end leaves them as variables */
		nir_assign_io_var_locations(s, nir_var_shader_in);
		nir_assign_io_var_locations(s, nir_var_shader_out);
		NIR_PASS(_, s, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
		NIR_PASS(_, s, nir_lower_io, nir_var_shader_in | nir_var_shader_out, attribute_slots,
		    nir_lower_io_use_interpolated_input_intrinsics);
		NIR_PASS(_, s, nir_opt_constant_folding);
	}
	NIR_PASS(_, s, nir_shader_instructions_pass, lower_res_indices, nir_metadata_control_flow, NULL);
	if (want == MESA_SHADER_VERTEX) {
		/* where the varyings go, compact, as panvk lays them out; the fragment stage is compiled against the same */
		varyings = calloc(1, sizeof(*varyings));
		if (varyings == NULL) {
			say(error, error_size, "no memory");
			goto done;
		}
		pan_varying_collect_formats(varyings, s, GPU_ID, true, true);
		pan_build_varying_layout_compact(varyings, s, GPU_ID);
		inputs.varying_layout = varyings;
		inputs.trust_varying_flat_highp_types = true;
	} else if (want == MESA_SHADER_FRAGMENT) {
		inputs.varying_layout = vs_varyings;
		inputs.trust_varying_flat_highp_types = true;
	}
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(s, &inputs, &info);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, want == MESA_SHADER_COMPUTE ? lower_sysvals : lower_draw_sysvals,
	    nir_metadata_control_flow, NULL);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_algebraic);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_dce);
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
		if ((r->kind == AIR_RESOURCE_DEVICE_BUFFER || r->kind == AIR_RESOURCE_CONSTANT_BUFFER) &&
		    r->index >= MALI_KERNEL_BUFFERS) {
			say(error, error_size, "%s: [[buffer(%u)]] (the Mali build binds buffers 0 to %u)", out->name, r->index,
			    MALI_KERNEL_BUFFERS - 1);
			goto done;
		}
		out->resources[out->num_resources++] = (struct mali_kernel_resource){ r->kind, r->index, r->size, r->read, r->write };
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
		for (unsigned n = 0; n < MALI_KERNEL_BUFFERS; n++) {
			if (!(st.buffers & 1u << n)) {
				continue;
			}
			if (out->num_resources == MALI_KERNEL_MAX_RESOURCES) {
				say(error, error_size, "%s: more than %u resources", out->name, MALI_KERNEL_MAX_RESOURCES);
				goto done;
			}
			out->resources[out->num_resources++] = (struct mali_kernel_resource){ AIR_RESOURCE_DEVICE_BUFFER, n, 0, 1, 0 };
		}
	}
	result = compile_nir(s, want, vs_varyings, out, error, error_size);
done:
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
mali_vertex_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    const struct mali_vertex_fetch *fetch, struct mali_kernel_binary *out, char *error, size_t error_size)
{
	return compile_stage(bitcode, size, constants, count, MESA_SHADER_VERTEX, NULL, fetch, out, error, error_size);
}

/*
 * The tile buffer's preload, as a draw (c_api_mali.cpp: a pass that loads draws this first, in the same Mali pass): one
 * triangle over the whole target from [[vertex_id]] 0..2, its fragment the target's own texel at the fragment's
 * position -- texture 0, read with txf (no sampler, no filtering). What panfrost's frame shaders do with a pre-frame
 * DCD, through the ordinary draw path instead: on a tiler a tile reads its own pixels before it writes them back.
 */
int
mali_preload_compile(struct mali_kernel_binary *vs, struct mali_kernel_binary *fs, char *error, size_t error_size)
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
	{
		nir_def *xy = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
		nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);

		tex->op = nir_texop_txf;
		tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
		tex->dest_type = nir_type_float32;
		tex->coord_components = 2;
		tex->texture_index = 0;
		tex->sampler_index = 0;
		tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, xy);
		tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
		nir_def_init(&tex->instr, &tex->def, 4, 32);
		nir_builder_instr_insert(&b, &tex->instr);

		var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
		var->data.location = FRAG_RESULT_DATA0;
		nir_store_var(&b, var, &tex->def, 0xf);
	}
	fs->stage = 2;
	snprintf(fs->name, sizeof(fs->name), "hakios_preload_fs");
	fs->num_resources = 1;
	fs->resources[0] = (struct mali_kernel_resource){ AIR_RESOURCE_TEXTURE, 0, 0, 1, 0 };
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

void
mali_kernel_binary_free(struct mali_kernel_binary *binary)
{
	free(binary->code);
	free(binary->varyings);
	binary->code = NULL;
	binary->varyings = NULL;
	binary->size = 0;
}
