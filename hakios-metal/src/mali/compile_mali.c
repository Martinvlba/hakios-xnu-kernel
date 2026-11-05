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

static pthread_once_t types_once = PTHREAD_ONCE_INIT;

static void
types_init(void)
{
	glsl_type_singleton_init_or_ref();
}

int
mali_kernel_compile(const void *bitcode, size_t size, const struct mali_constant *constants, unsigned count,
    struct mali_kernel_binary *out, char *error, size_t error_size)
{
	const nir_shader_compiler_options *options =
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_COMPUTE, false);
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID, .fau = { .reserved = MALI_KERNEL_RESERVED } };
	struct air_constant_value *values = NULL;
	struct air_to_nir_options air_options = { 0 };
	struct air_shader_info air_info;
	struct pan_shader_info info;
	struct util_dynarray binary;
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
	if (air_info.stage != MESA_SHADER_COMPUTE) {
		say(error, error_size, "%s is not a compute kernel", air_info.entry_point);
		goto done;
	}
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

	NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_buffers, nir_metadata_control_flow, NULL);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, widen_image_stores, nir_metadata_control_flow, NULL);
	bound_to_grid(s);
	pan_preprocess_nir(s, GPU_ID);
	NIR_PASS(_, s, nir_shader_instructions_pass, lower_res_indices, nir_metadata_control_flow, NULL);
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(s, &inputs, &info);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_sysvals, nir_metadata_control_flow, NULL);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_algebraic);
	NIR_PASS(_, s, nir_opt_constant_folding);
	NIR_PASS(_, s, nir_opt_dce);
	if ((why = refusal(s)) != NULL) {
		say(error, error_size, "%s: %s", out->name, why);
		goto done;
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
	result = 0;
done:
	ralloc_free(mem);
	if (result != 0) {
		mali_kernel_binary_free(out);
	}
	return result;
}

void
mali_kernel_binary_free(struct mali_kernel_binary *binary)
{
	free(binary->code);
	binary->code = NULL;
	binary->size = 0;
}
