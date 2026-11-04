/*
 * pan_blit_shaders: the four compute shaders hakios-metal's Mali back end does Metal's blit copies and fills with,
 * built in NIR and compiled by Mesa 26.2.5's panfrost compiler for the Pixel 8a's Mali-G715 (GPU_ID 0xb8a21020):
 *
 *   copy_u32, fill_u32    a 32-bit word an invocation (offsets and length multiples of 4)
 *   copy_u8,  fill_u8     a byte an invocation (anything else)
 *
 * 64 invocations a workgroup; invocation i acts if i < count. Push constants (8 words, the same for all four):
 * byte 0 the destination's address, 8 the source's (copies), 16 the value (fills: the byte, or it repeated in a word),
 * 20 the count of elements.
 *
 *   pan_blit_shaders OUT.h      (tools/mesa/pan/mk-pan-host.sh writes hakios-metal's src/mali/blit_shaders.h)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/model/pan_model.h"
#include "util/u_dynarray.h"

#define GPU_ID 0xb8a21020ULL

static void
emit(FILE *out, const char *name, bool copy, unsigned bits)
{
	const nir_shader_compiler_options *options =
	    pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_COMPUTE, false);
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID, .fau = { .reserved = 8 } };
	struct pan_shader_info info;
	struct util_dynarray binary;
	nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options, "%s", name);
	nir_def *gid, *count, *dst, *offset;

	b.shader->info.workgroup_size[0] = 64;
	b.shader->info.workgroup_size[1] = 1;
	b.shader->info.workgroup_size[2] = 1;
	gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
	count = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 20), .base = 0);
	nir_push_if(&b, nir_ult(&b, gid, count));
	offset = nir_u2u64(&b, nir_imul_imm(&b, gid, bits / 8));
	dst = nir_iadd(&b, nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0), .base = 0), offset);
	if (copy) {
		nir_def *src = nir_iadd(&b, nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 8), .base = 0), offset);
		nir_def *v = nir_load_global(&b, 1, bits, src, .align_mul = bits / 8);

		nir_store_global(&b, v, dst, .align_mul = bits / 8, .write_mask = 1);
	} else {
		nir_def *v = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 16), .base = 0);

		nir_store_global(&b, bits == 8 ? nir_u2u8(&b, v) : v, dst, .align_mul = bits / 8, .write_mask = 1);
	}
	nir_pop_if(&b, NULL);

	pan_preprocess_nir(b.shader, GPU_ID);
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(b.shader, &inputs, &info);
	binary = UTIL_DYNARRAY_INIT;
	pan_shader_compile(b.shader, &inputs, &binary, &info);
	printf("pan_blit_shaders: %s: %u bytes, %u work registers, preload 0x%llx, FAU %u words\n", name,
	    (unsigned)binary.size, info.work_reg_count, (unsigned long long)info.preload, info.fau.count);
	fprintf(out, "static const struct mali_blit_shader mali_%s = { %u, 0x%llxULL, %u, %u, {", name, info.work_reg_count,
	    (unsigned long long)info.preload, info.fau.count, (unsigned)binary.size);
	for (unsigned i = 0; i < binary.size; i++) {
		fprintf(out, "%s0x%02x,", i % 16 ? " " : "\n\t", ((unsigned char *)binary.data)[i]);
	}
	fprintf(out, "\n} };\n");
	util_dynarray_fini(&binary);
	ralloc_free(b.shader);
}

int
main(int argc, char **argv)
{
	FILE *out;

	if (argc != 2) {
		fprintf(stderr, "usage: pan_blit_shaders OUT.h\n");
		return 1;
	}
	glsl_type_singleton_init_or_ref();
	out = fopen(argv[1], "w");
	if (out == NULL) {
		perror(argv[1]);
		return 1;
	}
	fprintf(out, "/* made by hakios-kernel's tools/mesa/pan/pan_blit_shaders.c: Mesa 26.2.5's panfrost compiler for the "
	    "Mali-G715 (GPU_ID 0x%llx). Push constants: dst @0, src @8, value @16, count @20; 64 invocations a workgroup. */\n"
	    "#pragma once\n#include <stdint.h>\nstruct mali_blit_shader { unsigned work_regs; uint64_t preload; unsigned fau_words; "
	    "unsigned size; uint8_t code[512]; };\n", (unsigned long long)GPU_ID);
	emit(out, "copy_u32", true, 32);
	emit(out, "fill_u32", false, 32);
	emit(out, "copy_u8", true, 8);
	emit(out, "fill_u8", false, 8);
	fclose(out);
	return 0;
}
