/*
 * pan_compile_test: a compute shader built in NIR, compiled by Mesa 26.2.5's panfrost compiler for the Pixel 8a's
 * Mali-G715 (GPU_ID 0xb8a21020, arch 11), and written out as what a dispatch needs: the binary (as a C array) and
 * the facts of the shader program descriptor -- the register allocation, the preload mask (r48..r63), the push
 * constants' layout -- with Mesa's disassembly of it. The first shader compiled by Mesa's own compiler for this GPU
 * (tools/mesa/pan/mk-pan-host.sh builds it on the host; userland/src/mali_test.c runs its output on the phone).
 *
 * The shader: 64 invocations a workgroup; invocation i (global) stores 3 * i + 7 into buffer[i], the buffer's address
 * in push constant 0 (64 bits).
 *
 *   pan_compile_test OUT.h
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

int
main(int argc, char **argv)
{
	const nir_shader_compiler_options *options = pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_COMPUTE, false);
	nir_builder b;
	nir_def *gid, *base, *value, *address;
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID, .fau = { .reserved = 2 } };   /* push constant 0: 2 words */
	struct pan_shader_info info;
	struct util_dynarray binary;
	FILE *out;

	if (argc != 2) {
		fprintf(stderr, "usage: pan_compile_test OUT.h\n");
		return 1;
	}
	glsl_type_singleton_init_or_ref();
	b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options, "hakios_store_index");
	b.shader->info.workgroup_size[0] = 64;
	b.shader->info.workgroup_size[1] = 1;
	b.shader->info.workgroup_size[2] = 1;
	b.shader->num_uniforms = 8;

	gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
	base = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0), .base = 0);   /* (the compiler wants range 0) */
	value = nir_iadd_imm(&b, nir_imul_imm(&b, gid, 3), 7);
	address = nir_iadd(&b, base, nir_u2u64(&b, nir_imul_imm(&b, gid, 4)));
	nir_store_global(&b, value, address, .align_mul = 4, .write_mask = 1);

	pan_preprocess_nir(b.shader, GPU_ID);
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(b.shader, &inputs, &info);
	binary = UTIL_DYNARRAY_INIT;
	pan_shader_compile(b.shader, &inputs, &binary, &info);

	printf("pan_compile_test: %u bytes for arch %u; work registers %u, preload 0x%llx, FAU %u words, TLS %u, WLS %u\n",
	    (unsigned)binary.size, pan_arch(GPU_ID), info.work_reg_count, (unsigned long long)info.preload, info.fau.count,
	    info.tls_size, info.wls_size);
	{
		uint32_t h = 2166136261u;       /* FNV-1a of the binary: the host's and the phone's compiles compared */

		for (unsigned i = 0; i < binary.size; i++) {
			h = (h ^ ((unsigned char *)binary.data)[i]) * 16777619u;
		}
		printf("pan_compile_test: the binary's FNV-1a 0x%08x\n", h);
	}
	pan_disassemble(stdout, binary.data, binary.size, GPU_ID, false);

	out = fopen(argv[1], "w");
	if (out == NULL) {
		perror(argv[1]);
		return 1;
	}
	fprintf(out, "/* made by tools/mesa/pan/pan_compile_test (Mesa 26.2.5's panfrost compiler, GPU_ID 0x%llx) */\n",
	    (unsigned long long)GPU_ID);
	fprintf(out, "#define STORE_INDEX_WORK_REGS %u\n#define STORE_INDEX_PRELOAD 0x%llxULL\n#define STORE_INDEX_FAU_WORDS %u\n",
	    info.work_reg_count, (unsigned long long)info.preload, info.fau.count);
	fprintf(out, "static const unsigned char store_index_shader[%u] __attribute__((aligned(128))) = {", (unsigned)binary.size);
	for (unsigned i = 0; i < binary.size; i++) {
		fprintf(out, "%s0x%02x,", i % 16 ? " " : "\n\t", ((unsigned char *)binary.data)[i]);
	}
	fprintf(out, "\n};\n");
	fclose(out);
	util_dynarray_fini(&binary);
	ralloc_free(b.shader);
	return 0;
}
