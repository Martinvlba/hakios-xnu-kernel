/*
 * air_pan: Apple's shader bitcode (AIR) compiled for the Pixel 8a's Mali-G715 -- our AIR front end (tools/mesa/air)
 * into NIR, Metal's buffers made addresses, and Mesa 26.2.5's panfrost compiler for arch 11: the path the Metal
 * driver's Mali back end takes with every shader an app gives it, here from the command line.
 *
 *   air_pan FILE [FUNCTION] [OUT.h]     FILE: one module's bitcode, or a .metallib (FUNCTION picks the function)
 *
 * Metal's buffers: [[buffer(n)]] is an address. The front end makes a `device` buffer an SSBO and a `constant` one a
 * UBO at index n; here every access becomes a global one at base(n) + offset, base(n) being 64 bits of push
 * constants at words 2n, 2n + 1 (the first AIR_PAN_BUFFERS buffers; the dispatch puts the buffers' addresses there).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/air/nir_air.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/model/pan_model.h"
#include "util/os_file.h"
#include "util/u_dynarray.h"

#define GPU_ID          0xb8a21020ULL
#define AIR_PAN_BUFFERS 16

static nir_def *
buffer_address(nir_builder *b, nir_def *index, nir_def *offset)
{
	nir_def *base = nir_load_push_constant(b, 1, 64, nir_imul_imm(b, index, 8), .base = 0);

	return nir_iadd(b, base, nir_u2u64(b, offset));
}

/* load_ubo/load_ssbo/store_ssbo -> global loads and stores at the buffer's address */
static bool
lower_buffers(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	nir_def *address, *value;

	(void)data;
	switch (in->intrinsic) {
	case nir_intrinsic_load_ubo:
	case nir_intrinsic_load_ssbo:
		b->cursor = nir_before_instr(&in->instr);
		address = buffer_address(b, in->src[0].ssa, in->src[1].ssa);
		value = nir_load_global(b, in->def.num_components, in->def.bit_size, address,
		    .align_mul = nir_intrinsic_align_mul(in), .align_offset = nir_intrinsic_align_offset(in));
		nir_def_rewrite_uses(&in->def, value);
		nir_instr_remove(&in->instr);
		return true;
	case nir_intrinsic_store_ssbo:
		b->cursor = nir_before_instr(&in->instr);
		address = buffer_address(b, in->src[1].ssa, in->src[2].ssa);
		nir_store_global(b, in->src[0].ssa, address, .write_mask = nir_intrinsic_write_mask(in),
		    .align_mul = nir_intrinsic_align_mul(in), .align_offset = nir_intrinsic_align_offset(in));
		nir_instr_remove(&in->instr);
		return true;
	default:
		return false;
	}
}

int
main(int argc, char **argv)
{
	const nir_shader_compiler_options *options = pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), MESA_SHADER_COMPUTE, false);
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID, .fau = { .reserved = 2 * AIR_PAN_BUFFERS } };
	struct air_to_nir_options air_options = { 0 };
	struct air_shader_info air_info;
	struct pan_shader_info info;
	struct util_dynarray binary;
	const char *error = NULL, *function = argc > 2 && argv[2][0] ? argv[2] : NULL, *out_name = argc > 3 ? argv[3] : NULL;
	const void *bitcode;
	size_t size = 0, bitcode_size;
	void *file, *mem = ralloc_context(NULL);
	nir_shader *s;

	if (argc < 2) {
		fprintf(stderr, "usage: air_pan FILE [FUNCTION] [OUT.h]\n");
		return 1;
	}
	glsl_type_singleton_init_or_ref();
	file = os_read_file(argv[1], &size);
	if (file == NULL) {
		perror(argv[1]);
		return 1;
	}
	bitcode = file;
	bitcode_size = size;
	if (size >= 4 && memcmp(file, "MTLB", 4) == 0) {        /* a .metallib: its function's module */
		struct air_metallib lib;

		if (!air_metallib_parse(mem, file, size, &lib, &error)) {
			fprintf(stderr, "air_pan: %s\n", error);
			return 1;
		}
		bitcode = NULL;
		for (unsigned i = 0; i < lib.num_functions; i++) {
			if (function == NULL || strcmp(lib.functions[i].name, function) == 0) {
				bitcode = lib.functions[i].bitcode;
				bitcode_size = lib.functions[i].bitcode_size;
				function = lib.functions[i].name;
				break;
			}
		}
		if (bitcode == NULL) {
			fprintf(stderr, "air_pan: no function %s in %s\n", function, argv[1]);
			return 1;
		}
		function = NULL;                /* (the module's own entry point) */
	}
	s = air_to_nir(mem, bitcode, bitcode_size, function, &air_options, options, &air_info, &error);
	if (s == NULL) {
		fprintf(stderr, "air_pan: the front end: %s\n", error);
		return 1;
	}
	printf("air_pan: %s: %s, %u resources:", air_info.entry_point, mesa_shader_stage_name(air_info.stage),
	    air_info.num_resources);
	for (unsigned i = 0; i < air_info.num_resources; i++) {
		printf(" %s[%u]%s%s", air_info.resources[i].kind == AIR_RESOURCE_DEVICE_BUFFER ? "device" :
		    air_info.resources[i].kind == AIR_RESOURCE_CONSTANT_BUFFER ? "constant" : "other", air_info.resources[i].index,
		    air_info.resources[i].read ? "r" : "", air_info.resources[i].write ? "w" : "");
	}
	printf("; workgroup %ux%ux%u\n", s->info.workgroup_size[0], s->info.workgroup_size[1], s->info.workgroup_size[2]);

	NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_buffers, nir_metadata_control_flow, NULL);
	pan_preprocess_nir(s, GPU_ID);
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(s, &inputs, &info);
	binary = UTIL_DYNARRAY_INIT;
	pan_shader_compile(s, &inputs, &binary, &info);
	printf("air_pan: %u bytes for arch %u; work registers %u, preload 0x%llx, FAU %u words, TLS %u, WLS %u\n",
	    (unsigned)binary.size, pan_arch(GPU_ID), info.work_reg_count, (unsigned long long)info.preload, info.fau.count,
	    info.tls_size, info.wls_size);
	pan_disassemble(stdout, binary.data, binary.size, GPU_ID, false);
	if (out_name != NULL) {
		FILE *out = fopen(out_name, "w");

		if (out == NULL) {
			perror(out_name);
			return 1;
		}
		fprintf(out, "/* made by tools/mesa/pan/air_pan from %s (%s): AIR -> NIR -> Mesa 26.2.5's panfrost, GPU_ID "
		    "0x%llx; buffer n's address at push constant words 2n, 2n+1 */\n", argv[1], air_info.entry_point,
		    (unsigned long long)GPU_ID);
		fprintf(out, "#define AIR_KERNEL_WORK_REGS %u\n#define AIR_KERNEL_PRELOAD 0x%llxULL\n#define AIR_KERNEL_FAU_WORDS %u\n"
		    "#define AIR_KERNEL_WORKGROUP %u\n", info.work_reg_count, (unsigned long long)info.preload, info.fau.count,
		    s->info.workgroup_size[0] ? s->info.workgroup_size[0] : 1);
		fprintf(out, "static const unsigned char air_kernel_shader[%u] __attribute__((aligned(128))) = {",
		    (unsigned)binary.size);
		for (unsigned i = 0; i < binary.size; i++) {
			fprintf(out, "%s0x%02x,", i % 16 ? " " : "\n\t", ((unsigned char *)binary.data)[i]);
		}
		fprintf(out, "\n};\n");
		fclose(out);
	}
	util_dynarray_fini(&binary);
	ralloc_free(mem);
	return 0;
}
