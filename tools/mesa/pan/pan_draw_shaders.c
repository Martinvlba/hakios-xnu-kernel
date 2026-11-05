/*
 * pan_draw_shaders: the first draw's shaders for the Pixel 8a's Mali-G715 (GPU_ID 0xb8a21020), built in NIR and compiled
 * by Mesa 26.2.5's panfrost compiler -- mali_test's triangle (userland/src/mali_test.c, "the triangle"):
 *
 *   tri_vs   a vertex shader with no inputs: vertex 0, 1, 2 at (-0.5, -0.5), (0.5, -0.5), (0, 0.5) in clip space; the
 *            viewport transform (Mesa adds it) made constants for a 64 x 64 target. Compiled as IDVS: a position shader,
 *            and (when there are varyings) a varying shader at secondary_offset.
 *   tri_fs   a fragment shader writing (1, 0, 0, 1) to render target 0.
 *
 *   pan_draw_shaders OUT.h      (tools/mesa/pan/mk-pan-host.sh writes userland/src/mali_draw_shaders.h)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/model/pan_model.h"
#include "util/u_dynarray.h"

#define GPU_ID   0xb8a21020ULL
#define TARGET_W 64
#define TARGET_H 64

/* the viewport transform's inputs, which panvk gives as push constants: constants for a TARGET_W x TARGET_H target */
static bool
fixed_viewport(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
	nir_def *v;

	(void)data;
	b->cursor = nir_before_instr(&in->instr);
	if (in->intrinsic == nir_intrinsic_load_viewport_scale) {
		v = nir_imm_vec3(b, TARGET_W / 2.0f, TARGET_H / 2.0f, 1.0f);
	} else if (in->intrinsic == nir_intrinsic_load_viewport_offset) {
		v = nir_imm_vec3(b, TARGET_W / 2.0f, TARGET_H / 2.0f, 0.0f);
	} else {
		return false;
	}
	nir_def_rewrite_uses(&in->def, v);
	nir_instr_remove(&in->instr);
	return true;
}

static void
emit(FILE *out, const char *name, mesa_shader_stage stage)
{
	const nir_shader_compiler_options *options = pan_get_nir_shader_compiler_options(pan_arch(GPU_ID), stage, false);
	struct pan_compile_inputs inputs = { .gpu_id = GPU_ID };
	struct pan_shader_info info;
	struct pan_varying_layout varyings;
	struct util_dynarray binary;
	nir_builder b = nir_builder_init_simple_shader(stage, options, "%s", name);

	if (stage == MESA_SHADER_VERTEX) {
		nir_def *vid = nir_load_vertex_id(&b);
		nir_def *x = nir_bcsel(&b, nir_ieq_imm(&b, vid, 0), nir_imm_float(&b, -0.5f),
		    nir_bcsel(&b, nir_ieq_imm(&b, vid, 1), nir_imm_float(&b, 0.5f), nir_imm_float(&b, 0.0f)));
		nir_def *y = nir_bcsel(&b, nir_ieq_imm(&b, vid, 2), nir_imm_float(&b, 0.5f), nir_imm_float(&b, -0.5f));

		nir_store_output(&b, nir_vec4(&b, x, y, nir_imm_float(&b, 0.5f), nir_imm_float(&b, 1.0f)), nir_imm_int(&b, 0),
		    .base = 0, .write_mask = 0xf, .src_type = nir_type_float32,
		    .io_semantics = (nir_io_semantics){ .location = VARYING_SLOT_POS, .num_slots = 1 });
		b.shader->info.outputs_written = VARYING_BIT_POS;
	} else {
		/* (and a count of the fragments shaded, at the triangle's descriptors + 0x12000 -- mali_test's DESC_VA) */
		nir_global_atomic(&b, 32, nir_imm_int64(&b, 0x10400000ULL + 0x12000), nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
		nir_store_output(&b, nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f), nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
		    .src_type = nir_type_float32,
		    .io_semantics = (nir_io_semantics){ .location = FRAG_RESULT_DATA0, .num_slots = 1 });
		b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
	}

	pan_preprocess_nir(b.shader, GPU_ID);
	if (stage == MESA_SHADER_VERTEX) {
		/* where the varyings go (here only the position): as panvk lays them out, compact */
		pan_varying_collect_formats(&varyings, b.shader, GPU_ID, true, true);
		pan_build_varying_layout_compact(&varyings, b.shader, GPU_ID);
		inputs.varying_layout = &varyings;
		inputs.trust_varying_flat_highp_types = true;
	}
	memset(&info, 0, sizeof(info));
	pan_postprocess_nir(b.shader, &inputs, &info);
	NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, fixed_viewport, nir_metadata_control_flow, NULL);
	NIR_PASS(_, b.shader, nir_opt_constant_folding);
	NIR_PASS(_, b.shader, nir_opt_dce);
	binary = UTIL_DYNARRAY_INIT;
	pan_shader_compile(b.shader, &inputs, &binary, &info);
	printf("pan_draw_shaders: %s: %u bytes, %u work registers, preload 0x%llx, FAU %u words", name, (unsigned)binary.size,
	    info.work_reg_count, (unsigned long long)info.preload, info.fau.count);
	if (stage == MESA_SHADER_VERTEX) {
		printf("; IDVS %d, secondary %d at %u (%u registers, preload 0x%llx)", info.vs.idvs, info.vs.secondary_enable,
		    info.vs.secondary_offset, info.vs.secondary_work_reg_count, (unsigned long long)info.vs.secondary_preload);
	}
	printf("\n");
	pan_disassemble(stdout, binary.data, binary.size, GPU_ID, false);
	fprintf(out, "static const struct mali_draw_shader mali_%s = { %u, 0x%llxULL, %u, %u, %u, %u, 0x%llxULL, %u, {", name,
	    info.work_reg_count, (unsigned long long)info.preload, info.fau.count, stage == MESA_SHADER_VERTEX && info.vs.idvs,
	    stage == MESA_SHADER_VERTEX && info.vs.secondary_enable ? info.vs.secondary_offset : 0,
	    stage == MESA_SHADER_VERTEX ? info.vs.secondary_work_reg_count : 0,
	    stage == MESA_SHADER_VERTEX ? (unsigned long long)info.vs.secondary_preload : 0ULL, (unsigned)binary.size);
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
		fprintf(stderr, "usage: pan_draw_shaders OUT.h\n");
		return 1;
	}
	glsl_type_singleton_init_or_ref();
	out = fopen(argv[1], "w");
	if (out == NULL) {
		perror(argv[1]);
		return 1;
	}
	fprintf(out, "/* made by hakios-kernel's tools/mesa/pan/pan_draw_shaders.c: Mesa 26.2.5's panfrost compiler for the "
	    "Mali-G715 (GPU_ID 0x%llx): mali_test's triangle, %ux%u. */\n#pragma once\n#include <stdint.h>\n"
	    "struct mali_draw_shader { unsigned work_regs; uint64_t preload; unsigned fau_words, idvs, secondary_offset, "
	    "secondary_work_regs; uint64_t secondary_preload; unsigned size; uint8_t code[1024]; };\n", (unsigned long long)GPU_ID,
	    TARGET_W, TARGET_H);
	emit(out, "tri_vs", MESA_SHADER_VERTEX);
	emit(out, "tri_fs", MESA_SHADER_FRAGMENT);
	fclose(out);
	return 0;
}
