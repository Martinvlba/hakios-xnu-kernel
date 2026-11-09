/*
 * mali_draw_test: the first draws through hakios-metal's C interface on the Pixel 8a's Mali-G715 (hakios_metal.h's
 * programs and hakios_metal_cb_draw, src/mali/c_api_mali.cpp): hakios-metal's tri_color_vs + tri_color_fs (two triangles
 * over the whole target, a colour varying per vertex: red, green, blue) into a 64 x 64 RGBA8 texture cleared to grey.
 * Wanted: no grey left; top left blue, bottom left red, top right green (Metal's NDC: y up).
 *
 *   mali_draw_test
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "hakios-metal/hakios_metal.h"
#include "mali_draw_bc.h"

static const char *
colour(uint32_t p)
{
	unsigned r = p & 0xff, g = p >> 8 & 0xff, b = p >> 16 & 0xff;

	return r > g && r > b ? "red" : g > r && g > b ? "green" : b > r && b > g ? "blue" : "?";
}

/* one draw of PROGRAM's six vertices into a fresh 64 x 64 RGBA8 texture cleared to grey; the pixels into PX */
static int
draw_n(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *p, unsigned count, uint32_t *px, long *us)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	struct hakios_metal_render_pass pass;
	struct timespec t0, t1;
	int status;

	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	if (t == NULL || hakios_metal_cb_draw(cb, p, &pass, NULL, NULL, 3, 0, count, 1, NULL, NULL, 0, NULL, NULL, 0) != 0) {
		printf("mali_draw_test: the draw refused: %s\n", hakios_metal_error());
		return -1;
	}
	clock_gettime(CLOCK_MONOTONIC, &t0);
	hakios_metal_cb_commit(cb, NULL, NULL);
	hakios_metal_cb_wait(cb);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	*us = (long)((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000);
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 4);
	status = hakios_metal_cb_status(cb);
	if (hakios_metal_cb_error(cb)[0]) {
		printf("mali_draw_test: the command buffer: %s\n", hakios_metal_cb_error(cb));
	}
	hakios_metal_cb_free(cb);
	hakios_metal_texture_free(t);
	return status;
}

static int
draw(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *p, uint32_t *px, long *us)
{
	return draw_n(d, q, p, 6, px, us);
}

static unsigned
greys(const uint32_t *px)
{
	unsigned n = 0;

	for (unsigned i = 0; i < 64 * 64; i++) {
		n += px[i] == 0xff808080u;
	}
	return n;
}

static void
picture(const uint32_t *px)
{
	for (unsigned y = 0; y < 64; y += 8) {
		char row[33];
		for (unsigned x = 0; x < 64; x += 2) {
			const char *c = colour(px[y * 64 + x]);
			row[x / 2] = px[y * 64 + x] == 0xff808080u ? '.' : px[y * 64 + x] == 0xff804020u ? '#' : c[0];
		}
		row[32] = 0;
		printf("mali_draw_test:   %s\n", row);
	}
}

static uint32_t *
read_level(hakios_metal_texture *t, unsigned level, unsigned w, unsigned h)
{
	uint32_t *px = calloc((size_t)w * h, 4);

	if (hakios_metal_texture_read_level(t, level, 0, 0, w, h, px, w * 4) != 0) {
		printf("mali_draw_test: reading level %u: %s\n", level, hakios_metal_error());
	}
	return px;
}

static int
finish(hakios_metal_cb *cb, const char *what)
{
	int st;

	hakios_metal_cb_commit(cb, NULL, NULL);
	hakios_metal_cb_wait(cb);
	st = hakios_metal_cb_status(cb);
	if (st != HAKIOS_METAL_COMPLETED) {
		printf("mali_draw_test: %s: status %d: %s\n", what, st, hakios_metal_cb_error(cb));
	}
	hakios_metal_cb_free(cb);
	return st == HAKIOS_METAL_COMPLETED;
}

static int
near(uint32_t p, uint32_t want, unsigned tol)
{
	for (unsigned s = 0; s < 32; s += 8) {
		int a = (int)(p >> s & 0xff), b = (int)(want >> s & 0xff);
		if (abs(a - b) > (int)tol) return 0;
	}
	return 1;
}

static int
levels_and_blits(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	const unsigned rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	int ok = 1;

	/* 1. generateMipmaps: level 0 left half red, right half blue; level 1 the same halves, level 6 (1x1) their mean */
	{
		hakios_metal_texture *t = hakios_metal_texture_new_levels(d, 2, 70, 64, 64, 7, rt);
		uint32_t *px = calloc(64 * 64, 4), *l1, *l6;
		int good;

		for (unsigned i = 0; i < 64 * 64; i++) px[i] = (i % 64) < 32 ? 0xff0000ffu : 0xffff0000u;
		hakios_metal_texture_replace(t, 0, 0, 64, 64, px, 64 * 4);
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_generate_mipmaps(cb, t) != 0) printf("mali_draw_test: mipmaps refused: %s\n", hakios_metal_error());
		good = finish(cb, "mipmaps");
		l1 = read_level(t, 1, 32, 32);
		l6 = read_level(t, 6, 1, 1);
		good = good && hakios_metal_texture_levels(t) == 7 && l1[0] == 0xff0000ffu && l1[31] == 0xffff0000u &&
		    near(l6[0], 0xff7f007fu, 2);
		for (unsigned l = 2; l <= 5; l++) {
			unsigned w = 64 >> l;
			uint32_t *lv = read_level(t, l, w, w);
			printf("mali_draw_test: mipmaps: level %u (%ux%u): row 0 from 0x%08x to 0x%08x\n", l, w, w, lv[0], lv[w - 1]);
			free(lv);
		}
		printf("mali_draw_test: mipmaps: %u levels, level 1 left 0x%08x right 0x%08x, level 6 0x%08x: %s\n",
		    hakios_metal_texture_levels(t), l1[0], l1[31], l6[0], good ? "WORK" : "FAILED");
		ok = ok && good;

		/* 2. a draw into level 2 (16x16): every pixel tri_flat's, level 0 untouched */
		struct hakios_metal_render_pass pass;
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.color0_level = 2;
		pass.load_action = 2;
		pass.store_action = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: a draw into level 2 refused: %s\n", hakios_metal_error());
		good = finish(cb, "a draw into level 2");
		uint32_t *l2 = read_level(t, 2, 16, 16), *l0 = read_level(t, 0, 64, 64);
		unsigned n2 = 0, n0 = 0;
		for (unsigned i = 0; i < 16 * 16; i++) n2 += l2[i] == 0xff804020u;
		for (unsigned i = 0; i < 64 * 64; i++) n0 += l0[i] == px[i];
		good = good && n2 == 256 && n0 == 4096;
		printf("mali_draw_test: a draw into level 2: %u of 256 drawn, level 0 %u of 4096 kept: %s\n", n2, n0,
		    good ? "WORK" : "FAILED");
		ok = ok && good;
		free(px); free(l1); free(l6); free(l2); free(l0);
		hakios_metal_texture_free(t);
	}
	/* 3. a scaled blit: 16x16 green into (8, 8, 32, 32) of a 64x64 grey; and a 1x1 copy */
	{
		hakios_metal_texture *s = hakios_metal_texture_new(d, 2, 70, 16, 16, 1);
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 80, 64, 64, 1);
		uint32_t src[16 * 16], *px = calloc(64 * 64, 4);
		unsigned in = 0, out = 0;
		int good;

		for (unsigned i = 0; i < 256; i++) src[i] = 0xff00ff00u;
		for (unsigned i = 0; i < 64 * 64; i++) px[i] = 0xff808080u;
		hakios_metal_texture_replace(s, 0, 0, 16, 16, src, 64);
		hakios_metal_texture_replace(t, 0, 0, 64, 64, px, 256);
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_blit(cb, s, 0, 0, 0, 16, 16, t, 0, 8, 8, 32, 32, 1) != 0 ||
		    hakios_metal_cb_blit(cb, s, 0, 0, 0, 1, 1, t, 0, 60, 60, 1, 1, 0) != 0)
			printf("mali_draw_test: a blit refused: %s\n", hakios_metal_error());
		good = finish(cb, "blits");
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
		for (unsigned y = 0; y < 64; y++)
			for (unsigned x = 0; x < 64; x++) {
				int inside = x >= 8 && x < 40 && y >= 8 && y < 40;
				if (inside) in += px[y * 64 + x] == 0xff00ff00u;
				else if (!(x == 60 && y == 60)) out += px[y * 64 + x] == 0xff808080u;
			}
		good = good && in == 1024 && out == 4096 - 1024 - 1 && px[60 * 64 + 60] == 0xff00ff00u;
		printf("mali_draw_test: blits: %u of 1024 inside green, %u of 3071 outside kept, the 1x1 copy 0x%08x: %s\n", in, out,
		    px[60 * 64 + 60], good ? "WORK" : "FAILED");
		ok = ok && good;
		free(px);
		hakios_metal_texture_free(s);
		hakios_metal_texture_free(t);
	}
	/* 4. a pass that clears attachments 1 and 2 (red, blue) and loads attachment 0: 0 untouched */
	{
		hakios_metal_texture *t[3];
		uint32_t *px = calloc(64 * 64, 4);
		struct hakios_metal_render_pass pass;
		unsigned k0 = 0, r1 = 0, b2 = 0;
		int good;

		for (int i = 0; i < 3; i++) t[i] = hakios_metal_texture_new(d, 2, 70, 64, 64, rt);
		for (unsigned i = 0; i < 64 * 64; i++) px[i] = 0x11223344u;
		hakios_metal_texture_replace(t[0], 0, 0, 64, 64, px, 256);
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t[0];
		pass.load_action = 1;
		pass.store_action = 1;
		pass.more[0].texture = t[1];
		pass.more[0].load_action = 2;
		pass.more[0].store_action = 1;
		pass.more[0].clear_color[0] = pass.more[0].clear_color[3] = 1;
		pass.more[1].texture = t[2];
		pass.more[1].load_action = 2;
		pass.more[1].store_action = 1;
		pass.more[1].clear_color[2] = pass.more[1].clear_color[3] = 1;
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_clear(cb, &pass) != 0) printf("mali_draw_test: the clear refused: %s\n", hakios_metal_error());
		good = finish(cb, "clears past 0");
		for (int i = 0; i < 3; i++) {
			hakios_metal_texture_read(t[i], 0, 0, 64, 64, px, 256);
			for (unsigned k = 0; k < 64 * 64; k++) {
				k0 += i == 0 && px[k] == 0x11223344u;
				r1 += i == 1 && px[k] == 0xff0000ffu;
				b2 += i == 2 && px[k] == 0xffff0000u;
			}
		}
		good = good && k0 == 4096 && r1 == 4096 && b2 == 4096;
		printf("mali_draw_test: clears past 0: attachment 0 %u kept, 1 %u red, 2 %u blue (of 4096): %s\n", k0, r1, b2,
		    good ? "WORK" : "FAILED");
		ok = ok && good;
		free(px);
		for (int i = 0; i < 3; i++) hakios_metal_texture_free(t[i]);
	}
	/* 5. ASTC 4x4 (MTLPixelFormat 204): a void-extent block (one colour: red 1, green 0, blue 0.5, alpha 1) in all four
	 * blocks of an 8x8, drawn scaled into a 16x16 RGBA8 */
	{
		static const uint8_t block[16] = { 0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0x80, 0xff, 0xff };
		hakios_metal_texture *a = hakios_metal_texture_new(d, 2, 204, 8, 8, 1);
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 16, 16, 1);
		uint8_t blocks[64];
		uint32_t px[256];
		unsigned n = 0;
		int good;

		if (a == NULL || t == NULL) {
			printf("mali_draw_test: ASTC: no texture: %s\n", hakios_metal_error());
			return 0;
		}
		for (int i = 0; i < 4; i++) memcpy(blocks + 16 * i, block, 16);
		hakios_metal_texture_replace(a, 0, 0, 8, 8, blocks, 32);
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_blit(cb, a, 0, 0, 0, 8, 8, t, 0, 0, 0, 16, 16, 0) != 0)
			printf("mali_draw_test: the ASTC blit refused: %s\n", hakios_metal_error());
		good = finish(cb, "ASTC");
		hakios_metal_texture_read(t, 0, 0, 16, 16, px, 64);
		for (unsigned i = 0; i < 256; i++) n += near(px[i], 0xff8000ffu, 2);
		good = good && n == 256;
		printf("mali_draw_test: ASTC 4x4: %u of 256 pixels 0xff8000ff (first 0x%08x): %s\n", n, px[0], good ? "WORK" : "FAILED");
		ok = ok && good;
		hakios_metal_texture_free(a);
		hakios_metal_texture_free(t);
	}
	return ok;
}

/* 6. depth: tri_flat (z 0) into a Depth32Float cleared to 1, compare Less and written; then tri_color over it at the same
 * z, compare Less: it must fail everywhere (every pixel still tri_flat's); the same with no depth state (always): drawn */
static int
depth_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat, hakios_metal_program *col, unsigned level)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	/* (LEVEL 1: level 1 of a 128x128 depth texture of two levels -- its level 0 must stay as it was) */
	hakios_metal_texture *z = hakios_metal_texture_new_levels(d, 2, 252, 64 << level, 64 << level, level + 1,
	    HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static float z0[128 * 128], z0_after[128 * 128];
	struct hakios_metal_render_pass pass;
	uint32_t px[64 * 64];
	unsigned kept = 0, over = 0;
	int ok = 1;

	if (t == NULL || z == NULL) {
		printf("mali_draw_test: depth: no textures (%s)\n", hakios_metal_error());
		return 0;
	}
	if (level) {
		for (unsigned i = 0; i < 128 * 128; i++) z0[i] = 0.25f;
		hakios_metal_texture_replace(z, 0, 0, 128, 128, z0, 128 * 4);
	}
	for (int round = 0; round < 2; round++) {
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.depth = z;
		pass.depth_level = level;
		pass.depth_load_action = 2;
		pass.depth_store_action = 1;
		pass.clear_depth = 1.0;
		pass.depth_state_set = 1;
		pass.depth_compare = 1;         /* Less */
		pass.depth_write = 1;
		if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: depth: the first draw refused: %s\n", hakios_metal_error());
		pass.load_action = 1;
		pass.depth_load_action = 1;
		pass.depth_state_set = round == 0;
		if (hakios_metal_cb_draw(cb, col, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: depth: the second draw refused: %s\n", hakios_metal_error());
		ok = finish(cb, "depth") && ok;
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
		unsigned n = 0;
		for (unsigned i = 0; i < 64 * 64; i++) n += px[i] == 0xff804020u;
		if (round == 0) kept = n; else over = 4096 - n;
	}
	ok = ok && kept == 4096 && over == 4096;
	if (level) {
		unsigned same = 0;
		hakios_metal_texture_read(z, 0, 0, 128, 128, z0_after, 128 * 4);
		for (unsigned i = 0; i < 128 * 128; i++) same += z0_after[i] == 0.25f;
		ok = ok && same == 128 * 128;
		printf("mali_draw_test: depth at level 1: compare Less kept %u of 4096, no test drew over %u of 4096, level 0 kept "
		    "%u of 16384: %s\n", kept, over, same, ok ? "WORK" : "FAILED");
	} else
	printf("mali_draw_test: depth: compare Less kept %u of 4096, no test drew over %u of 4096: %s\n", kept, over,
	    ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	hakios_metal_texture_free(z);
	return ok;
}

/* 6b. stencil (Depth32Float_Stencil8; LEVEL 1: level 1 of a 128x128 one of two levels): the lower-left half writes
 * stencil 1 (compare Always, Replace) and draws flat; tri_color over the whole target where stencil Equal 1 -- the
 * lower-left half all tri_color, the rest the grey clear; then, a new command buffer with the stencil loaded, flat where
 * NotEqual 1 -- the upper-right half flat, tri_color kept */
static int
stencil_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat, hakios_metal_program *col,
    unsigned level)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *z = hakios_metal_texture_new_levels(d, 2, 260, 64 << level, 64 << level, level + 1,
	    HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	struct hakios_metal_render_pass pass;
	static uint32_t px[64 * 64];
	unsigned grey_a = 0, flat_a = 0, grey_b = 0, flat_b = 0;
	int ok = 1;

	if (t == NULL || z == NULL) {
		printf("mali_draw_test: stencil: no textures (%s)\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	pass.depth = z;
	pass.depth_level = level;
	pass.depth_load_action = 2;
	pass.clear_depth = 1.0;
	pass.stencil = z;
	pass.stencil_load_action = 2;
	pass.stencil_store_action = 1;
	pass.clear_stencil = 0;
	pass.stencil_state_set = 1;
	pass.stencil_front.compare = pass.stencil_back.compare = 7;                     /* Always */
	pass.stencil_front.pass = pass.stencil_back.pass = 2;                           /* Replace */
	pass.stencil_front.read_mask = pass.stencil_back.read_mask = 0xff;
	pass.stencil_front.write_mask = pass.stencil_back.write_mask = 0xff;
	pass.stencil_front_reference = pass.stencil_back_reference = 1;
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: stencil: the first draw refused: %s\n", hakios_metal_error());
	pass.load_action = 1;
	pass.depth_load_action = 1;
	pass.stencil_load_action = 1;
	pass.stencil_front.compare = pass.stencil_back.compare = 2;                     /* Equal */
	pass.stencil_front.pass = pass.stencil_back.pass = 0;                           /* Keep */
	if (hakios_metal_cb_draw(cb, col, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: stencil: the Equal draw refused: %s\n", hakios_metal_error());
	ok = finish(cb, "stencil") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 64 * 64; i++) {
		grey_a += px[i] == 0xff808080u;
		flat_a += px[i] == 0xff804020u;
	}
	cb = hakios_metal_cb_new(q);
	pass.stencil_front.compare = pass.stencil_back.compare = 5;                     /* NotEqual */
	if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: stencil: the NotEqual draw refused: %s\n", hakios_metal_error());
	ok = finish(cb, "stencil") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 64 * 64; i++) {
		grey_b += px[i] == 0xff808080u;
		flat_b += px[i] == 0xff804020u;
	}
	/* (the half: 2016 of 4096 drawn by the 3-vertex triangle, as the geometry passes count it) */
	ok = ok && grey_a > 1900 && grey_a < 2200 && flat_a == 0 && grey_b == 0 && flat_b == grey_a;
	printf("mali_draw_test: stencil%s: Equal left grey %u, flat %u (0 wanted); NotEqual, the stencil loaded: grey %u, flat "
	    "%u (the grey before wanted): %s\n", level ? " at level 1" : "", grey_a, flat_a, grey_b, flat_b, ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	hakios_metal_texture_free(z);
	return ok;
}

/* `mali_draw_test zsdiag`: what a pass stores of stencil (Stencil8) and depth (Depth32Float), read back by the CPU; then
 * what a pass that loads them sees (NotEqual 1 / Less 0.5 drawing flat over a grey target) */
static int
zs_diag(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *s8 = hakios_metal_texture_new(d, 2, 253, 64, 64, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *z = hakios_metal_texture_new(d, 2, 252, 64, 64, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static uint8_t sb[64 * 64];
	static float zb[64 * 64];
	static uint32_t px[64 * 64];
	struct hakios_metal_render_pass pass;
	unsigned ones = 0, zero_z = 0, one_z = 0, flat_n = 0;

	for (int what = 0; what < 2; what++) {
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
		pass.clear_color[3] = 1;
		if (what == 0) {
			pass.stencil = s8;
			pass.stencil_load_action = 2;
			pass.stencil_store_action = 1;
			pass.stencil_state_set = 1;
			pass.stencil_front.compare = pass.stencil_back.compare = 7;
			pass.stencil_front.pass = pass.stencil_back.pass = 2;
			pass.stencil_front.read_mask = pass.stencil_back.read_mask = 0xff;
			pass.stencil_front.write_mask = pass.stencil_back.write_mask = 0xff;
			pass.stencil_front_reference = pass.stencil_back_reference = 1;
		} else {
			pass.depth = z;
			pass.depth_load_action = 2;
			pass.depth_store_action = 1;
			pass.clear_depth = 1.0;
			pass.depth_state_set = 1;
			pass.depth_compare = 7;
			pass.depth_write = 1;
		}
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0);
		finish(cb, "zsdiag");
		if (what == 0) {
			hakios_metal_texture_read(s8, 0, 0, 64, 64, sb, 64);
			for (unsigned i = 0; i < 64 * 64; i++) ones += sb[i] == 1;
		} else {
			hakios_metal_texture_read(z, 0, 0, 64, 64, zb, 256);
			for (unsigned i = 0; i < 64 * 64; i++) { zero_z += zb[i] == 0.0f; one_z += zb[i] == 1.0f; }
		}
		/* then loaded: flat over the whole target where stencil NotEqual 1 / depth Less (flat's z is 0: Less than the
		 * stored 0 fails, than 1 passes) -- the upper-right half wanted */
		pass.load_action = 1;
		pass.stencil_load_action = 1;
		pass.depth_load_action = 1;
		pass.stencil_front.compare = pass.stencil_back.compare = 5;
		pass.stencil_front.pass = pass.stencil_back.pass = 0;
		pass.depth_compare = 1;
		pass.depth_write = 0;
		pass.clear_color[0] = 0;
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
		for (unsigned i = 0; i < 64 * 64; i++) px[i] = 0;
		cb = hakios_metal_cb_new(q);
		hakios_metal_cb_clear(cb, &(struct hakios_metal_render_pass){ .color0 = t, .load_action = 2, .store_action = 1 });
		hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0);
		finish(cb, "zsdiag load");
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
		flat_n = 0;
		for (unsigned i = 0; i < 64 * 64; i++) flat_n += px[i] == 0xff804020u;
		if (what == 0) {
			printf("mali_draw_test: zsdiag: stencil stored: %u of 4096 are 1 (about 2016 wanted); a loading pass drew %u "
			    "(about 2080 wanted)\n", ones, flat_n);
		} else {
			printf("mali_draw_test: zsdiag: depth stored: %u of 4096 are 0, %u are 1 (about 2016 and 2080 wanted); a loading "
			    "pass drew %u (about 2080 wanted)\n", zero_z, one_z, flat_n);
		}
	}
	return 1;
}

/* 7. what a full-screen (1080x2340) pass costs with one tiny draw: clear+store, clear alone, load+store, load+store of
 * two attachments; each the mean of 20 command buffers, wall time commit to done */
static void
pass_bench(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	const unsigned w = 1080, h = 2340, rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	hakios_metal_texture *a = hakios_metal_texture_new(d, 2, 80, w, h, rt), *b = hakios_metal_texture_new(d, 2, 80, w, h, rt);
	struct hakios_metal_scissor sc = { 0, 0, 16, 16 };
	hakios_metal_buffer *small = hakios_metal_buffer_new(d, 256);
	static const char *names[5] = { "empty command buffer", "clear+store", "clear, no store", "load+store",
	    "load+store, 2 attachments" };

	if (a == NULL || b == NULL) {
		printf("mali_draw_test: bench: no textures (%s)\n", hakios_metal_error());
		return;
	}
	for (int v = 0; v < 5; v++) {
		long total = 0;
		for (int r = 0; r < 21; r++) {
			struct hakios_metal_render_pass pass;
			struct timespec t0, t1;
			hakios_metal_cb *cb = hakios_metal_cb_new(q);

			memset(&pass, 0, sizeof(pass));
			pass.color0 = a;
			pass.load_action = v == 1 || v == 2 ? 2 : 1;
			pass.store_action = v == 2 ? 0 : 1;
			if (v == 4) {
				pass.more[0].texture = b;
				pass.more[0].load_action = 1;
				pass.more[0].store_action = 1;
			}
			if (v > 0) hakios_metal_cb_draw(cb, flat, &pass, NULL, &sc, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0);
			else hakios_metal_cb_fill(cb, small, 0, 256, 0);
			clock_gettime(CLOCK_MONOTONIC, &t0);
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			clock_gettime(CLOCK_MONOTONIC, &t1);
			if (r > 0) total += (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
			hakios_metal_cb_free(cb);
		}
		printf("mali_draw_test: bench 1080x2340: %s: %ld us\n", names[v], total / 20);
	}
	hakios_metal_texture_free(a);
	hakios_metal_texture_free(b);
	hakios_metal_buffer_free(small);
}

/* 9. the tiler heap's growth: a full-screen draw of many instances (10000: 30 MB of polygon lists, the heap's first
 * 10 MB grown to 15 chunks) must complete and draw. `mali_draw_test heap N [MOST]` runs it alone (past 60000
 * instances, or MOST chunks, the heap is not enough: incremental rendering) */
/* (irdepth / irstencil: the lower-left half writes depth 0 / stencil 1 and draws flat; then INSTANCES full-screen
 * tri_color where depth Less / stencil NotEqual 1 -- through every flush the lower-left must stay flat: each later part
 * loads depth/stencil again) */
static int
zs_ir_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat, hakios_metal_program *col,
    unsigned instances, int stencil)
{
	const unsigned w = 1080, h = 2340;
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, w, h, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *z = hakios_metal_texture_new(d, 2, 260, w, h, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	struct hakios_metal_render_pass pass;
	struct timespec t0, t1;
	uint32_t bl[64], tr[64];
	int st, ok;

	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	pass.depth = z;
	pass.depth_load_action = 2;
	pass.depth_store_action = 1;
	pass.clear_depth = 1.0;
	pass.stencil = z;
	pass.stencil_load_action = 2;
	pass.stencil_store_action = 1;
	if (stencil) {
		pass.stencil_state_set = 1;
		pass.stencil_front.compare = pass.stencil_back.compare = 7;
		pass.stencil_front.pass = pass.stencil_back.pass = 2;
		pass.stencil_front.read_mask = pass.stencil_back.read_mask = 0xff;
		pass.stencil_front.write_mask = pass.stencil_back.write_mask = 0xff;
		pass.stencil_front_reference = pass.stencil_back_reference = 1;
	} else {
		pass.depth_state_set = 1;
		pass.depth_compare = 7;
		pass.depth_write = 1;
	}
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0);
	pass.load_action = 1;
	pass.depth_load_action = 1;
	pass.stencil_load_action = 1;
	pass.stencil_front.compare = pass.stencil_back.compare = 5;
	pass.stencil_front.pass = pass.stencil_back.pass = 0;
	pass.depth_compare = 1;
	pass.depth_write = 0;
	if (hakios_metal_cb_draw(cb, col, &pass, NULL, NULL, 3, 0, 6, instances, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: ir%s: refused: %s\n", stencil ? "stencil" : "depth", hakios_metal_error());
	clock_gettime(CLOCK_MONOTONIC, &t0);
	hakios_metal_cb_commit(cb, NULL, NULL);
	hakios_metal_cb_wait(cb);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	st = hakios_metal_cb_status(cb);
	hakios_metal_cb_free(cb);
	hakios_metal_texture_read(t, 0, h - 1, 64, 1, bl, sizeof(bl));
	hakios_metal_texture_read(t, w - 64, 0, 64, 1, tr, sizeof(tr));
	ok = st == HAKIOS_METAL_COMPLETED && bl[0] == 0xff804020u && tr[63] != 0xff808080u && tr[63] != 0xff804020u;
	printf("mali_draw_test: %u instances behind a lower-left half's %s: status %d in %ld ms, bottom left 0x%08x (flat "
	    "wanted), top right 0x%08x (tri_color wanted): %s\n", instances, stencil ? "stencil" : "depth", st,
	    (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000), bl[0], tr[63], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	hakios_metal_texture_free(z);
	return ok;
}

static int
heap_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat, unsigned instances, int loaded)
{
	int ok;
	struct timespec t0, t1;
	hakios_metal_cb *cb;
	const unsigned w = 1080, h = 2340;
	{
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, w, h, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		struct hakios_metal_render_pass pass;
		uint32_t row[64];
		int st;

		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		cb = hakios_metal_cb_new(q);
		if (loaded) {
			/* the target grey first; then loaded, and the lower-left half drawn many times: where nothing is drawn the
			 * grey must stay -- through every flush of incremental rendering (each later part loads what the one before
			 * stored, by the pre-frame shader) */
			pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
			pass.clear_color[3] = 1;
			hakios_metal_cb_clear(cb, &pass);
			pass.load_action = 1;
		}
		hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, loaded ? 3 : 6, instances, NULL, NULL, 0, NULL, NULL, 0);
		clock_gettime(CLOCK_MONOTONIC, &t0);
		hakios_metal_cb_commit(cb, NULL, NULL);
		hakios_metal_cb_wait(cb);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		st = hakios_metal_cb_status(cb);
		hakios_metal_texture_read(t, 0, h - 1, 64, 1, row, sizeof(row));
		if (loaded) {
			uint32_t tr[64];
			hakios_metal_texture_read(t, w - 64, 0, 64, 1, tr, sizeof(tr));
			ok = st == HAKIOS_METAL_COMPLETED && row[0] == 0xff804020u && tr[63] == 0xff808080u;
			printf("mali_draw_test: %u lower-left halves over a loaded grey target: status %d in %ld ms, bottom left 0x%08x, top "
			    "right 0x%08x: %s\n", instances, st, (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000),
			    row[0], tr[63], ok ? "WORK" : "FAILED");
		} else {
			ok = st == HAKIOS_METAL_COMPLETED && row[0] == 0xff804020u;
			printf("mali_draw_test: %u full-screen instances at 1080x2340: status %d in %ld ms, the last row 0x%08x: %s\n",
			    instances, st, (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000), row[0],
			    ok ? "WORK" : "FAILED");
		}
		hakios_metal_cb_free(cb);
		hakios_metal_texture_free(t);
	}
	return ok;
}

/* 8. the kernel's recovery: a fill into an address nothing maps (HAKIOS_MALI_FAULT_TEST) faults the GPU; the command
 * buffer must come back (not hang 10 s), and a draw after it must draw. */
static int
recovery_tests(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	uint32_t px[64 * 64];
	long us;
	int ok = 1;
	struct timespec t0, t1;
	hakios_metal_buffer *b = hakios_metal_buffer_new(d, 4096);
	hakios_metal_cb *cb = hakios_metal_cb_new(q);

	setenv("HAKIOS_MALI_FAULT_TEST", "1", 1);
	hakios_metal_cb_fill(cb, b, 0, 4096, 0x55);
	unsetenv("HAKIOS_MALI_FAULT_TEST");
	clock_gettime(CLOCK_MONOTONIC, &t0);
	hakios_metal_cb_commit(cb, NULL, NULL);
	hakios_metal_cb_wait(cb);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	printf("mali_draw_test: a faulting fill: status %d after %ld ms (%s)\n", hakios_metal_cb_status(cb),
	    (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000), hakios_metal_cb_error(cb));
	hakios_metal_cb_free(cb);
	hakios_metal_buffer_free(b);
	{
		int st = draw(d, q, flat, px, &us);
		unsigned n = 0;
		for (unsigned i = 0; i < 64 * 64; i++) n += px[i] == 0xff804020u;
		printf("mali_draw_test: a draw after the fault: status %d, %u of 4096 drawn: %s\n", st, n,
		    st == HAKIOS_METAL_COMPLETED && n == 4096 ? "WORK" : "FAILED");
		ok = ok && st == HAKIOS_METAL_COMPLETED && n == 4096;
	}
	ok = heap_test(d, q, flat, 10000, 0) && ok;
	return ok;
}

int
main(int argc, char **argv)
{
	/* `mali_draw_test heap|irload N [MOST]`: the heap test alone, N instances, the heap's most chunks MOST (5-64: below
	 * what N needs, the pass is drawn by incremental rendering); irload: over a loaded target */
	const int alone = argc >= 3 && (!strcmp(argv[1], "heap") || !strcmp(argv[1], "irload") || !strcmp(argv[1], "irdepth") ||
	    !strcmp(argv[1], "irstencil"));
	if (alone && argc >= 4) {
		setenv("HAKIOS_MALI_HEAP_CHUNKS", argv[3], 1);
	}
	hakios_metal_device *d = hakios_metal_open();
	uint32_t px[64 * 64];
	long us;
	int right = 1;

	if (d == NULL) {
		printf("mali_draw_test: no device (%s)\n", hakios_metal_error());
		return 1;
	}
	hakios_metal_queue *q = hakios_metal_queue_new(d);
	hakios_metal_program *flat = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 70);
	hakios_metal_program *col = hakios_metal_program_new_bitcode(tri_color_vs_bc, tri_color_vs_bc_len, tri_color_fs_bc,
	    tri_color_fs_bc_len, 70);
	/* (the same tri_flat with the viewport baked into its code: push constants out of the question) */
	setenv("MALI_FIXED_VIEWPORT", "64", 1);
	hakios_metal_program *fixed = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 70);
	unsetenv("MALI_FIXED_VIEWPORT");
	if (flat == NULL || col == NULL) {
		printf("mali_draw_test: no program (%s)\n", hakios_metal_error());
		return 1;
	}
	if (argc == 2 && !strcmp(argv[1], "zsdiag")) {
		return zs_diag(d, q, flat) ? 0 : 1;
	}
	if (alone && !strncmp(argv[1], "irdepth", 7)) {
		return zs_ir_test(d, q, flat, col, (unsigned)strtoul(argv[2], NULL, 0), 0) ? 0 : 1;
	}
	if (alone && !strcmp(argv[1], "irstencil")) {
		return zs_ir_test(d, q, flat, col, (unsigned)strtoul(argv[2], NULL, 0), 1) ? 0 : 1;
	}
	if (alone) {
		return heap_test(d, q, flat, (unsigned)strtoul(argv[2], NULL, 0), !strcmp(argv[1], "irload")) ? 0 : 1;
	}
	/* one pass after another with other geometry: 6 vertices (the whole target) or 3 (its lower-left half), so that
	 * what each pass shows says whose primitives it binned */
	{
		static const unsigned counts[] = { 6, 3, 3, 6, 3, 6 };
		for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
			int st = draw_n(d, q, flat, counts[i], px, &us);
			unsigned want = 0;
			for (unsigned k = 0; k < 64 * 64; k++) {
				want += px[k] == 0xff804020u;
			}
			printf("mali_draw_test: geometry pass %u, %u vertices: status %d, %u of 4096 drawn (%s)\n", i + 1, counts[i], st,
			    want, want == 4096 ? "whole" : want > 1500 && want < 2600 ? "half" : want == 0 ? "none" : "other");
		}
	}
	/* tri_color: a colour varying per vertex: no grey left; top left blue, bottom left red, top right green */
	for (int round = 0; round < 2; round++) {
		int st = draw(d, q, col, px, &us);
		uint32_t tl = px[0], tr = px[63], bl = px[63 * 64];
		printf("mali_draw_test: tri_color, draw %d: status %d, %u grey; top left 0x%08x (%s), top right 0x%08x (%s), bottom "
		    "left 0x%08x (%s); %ld us\n", round + 1, st, greys(px), tl, colour(tl), tr, colour(tr), bl, colour(bl), us);
		if (round == 1) {
			picture(px);
		}
		right = right && st == HAKIOS_METAL_COMPLETED && greys(px) == 0 && !strcmp(colour(tl), "blue") &&
		    !strcmp(colour(bl), "red") && !strcmp(colour(tr), "green");
	}
	/* tri_flat: no varyings, a constant colour (32, 64, 128): every pixel 0xff804020 wanted */
	for (int round = 0; round < 2; round++) {
		int st = draw(d, q, flat, px, &us);
		unsigned want = 0;
		for (unsigned i = 0; i < 64 * 64; i++) {
			want += px[i] == 0xff804020u;
		}
		printf("mali_draw_test: tri_flat, draw %d: status %d, %u of 4096 pixels 0xff804020, %u grey; %ld us\n", round + 1, st,
		    want, greys(px), us);
		picture(px);
		right = right && st == HAKIOS_METAL_COMPLETED && want == 4096;
	}
	for (int round = 0; round < 2 && fixed != NULL; round++) {
		int st = draw(d, q, fixed, px, &us);
		unsigned want = 0;
		for (unsigned i = 0; i < 64 * 64; i++) {
			want += px[i] == 0xff804020u;
		}
		printf("mali_draw_test: tri_flat, viewport baked in, draw %d: status %d, %u of 4096 pixels 0xff804020, %u grey\n",
		    round + 1, st, want, greys(px));
	}
	/* two passes in one command buffer */
	{
		hakios_metal_texture *t1 = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		hakios_metal_texture *t2 = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		struct hakios_metal_render_pass pass;
		uint32_t px2[64 * 64];
		unsigned a1 = 0, a2 = 0;

		memset(&pass, 0, sizeof(pass));
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
		pass.clear_color[3] = 1;
		pass.color0 = t1;
		hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0);
		pass.color0 = t2;
		hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0);
		hakios_metal_cb_commit(cb, NULL, NULL);
		hakios_metal_cb_wait(cb);
		hakios_metal_texture_read(t1, 0, 0, 64, 64, px, 64 * 4);
		hakios_metal_texture_read(t2, 0, 0, 64, 64, px2, 64 * 4);
		for (unsigned i = 0; i < 64 * 64; i++) {
			a1 += px[i] == 0xff804020u;
			a2 += px2[i] == 0xff804020u;
		}
		printf("mali_draw_test: two passes in one command buffer: status %d; the first %u of 4096 drawn, "
		    "the second %u\n", hakios_metal_cb_status(cb), a1, a2);
		picture(px);
		picture(px2);
		hakios_metal_cb_free(cb);
		hakios_metal_texture_free(t1);
		hakios_metal_texture_free(t2);
	}
	/* mip levels, blits, clears past attachment 0, ASTC (hakios-metal 11-09) */
	right = levels_and_blits(d, q, flat) && right;
	right = depth_test(d, q, flat, col, 0) && right;
	right = depth_test(d, q, flat, col, 1) && right;
	right = stencil_test(d, q, flat, col, 0) && right;
	right = stencil_test(d, q, flat, col, 1) && right;
	pass_bench(d, q, flat);
	right = recovery_tests(d, q, flat) && right;
	/* sustained load, for the GPU's clock governor: 300 draws back to back, the first and last 20 timed */
	{
		long first = 0, last = 0;
		for (int i = 0; i < 300; i++) {
			draw(d, q, col, px, &us);
			if (i < 20) {
				first += us;
			} else if (i >= 280) {
				last += us;
			}
		}
		printf("mali_draw_test: 300 draws in a row: the first 20 %ld us each, the last 20 %ld us each\n", first / 20,
		    last / 20);
	}
	printf("mali_draw_test: the first Metal draws %s\n", right ? "WORK" : "FAILED");
	hakios_metal_program_free(flat);
	hakios_metal_program_free(col);
	hakios_metal_queue_free(q);
	hakios_metal_close(d);
	return right ? 0 : 1;
}
