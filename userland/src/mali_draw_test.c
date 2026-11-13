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
#include "cover1_bc.h"
#include "cube_fs_bc.h"
#include "link_bc.h"

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

/* 6a2. a pass with a depth buffer and NO colour attachment that only clears (ANGLE's first clear of a WebGL depth
 * buffer; refused until 11-13): the depth buffer, filled with 0.75 from here, reads 0.5 everywhere afterwards */
static int
depth_alone_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *z = hakios_metal_texture_new(d, 2, 252, 64, 64, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static float before[64 * 64], after[64 * 64];
	struct hakios_metal_render_pass pass;
	unsigned cleared = 0;
	int ok = 1;

	if (z == NULL) {
		printf("mali_draw_test: a depth buffer cleared alone: no texture (%s)\n", hakios_metal_error());
		return 0;
	}
	for (unsigned i = 0; i < 64 * 64; i++) before[i] = 0.75f;
	hakios_metal_texture_replace(z, 0, 0, 64, 64, before, 64 * 4);
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	memset(&pass, 0, sizeof(pass));
	pass.depth = z;
	pass.depth_load_action = 2;
	pass.depth_store_action = 1;
	pass.clear_depth = 0.5;
	if (hakios_metal_cb_clear(cb, &pass) != 0) {
		printf("mali_draw_test: a depth buffer cleared alone: refused: %s\n", hakios_metal_error());
		ok = 0;
	}
	ok = finish(cb, "a depth buffer cleared alone") && ok;
	hakios_metal_texture_read(z, 0, 0, 64, 64, after, 64 * 4);
	for (unsigned i = 0; i < 64 * 64; i++) cleared += after[i] == 0.5f;
	ok = ok && cleared == 4096;
	printf("mali_draw_test: a depth buffer cleared alone (no colour attachment): %u of 4096 read 0.5 (the first %g): %s\n", cleared,
	    (double)after[0], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(z);
	return ok;
}

/* 6a3. ANGLE's upload (the WebGL aquarium's big textures came out striped, 11-13): the picture replaced into a
 * one-level texture, copied on the GPU into level 0 of a texture with all its levels, the levels generated; level 0
 * read back is the picture */
static int
upload_copy_test(hakios_metal_device *d, hakios_metal_queue *q, unsigned w, unsigned h)
{
	unsigned levels = 1, right = 0;
	for (unsigned m = w > h ? w : h; m > 1; m >>= 1) levels++;
	hakios_metal_texture *src = hakios_metal_texture_new(d, 2, 70, w, h, 1);
	hakios_metal_texture *dst = hakios_metal_texture_new_levels(d, 2, 70, w, h, levels, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	uint32_t *px = malloc((size_t)w * h * 4), *back = calloc((size_t)w * h, 4);
	int ok = src != NULL && dst != NULL && px != NULL && back != NULL;

	if (!ok) {
		printf("mali_draw_test: an upload by copy, %ux%u: no textures (%s)\n", w, h, hakios_metal_error());
		return 0;
	}
	for (unsigned y = 0; y < h; y++) {
		for (unsigned x = 0; x < w; x++) {
			px[y * w + x] = 0xff000000u | (x & 255) | (y & 255) << 8 | ((x >> 8) << 4 | (y >> 8)) << 16;
		}
	}
	hakios_metal_texture_replace(src, 0, 0, w, h, px, (uint64_t)w * 4);
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_blit(cb, src, 0, 0, 0, w, h, dst, 0, 0, 0, w, h, 0) != 0) {
		printf("mali_draw_test: an upload by copy, %ux%u: the copy refused: %s\n", w, h, hakios_metal_error());
		ok = 0;
	}
	if (hakios_metal_cb_generate_mipmaps(cb, dst) != 0) {
		printf("mali_draw_test: an upload by copy, %ux%u: the levels refused: %s\n", w, h, hakios_metal_error());
		ok = 0;
	}
	ok = finish(cb, "an upload by copy") && ok;
	hakios_metal_texture_read(dst, 0, 0, w, h, back, (uint64_t)w * 4);
	unsigned first = w * h;
	for (unsigned i = 0; i < w * h; i++) {
		if (back[i] == px[i]) right++; else if (first == w * h) first = i;
	}
	ok = ok && right == w * h;
	if (first < w * h)
		printf("mali_draw_test: an upload by copy, %ux%u: the first wrong pixel at (%u, %u): %#x for %#x\n", w, h, first % w,
		    first / w, back[first], px[first]);
	printf("mali_draw_test: an upload by copy, %ux%u into level 0 of %u levels, the levels generated: %u of %u pixels right: %s\n", w, h,
	    levels, right, w * h, ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(src);
	hakios_metal_texture_free(dst);
	free(px);
	free(back);
	return ok;
}

/* 6a4. ANGLE's cube upload (the aquarium's skybox, 11-13): six one-level 2D pictures, each of one colour with its
 * face's number in it, copied on the GPU into the faces of a cube of SIZE with all its levels, the levels generated;
 * each face's level 0, level 1 and a small level read back are that face's colour */
static int
cube_upload_test(hakios_metal_device *d, hakios_metal_queue *q, unsigned size)
{
	unsigned levels = 1, right = 0, asked = 0;
	for (unsigned m = size; m > 1; m >>= 1) levels++;
	hakios_metal_texture *cube = hakios_metal_texture_new_layered(d, 5, 70, size, size, 6, levels, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *src[6] = { NULL };
	uint32_t *px = malloc((size_t)size * size * 4), *back = malloc((size_t)size * size * 4);
	int ok = cube != NULL && px != NULL && back != NULL;
	hakios_metal_cb *cb;

	if (!ok) {
		printf("mali_draw_test: a cube uploaded by copies, %u: no cube (%s)\n", size, hakios_metal_error());
		return 0;
	}
	cb = hakios_metal_cb_new(q);
	for (unsigned f = 0; f < 6; f++) {
		const uint32_t colour = 0xff000000u | (40 * (f + 1)) | (250 - 40 * f) << 8 | (f * 17 + 5) << 16;

		src[f] = hakios_metal_texture_new(d, 2, 70, size, size, 1);
		for (unsigned i = 0; i < size * size; i++) px[i] = colour;
		if (src[f] == NULL || hakios_metal_texture_replace(src[f], 0, 0, size, size, px, (uint64_t)size * 4) != 0 ||
		    hakios_metal_cb_blit_layer(cb, src[f], 0, 0, 0, 0, size, size, cube, 0, f, 0, 0) != 0) {
			printf("mali_draw_test: a cube uploaded by copies, %u: face %u refused: %s\n", size, f, hakios_metal_error());
			ok = 0;
		}
	}
	if (hakios_metal_cb_generate_mipmaps(cb, cube) != 0) {
		printf("mali_draw_test: a cube uploaded by copies, %u: the levels refused: %s\n", size, hakios_metal_error());
		ok = 0;
	}
	ok = finish(cb, "a cube uploaded by copies") && ok;
	for (unsigned f = 0; f < 6; f++) {
		const uint32_t colour = 0xff000000u | (40 * (f + 1)) | (250 - 40 * f) << 8 | (f * 17 + 5) << 16;
		const unsigned at[3] = { 0, 1, levels - 3 };

		for (unsigned k = 0; k < 3; k++) {
			const unsigned l = at[k], n = size >> l;
			unsigned good = 0;

			memset(back, 0, (size_t)n * n * 4);
			if (hakios_metal_texture_read_3d(cube, l, 0, 0, f, n, n, 1, back, (uint64_t)n * 4, 0) != 0) {
				printf("mali_draw_test: a cube uploaded by copies, %u: face %u level %u not read: %s\n", size, f, l, hakios_metal_error());
			}
			for (unsigned i = 0; i < n * n; i++) good += back[i] == colour;
			asked++;
			right += good == n * n;
			if (good != n * n)
				printf("mali_draw_test: a cube uploaded by copies, %u: face %u level %u: %u of %u pixels its colour %#x (the first %#x)\n",
				    size, f, l, good, n * n, colour, back[0]);
		}
	}
	ok = ok && right == asked;
	printf("mali_draw_test: a cube of %u with %u levels uploaded by copies into its faces, the levels generated: %u of %u (face, level) "
	    "read back right: %s\n", size, levels, right, asked, ok ? "WORK" : "FAILED");
	for (unsigned f = 0; f < 6; f++) if (src[f] != NULL) hakios_metal_texture_free(src[f]);
	hakios_metal_texture_free(cube);
	free(px);
	free(back);
	return ok;
}

/* 6a5. a cube SAMPLED (11-13: never checked on the phone until the aquarium's skybox): the cube of 6a4 -- faces of one
 * colour each, all its levels -- sampled over a 16x16 target along a direction from the fragment stage's buffer, off
 * the axis: the target is the colour of the face the direction's biggest component names (+X -X +Y -Y +Z -Z) */
static int
cube_sample_test(hakios_metal_device *d, hakios_metal_queue *q, unsigned size, unsigned mip_filter)
{
	unsigned levels = 1, right = 0;
	for (unsigned m = size; m > 1; m >>= 1) levels++;
	hakios_metal_texture *cube = hakios_metal_texture_new_layered(d, 5, 70, size, size, 6, levels, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 16, 16, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *p = hakios_metal_program_new_bitcode(tri_cover_vs_bc, tri_cover_vs_bc_len, tri_cube_fs_bc, tri_cube_fs_bc_len, 70);
	hakios_metal_buffer *params = hakios_metal_buffer_new(d, 4096), *dir = hakios_metal_buffer_new(d, 4096);
	struct hakios_metal_sampler_desc sd;
	hakios_metal_sampler *smp;
	uint32_t *px = malloc((size_t)size * size * 4), got[6] = { 0 }, back[256];
	static const float dirs[6][3] = { { 1, 0.4f, -0.3f }, { -1, -0.5f, 0.2f }, { 0.3f, 1, 0.6f }, { -0.2f, -1, -0.7f }, { 0.5f, -0.4f, 1 }, { -0.6f, 0.3f, -1 } };
	int ok;
	hakios_metal_cb *cb;

	memset(&sd, 0, sizeof(sd));
	sd.normalized_coordinates = 1;
	sd.min_filter = sd.mag_filter = 1;
	sd.mip_filter = mip_filter;
	sd.lod_max_clamp = 1000.0f;
	sd.max_anisotropy = 1;
	smp = hakios_metal_sampler_new(d, &sd);
	if (cube == NULL || t == NULL || p == NULL || params == NULL || dir == NULL || smp == NULL || px == NULL) {
		printf("mali_draw_test: a cube sampled: not made (%s): FAILED\n", hakios_metal_error());
		return 0;
	}
	ok = 1;
	for (unsigned f = 0; f < 6; f++) {
		const uint32_t colour = 0xff000000u | (40 * (f + 1)) | (250 - 40 * f) << 8 | (f * 17 + 5) << 16;

		for (unsigned i = 0; i < size * size; i++) px[i] = colour;
		ok = hakios_metal_texture_replace_3d(cube, 0, 0, 0, f, size, size, 1, px, (uint64_t)size * 4, 0) == 0 && ok;
	}
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_generate_mipmaps(cb, cube) != 0) ok = 0;
	ok = finish(cb, "a cube sampled: its levels") && ok;
	{
		float *f = hakios_metal_buffer_contents(params);

		memset(f, 0, 48);
		f[2] = 2.0f;
		f[5] = 2.0f;
	}
	for (unsigned f = 0; f < 6; f++) {
		const uint32_t colour = 0xff000000u | (40 * (f + 1)) | (250 - 40 * f) << 8 | (f * 17 + 5) << 16;
		struct hakios_metal_render_pass pass;
		hakios_metal_buffer *vb[1] = { params }, *fb[1] = { dir };
		uint64_t offs[1] = { 0 };
		hakios_metal_texture *texs[1] = { cube };
		hakios_metal_sampler *smps[1] = { smp };
		float *v = hakios_metal_buffer_contents(dir);
		unsigned good = 0;

		v[0] = dirs[f][0]; v[1] = dirs[f][1]; v[2] = dirs[f][2]; v[3] = 0;
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[3] = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw2(cb, p, &pass, NULL, NULL, 3, 0, 3, 1, vb, offs, 1, fb, offs, 1, texs, 1, smps, 1) != 0)
			printf("mali_draw_test: a cube sampled, face %u: refused: %s\n", f, hakios_metal_error());
		ok = finish(cb, "a cube sampled") && ok;
		memset(back, 0xee, sizeof(back));
		hakios_metal_texture_read(t, 0, 0, 16, 16, back, 64);
		for (unsigned i = 0; i < 256; i++) good += back[i] == colour;
		got[f] = back[0];
		right += good == 256;
		if (good != 256)
			printf("mali_draw_test: a cube sampled, direction (%g, %g, %g), face %u: %u of 256 its colour %#x, pixel 0 is %#x\n",
			    (double)dirs[f][0], (double)dirs[f][1], (double)dirs[f][2], f, good, colour, back[0]);
	}
	ok = ok && right == 6;
	printf("mali_draw_test: a cube of %u with %u levels sampled (mip filter %u) along six directions: %u of 6 give their face's colour "
	    "(%#x %#x %#x %#x %#x %#x): %s\n", size, levels, mip_filter, right, got[0], got[1], got[2], got[3], got[4], got[5], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(cube);
	hakios_metal_texture_free(t);
	free(px);
	return ok;
}

/* 6a6. what a buffer costs (11-13: WebKit's WebGL makes and frees one for every draw; the aquarium's 618 a frame took
 * seconds): COUNT buffers of BYTES made, then freed -- microseconds each. With IDLE 0 a command buffer is at work
 * meanwhile (a clear of a 1024x1024 target every round). Not a check: said. */
static void
buffer_cost(hakios_metal_device *d, hakios_metal_queue *q, unsigned count, uint64_t bytes, int busy)
{
	hakios_metal_buffer **b = calloc(count, sizeof(*b));
	hakios_metal_texture *t = busy ? hakios_metal_texture_new(d, 2, 70, 1024, 1024, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET) : NULL;
	struct timespec t0, t1, t2;
	unsigned made = 0;

	if (b == NULL) return;
	clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
	for (unsigned i = 0; i < count; i++) {
		if (busy && t != NULL && i % 16 == 0) {
			struct hakios_metal_render_pass pass;
			hakios_metal_cb *cb = hakios_metal_cb_new(q);

			memset(&pass, 0, sizeof(pass));
			pass.color0 = t;
			pass.load_action = 2;
			pass.store_action = 1;
			hakios_metal_cb_clear(cb, &pass);
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_free(cb);
		}
		b[i] = hakios_metal_buffer_new(d, bytes);
		made += b[i] != NULL;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	for (unsigned i = 0; i < count; i++) {
		if (b[i] != NULL) hakios_metal_buffer_free(b[i]);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
	printf("mali_draw_test: what a buffer costs: %u of %llu bytes%s: %.0f us each to make, %.0f us each to free\n", made,
	    (unsigned long long)bytes, busy ? ", command buffers at work meanwhile" : "",
	    ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3) / count,
	    ((t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3) / count);
	if (t != NULL) hakios_metal_texture_free(t);
	free(b);
}

/* 6a7. THE STAGES LINKED BY NAME (11-13, the WebGL aquarium): tri_link_vs gives seven varyings, tri_link_fs takes four
 * of them (the first two, the fifth and the seventh) and returns (ve.x, vg.y, ve.z, 1) = (0.25, 0.375, 0.75, 1) */
static int
link_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 32, 32, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *p = hakios_metal_program_new_bitcode(tri_link_vs_bc, tri_link_vs_bc_len, tri_link_fs_bc, tri_link_fs_bc_len, 70);
	struct hakios_metal_render_pass pass;
	uint32_t px[32 * 32];
	unsigned right = 0;
	int ok;
	hakios_metal_cb *cb;

	if (t == NULL || p == NULL) {
		printf("mali_draw_test: the stages linked by name: not made (%s): FAILED\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[3] = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, p, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: the stages linked by name: refused: %s\n", hakios_metal_error());
	ok = finish(cb, "the stages linked by name");
	memset(px, 0xee, sizeof(px));
	hakios_metal_texture_read(t, 0, 0, 32, 32, px, 128);
	for (unsigned i = 0; i < 1024; i++) {
		int r = (int)(px[i] & 255) - 64, g = (int)(px[i] >> 8 & 255) - 96, b = (int)(px[i] >> 16 & 255) - 191;
		right += r >= -1 && r <= 1 && g >= -1 && g <= 1 && b >= -1 && b <= 1 && px[i] >> 24 == 255;
	}
	ok = ok && right == 1024;
	printf("mali_draw_test: the stages linked by name (seven varyings given, four taken): %u of 1024 pixels (0.25, 0.375, 0.75, 1), "
	    "pixel 0 is 0x%08x for 0xffbf6040: %s\n", right, px[0], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
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

/* 6c. blending in the shader: the blend colour (flat over grey, BlendColor / OneMinusBlendColor, BlendAlpha /
 * OneMinusBlendAlpha, colour (0.25, 0.5, 0.75, 1): (104, 96, 128, 255) wanted) and dual source (tri_color through dual_fs,
 * Source1Color / Source1Alpha times the source, plus zero: each channel times (0.5, 0.25, 1, 0.75) -- green top right a
 * quarter, alpha 191) */
static int
near8(uint32_t px, unsigned r, unsigned g, unsigned b, unsigned a)
{
	int d[4] = { (int)(px & 0xff) - (int)r, (int)((px >> 8) & 0xff) - (int)g, (int)((px >> 16) & 0xff) - (int)b,
	    (int)(px >> 24) - (int)a };
	for (int i = 0; i < 4; i++) if (d[i] < -3 || d[i] > 3) return 0;
	return 1;
}

static int
blend_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *bl = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 70);
	hakios_metal_program *du = hakios_metal_program_new_bitcode(tri_color_vs_bc, tri_color_vs_bc_len, dual_fs_bc,
	    dual_fs_bc_len, 70);
	struct hakios_metal_render_pass pass;
	static uint32_t px[64 * 64];
	unsigned right = 0;
	int ok = 1;

	if (t == NULL || bl == NULL || du == NULL) {
		printf("mali_draw_test: blend: no texture or program (%s)\n", hakios_metal_error());
		return 0;
	}
	hakios_metal_program_set_blend(bl, 1, 0, 0, 11, 12, 13, 14, 0xf);
	hakios_metal_program_set_blend(du, 1, 0, 0, 15, 0, 17, 0, 0xf);
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	pass.blend_color[0] = 0.25f;
	pass.blend_color[1] = 0.5f;
	pass.blend_color[2] = 0.75f;
	pass.blend_color[3] = 1;
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, bl, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: blend colour: refused: %s\n", hakios_metal_error());
	ok = finish(cb, "blend colour") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 64 * 64; i++) right += near8(px[i], 104, 96, 128, 255);
	ok = ok && right == 4096;
	printf("mali_draw_test: blend colour: %u of 4096 near (104, 96, 128, 255), the first 0x%08x: %s\n", right, px[0],
	    right == 4096 ? "WORK" : "FAILED");

	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, du, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: dual source: refused: %s\n", hakios_metal_error());
	ok = finish(cb, "dual source") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	{
		/* tri_color alone: top left 0xfffd0002 (blue), top right 0xff02fb02 (green) */
		uint32_t tl = px[0], tr = px[63];
		int good = near8(tl, 1, 0, 253, 191) && near8(tr, 1, 63, 2, 191);
		ok = ok && good;
		printf("mali_draw_test: dual source: top left 0x%08x (0xbffd0001 wanted), top right 0x%08x (0xbf023f01 wanted): %s\n",
		    tl, tr, good ? "WORK" : "FAILED");
	}
	hakios_metal_texture_free(t);
	return ok;
}

/* 6d. MSAA (4 samples): the lower-left half into a multisample target, resolved (hakios_metal_cb_blit) into one of a
 * sample: the inside tri_flat, the outside black, and along the diagonal pixels a quarter, half or three quarters
 * covered -- none without the samples; then the target loaded in a new command buffer (the per-sample preload), a draw
 * that covers nothing, stored and resolved again: the same picture */
static int
msaa_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *ms = hakios_metal_texture_new_multisample(d, 70, 64, 64, 4, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *rs = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *f4 = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 70);
	struct hakios_metal_render_pass pass;
	struct hakios_metal_scissor corner = { 63, 0, 1, 1 };
	static uint32_t a[64 * 64], b[64 * 64];
	unsigned inside = 0, outside = 0, edge = 0, odd = 0, same = 0;
	int ok = 1;

	if (ms == NULL || rs == NULL || f4 == NULL) {
		printf("mali_draw_test: msaa: no textures or program (%s)\n", hakios_metal_error());
		return 0;
	}
	hakios_metal_program_set_samples(f4, 4);
	memset(&pass, 0, sizeof(pass));
	pass.color0 = ms;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[3] = 1;
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, f4, &pass, NULL, NULL, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: msaa: the draw refused: %s\n", hakios_metal_error());
	if (hakios_metal_cb_blit(cb, ms, 0, 0, 0, 64, 64, rs, 0, 0, 0, 64, 64, 0) != 0)
		printf("mali_draw_test: msaa: the resolve refused: %s\n", hakios_metal_error());
	ok = finish(cb, "msaa") && ok;
	hakios_metal_texture_read(rs, 0, 0, 64, 64, a, 256);
	for (unsigned i = 0; i < 64 * 64; i++) {
		unsigned r = a[i] & 0xff, g = (a[i] >> 8) & 0xff, bl = (a[i] >> 16) & 0xff;
		if (a[i] == 0xff804020u) inside++;
		else if (a[i] == 0xff000000u) outside++;
		else if (((r >= 7 && r <= 9) || (r >= 15 && r <= 17) || (r >= 23 && r <= 25)) && g >= r * 2 - 2 && g <= r * 2 + 2 &&
		         bl >= r * 4 - 3 && bl <= r * 4 + 3) edge++;
		else odd++;
	}
	pass.load_action = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, f4, &pass, NULL, &corner, 3, 0, 3, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: msaa: the loading draw refused: %s\n", hakios_metal_error());
	hakios_metal_cb_blit(cb, ms, 0, 0, 0, 64, 64, rs, 0, 0, 0, 64, 64, 0);
	ok = finish(cb, "msaa load") && ok;
	hakios_metal_texture_read(rs, 0, 0, 64, 64, b, 256);
	for (unsigned i = 0; i < 64 * 64; i++) same += a[i] == b[i];
	ok = ok && inside > 1900 && outside > 1900 && edge >= 32 && odd == 0 && same == 4096;
	printf("mali_draw_test: msaa: resolved: inside %u, outside %u, edge pixels partly covered %u, other %u (0 wanted); loaded "
	    "and resolved again: %u of 4096 the same: %s\n", inside, outside, edge, odd, same, ok ? "WORK" : "FAILED");
	{
		/* a multisample depth attachment: tri_flat written at z 0 (Less), stored; in a new command buffer, the depth
		 * loaded (a sample a run), tri_color at the same z, Less: fails everywhere -- resolved, all tri_flat */
		hakios_metal_texture *z4 = hakios_metal_texture_new_multisample(d, 252, 64, 64, 4, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		hakios_metal_program *c4 = hakios_metal_program_new_bitcode(tri_color_vs_bc, tri_color_vs_bc_len, tri_color_fs_bc,
		    tri_color_fs_bc_len, 70);
		unsigned kept = 0;
		hakios_metal_program_set_samples(c4, 4);
		memset(&pass, 0, sizeof(pass));
		pass.color0 = ms;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.depth = z4;
		pass.depth_load_action = 2;
		pass.depth_store_action = 1;
		pass.clear_depth = 1.0;
		pass.depth_state_set = 1;
		pass.depth_compare = 1;
		pass.depth_write = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, f4, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: msaa depth: refused: %s\n", hakios_metal_error());
		ok = finish(cb, "msaa depth") && ok;
		pass.load_action = 1;
		pass.depth_load_action = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, c4, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: msaa depth: the second draw refused: %s\n", hakios_metal_error());
		hakios_metal_cb_blit(cb, ms, 0, 0, 0, 64, 64, rs, 0, 0, 0, 64, 64, 0);
		ok = finish(cb, "msaa depth load") && ok;
		hakios_metal_texture_read(rs, 0, 0, 64, 64, a, 256);
		for (unsigned i = 0; i < 64 * 64; i++) kept += a[i] == 0xff804020u;
		ok = ok && kept == 4096;
		printf("mali_draw_test: msaa depth: loaded in a new command buffer, Less kept %u of 4096: %s\n", kept,
		    kept == 4096 ? "WORK" : "FAILED");
		hakios_metal_texture_free(z4);
	}
	hakios_metal_texture_free(ms);
	hakios_metal_texture_free(rs);
	return ok;
}

/* 6e. sampling an RGBA16Float texture (CoreAnimation's 256x1 gradient tables, format 115, in the draws that turned the
 * screen cyan/blue, 11-09): a 256x1 gradient of halves (r = i/255, g = 1 - i/255, b = 0.5, a = 1) drawn scaled into a
 * 128x1 RGBA8 texture (the driver's blit samples it): each byte within 3 of what it should be */
static int
half_texture_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *h = hakios_metal_texture_new(d, 2, 115, 256, 1, 1);
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 128, 1, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static __fp16 src[256 * 4];
	uint32_t out[128];
	unsigned right = 0;
	int ok;

	if (h == NULL || t == NULL) {
		printf("mali_draw_test: RGBA16Float: no textures (%s)\n", hakios_metal_error());
		return 0;
	}
	for (int i = 0; i < 256; i++) {
		src[4 * i] = (__fp16)(i / 255.0f);
		src[4 * i + 1] = (__fp16)(1.0f - i / 255.0f);
		src[4 * i + 2] = (__fp16)0.5f;
		src[4 * i + 3] = (__fp16)1.0f;
	}
	hakios_metal_texture_replace(h, 0, 0, 256, 1, src, sizeof(src));
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_blit(cb, h, 0, 0, 0, 256, 1, t, 0, 0, 0, 128, 1, 0) != 0)
		printf("mali_draw_test: RGBA16Float: the blit refused: %s\n", hakios_metal_error());
	ok = finish(cb, "RGBA16Float");
	hakios_metal_texture_read(t, 0, 0, 128, 1, out, sizeof(out));
	for (int x = 0; x < 128; x++) {
		/* destination pixel x samples the source at (x + 0.5) * 2 -- texel 2x or 2x+1, nearest */
		int i = 2 * x + 1, r = (int)(out[x] & 0xff), g = (int)((out[x] >> 8) & 0xff), b = (int)((out[x] >> 16) & 0xff);
		int a = (int)(out[x] >> 24);
		right += abs(r - i) <= 3 && abs(g - (255 - i)) <= 3 && abs(b - 128) <= 3 && a == 255;
	}
	ok = ok && right >= 120;
	printf("mali_draw_test: RGBA16Float sampled: %u of 128 right (pixel 0 0x%08x, 64 0x%08x, 127 0x%08x): %s\n", right, out[0],
	    out[64], out[127], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(h);
	hakios_metal_texture_free(t);
	return ok;
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

/* (irmsaa: a 4-sample 1080x2340 target grey, then loaded, INSTANCES lower-left halves -- through incremental
 * rendering's flushes, each later part loading every sample -- resolved: the top right still grey) */
static int
ms_ir_test(hakios_metal_device *d, hakios_metal_queue *q, unsigned instances)
{
	const unsigned w = 1080, h = 2340;
	hakios_metal_texture *ms = hakios_metal_texture_new_multisample(d, 70, w, h, 4, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *rs = hakios_metal_texture_new(d, 2, 70, w, h, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *f4 = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 70);
	struct hakios_metal_render_pass pass;
	struct timespec t0, t1;
	uint32_t bl[64], tr[64];
	int st, ok;

	if (ms == NULL || rs == NULL || f4 == NULL) {
		printf("mali_draw_test: irmsaa: no textures (%s)\n", hakios_metal_error());
		return 0;
	}
	hakios_metal_program_set_samples(f4, 4);
	memset(&pass, 0, sizeof(pass));
	pass.color0 = ms;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	hakios_metal_cb *cb = hakios_metal_cb_new(q);
	hakios_metal_cb_clear(cb, &pass);
	pass.load_action = 1;
	if (hakios_metal_cb_draw(cb, f4, &pass, NULL, NULL, 3, 0, 3, instances, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: irmsaa: refused: %s\n", hakios_metal_error());
	hakios_metal_cb_blit(cb, ms, 0, 0, 0, w, h, rs, 0, 0, 0, w, h, 0);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	hakios_metal_cb_commit(cb, NULL, NULL);
	hakios_metal_cb_wait(cb);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	st = hakios_metal_cb_status(cb);
	hakios_metal_cb_free(cb);
	hakios_metal_texture_read(rs, 0, h - 1, 64, 1, bl, sizeof(bl));
	hakios_metal_texture_read(rs, w - 64, 0, 64, 1, tr, sizeof(tr));
	ok = st == HAKIOS_METAL_COMPLETED && bl[0] == 0xff804020u && tr[63] == 0xff808080u;
	printf("mali_draw_test: %u lower-left halves over a loaded grey 4-sample target: status %d in %ld ms, bottom left 0x%08x, "
	    "top right 0x%08x: %s\n", instances, st, (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000),
	    bl[0], tr[63], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(ms);
	hakios_metal_texture_free(rs);
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

/* 7b. colour attachments of one and two 8-bit channels (R8Unorm 10, RG8Unorm 30; 11-12: what widgets draw into): the
 * draw of draw() into each, every pixel the red (and green) of the RGBA8 picture */
static int
narrow_target_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	static const struct { unsigned format, bytes; const char *name; } kinds[2] = { { 10, 1, "R8Unorm" }, { 30, 2, "RG8Unorm" } };
	uint32_t ref[64 * 64];
	long us;
	int ok = 1;

	if (draw(d, q, flat, ref, &us) != HAKIOS_METAL_COMPLETED) {
		return 0;
	}
	for (int k = 0; k < 2; k++) {
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, kinds[k].format, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		struct hakios_metal_render_pass pass;
		uint8_t px[64 * 64 * 2];
		unsigned right = 0, drawn = 0;
		int st = -1;

		memset(&pass, 0, sizeof(pass));
		memset(px, 0xee, sizeof(px));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
		pass.clear_color[3] = 1;
		if (t == NULL || hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0) {
			printf("mali_draw_test: a %s colour attachment: refused: %s: FAILED\n", kinds[k].name, hakios_metal_error());
			ok = 0;
		} else {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st = hakios_metal_cb_status(cb);
			hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * kinds[k].bytes);
			for (unsigned i = 0; i < 64 * 64; i++) {
				uint32_t want = ref[i] & (kinds[k].bytes == 1 ? 0xffu : 0xffffu);
				uint32_t got = kinds[k].bytes == 1 ? px[i] : (uint32_t)px[i * 2] | (uint32_t)px[i * 2 + 1] << 8;
				right += got == want;
				drawn += ref[i] == 0xff804020u;
			}
			printf("mali_draw_test: a %s colour attachment: status %d, %u of 4096 pixels right (%u of them drawn, pixel 0 0x%x, the "
			    "middle 0x%x): %s\n", kinds[k].name, st, right, drawn, kinds[k].bytes == 1 ? px[0] : px[0] | px[1] << 8,
			    kinds[k].bytes == 1 ? px[32 * 64 + 32] : px[(32 * 64 + 32) * 2] | px[(32 * 64 + 32) * 2 + 1] << 8,
			    st == HAKIOS_METAL_COMPLETED && right == 4096 ? "WORK" : "FAILED");
			ok = ok && st == HAKIOS_METAL_COMPLETED && right == 4096;
		}
		hakios_metal_cb_free(cb);
		if (t) hakios_metal_texture_free(t);
	}
	return ok;
}

/* 7b'. a half-float colour attachment (RG16Float 65, R16Float 25; 11-12: the widgets' accumulators): a clear alone to
 * (0.25, 0.75) -- halves 0x3400 and 0x3a00 --, then the flat draw (red 0x20/255, green 0x40/255) loaded over it */
static float
half_value(uint16_t h)
{
	int e = (h >> 10) & 31;
	float m = (float)(h & 0x3ff) / 1024.0f, v = e == 0 ? m / 16384.0f : (1.0f + m) * (e >= 15 ? (float)(1u << (e - 15)) : 1.0f / (float)(1u << (15 - e)));

	return h & 0x8000 ? -v : v;
}

/* 7b''. an RG11B10Float colour attachment (92; 11-13): a clear to (0.25, 0.75, 0.125) -- the packed word 0x601d0340 --,
 * then the flat draw (red 0x20/255, green 0x40/255, blue 0x80/255) loaded over it, each channel unpacked */
static float
small_float(uint32_t bits, unsigned mantissa_bits)
{
	unsigned e = bits >> mantissa_bits, m = bits & ((1u << mantissa_bits) - 1);
	float f = (float)m / (float)(1u << mantissa_bits);

	return e == 0 ? f / 16384.0f : (1.0f + f) * (e >= 15 ? (float)(1u << (e - 15)) : 1.0f / (float)(1u << (15 - e)));
}

static int
packed_float_target_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 92, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static uint32_t px[64 * 64];
	struct hakios_metal_render_pass pass;
	unsigned cleared = 0, drawn = 0;
	int st1 = -1, st2 = -1, ok;
	hakios_metal_cb *cb;

	if (t == NULL) {
		printf("mali_draw_test: an RG11B10Float colour attachment: no texture: %s: FAILED\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = 0.25;
	pass.clear_color[1] = 0.75;
	pass.clear_color[2] = 0.125;
	pass.clear_color[3] = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_clear(cb, &pass) == 0) {
		hakios_metal_cb_commit(cb, NULL, NULL);
		hakios_metal_cb_wait(cb);
		st1 = hakios_metal_cb_status(cb);
		memset(px, 0xee, sizeof(px));
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
		for (unsigned i = 0; i < 4096; i++) cleared += px[i] == 0x601d0340u;
	} else {
		printf("mali_draw_test: an RG11B10Float colour attachment: the clear refused: %s\n", hakios_metal_error());
	}
	hakios_metal_cb_free(cb);
	{
		uint32_t first = px[0];

		pass.load_action = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0) {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st2 = hakios_metal_cb_status(cb);
			memset(px, 0xee, sizeof(px));
			hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
			for (unsigned i = 0; i < 4096; i++) {
				float r = small_float(px[i] & 0x7ff, 6) - 32.0f / 255.0f, g = small_float((px[i] >> 11) & 0x7ff, 6) - 64.0f / 255.0f;
				float b = small_float(px[i] >> 22, 5) - 128.0f / 255.0f;

				drawn += r > -0.004f && r < 0.004f && g > -0.006f && g < 0.006f && b > -0.02f && b < 0.02f;
			}
		} else {
			printf("mali_draw_test: an RG11B10Float colour attachment: the draw refused: %s\n", hakios_metal_error());
		}
		hakios_metal_cb_free(cb);
		ok = st1 == HAKIOS_METAL_COMPLETED && st2 == HAKIOS_METAL_COMPLETED && cleared == 4096 && drawn == 4096;
		printf("mali_draw_test: an RG11B10Float colour attachment: the clear status %d, %u of 4096 cleared (pixel 0 0x%08x, "
		    "0x601d0340 expected); the draw status %d, %u of 4096 drawn (pixel 0 0x%08x): %s\n", st1, cleared, first, st2, drawn,
		    px[0], ok ? "WORK" : "FAILED");
	}
	hakios_metal_texture_free(t);
	return ok;
}

/* 7d. A MAP'S PASS (11-13: VectorKit's, black on the phone): into a BGRA8 sRGB target with depth+stencil (260), 1 or 4
 * samples, nothing loaded and depth/stencil not stored, as Maps asks -- (1) the flat quad, depth Always and written,
 * stencil Always/Replace reference 128; (2) the colour quad, depth LessEqual unwritten, stencil Equal 128 under read
 * mask 0x80: drawn everywhere; (3) the flat quad again with colour write mask 0, no stencil: changes nothing. Told
 * apart by what a pass of (1) alone leaves. */
static int
map_pass_test(hakios_metal_device *d, hakios_metal_queue *q, unsigned samples)
{
	const unsigned rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	hakios_metal_texture *shown = hakios_metal_texture_new(d, 2, 81, 64, 64, rt);
	hakios_metal_texture *t = samples > 1 ? hakios_metal_texture_new_multisample(d, 81, 64, 64, samples, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET) : shown;
	hakios_metal_texture *z = samples > 1 ? hakios_metal_texture_new_multisample(d, 260, 64, 64, samples, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET)
	    : hakios_metal_texture_new_levels(d, 2, 260, 64, 64, 1, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *flat = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc, tri_flat_fs_bc_len, 81);
	hakios_metal_program *masked = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc, tri_flat_fs_bc_len, 81);
	hakios_metal_program *col = hakios_metal_program_new_bitcode(tri_color_vs_bc, tri_color_vs_bc_len, tri_color_fs_bc, tri_color_fs_bc_len, 81);
	static uint32_t base[64 * 64], px[64 * 64];
	unsigned counts[3] = { 0, 0, 0 };
	int ok = 1;

	if (shown == NULL || t == NULL || z == NULL || flat == NULL || masked == NULL || col == NULL) {
		printf("mali_draw_test: a map's pass, %u sample(s): not made (%s): FAILED\n", samples, hakios_metal_error());
		return 0;
	}
	if (samples > 1) {
		hakios_metal_program_set_samples(flat, samples);
		hakios_metal_program_set_samples(masked, samples);
		hakios_metal_program_set_samples(col, samples);
	}
	hakios_metal_program_set_blend(masked, 0, 0, 0, 1, 0, 1, 0, 0);          /* (no blending, write mask 0) */
	/* stage 0: (1) alone; stage 1: (1) (2); stage 2: (1) (2) (3) -- each in one command buffer, as one pass */
	for (int stage = 0; stage < 3; stage++) {
		struct hakios_metal_render_pass pass;
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		int asked;

		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 0;
		pass.store_action = 1;
		pass.depth = z;
		pass.stencil = z;
		pass.depth_state_set = 1;
		pass.depth_compare = 7;
		pass.depth_write = 1;
		pass.stencil_state_set = 1;
		pass.stencil_front.compare = pass.stencil_back.compare = 7;
		pass.stencil_front.pass = pass.stencil_back.pass = 2;
		pass.stencil_front.depth_fail = pass.stencil_back.depth_fail = 2;
		pass.stencil_front.read_mask = pass.stencil_back.read_mask = 0;
		pass.stencil_front.write_mask = pass.stencil_back.write_mask = 0xff;
		pass.stencil_front_reference = pass.stencil_back_reference = 128;
		asked = hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0;
		pass.load_action = pass.depth_load_action = pass.stencil_load_action = 1;
		if (stage >= 1) {
			pass.depth_compare = 3;
			pass.depth_write = 0;
			pass.stencil_front.compare = pass.stencil_back.compare = 2;
			pass.stencil_front.pass = pass.stencil_back.pass = 0;
			pass.stencil_front.depth_fail = pass.stencil_back.depth_fail = 0;
			pass.stencil_front.read_mask = pass.stencil_back.read_mask = 0x80;
			pass.stencil_front.write_mask = pass.stencil_back.write_mask = 0;
			asked = asked && hakios_metal_cb_draw(cb, col, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0;
		}
		if (stage >= 2) {
			pass.depth_compare = 7;
			pass.stencil_state_set = 0;
			asked = asked && hakios_metal_cb_draw(cb, masked, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0;
		}
		if (!asked) {
			printf("mali_draw_test: a map's pass, %u sample(s), stage %d: refused: %s\n", samples, stage, hakios_metal_error());
			ok = 0;
		}
		if (samples > 1) hakios_metal_cb_blit(cb, t, 0, 0, 0, 64, 64, shown, 0, 0, 0, 64, 64, 0);
		ok = finish(cb, "a map's pass") && ok;
		hakios_metal_texture_read(shown, 0, 0, 64, 64, stage == 0 ? base : px, 256);
		for (unsigned i = 0; i < 4096; i++) {
			counts[stage] += stage == 0 ? base[i] == base[0] : px[i] != base[i];
		}
	}
	ok = ok && counts[0] == 4096 && counts[1] == 4096 && counts[2] == 4096;
	printf("mali_draw_test: a map's pass, %u sample(s): the flat quad alone one colour over %u of 4096 (0x%08x); the colour quad "
	    "through the stencil it wrote: %u of 4096 drawn; a masked quad after it: %u of 4096 still the colour quad's (pixel 0 "
	    "0x%08x): %s\n", samples, counts[0], base[0], counts[1], counts[2], px[0], ok ? "WORK" : "FAILED");
	if (t != shown) hakios_metal_texture_free(t);
	hakios_metal_texture_free(shown);
	hakios_metal_texture_free(z);
	return ok;
}

static int
float_target_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	static const struct { unsigned format, n; const char *name; } kinds[3] = { { 65, 2, "RG16Float" }, { 25, 1, "R16Float" },
	    { 115, 4, "RGBA16Float" } };
	int ok = 1;

	for (int k = 0; k < 3; k++) {
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, kinds[k].format, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		static uint16_t px[64 * 64 * 4];
		struct hakios_metal_render_pass pass;
		unsigned cleared = 0, drawn = 0, n = kinds[k].n;
		int st1 = -1, st2 = -1;
		hakios_metal_cb *cb;

		if (t == NULL) {
			printf("mali_draw_test: a %s colour attachment: no texture: %s: FAILED\n", kinds[k].name, hakios_metal_error());
			ok = 0;
			continue;
		}
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[0] = 0.25;
		pass.clear_color[1] = 0.75;
		pass.clear_color[2] = 0.125;            /* (halves 0x3000 and 0x3c00: an RGBA16Float's second word) */
		pass.clear_color[3] = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_clear(cb, &pass) == 0) {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st1 = hakios_metal_cb_status(cb);
			memset(px, 0xee, sizeof(px));
			hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 2 * n);
			for (unsigned i = 0; i < 4096; i++) {
				cleared += px[i * n] == 0x3400 && (n == 1 || px[i * n + 1] == 0x3a00) &&
				    (n < 4 || (px[i * n + 2] == 0x3000 && px[i * n + 3] == 0x3c00));
			}
		} else {
			printf("mali_draw_test: a %s colour attachment: the clear refused: %s\n", kinds[k].name, hakios_metal_error());
		}
		hakios_metal_cb_free(cb);
		pass.load_action = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0) {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st2 = hakios_metal_cb_status(cb);
			memset(px, 0xee, sizeof(px));
			hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 2 * n);
			for (unsigned i = 0; i < 4096; i++) {
				float r = half_value(px[i * n]) - 32.0f / 255.0f, g = n == 1 ? 0 : half_value(px[i * n + 1]) - 64.0f / 255.0f;
				float b = n < 4 ? 0 : half_value(px[i * n + 2]) - 128.0f / 255.0f, a = n < 4 ? 0 : half_value(px[i * n + 3]) - 1.0f;
				drawn += r > -0.004f && r < 0.004f && g > -0.004f && g < 0.004f && b > -0.004f && b < 0.004f && a > -0.004f &&
				    a < 0.004f;
			}
		} else {
			printf("mali_draw_test: a %s colour attachment: the draw refused: %s\n", kinds[k].name, hakios_metal_error());
		}
		hakios_metal_cb_free(cb);
		printf("mali_draw_test: a %s colour attachment: the clear status %d, %u of 4096 cleared; the draw status %d, %u of 4096 "
		    "drawn (pixel 0 0x%04x 0x%04x): %s\n", kinds[k].name, st1, cleared, st2, drawn, px[0], n == 1 ? 0 : px[1],
		    st1 == HAKIOS_METAL_COMPLETED && st2 == HAKIOS_METAL_COMPLETED && cleared == 4096 && drawn == 4096 ? "WORK" : "FAILED");
		ok = ok && st1 == HAKIOS_METAL_COMPLETED && st2 == HAKIOS_METAL_COMPLETED && cleared == 4096 && drawn == 4096;
		hakios_metal_texture_free(t);
	}
	return ok;
}

/* 7b*. A BLEND THE SHADER COMPUTES, INTO HALF FLOATS (11-13): the blend colour's factors (fixed function has none: the
 * shader reads the target and blends) into an RG16Float and an RGBA16Float cleared to 0.5 -- red 0.1255 x 0.25 + 0.5 x
 * 0.75 = 0.4064, green 0.251 x 0.5 + 0.5 x 0.5 = 0.3755, as "blend colour" has them in 8 bits */
static int
float_shader_blend_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	static const struct { unsigned format, n; const char *name; } kinds[2] = { { 65, 2, "RG16Float" }, { 115, 4, "RGBA16Float" } };
	int ok = 1;

	for (int k = 0; k < 2; k++) {
		const unsigned n = kinds[k].n;
		hakios_metal_texture *t = hakios_metal_texture_new(d, 2, kinds[k].format, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
		hakios_metal_program *bl = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
		    tri_flat_fs_bc_len, kinds[k].format);
		struct hakios_metal_render_pass pass;
		static uint16_t px[64 * 64 * 4];
		unsigned right = 0;
		int good;
		hakios_metal_cb *cb;

		if (t == NULL || bl == NULL) {
			printf("mali_draw_test: a shader's blend into %s: no texture or program (%s): FAILED\n", kinds[k].name, hakios_metal_error());
			ok = 0;
			continue;
		}
		hakios_metal_program_set_attachment_format(bl, 0, kinds[k].format);
		hakios_metal_program_set_blend(bl, 1, 0, 0, 11, 12, 13, 14, 0xf);
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
		pass.clear_color[3] = 1;
		pass.blend_color[0] = 0.25f;
		pass.blend_color[1] = 0.5f;
		pass.blend_color[2] = 0.75f;
		pass.blend_color[3] = 1;
		cb = hakios_metal_cb_new(q);
		if (hakios_metal_cb_draw(cb, bl, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: a shader's blend into %s: refused: %s\n", kinds[k].name, hakios_metal_error());
		good = finish(cb, "a shader's blend into half floats");
		memset(px, 0xee, sizeof(px));
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 2 * n);
		for (unsigned i = 0; i < 4096; i++) {
			float r = half_value(px[i * n]) - 0.4064f, g = half_value(px[i * n + 1]) - 0.3755f;
			right += r > -0.006f && r < 0.006f && g > -0.006f && g < 0.006f;
		}
		good = good && right == 4096;
		printf("mali_draw_test: a shader's blend into %s: %u of 4096 near (0.4064, 0.3755), the first %d %d /10000: %s\n",
		    kinds[k].name, right, (int)(half_value(px[0]) * 10000.0f), (int)(half_value(px[1]) * 10000.0f), good ? "WORK" : "FAILED");
		ok = ok && good;
		hakios_metal_texture_free(t);
	}
	return ok;
}

/* 7b**. AN sRGB TARGET (11-13; RGBA8Unorm_sRGB, 71): what a shader or a clear gives is linear and the memory holds it
 * encoded -- a clear to (0.125, 0.25, 0.5) is 0x63 0x89 0xbc; a blend the shader computes over a clear to 0.5 (it reads
 * the target back decoded): 0.4064, 0.3755, 0.5015 linear, 171 165 188 encoded */
static int
srgb_target_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 71, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *bl = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc, tri_flat_fs_bc_len, 71);
	struct hakios_metal_render_pass pass;
	static uint32_t px[64 * 64];
	unsigned cleared = 0, blended = 0;
	uint32_t first;
	int ok;
	hakios_metal_cb *cb;

	if (t == NULL || bl == NULL) {
		printf("mali_draw_test: an sRGB target: no texture or program (%s): FAILED\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = 0.125;
	pass.clear_color[1] = 0.25;
	pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_clear(cb, &pass) != 0) printf("mali_draw_test: an sRGB target: the clear refused: %s\n", hakios_metal_error());
	ok = finish(cb, "an sRGB target's clear");
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) cleared += px[i] == 0xffbc8963u;
	first = px[0];
	hakios_metal_program_set_attachment_format(bl, 0, 71);
	hakios_metal_program_set_blend(bl, 1, 0, 0, 11, 12, 13, 14, 0xf);
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.blend_color[0] = 0.25f;
	pass.blend_color[1] = 0.5f;
	pass.blend_color[2] = 0.75f;
	pass.blend_color[3] = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, bl, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: an sRGB target: the blended draw refused: %s\n", hakios_metal_error());
	ok = finish(cb, "an sRGB target's shader blend") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) {
		int r = (int)(px[i] & 0xff) - 171, g = (int)(px[i] >> 8 & 0xff) - 165, b = (int)(px[i] >> 16 & 0xff) - 188;
		blended += r >= -3 && r <= 3 && g >= -3 && g <= 3 && b >= -3 && b <= 3;
	}
	ok = ok && cleared == 4096 && blended == 4096;
	printf("mali_draw_test: an sRGB target: cleared %u of 4096 to 0xffbc8963 (pixel 0 0x%08x); a shader's blend over it %u of 4096 "
	    "near 171 165 188 (pixel 0 0x%08x): %s\n", cleared, first, blended, px[0], ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	return ok;
}

/* 7e. ONE TRIANGLE OVER THE TARGET, A TEXTURE THROUGH A BOUND SAMPLER (11-13: tri_cover_vs/fs, the shape of a GL
 * layer's blit -- WebKit's WebGL through ANGLE left its canvas empty): a 64x64 texture of four quadrants (red, green /
 * blue, white) drawn 1:1 into a 64x64 target cleared to grey, the coordinates (0,0) (2,0) (0,2) from the buffer both
 * stages read */
static int
cover_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	const unsigned rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, rt), *from = hakios_metal_texture_new(d, 2, 70, 64, 64, 1);
	hakios_metal_program *p = hakios_metal_program_new_bitcode(tri_cover_vs_bc, tri_cover_vs_bc_len, tri_cover_fs_bc, tri_cover_fs_bc_len, 70);
	hakios_metal_buffer *params = hakios_metal_buffer_new(d, 4096);
	struct hakios_metal_sampler_desc sd;
	hakios_metal_sampler *smp;
	struct hakios_metal_render_pass pass;
	static uint32_t px[64 * 64], want[64 * 64];
	unsigned right = 0, grey = 0;
	int ok;
	hakios_metal_cb *cb;

	memset(&sd, 0, sizeof(sd));
	sd.normalized_coordinates = 1;
	sd.lod_max_clamp = 1000.0f;
	sd.max_anisotropy = 1;
	smp = hakios_metal_sampler_new(d, &sd);
	if (t == NULL || from == NULL || p == NULL || params == NULL || smp == NULL) {
		printf("mali_draw_test: a covering triangle: not made (%s): FAILED\n", hakios_metal_error());
		return 0;
	}
	for (unsigned i = 0; i < 4096; i++) {
		unsigned x = i % 64, y = i / 64;
		want[i] = y < 32 ? (x < 32 ? 0xff0000ffu : 0xff00ff00u) : (x < 32 ? 0xffff0000u : 0xffffffffu);
	}
	hakios_metal_texture_replace(from, 0, 0, 64, 64, want, 256);
	{
		float *f = hakios_metal_buffer_contents(params);

		memset(f, 0, 48);
		f[2] = 2.0f;            /* uv[1] = (2, 0) */
		f[5] = 2.0f;            /* uv[2] = (0, 2) */
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = pass.clear_color[1] = pass.clear_color[2] = 0.5;
	pass.clear_color[3] = 1;
	cb = hakios_metal_cb_new(q);
	{
		hakios_metal_buffer *bufs[1] = { params };
		uint64_t offs[1] = { 0 };
		hakios_metal_texture *texs[1] = { from };
		hakios_metal_sampler *smps[1] = { smp };

		if (hakios_metal_cb_draw2(cb, p, &pass, NULL, NULL, 3, 0, 3, 1, bufs, offs, 1, bufs, offs, 1, texs, 1, smps, 1) != 0)
			printf("mali_draw_test: a covering triangle: refused: %s\n", hakios_metal_error());
	}
	ok = finish(cb, "a covering triangle");
	memset(px, 0xee, sizeof(px));
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) {
		right += px[i] == want[i];
		grey += px[i] == 0xff808080u;
	}
	/* (finding what stops it, 11-13: the switches the other way, and corners inside clip space) */
	{
		extern unsigned char tri_cover1_vs_bc[];
		extern unsigned int tri_cover1_vs_bc_len;
		hakios_metal_program *p1 = hakios_metal_program_new_bitcode(tri_cover1_vs_bc, tri_cover1_vs_bc_len, tri_cover_fs_bc, tri_cover_fs_bc_len, 70);

		for (int v = 0; v < 6; v++) {
			unsigned char *b8 = hakios_metal_buffer_contents(params);
			hakios_metal_buffer *bufs[1] = { params };
			uint64_t offs[1] = { 0 };
			hakios_metal_texture *texs[1] = { from };
			hakios_metal_sampler *smps[1] = { smp };
			unsigned drawn = 0;

			b8[32] = (v % 3) == 2;          /* flip_x */
			b8[33] = (v % 3) == 1;          /* flip_y */
			cb = hakios_metal_cb_new(q);
			if (hakios_metal_cb_draw2(cb, v < 3 ? p : p1, &pass, NULL, NULL, 3, 0, 3, 1, bufs, offs, 1, bufs, offs, 1, texs, 1, smps, 1) != 0)
				printf("mali_draw_test: a covering triangle, variant %d: refused: %s\n", v, hakios_metal_error());
			finish(cb, "a covering triangle's variant");
			hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
			for (unsigned i = 0; i < 4096; i++) drawn += px[i] != 0xff808080u;
			printf("mali_draw_test: a covering triangle, variant %d (%s corners, flip_x %d flip_y %d): %u of 4096 drawn, pixels 0 63 "
			    "4032 4095: 0x%08x 0x%08x 0x%08x 0x%08x\n", v, v < 3 ? "3.0" : "1.0", b8[32], b8[33], drawn, px[0], px[63], px[4032], px[4095]);
			b8[32] = b8[33] = 0;
		}
	}
	ok = ok && right == 4096;
	printf("mali_draw_test: a covering triangle with a texture through a bound sampler: %u of 4096 the texture's, %u still the "
	    "clear's (pixels 0, 63, 4032, 4095: 0x%08x 0x%08x 0x%08x 0x%08x): %s\n", right, grey, px[0], px[63], px[4032], px[4095],
	    ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	hakios_metal_texture_free(from);
	return ok;
}

/* 7b+. A WIDE TARGET AMONG OTHERS (11-13): RGBA8, RGBA16Float, RGBA8 cleared in one pass -- the RGBA16Float takes two of
 * the tile buffer's kilobytes, so the third target's part starts further on: each must come back its own clear */
static int
wide_among_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	const unsigned rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	hakios_metal_texture *a = hakios_metal_texture_new(d, 2, 70, 64, 64, rt), *w = hakios_metal_texture_new(d, 2, 115, 64, 64, rt);
	hakios_metal_texture *c = hakios_metal_texture_new(d, 2, 70, 64, 64, rt);
	struct hakios_metal_render_pass pass;
	static uint32_t px[64 * 64];
	static uint16_t hx[64 * 64 * 4];
	unsigned ra = 0, rw = 0, rc = 0;
	int ok;
	hakios_metal_cb *cb;

	if (a == NULL || w == NULL || c == NULL) {
		printf("mali_draw_test: a wide target among others: no texture: %s: FAILED\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = a;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = 1;                /* red */
	pass.clear_color[3] = 1;
	pass.more[0].texture = w;
	pass.more[0].load_action = 2;
	pass.more[0].store_action = 1;
	pass.more[0].clear_color[0] = 0.25;
	pass.more[0].clear_color[1] = 0.75;
	pass.more[0].clear_color[2] = 0.125;
	pass.more[0].clear_color[3] = 1;
	pass.more[1].texture = c;
	pass.more[1].load_action = 2;
	pass.more[1].store_action = 1;
	pass.more[1].clear_color[2] = pass.more[1].clear_color[3] = 1;      /* blue */
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_clear(cb, &pass) != 0) printf("mali_draw_test: a wide target among others: refused: %s\n", hakios_metal_error());
	ok = finish(cb, "a wide target among others");
	hakios_metal_texture_read(a, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) ra += px[i] == 0xff0000ffu;
	hakios_metal_texture_read(c, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) rc += px[i] == 0xffff0000u;
	hakios_metal_texture_read(w, 0, 0, 64, 64, hx, 512);
	for (unsigned i = 0; i < 4096; i++) rw += hx[i * 4] == 0x3400 && hx[i * 4 + 1] == 0x3a00 && hx[i * 4 + 2] == 0x3000 && hx[i * 4 + 3] == 0x3c00;
	ok = ok && ra == 4096 && rw == 4096 && rc == 4096;
	printf("mali_draw_test: a wide target among others: RGBA8 %u red, RGBA16Float %u its clear (first %04x %04x %04x %04x), RGBA8 "
	    "%u blue (of 4096): %s\n", ra, rw, hx[0], hx[1], hx[2], hx[3], rc, ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(a);
	hakios_metal_texture_free(w);
	hakios_metal_texture_free(c);
	return ok;
}

/* 7b''. a raw 32-bit float colour attachment (R32Float 55; 11-12 night: WebKit's paths): cleared to 0.25, then the flat
 * draw's red (0x20/255) loaded over it */
static int
raw_float_target_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 55, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	static float px[64 * 64];
	struct hakios_metal_render_pass pass;
	unsigned cleared = 0, drawn = 0;
	int st1 = -1, st2 = -1;
	hakios_metal_cb *cb;

	if (t == NULL) {
		printf("mali_draw_test: an R32Float colour attachment: no texture: %s: FAILED\n", hakios_metal_error());
		return 0;
	}
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = 0.25;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_clear(cb, &pass) == 0) {
		hakios_metal_cb_commit(cb, NULL, NULL);
		hakios_metal_cb_wait(cb);
		st1 = hakios_metal_cb_status(cb);
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 4);
		for (unsigned i = 0; i < 4096; i++) cleared += px[i] == 0.25f;
	} else {
		printf("mali_draw_test: an R32Float colour attachment: the clear refused: %s\n", hakios_metal_error());
	}
	hakios_metal_cb_free(cb);
	pass.load_action = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) == 0) {
		hakios_metal_cb_commit(cb, NULL, NULL);
		hakios_metal_cb_wait(cb);
		st2 = hakios_metal_cb_status(cb);
		hakios_metal_texture_read(t, 0, 0, 64, 64, px, 64 * 4);
		for (unsigned i = 0; i < 4096; i++) {
			float r = px[i] - 32.0f / 255.0f;
			drawn += r > -0.004f && r < 0.004f;
		}
	} else {
		printf("mali_draw_test: an R32Float colour attachment: the draw refused: %s\n", hakios_metal_error());
	}
	hakios_metal_cb_free(cb);
	printf("mali_draw_test: an R32Float colour attachment: the clear status %d, %u of 4096 cleared; the draw status %d, %u of 4096 "
	    "drawn (pixel 0 %d/100000): %s\n", st1, cleared, st2, drawn, (int)(px[0] * 100000.0f),
	    st1 == HAKIOS_METAL_COMPLETED && st2 == HAKIOS_METAL_COMPLETED && cleared == 4096 && drawn == 4096 ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	return st1 == HAKIOS_METAL_COMPLETED && st2 == HAKIOS_METAL_COMPLETED && cleared == 4096 && drawn == 4096;
}

/* 7b'''. blending INTO an R32Float attachment (11-12 night: WebKit's path pipelines add coverage into one, source ONE,
 * destination ONE, red alone): cleared to 0.25, the flat draw (red 0x20/255) added twice in one pass and once more in
 * a pass that loads it: 0.25 + 3 x 0.1255 */
static int
raw_blend_test(hakios_metal_device *d, hakios_metal_queue *q)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 55, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_program *add = hakios_metal_program_new_bitcode(tri_flat_vs_bc, tri_flat_vs_bc_len, tri_flat_fs_bc,
	    tri_flat_fs_bc_len, 55);
	struct hakios_metal_render_pass pass;
	static float px[64 * 64];
	unsigned right1 = 0, right2 = 0;
	float first1 = 0;
	int ok;
	hakios_metal_cb *cb;

	if (t == NULL || add == NULL) {
		printf("mali_draw_test: blending into R32Float: no texture or program (%s): FAILED\n", hakios_metal_error());
		return 0;
	}
	hakios_metal_program_set_attachment_format(add, 0, 55);
	hakios_metal_program_set_blend(add, 1, 0, 0, 1, 1, 1, 0, 0x8);
	memset(&pass, 0, sizeof(pass));
	pass.color0 = t;
	pass.load_action = 2;
	pass.store_action = 1;
	pass.clear_color[0] = 0.25;
	cb = hakios_metal_cb_new(q);
	for (int k = 0; k < 2; k++) {
		/* (the second draw of the pass: the attachment as the first left it -- Load, as the plug-in passes a pass's
		 * later draws; with Clear again the core takes it for a new pass and clears) */
		pass.load_action = k == 0 ? 2 : 1;
		if (hakios_metal_cb_draw(cb, add, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
			printf("mali_draw_test: blending into R32Float: refused: %s\n", hakios_metal_error());
	}
	ok = finish(cb, "blending into R32Float");
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	first1 = px[0];
	for (unsigned i = 0; i < 4096; i++) {
		float e = px[i] - (0.25f + 2 * 32.0f / 255.0f);
		right1 += e > -0.004f && e < 0.004f;
	}
	pass.load_action = 1;
	cb = hakios_metal_cb_new(q);
	if (hakios_metal_cb_draw(cb, add, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0)
		printf("mali_draw_test: blending into R32Float (loaded): refused: %s\n", hakios_metal_error());
	ok = finish(cb, "blending into R32Float, loaded") && ok;
	hakios_metal_texture_read(t, 0, 0, 64, 64, px, 256);
	for (unsigned i = 0; i < 4096; i++) {
		float e = px[i] - (0.25f + 3 * 32.0f / 255.0f);
		right2 += e > -0.004f && e < 0.004f;
	}
	printf("mali_draw_test: blending into R32Float: two draws added %u of 4096 right (pixel 0 %d/100000, wanted 50098); a "
	    "third in a pass that loads it %u of 4096 (pixel 0 %d/100000, wanted 62647): %s\n", right1, (int)(first1 * 100000.0f),
	    right2, (int)(px[0] * 100000.0f), ok && right1 == 4096 && right2 == 4096 ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	return ok && right1 == 4096 && right2 == 4096;
}

/* 7c. layered textures (11-12): a 2D array of 3 layers with 3 levels and a cube -- each layer and face written and read
 * back alone; a draw into layer 1 leaves layers 0 and 2 as they were; a view of layer 2 reads layer 2 */
static int
layered_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat)
{
	const unsigned rt = 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET;
	hakios_metal_texture *a = hakios_metal_texture_new_layered(d, 3, 70, 64, 64, 3, 3, rt);
	hakios_metal_texture *cube = hakios_metal_texture_new_layered(d, 5, 70, 16, 16, 6, 1, 1);
	static uint32_t px[64 * 64], got[64 * 64];
	unsigned right = 0, want = 0;
	int ok = 1;

	if (a == NULL || cube == NULL) {
		printf("mali_draw_test: layered textures: refused: %s: FAILED\n", hakios_metal_error());
		return 0;
	}
	/* every layer's every level its own colour, then all read back */
	for (unsigned pass = 0; pass < 2; pass++) {
		for (unsigned z = 0; z < 3; z++) {
			for (unsigned l = 0; l < 3; l++) {
				unsigned n = 64u >> l;
				uint32_t colour = 0xff000000u | (z + 1) << 20 | (l + 1) << 4;
				if (pass == 0) {
					for (unsigned i = 0; i < n * n; i++) px[i] = colour;
					ok = hakios_metal_texture_replace_3d(a, l, 0, 0, z, n, n, 1, px, n * 4, 0) == 0 && ok;
				} else {
					memset(got, 0, sizeof(got));
					ok = hakios_metal_texture_read_3d(a, l, 0, 0, z, n, n, 1, got, n * 4, 0) == 0 && ok;
					for (unsigned i = 0; i < n * n; i++) right += got[i] == colour;
					want += n * n;
				}
			}
		}
	}
	for (unsigned pass = 0; pass < 2; pass++) {
		for (unsigned f = 0; f < 6; f++) {
			uint32_t colour = 0xff102030u + f;
			if (pass == 0) {
				for (unsigned i = 0; i < 256; i++) px[i] = colour;
				ok = hakios_metal_texture_replace_3d(cube, 0, 0, 0, f, 16, 16, 1, px, 64, 0) == 0 && ok;
			} else {
				memset(got, 0, sizeof(got));
				ok = hakios_metal_texture_read_3d(cube, 0, 0, 0, f, 16, 16, 1, got, 64, 0) == 0 && ok;
				for (unsigned i = 0; i < 256; i++) right += got[i] == colour;
				want += 256;
			}
		}
	}
	printf("mali_draw_test: layered textures: a 2D array's 3 layers x 3 levels and a cube's 6 faces written and read back: %u of "
	    "%u pixels right%s%s: %s\n", right, want, ok ? "" : ", a copy refused: ", ok ? "" : hakios_metal_error(),
	    ok && right == want ? "WORK" : "FAILED");
	ok = ok && right == want;
	/* a draw into layer 1 */
	{
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		struct hakios_metal_render_pass pass;
		unsigned drawn = 0, kept = 0, viewed = 0;
		int st = -1;
		hakios_metal_texture *v;

		memset(&pass, 0, sizeof(pass));
		pass.color0 = a;
		pass.color0_slice = 1;
		pass.load_action = 1;
		pass.store_action = 1;
		if (hakios_metal_cb_draw(cb, flat, &pass, NULL, NULL, 3, 0, 6, 1, NULL, NULL, 0, NULL, NULL, 0) != 0) {
			printf("mali_draw_test: a draw into an array's layer: refused: %s: FAILED\n", hakios_metal_error());
			ok = 0;
		} else {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st = hakios_metal_cb_status(cb);
			hakios_metal_texture_read_3d(a, 0, 0, 0, 1, 64, 64, 1, got, 256, 0);
			for (unsigned i = 0; i < 4096; i++) drawn += got[i] == 0xff804020u;
			hakios_metal_texture_read_3d(a, 0, 0, 0, 0, 64, 64, 1, got, 256, 0);
			for (unsigned i = 0; i < 4096; i++) kept += got[i] == (0xff000000u | 1 << 20 | 1 << 4);
			hakios_metal_texture_read_3d(a, 0, 0, 0, 2, 64, 64, 1, got, 256, 0);
			for (unsigned i = 0; i < 4096; i++) kept += got[i] == (0xff000000u | 3 << 20 | 1 << 4);
			v = hakios_metal_texture_new_view(a, 70, 2, 0, 1, 2, 1);
			if (v != NULL) {
				hakios_metal_texture_read(v, 0, 0, 64, 64, got, 256);
				for (unsigned i = 0; i < 4096; i++) viewed += got[i] == (0xff000000u | 3 << 20 | 1 << 4);
				hakios_metal_texture_free(v);
			}
			printf("mali_draw_test: a draw into an array's layer 1: status %d, %u of 4096 drawn there, %u of 8192 kept in layers 0 "
			    "and 2, a view of layer 2 reads %u of 4096: %s\n", st, drawn, kept, viewed,
			    st == HAKIOS_METAL_COMPLETED && drawn == 4096 && kept == 8192 && viewed == 4096 ? "WORK" : "FAILED");
			ok = ok && st == HAKIOS_METAL_COMPLETED && drawn == 4096 && kept == 8192 && viewed == 4096;
		}
		hakios_metal_cb_free(cb);
	}
	/* (11-13) copies between layers: the cube's face 3 into the array's layer 1, level 2 (16x16, whole); the array's
	 * layer 2 into its layer 0, a 40x40 part at (8, 8) of level 0 (a row at a time); layer 2 itself as it was */
	{
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		unsigned face = 0, part = 0, around = 0, source = 0;
		int st = -1;
		int asked = hakios_metal_cb_blit_layer(cb, cube, 0, 3, 0, 0, 16, 16, a, 2, 1, 0, 0) == 0 &&
		    hakios_metal_cb_blit_layer(cb, a, 0, 2, 8, 8, 40, 40, a, 0, 0, 8, 8) == 0;

		if (!asked) {
			printf("mali_draw_test: copies between layers: refused: %s: FAILED\n", hakios_metal_error());
			ok = 0;
		} else {
			hakios_metal_cb_commit(cb, NULL, NULL);
			hakios_metal_cb_wait(cb);
			st = hakios_metal_cb_status(cb);
			hakios_metal_texture_read_3d(a, 2, 0, 0, 1, 16, 16, 1, got, 64, 0);
			for (unsigned i = 0; i < 256; i++) face += got[i] == 0xff102033u;
			hakios_metal_texture_read_3d(a, 0, 0, 0, 0, 64, 64, 1, got, 256, 0);
			for (unsigned i = 0; i < 4096; i++) {
				unsigned x = i % 64, y = i / 64;
				if (x >= 8 && x < 48 && y >= 8 && y < 48) part += got[i] == (0xff000000u | 3 << 20 | 1 << 4);
				else around += got[i] == (0xff000000u | 1 << 20 | 1 << 4);
			}
			hakios_metal_texture_read_3d(a, 0, 0, 0, 2, 64, 64, 1, got, 256, 0);
			for (unsigned i = 0; i < 4096; i++) source += got[i] == (0xff000000u | 3 << 20 | 1 << 4);
			printf("mali_draw_test: copies between layers: status %d; a cube's face into an array's layer 1 level 2: %u of 256; "
			    "40x40 of layer 2 into layer 0: %u of 1600 there, %u of 2496 around it kept, %u of 4096 of layer 2 as it was: %s\n",
			    st, face, part, around, source,
			    st == HAKIOS_METAL_COMPLETED && face == 256 && part == 1600 && around == 2496 && source == 4096 ? "WORK" : "FAILED");
			ok = ok && st == HAKIOS_METAL_COMPLETED && face == 256 && part == 1600 && around == 2496 && source == 4096;
		}
		hakios_metal_cb_free(cb);
	}
	hakios_metal_texture_free(a);
	hakios_metal_texture_free(cube);
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
	/* (11-12: it ends in error -- the kernel says which submissions its restart counted done without their work) */
	printf("mali_draw_test: a faulting fill: status %d after %ld ms (%s): %s\n", hakios_metal_cb_status(cb),
	    (long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000), hakios_metal_cb_error(cb),
	    hakios_metal_cb_status(cb) == HAKIOS_METAL_ERROR ? "WORK" : "FAILED");
	ok = ok && hakios_metal_cb_status(cb) == HAKIOS_METAL_ERROR;
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
	    !strcmp(argv[1], "irstencil") || !strcmp(argv[1], "irmsaa"));
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
	if (alone && !strcmp(argv[1], "irmsaa")) {
		return ms_ir_test(d, q, (unsigned)strtoul(argv[2], NULL, 0)) ? 0 : 1;
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
	right = depth_alone_test(d, q) && right;
	right = upload_copy_test(d, q, 256, 256) && right;
	right = upload_copy_test(d, q, 1024, 128) && right;
	right = upload_copy_test(d, q, 1024, 1024) && right;
	right = cube_upload_test(d, q, 64) && right;
	right = cube_upload_test(d, q, 512) && right;
	right = cube_sample_test(d, q, 64, 0) && right;
	right = cube_sample_test(d, q, 512, 2) && right;
	right = cube_sample_test(d, q, 16, 0) && right;
	right = link_test(d, q) && right;
	buffer_cost(d, q, 400, 40000, 0);
	buffer_cost(d, q, 400, 4096, 0);
	buffer_cost(d, q, 100, 1 << 20, 0);
	buffer_cost(d, q, 400, 40000, 1);
	right = stencil_test(d, q, flat, col, 0) && right;
	right = stencil_test(d, q, flat, col, 1) && right;
	right = blend_test(d, q) && right;
	right = msaa_test(d, q) && right;
	right = half_texture_test(d, q) && right;
	right = narrow_target_test(d, q, flat) && right;
	right = float_target_test(d, q, flat) && right;
	right = raw_float_target_test(d, q, flat) && right;
	right = packed_float_target_test(d, q, flat) && right;
	right = srgb_target_test(d, q) && right;
	right = cover_test(d, q) && right;
	right = map_pass_test(d, q, 1) && right;
	right = map_pass_test(d, q, 4) && right;
	right = raw_blend_test(d, q) && right;
	right = wide_among_test(d, q) && right;
	right = float_shader_blend_test(d, q) && right;
	right = layered_test(d, q, flat) && right;
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
