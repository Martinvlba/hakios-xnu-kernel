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
depth_test(hakios_metal_device *d, hakios_metal_queue *q, hakios_metal_program *flat, hakios_metal_program *col)
{
	hakios_metal_texture *t = hakios_metal_texture_new(d, 2, 70, 64, 64, 1 | HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	hakios_metal_texture *z = hakios_metal_texture_new(d, 2, 252, 64, 64, HAKIOS_METAL_TEXTURE_USAGE_RENDER_TARGET);
	struct hakios_metal_render_pass pass;
	uint32_t px[64 * 64];
	unsigned kept = 0, over = 0;
	int ok = 1;

	if (t == NULL || z == NULL) {
		printf("mali_draw_test: depth: no textures (%s)\n", hakios_metal_error());
		return 0;
	}
	for (int round = 0; round < 2; round++) {
		hakios_metal_cb *cb = hakios_metal_cb_new(q);
		memset(&pass, 0, sizeof(pass));
		pass.color0 = t;
		pass.load_action = 2;
		pass.store_action = 1;
		pass.depth = z;
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
	printf("mali_draw_test: depth: compare Less kept %u of 4096, no test drew over %u of 4096: %s\n", kept, over,
	    ok ? "WORK" : "FAILED");
	hakios_metal_texture_free(t);
	hakios_metal_texture_free(z);
	return ok;
}

int
main(void)
{
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
	right = depth_test(d, q, flat, col) && right;
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
