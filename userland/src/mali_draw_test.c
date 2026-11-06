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

int
main(void)
{
	setenv("HAKIOS_MALI_TRACE", "1", 1);         /* (the back end says what the tiler did with each draw) */
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
	printf("mali_draw_test: the first Metal draws %s\n", right ? "WORK" : "FAILED");
	hakios_metal_program_free(flat);
	hakios_metal_program_free(col);
	hakios_metal_queue_free(q);
	hakios_metal_close(d);
	return right ? 0 : 1;
}
