/*
 * ispdev: libhakios.media's picture processor (userland/src/camera_isp.c, the very code the phone runs) on the host, on
 * a frame kept from the phone (camera_test raw N keepraw: /private/var/mobile/Media/hakios-klog/camera-S.raw, fetched by
 * devices/akita/tools/usbmux-pull-klog.py OUT --camera): the white balance and the exposure's digital gain settled on
 * that one frame as they settle on the phone's stream, the colour tuning applied as there, a picture written.
 *
 * usage: ispdev FRAME.raw OUT.ppm [TUNING.isp|-] [WIDTH HEIGHT] [LUX]
 * build: cc -O2 -Iuserland/src -o /tmp/ispdev tools/ispdev.c userland/src/camera_isp.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "camera_isp.h"

int
main(int argc, char **argv)
{
	FILE *in, *out;
	struct hakios_camera_frame head, *f;
	struct isp isp;
	struct isp_stats s;
	struct hakios_camera_command c;
	uint32_t width, height;
	size_t bytes;
	uint8_t *raw, *bgra;
	float lux = argc > 6 ? (float)atof(argv[6]) : 300, cct = 0;

	if (argc < 3 || (in = fopen(argv[1], "rb")) == NULL || fread(&head, sizeof(head), 1, in) != 1) {
		fprintf(stderr, "usage: ispdev FRAME.raw OUT.ppm [TUNING.isp|-] [WIDTH HEIGHT] [LUX]\n");
		return 1;
	}
	bytes = head.header_bytes + (size_t)head.stride * head.height;
	f = malloc(bytes);
	rewind(in);
	if (fread(f, 1, bytes, in) != bytes) {
		fprintf(stderr, "ispdev: the frame is short\n");
		return 1;
	}
	raw = (uint8_t *)f + f->header_bytes;
	width = argc > 5 ? (uint32_t)atoi(argv[4]) : f->width / 4;
	height = argc > 5 ? (uint32_t)atoi(argv[5]) : f->height / 4;
	isp_init(&isp);
	if (argc > 3 && strcmp(argv[3], "-") != 0 && !isp_load_tuning(&isp, argv[3])) {
		fprintf(stderr, "ispdev: %s is no tuning\n", argv[3]);
		return 1;
	}
	for (int i = 0; i < 80; i++) {
		isp_measure(&isp, f, raw, &s);
		isp_awb(&isp, &s);
		(void)isp_ae(&isp, f, &s, &c);
		if (isp.tuning != NULL) {
			cct = isp_cct(&isp);
			isp_tune(&isp, lux, cct);
		}
	}
	printf("ispdev: %ux%u bayer %u, %u lines, gain %.1fx; green %.1f above black; white balance R %.2f B %.2f, digital %.2fx, "
	    "about %.0f K\n", f->width, f->height, f->bayer, f->exposure_lines, (double)f->gain / f->gain_unity, s.mean[1], isp.wb[0],
	    isp.wb[2], isp.digital, cct);
	printf("ispdev: matrix %.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f\n", isp.ccm[0][0], isp.ccm[0][1], isp.ccm[0][2],
	    isp.ccm[1][0], isp.ccm[1][1], isp.ccm[1][2], isp.ccm[2][0], isp.ccm[2][1], isp.ccm[2][2]);
	bgra = malloc((size_t)width * height * 4);
	isp_render_bgra(&isp, f, raw, bgra, (size_t)width * 4, width, height, 0, height);
	if ((out = fopen(argv[2], "wb")) == NULL) {
		return 1;
	}
	fprintf(out, "P6\n%u %u\n255\n", width, height);
	for (size_t i = 0; i < (size_t)width * height; i++) {
		fputc(bgra[i * 4 + 2], out);
		fputc(bgra[i * 4 + 1], out);
		fputc(bgra[i * 4], out);
	}
	fclose(out);
	return 0;
}
