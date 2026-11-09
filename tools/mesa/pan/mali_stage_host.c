/*
 * mali_stage_host: a render pipeline's two stages compiled on this machine exactly as the phone compiles them --
 * hakios-metal's own compile_mali.c (mali_stage_compile: our AIR front end, then Mesa's panfrost compiler for the
 * Mali-G715), linked with the host Mesa build (tools/mesa/pan/mk-pan-host.sh). For trying a shader's function
 * constants without booting the phone: CoreAnimation's generic uber shaders specialised per pipeline.
 *
 *   mali_stage_host LIB.metallib VERTEX FRAGMENT "VS-CONSTANTS" "FS-CONSTANTS"
 *
 * CONSTANTS: "index=bits index=bits ..." as the plug-in prints them ("fragment constants not 0: 4=1 6=2 58=1541"); a
 * vector constant's component i at bit i * its width (58=1541: uchar3 5, 6, 0). Prints each stage's outcome, its
 * code size and the time it took.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include "compiler/glsl_types.h"
#include "compiler/air/nir_air.h"
#include "util/os_file.h"
#include "util/ralloc.h"

#include "compile_mali.h"

static unsigned
parse(const char *spec, struct mali_constant *out, unsigned max)
{
	unsigned n = 0;
	const char *p = spec;

	while (p && *p && n < max) {
		char *end;
		unsigned long index = strtoul(p, &end, 10);

		if (end == p || *end != '=') break;
		out[n].index = (unsigned)index;
		out[n].bits = strtoull(end + 1, &end, 0);
		n++;
		p = end;
		while (*p == ' ' || *p == ',') p++;
	}
	return n;
}

static const void *
function(struct air_metallib *lib, const char *name, size_t *size)
{
	for (unsigned i = 0; i < lib->num_functions; i++) {
		if (strcmp(lib->functions[i].name, name) == 0) {
			*size = lib->functions[i].bitcode_size;
			return lib->functions[i].bitcode;
		}
	}
	return NULL;
}

/* a crash's frames (no debugger on this machine) */
static void
crashed(int sig)
{
	void *frames[64];
	int n = backtrace(frames, 64);

	fprintf(stderr, "mali_stage_host: signal %d\n", sig);
	backtrace_symbols_fd(frames, n, 2);
	_exit(3);
}

static double
now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

int
main(int argc, char **argv)
{
	struct mali_constant vc[64], fc[64];
	struct mali_kernel_binary vs, fs;
	struct air_metallib lib;
	const char *error = NULL;
	char why[1024] = "";
	size_t size = 0, vs_size = 0, fs_size = 0;
	void *file, *mem = ralloc_context(NULL);

	if (argc < 4) {
		fprintf(stderr, "usage: mali_stage_host LIB.metallib VERTEX FRAGMENT [\"VS-CONSTANTS\"] [\"FS-CONSTANTS\"]\n");
		return 1;
	}
	if (getenv("MALI_STAGE_BT")) {
		signal(SIGSEGV, crashed);
		signal(SIGABRT, crashed);
	}
	glsl_type_singleton_init_or_ref();
	file = os_read_file(argv[1], &size);
	if (file == NULL || !air_metallib_parse(mem, file, size, &lib, &error)) {
		fprintf(stderr, "mali_stage_host: %s: %s\n", argv[1], file ? error : "unreadable");
		return 1;
	}
	unsigned vn = parse(argc > 4 ? argv[4] : "", vc, 64), fn = parse(argc > 5 ? argv[5] : "", fc, 64);
	const void *vb = function(&lib, argv[2], &vs_size), *fb = function(&lib, argv[3], &fs_size);
	if (vb == NULL || fb == NULL) {
		fprintf(stderr, "mali_stage_host: %s not in the library\n", vb == NULL ? argv[2] : argv[3]);
		return 1;
	}
	double t0 = now();
	if (mali_stage_compile(vb, vs_size, vc, vn, 0, NULL, &vs, why, sizeof(why)) != 0) {
		printf("vertex %s (%zu bytes of AIR, %u constants): REFUSED in %.2f s: %s\n", argv[2], vs_size, vn, now() - t0, why);
		return 2;
	}
	printf("vertex %s (%zu bytes of AIR, %u constants): %zu bytes of code in %.2f s\n", argv[2], vs_size, vn, vs.size, now() - t0);
	t0 = now();
	if (mali_stage_compile(fb, fs_size, fc, fn, 1, &vs, &fs, why, sizeof(why)) != 0) {
		printf("fragment %s (%zu bytes of AIR, %u constants): REFUSED in %.2f s: %s\n", argv[3], fs_size, fn, now() - t0, why);
		return 2;
	}
	printf("fragment %s (%zu bytes of AIR, %u constants): %zu bytes of code in %.2f s\n", argv[3], fs_size, fn, fs.size, now() - t0);
	return 0;
}
