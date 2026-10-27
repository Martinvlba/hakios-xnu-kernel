/*
 * userland/src/hakios_metallib.c on this machine (tools/metallib-test.sh builds it with the address and undefined-behaviour
 * sanitizers and runs it).
 *
 *   metallib_test list FILE    the functions as tools/metallib.py list prints them (name, type, AIR version, module size),
 *                              so that the script can compare the two readings line for line
 *   metallib_test cuts         a small metallib made here, then every way of cutting it short and a set of lengths and offsets
 *                              made to point outside it: each must be refused (or read), never read past the end
 */
#include "hakios_metallib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void
put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void
put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> 8 * i); }
static void
put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> 8 * i); }

static size_t
tag(uint8_t *p, const char *name, const void *value, uint16_t length)
{
	memcpy(p, name, 4);
	put16(p + 4, length);
	memcpy(p + 6, value, length);
	return 6 + (size_t)length;
}

/* two functions, the second's module in the bitcode wrapper; returns the size */
static size_t
make(uint8_t *b)
{
	size_t list = 0x58, p, entry, modules;
	uint8_t vers[8] = { 2, 0, 8, 0, 4, 0, 0, 0 }, offt[24] = { 0 }, mdsz[8] = { 0 }, type;

	memset(b, 0, 1024);
	memcpy(b, "MTLB", 4);
	put16(b + 4, 0x8001);
	put16(b + 6, 2);
	p = list + 4;
	put32(b + list, 2);
	for (int i = 0; i < 2; i++) {
		entry = p;
		p += 4;
		p += tag(b + p, "NAME", i == 0 ? "first\0" : "second_one\0", i == 0 ? 6 : 11);
		type = i == 0 ? 2 : 1;
		p += tag(b + p, "TYPE", &type, 1);
		p += tag(b + p, "VERS", vers, 8);
		put64(offt + 16, i == 0 ? 0 : 16);
		p += tag(b + p, "OFFT", offt, 24);
		put64(mdsz, i == 0 ? 16 : 40);
		p += tag(b + p, "MDSZ", mdsz, 8);
		memcpy(b + p, "ENDT", 4);
		p += 4;
		put32(b + entry, (uint32_t)(p - entry));
	}
	put64(b + 0x18, list);
	put64(b + 0x20, p - list - 4);
	modules = p;
	memset(b + modules, 0xab, 16);                          /* the first module, bare */
	put32(b + modules + 16, 0x0B17C0DE);                    /* the second, wrapped: 20 bytes of header, then 20 of bitcode */
	put32(b + modules + 16 + 8, 20);
	put32(b + modules + 16 + 12, 20);
	memcpy(b + modules + 36, "BC\xc0\xde", 4);
	put64(b + 0x48, modules);
	put64(b + 0x50, 56);
	put64(b + 0x10, modules + 56);
	return modules + 56;
}

static void
check(const char *what, int ok)
{
	if (!ok) {
		printf("FAIL %s\n", what);
		failures++;
	}
}

static int
cuts(void)
{
	uint8_t good[1024];
	size_t size = make(good);
	struct hakios_metallib info;
	struct hakios_metallib_function f[4];
	const char *why = NULL;
	const uint8_t *bc;
	uint64_t length;

	check("the made file reads", hakios_metallib_parse(good, size, &info, f, 4, &why) == 0 && info.count == 2);
	check("names, types, versions", f[0].name_length == 5 && memcmp(f[0].name, "first", 5) == 0 && f[1].name_length == 10 &&
	    f[0].type == HAKIOS_METALLIB_KERNEL && f[1].type == HAKIOS_METALLIB_FRAGMENT && f[0].air[0] == 2 && f[0].air[1] == 8);
	check("a bare module is itself", hakios_metallib_bitcode(good, size, &f[0], &bc, &length) == 0 && length == 16);
	check("a wrapped one is what the wrapper points at", hakios_metallib_bitcode(good, size, &f[1], &bc, &length) == 0 &&
	    length == 20 && memcmp(bc, "BC\xc0\xde", 4) == 0);
	check("counting without keeping", hakios_metallib_parse(good, size, &info, NULL, 0, &why) == 0 && info.count == 2);

	/* every length short of the whole: refused, and (under the sanitizers) nothing read past it */
	int refused = 0;
	for (size_t cut = 0; cut < size; cut++) {
		uint8_t *copy = malloc(cut ? cut : 1);

		memcpy(copy, good, cut);
		refused += hakios_metallib_parse(copy, cut, &info, f, 4, &why) != 0;
		free(copy);
	}
	check("every file cut short is refused", refused == (int)size);

	/* offsets and lengths made to point outside, each alone */
	static const struct { size_t at; int width; uint64_t value; const char *what; } bad[] = {
		{ 0x18, 8, 0xfffffffffffffff0ull, "the list's start" },
		{ 0x20, 8, 0xffffffffffffffffull, "the list's length" },
		{ 0x48, 8, 0xfffffffffffffff8ull, "the modules' start" },
		{ 0x50, 8, 0x7fffffffffffffffull, "the modules' length" },
		{ 0x58, 4, 0xffffffffu, "the count of functions" },
		{ 0x5c, 4, 0xfffffff0u, "an entry's size" },
		{ 0x5c, 4, 2, "an entry's size, too small" },
	};
	for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
		uint8_t copy[1024];

		memcpy(copy, good, size);
		if (bad[i].width == 8) {
			put64(copy + bad[i].at, bad[i].value);
		} else {
			put32(copy + bad[i].at, (uint32_t)bad[i].value);
		}
		char label[96];
		snprintf(label, sizeof(label), "a bad %s is refused", bad[i].what);
		check(label, hakios_metallib_parse(copy, size, &info, f, 4, &why) != 0);
	}
	/* a tag's length that runs past its entry, and a module past the modules */
	uint8_t copy[1024];
	memcpy(copy, good, size);
	put16(copy + 0x58 + 4 + 4 + 4, 0xffff);                  /* NAME's length of the first function */
	check("a tag running past its entry is refused", hakios_metallib_parse(copy, size, &info, f, 4, &why) != 0);
	memcpy(copy, good, size);
	hakios_metallib_parse(copy, size, &info, f, 4, &why);
	put64(copy + info.file_size - 56 - 1, 0);               /* (no-op: keeps the layout) */
	/* the wrapper of the second module pointing past it */
	memcpy(copy, good, size);
	put32(copy + size - 40 + 12, 1000);
	check("a wrapper pointing past its module is refused", hakios_metallib_parse(copy, size, &info, f, 4, &why) == 0 &&
	    hakios_metallib_bitcode(copy, size, &f[1], &bc, &length) != 0);
	/* a universal file: an Apple-GPU slice first, then the AIR64 one, which is what is taken */
	uint8_t fat[2048] = { 0 };
	const uint8_t *thin;
	size_t thin_size;
	static const uint8_t header[] = { 0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 2,
		0x01, 0x00, 0x00, 0x13, 0, 0, 0, 0, 0, 0, 0x01, 0x00, 0, 0, 0, 0x10, 0, 0, 0, 0,        /* an Apple GPU's: 16 bytes at 256 */
		0x01, 0x00, 0x00, 0x17, 0, 0, 0, 0, 0, 0, 0x02, 0x00, 0, 0, 0, 0, 0, 0, 0, 0 };         /* AIR64 at 512, its length below */
	memcpy(fat, header, sizeof(header));
	fat[8 + 20 + 12] = (uint8_t)(size >> 24);
	fat[8 + 20 + 13] = (uint8_t)(size >> 16);
	fat[8 + 20 + 14] = (uint8_t)(size >> 8);
	fat[8 + 20 + 15] = (uint8_t)size;
	memcpy(fat + 512, good, size);
	check("a universal file: its AIR64 slice is taken, and reads", hakios_metallib_thin(fat, 512 + size, &thin, &thin_size, &why) == 0 &&
	    thin == fat + 512 && thin_size == size && hakios_metallib_parse(thin, thin_size, &info, f, 4, &why) == 0 && info.count == 2);
	check("a thin file is itself", hakios_metallib_thin(good, size, &thin, &thin_size, &why) == 0 && thin == good && thin_size == size);
	check("a universal file cut short of its slice is refused", hakios_metallib_thin(fat, 512 + size - 1, &thin, &thin_size, &why) != 0);
	fat[8 + 20 + 3] = 0x13;                                 /* no AIR64 slice any more */
	check("a universal file with no AIR64 slice is refused", hakios_metallib_thin(fat, 512 + size, &thin, &thin_size, &why) != 0);
	for (size_t cut = 0; cut < 64; cut++) {
		hakios_metallib_thin(fat, cut, &thin, &thin_size, &why);      /* under the sanitizers: nothing read past CUT */
	}
	printf("metallib_test: %s\n", failures ? "FAILED" : "the made file reads, thin and universal; every cut and every bad offset is refused");
	return failures != 0;
}

static int
list(const char *path)
{
	FILE *file = fopen(path, "rb");
	uint8_t *data;
	long size;
	struct hakios_metallib info;
	struct hakios_metallib_function *f;
	const char *why = NULL;

	if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
		perror(path);
		return 1;
	}
	data = malloc((size_t)size + 1);
	if (fread(data, 1, (size_t)size, file) != (size_t)size) {
		perror(path);
		return 1;
	}
	fclose(file);
	if (hakios_metallib_parse(data, (size_t)size, &info, NULL, 0, &why) != 0) {
		printf("%s: %s\n", path, why);
		return 1;
	}
	f = calloc(info.count ? info.count : 1, sizeof(*f));
	hakios_metallib_parse(data, (size_t)size, &info, f, info.count, &why);
	printf("%u functions\n", info.count);
	for (uint32_t i = 0; i < info.count; i++) {
		char air[32];
		const uint8_t *bc;
		uint64_t length;

		snprintf(air, sizeof(air), "%u.%u.%u.%u", f[i].air[0], f[i].air[1], f[i].air[2], f[i].air[3]);
		if (hakios_metallib_bitcode(data, (size_t)size, &f[i], &bc, &length) != 0) {
			printf("%.*s: its wrapper points outside its module\n", (int)f[i].name_length, f[i].name);
			return 1;
		}
		printf("  %-52.*s %-13s air %s  module %llu\n", (int)f[i].name_length, f[i].name,
		    f[i].type == HAKIOS_METALLIB_NO_TYPE ? "?" : f[i].type < 7 ? hakios_metallib_type_name(f[i].type) : "type?",
		    air, (unsigned long long)f[i].module_size);
	}
	free(f);
	free(data);
	return 0;
}

int
main(int argc, char **argv)
{
	if (argc == 3 && strcmp(argv[1], "list") == 0) {
		return list(argv[2]);
	}
	if (argc == 2 && strcmp(argv[1], "cuts") == 0) {
		return cuts();
	}
	fprintf(stderr, "usage: metallib_test list FILE | cuts\n");
	return 2;
}
