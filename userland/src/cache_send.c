/*
 * cache_send: the Mali shader cache handed to the host over USB, for a seed in the next image (TODO 21, 11-11).
 *
 *   cache_send [SECONDS [DIR]]
 *
 * Listens on 127.0.0.1:31415 for SECONDS (120 by default) -- usbmux reaches a phone's local TCP port as it reaches
 * lockdownd's (drivers/usb/hakios.akita_usbmux.c) -- and to each connection writes every *.prog file of DIR
 * (hakios-metal's cache: /private/var/mobile/Library/Caches/com.hakios.metal-mali, else /private/var/tmp/...):
 * "HKMC0002", then per file a little-endian u32 name length, the name, a u64 size, a u64 modification time (seconds:
 * the cache touches a file whenever it reads it, so the time says which files a recent boot used -- those compiled by an
 * older build of the compiler are never read again) and the bytes; a zero length ends it. Reads only. devices/akita/tools/usbmux-pull-cache.py is the host's side; userland/build.sh puts what it fetched
 * into the image at /redwood/share/hakios-metal-cache-mali, where c_api_mali.cpp's cacheLoad looks first.
 */
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PORT 31415

static int
write_all(int fd, const void *p, size_t n)
{
	const uint8_t *b = p;

	while (n != 0) {
		ssize_t w = write(fd, b, n);

		if (w < 0 && errno == EINTR) {
			continue;
		}
		if (w <= 0) {
			return -1;
		}
		b += w;
		n -= (size_t)w;
	}
	return 0;
}

/* one file as an entry; 0 on success (a file that cannot be read is skipped, not an error) */
static int
send_file(int out, const char *dir, const char *name, unsigned *files, uint64_t *bytes)
{
	char path[1024];
	struct stat st;
	int fd;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	if ((fd = open(path, O_RDONLY)) < 0) {
		return 0;
	}
	if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 || st.st_size > (16 << 20)) {
		close(fd);
		return 0;
	}
	uint8_t *data = malloc((size_t)st.st_size);
	size_t got = 0;

	while (data != NULL && got < (size_t)st.st_size) {
		ssize_t r = read(fd, data + got, (size_t)st.st_size - got);

		if (r < 0 && errno == EINTR) {
			continue;
		}
		if (r <= 0) {
			break;
		}
		got += (size_t)r;
	}
	close(fd);
	if (data == NULL || got != (size_t)st.st_size) {
		free(data);
		return 0;
	}
	uint32_t len = (uint32_t)strlen(name);
	uint64_t size = got, mtime = (uint64_t)st.st_mtime;
	int r = write_all(out, &len, 4) || write_all(out, name, len) || write_all(out, &size, 8) || write_all(out, &mtime, 8) ||
	    write_all(out, data, got);

	free(data);
	if (r == 0) {
		(*files)++;
		*bytes += got;
	}
	return r;
}

static void
serve(int out, const char *dir)
{
	DIR *d = opendir(dir);
	struct dirent *e;
	unsigned files = 0;
	uint64_t bytes = 0;
	uint32_t end = 0;

	if (write_all(out, "HKMC0002", 8) != 0) {
		return;
	}
	while (d != NULL && (e = readdir(d)) != NULL) {
		size_t n = strlen(e->d_name);

		if (n > 5 && strcmp(e->d_name + n - 5, ".prog") == 0 && send_file(out, dir, e->d_name, &files, &bytes) != 0) {
			break;
		}
	}
	if (d != NULL) {
		closedir(d);
	}
	write_all(out, &end, 4);
	printf("cache_send: %u files, %llu bytes of %s sent\n", files, (unsigned long long)bytes, dir);
	fflush(stdout);
}

int
main(int argc, char **argv)
{
	int seconds = argc > 1 ? atoi(argv[1]) : 120;
	const char *dir = argc > 2 ? argv[2] : NULL;
	struct sockaddr_in a;
	int s, one = 1;
	time_t until = time(NULL) + (seconds > 0 ? seconds : 120);

	if (dir == NULL) {
		struct stat st;

		dir = stat("/private/var/mobile/Library/Caches/com.hakios.metal-mali", &st) == 0 ?
		    "/private/var/mobile/Library/Caches/com.hakios.metal-mali" : "/private/var/tmp/com.hakios.metal-mali";
	}
	if ((s = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		perror("cache_send: socket");
		return 1;
	}
	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	memset(&a, 0, sizeof(a));
	a.sin_family = AF_INET;
	a.sin_port = htons(PORT);
	a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if (bind(s, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(s, 2) != 0) {
		perror("cache_send: bind/listen");
		return 1;
	}
	printf("cache_send: %s on 127.0.0.1:%d for %d s\n", dir, PORT, seconds);
	fflush(stdout);
	while (time(NULL) < until) {
		fd_set r;
		struct timeval tv = { 1, 0 };

		FD_ZERO(&r);
		FD_SET(s, &r);
		if (select(s + 1, &r, NULL, NULL, &tv) > 0) {
			int c = accept(s, NULL, NULL);

			if (c >= 0) {
				serve(c, dir);
				close(c);
			}
		}
	}
	close(s);
	return 0;
}
