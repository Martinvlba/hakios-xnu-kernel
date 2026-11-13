/*
 * filemap_test: a file written and shared the way WebKit's network cache does it (rc.jobs.filemap) -- Speedometer 3 in
 * Safari on the Pixel 8a stayed a blank white page whenever its files came from the disk cache ("Cached entry is
 * directly usable" in the Networking process, then "Download decoding failed" in the WebContent process; from the
 * network the same page ran), 11-13.
 *
 *   store  (NetworkCache::Data::mapToFile)  open O_CREAT|O_EXCL|O_RDWR, ftruncate, mmap PROT_WRITE MAP_SHARED, memcpy,
 *          mprotect PROT_READ, msync MS_ASYNC, the mapping kept; a hard link to it under the record's name
 *   read   (mapFile)                        open O_RDONLY, mmap PROT_READ MAP_PRIVATE
 *   share  (SharedMemory::wrapMap)          mach_make_memory_entry_64(..., VM_PROT_READ | VM_PROT_IS_MASK |
 *          MAP_MEM_VM_SHARE) of that mapping, the port sent to the WebContent process, mach_vm_map there
 *
 * Each step's bytes are compared with what was written (a pattern by offset and a seed); the share is mapped in this
 * task and in a child (the port handed over as a registered port). And what the boot before left
 * (DIR/persist.blob, written so and never fsynced) is read first: did the mapped writes reach the disk.
 *
 *   filemap_test [DIR [BYTES]]      (default /private/var/mobile/Library/Caches/hakios-filemap, 3 MB + 1234 bytes)
 */
#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* (mach/mach_vm.h is not for iOS in the SDK; the call is in libSystem) */
extern kern_return_t mach_vm_map(vm_map_t target, mach_vm_address_t *address, mach_vm_size_t size, mach_vm_offset_t mask,
    int flags, mem_entry_name_port_t object, memory_object_offset_t offset, boolean_t copy, vm_prot_t cur, vm_prot_t max,
    vm_inherit_t inheritance);

#ifndef MAP_MEM_VM_SHARE
#define MAP_MEM_VM_SHARE 0x400000
#endif
#ifndef VM_PROT_IS_MASK
#define VM_PROT_IS_MASK ((vm_prot_t)0x40)
#endif

static int works = 0, fails = 0;

static uint8_t
pattern(size_t i, unsigned seed)
{
	return (uint8_t)((i * 2654435761u + seed * 97u) >> 13 ^ i);
}

/* how many bytes differ; the first place and what is there */
static void
judge(const char *what, const uint8_t *p, size_t n, unsigned seed)
{
	size_t bad = 0, first = 0, zeros = 0;

	for (size_t i = 0; i < n; i++) {
		if (p[i] != pattern(i, seed)) {
			if (bad++ == 0) {
				first = i;
			}
			zeros += p[i] == 0;
		}
	}
	if (bad == 0) {
		works++;
		printf("filemap_test: %s: WORKS (%zu bytes as written)\n", what, n);
	} else {
		fails++;
		printf("filemap_test: %s: FAILED -- %zu of %zu bytes differ (%zu of them zero), the first at %zu: 0x%02x for 0x%02x\n",
		    what, bad, n, zeros, first, p[first], pattern(first, seed));
	}
}

/* a memory entry of N bytes at P (FLAGS beside VM_PROT_READ), mapped in this task and in a child: both judged */
static void
share_check(const char *what, const uint8_t *p, size_t n, unsigned seed, int flags, int touch_first)
{
	memory_object_size_t size = n;
	mach_port_t port = MACH_PORT_NULL, old = MACH_PORT_NULL;
	char name[200];
	kern_return_t kr = mach_make_memory_entry_64(mach_task_self(), &size, (memory_object_offset_t)(uintptr_t)p,
	    VM_PROT_READ | flags, &port, MACH_PORT_NULL);

	if (kr != KERN_SUCCESS) {
		fails++;
		printf("filemap_test: %s: mach_make_memory_entry_64: FAILED -- 0x%x\n", what, kr);
		return;
	}
	mach_vm_address_t at = 0;
	kr = mach_vm_map(mach_task_self(), &at, n, 0, VM_FLAGS_ANYWHERE, port, 0, FALSE, VM_PROT_READ, VM_PROT_READ, VM_INHERIT_NONE);
	snprintf(name, sizeof(name), "%s, mapped in this task", what);
	if (kr == KERN_SUCCESS) {
		judge(name, (const uint8_t *)(uintptr_t)at, n, seed);
	} else {
		fails++;
		printf("filemap_test: %s: mach_vm_map: FAILED -- 0x%x\n", name, kr);
	}
	/* into another task: a child sends its task port here (to a port that is its bootstrap port for the moment), this
	 * task maps the entry INTO THE CHILD and tells it where (a pipe); the child judges what it reads there. (The
	 * first two versions handed the entry's port to the child, as its bootstrap port and as a registered port: the
	 * child's mach_vm_map gave zeroes or KERN_INVALID_OBJECT for anonymous memory too -- the handing over, not the
	 * memory.) */
	mach_port_t rcv = MACH_PORT_NULL;
	int to_child[2];

	(void)old;
	if (mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &rcv) != KERN_SUCCESS ||
	    mach_port_insert_right(mach_task_self(), rcv, rcv, MACH_MSG_TYPE_MAKE_SEND) != KERN_SUCCESS || pipe(to_child) != 0) {
		fails++;
		printf("filemap_test: %s: no port or pipe for the child\n", what);
		return;
	}
	fflush(stdout);
	pid_t child = fork();
	if (child == 0) {
		uint64_t there = 0;

		works = fails = 0;
		close(to_child[1]);
		if (read(to_child[0], &there, sizeof(there)) != sizeof(there) || there == 0) {
			printf("filemap_test: %s: the child was told no address\n", what);
			_exit(2);
		}
		if (touch_first) {
			snprintf(name, sizeof(name), "%s, the child's inherited view", what);
			judge(name, p, n, seed);
		}
		snprintf(name, sizeof(name), "%s, mapped into a child", what);
		judge(name, (const uint8_t *)(uintptr_t)there, n, seed);
		_exit(fails ? 1 : 0);
	}
	close(to_child[0]);
	int status = 0;
	if (child > 0) {
		uint64_t there = 0;
		mach_port_t child_task = MACH_PORT_NULL;
		/* (the child's task port: task_for_pid, as root; a child's message to a port set as the bootstrap port never
		 * came -- the third version) */
		kern_return_t got = task_for_pid(mach_task_self(), child, &child_task);

		if (got == KERN_SUCCESS && child_task != MACH_PORT_NULL) {
			mach_vm_address_t in_child = 0;

			kr = mach_vm_map(child_task, &in_child, n, 0, VM_FLAGS_ANYWHERE, port, 0, FALSE, VM_PROT_READ, VM_PROT_READ, VM_INHERIT_NONE);
			if (kr == KERN_SUCCESS) {
				there = in_child;
			} else {
				printf("filemap_test: %s: mach_vm_map into the child: 0x%x\n", what, kr);
			}
		} else {
			printf("filemap_test: %s: task_for_pid of the child: 0x%x (uid %d)\n", what, got, (int)getuid());
		}
		(void)write(to_child[1], &there, sizeof(there));
		close(to_child[1]);
		waitpid(child, &status, 0);
		if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
			works++;
		} else {
			fails++;
			if (!WIFEXITED(status) || WEXITSTATUS(status) == 2) {
				printf("filemap_test: %s: (the child's check did not run: the test's own plumbing)\n", what);
			}
		}
	}
	mach_port_mod_refs(mach_task_self(), rcv, MACH_PORT_RIGHT_RECEIVE, -1);
	snprintf(name, sizeof(name), "%s, this task's mapping after the child", what);
	if (at != 0) {
		judge(name, (const uint8_t *)(uintptr_t)at, n, seed);
	}
}

static void *
store(const char *path, size_t n, unsigned seed)
{
	int fd;
	uint8_t *map;

	unlink(path);
	fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (fd < 0 || ftruncate(fd, (off_t)n) != 0) {
		printf("filemap_test: %s: open/ftruncate: %s\n", path, strerror(errno));
		fails++;
		return NULL;
	}
	map = mmap(NULL, n, PROT_WRITE, MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		printf("filemap_test: %s: mmap for writing: %s\n", path, strerror(errno));
		fails++;
		close(fd);
		return NULL;
	}
	for (size_t i = 0; i < n; i++) {
		map[i] = pattern(i, seed);
	}
	mprotect(map, n, PROT_READ);
	msync(map, n, MS_ASYNC);
	close(fd);              /* (WebKit keeps the descriptor with the mapping; the mapping holds the file either way) */
	return map;
}

static void
read_back(const char *what, const char *path, size_t n, unsigned seed)
{
	char name[160];
	int fd = open(path, O_RDONLY);
	uint8_t *buf = malloc(n);
	ssize_t got = 0, part;
	struct stat st;

	if (fd < 0 || buf == NULL) {
		printf("filemap_test: %s: %s: %s\n", what, path, strerror(errno));
		fails++;
		free(buf);
		return;
	}
	fstat(fd, &st);
	while ((size_t)got < n && (part = read(fd, buf + got, n - (size_t)got)) > 0) {
		got += part;
	}
	close(fd);
	snprintf(name, sizeof(name), "%s, read(2) (the file is %lld bytes, %zd read)", what, (long long)st.st_size, got);
	judge(name, buf, got > 0 ? (size_t)got : 0, seed);
	if ((size_t)got != n) {
		fails++;
		printf("filemap_test: %s: FAILED -- %zd bytes where %zu were written\n", what, got, n);
	}
	free(buf);
}

int
main(int argc, char **argv)
{
	const char *dir = argc > 1 ? argv[1] : "/private/var/mobile/Library/Caches/hakios-filemap";
	size_t n = argc > 2 ? (size_t)atol(argv[2]) : 3 * 1024 * 1024 + 1234;
	char blob[512], linked[512], persist[512], seedfile[512];
	unsigned seed = (unsigned)getpid() * 7u + 3u, before = 0;
	uint8_t *written, *map;
	FILE *f;
	int fd;

	setvbuf(stdout, NULL, _IOLBF, 0);
	mkdir(dir, 0755);
	snprintf(blob, sizeof(blob), "%s/blob", dir);
	snprintf(linked, sizeof(linked), "%s/record-body", dir);
	snprintf(persist, sizeof(persist), "%s/persist.blob", dir);
	snprintf(seedfile, sizeof(seedfile), "%s/persist.seed", dir);

	/* what the boot before left */
	if ((f = fopen(seedfile, "r")) != NULL) {
		if (fscanf(f, "%u", &before) == 1) {
			read_back("the boot before's mapped write (never fsynced)", persist, n, before);
		}
		fclose(f);
	} else {
		printf("filemap_test: no file of an earlier boot in %s (the first run)\n", dir);
	}

	/* store */
	written = store(blob, n, seed);
	if (written == NULL) {
		return 1;
	}
	judge("the writer's own mapping after mprotect", written, n, seed);
	unlink(linked);
	if (link(blob, linked) != 0) {
		fails++;
		printf("filemap_test: link(blob, record-body): FAILED -- %s\n", strerror(errno));
		snprintf(linked, sizeof(linked), "%s", blob);
	} else {
		works++;
		printf("filemap_test: link(blob, record-body): WORKS\n");
	}
	read_back("the hard link while the writer's mapping is there", linked, n, seed);

	/* read: a private read-only mapping of the link */
	fd = open(linked, O_RDONLY);
	map = fd < 0 ? MAP_FAILED : mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) {
		fails++;
		printf("filemap_test: mmap PROT_READ MAP_PRIVATE of the link: FAILED -- %s\n", strerror(errno));
		return 1;
	}
	judge("mmap PROT_READ MAP_PRIVATE of the link", map, n, seed);

	/* share: a memory entry of that mapping, mapped again here and in a child */
	share_check("the whole file MAP_PRIVATE, MAP_MEM_VM_SHARE (WebKit's)", map, n, seed, VM_PROT_IS_MASK | MAP_MEM_VM_SHARE, 0);

	/* which kind of memory, which kind of entry  */
	{
		const size_t m = n < 400000 ? n : 400000;
		uint8_t *anon = mmap(NULL, m, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
		int fd2 = open(linked, O_RDONLY);
		uint8_t *shared = fd2 < 0 ? MAP_FAILED : mmap(NULL, m, PROT_READ, MAP_SHARED, fd2, 0);

		if (anon != MAP_FAILED) {
			for (size_t i = 0; i < m; i++) {
				anon[i] = pattern(i, seed);
			}
			share_check("anonymous memory, MAP_MEM_VM_SHARE", anon, m, seed, VM_PROT_IS_MASK | MAP_MEM_VM_SHARE, 0);
			share_check("anonymous memory, a plain entry", anon, m, seed, 0, 0);
		}
		share_check("the file MAP_PRIVATE, MAP_MEM_VM_SHARE, the child reading its own view first", map, m, seed,
		    VM_PROT_IS_MASK | MAP_MEM_VM_SHARE, 1);
		share_check("the file MAP_PRIVATE, a plain entry", map, m, seed, 0, 0);
		if (shared != MAP_FAILED) {
			share_check("the file MAP_SHARED, MAP_MEM_VM_SHARE", shared, m, seed, VM_PROT_IS_MASK | MAP_MEM_VM_SHARE, 0);
			share_check("the file MAP_SHARED, a plain entry", shared, m, seed, 0, 0);
		}
	}

	/* a second process reading the file as the next start of the browser would, the writer's mapping gone */
	munmap(written, n);
	munmap(map, n);
	close(fd);
	read_back("the link after every mapping is gone", linked, n, seed);

	/* for the next boot: stored the same way, not synced */
	{
		unsigned next = seed + 1;
		void *p = store(persist, n, next);

		if (p != NULL && (f = fopen(seedfile, "w")) != NULL) {
			fprintf(f, "%u\n", next);
			fclose(f);
			printf("filemap_test: %s written for the next boot to read (mapped, msync MS_ASYNC, no fsync)\n", persist);
		}
	}
	printf("filemap_test: %d checks WORK, %d FAILED\n", works, fails);
	return fails != 0;
}
