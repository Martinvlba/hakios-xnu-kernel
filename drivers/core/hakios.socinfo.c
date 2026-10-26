/*
 * What chip this is, from the chip itself: sysctl kern.hakios.soc_*. Only with boot-arg socinfo=1.
 *
 * Android's /sys/devices/soc0/{soc_id,revision,serial_number,machine,family,...} come from one item of the
 * SoC's shared memory ("SMEM"), which the boot firmware fills before anything else runs: the "socinfo"
 * structure (SMEM item 137). Linux's drivers/soc/qcom/socinfo.c reads it; so does this. It says which
 * chip and revision this is, which board (hardware platform and its subtype), which foundry made the die,
 * and the chip's serial number. (The structure also lists the board's power-management chips; this does not
 * read that part yet.) A kernel that is built for one SoC family can then check the phone it was put on really is one, instead of
 * trusting that the device tree handed to it came from this phone (docs/port-pkg110.md: a phone that
 * reports itself as another model is the case that made this worth writing).
 *
 * THIS DRIVER ONLY READS. The SMEM region is firmware's: 0x200000 bytes at soc_smem (the phone's device tree
 * reserves it as smem@80900000, no-map). It is read with byte loads only, because the memory is mapped as
 * device memory here and an unaligned or wide load of it would fault; every offset is checked against the
 * region before it is read, and a header that is not what it should be ends the search with a line that says
 * what was found instead.
 *
 * Where things are in SMEM (Linux drivers/soc/qcom/smem.c):
 *   - a table of "partitions" in the last 4 KB of the region ("$TOC"), one of which is the GLOBAL one (host
 *     0xfffe on both sides): a header ("$PRT") and, from its start, a chain of items, each a 16-byte entry
 *     (0xa5a5 canary, item number, size, padding) and then its data;
 *   - or, on the older layout, a header at the start of the region with a fixed table of 512 items.
 * Both are tried, the partition table first.
 *
 * The structure grows by appending; a field is read only when the item is long enough to hold it.
 *
 * Status (what has and has not been seen): written from Linux's smem.c and socinfo.c as remembered, not from a
 * copy of them. RUN ON THE PHONE (POCO X5 Pro, 2026-10-27, five runs): the item is found in the global partition and
 * says format 0xf, SoC 475 (the device tree's msm-id 0x1db), revision 1.0, board platform 11, foundry 1, 272 bytes, raw id
 * 402. What the first runs got wrong, and the diagnostic that found it: a partition-table entry is 48 bytes, not 32;
 * the older table at the start of the region is not there on this phone (its words are text), so the lookup in it
 * finds nothing and that is harmless. The firmware build string came out empty, and the PMIC list at the end of the
 * structure is not read. Not run in QEMU: QEMU has no SMEM, and the driver then says so and does nothing.
 *
 * Boot-args:
 *   socinfo=1        read it
 *   soc_smem=<pa>    where SMEM is (also the modem driver's argument; 0x80900000 on the reference phone)
 *   soc_smem_size=<n> its size (0x200000)
 */
#include <kern/startup.h>
#include <libkern/libkern.h>
#include <mach/mach_types.h>
#include <pexpert/pexpert.h>
#include <sys/sysctl.h>
#include <sys/systm.h>

extern vm_offset_t ml_io_map(vm_offset_t phys_addr, vm_size_t size);
extern void loader_log_putc(char c);

SYSCTL_DECL(_kern_hakios);

static int say(const char *format, ...) __printflike(1, 2);

static int
say(const char *format, ...)
{
	char line[300];
	va_list list;

	va_start(list, format);
	vsnprintf(line, sizeof(line), format, list);
	va_end(list);
	for (const char *c = line; *c != 0; c++) {
		loader_log_putc(*c);
	}
	return printf("%s", line);
}

#define SMEM_DEFAULT_PA         0x80900000UL
#define SMEM_DEFAULT_SIZE       0x200000UL
#define SMEM_HW_SW_BUILD_ID     137             /* the socinfo structure */

#define PTABLE_MAGIC            0x434f5424u     /* "$TOC" */
#define PART_MAGIC              0x54525024u     /* "$PRT" */
#define PTABLE_ENTRY            48              /* offset, size, flags, host0, host1, cacheline, reserved[7] (Linux smem.c) */
#define GLOBAL_HOST             0xfffeu
#define PRIVATE_CANARY          0xa5a5u
#define LEGACY_ITEMS            512
#define LEGACY_TOC_OFFSET       208             /* 4 proc_comm (64) + version[32] (128) + 4 words (16) */
#define LEGACY_SBL_VERSION      7               /* version[7]: 12 means the partitioned layout */

static vm_offset_t smem;
static vm_size_t smem_size;

static bool
inside(uint64_t offset, uint64_t length)
{
	return offset <= smem_size && length <= smem_size - offset;
}

static uint8_t
b8(uint64_t offset)
{
	return inside(offset, 1) ? *(volatile uint8_t *)(smem + offset) : 0;
}

static uint32_t
le16(uint64_t offset)
{
	return b8(offset) | (uint32_t)b8(offset + 1) << 8;
}

static uint32_t
le32(uint64_t offset)
{
	return le16(offset) | le16(offset + 2) << 16;
}

/*
 * Item `item` of the shared memory: its offset and length in the region, or false. `how` says
 * which layout it was found in, for the log.
 */
static bool
smem_item(uint32_t item, uint64_t *offset, uint64_t *length, const char **how)
{
	uint64_t table = smem_size - 0x1000;

	if (smem_size >= 0x2000 && le32(table) == PTABLE_MAGIC) {
		uint32_t entries = le32(table + 8);

		for (uint32_t i = 0; i < entries && i < 32; i++) {
			uint64_t entry = table + 32 + PTABLE_ENTRY * (uint64_t)i;
			uint32_t part_offset = le32(entry), part_size = le32(entry + 4);

			if (le16(entry + 12) != GLOBAL_HOST || le16(entry + 14) != GLOBAL_HOST || part_size == 0) {
				continue;
			}
			if (!inside(part_offset, part_size) || le32(part_offset) != PART_MAGIC) {
				say("socinfo: the global partition (offset %#x, size %#x) has no \"$PRT\" header (%#x)\n", part_offset,
				    part_size, inside(part_offset, 4) ? le32(part_offset) : 0);
				continue;
			}
			uint64_t at = part_offset + 32, end = part_offset + le32(part_offset + 16);       /* offset_free_uncached */

			if (end > (uint64_t)part_offset + part_size) {
				end = (uint64_t)part_offset + part_size;
			}
			while (at + 16 <= end) {
				uint32_t size = le32(at + 4), padding_hdr = le16(at + 10), padding_data = le16(at + 8);
				uint64_t data = at + 16 + padding_hdr;

				if (le16(at) != PRIVATE_CANARY) {
					say("socinfo: item chain broken at %#llx (canary %#x)\n", (unsigned long long)at, le16(at));
					break;
				}
				if (le16(at + 2) == item && size >= padding_data && inside(data, size - padding_data)) {
					*offset = data;
					*length = size - padding_data;
					*how = "the global partition";
					return true;
				}
				at = data + size;
			}
		}
	}
	/*
	 * The older layout, which Linux's qcom_smem_get_global() uses for every item that is not in a partition (and the
	 * socinfo is one): a header at the start of the region with a fixed table of 512 items. Not gated on the
	 * firmware's version word: the partitioned layout keeps this table for the items it has not moved.
	 */
	if (item < LEGACY_ITEMS) {
		uint64_t entry = LEGACY_TOC_OFFSET + 16 * (uint64_t)item;

		if (le32(entry) != 0 && inside(le32(entry + 4), le32(entry + 8))) {
			*offset = le32(entry + 4);
			*length = le32(entry + 8);
			*how = "the legacy table";
			return true;
		}
	}
	return false;
}

/* For a layout that is not what was expected: which items the legacy table has allocated (item number, size). */
static void
list_items(void)
{
	char line[200];
	unsigned shown = 0;

	line[0] = 0;
	for (uint32_t item = 0; item < LEGACY_ITEMS && shown < 24; item++) {
		uint64_t entry = LEGACY_TOC_OFFSET + 16 * (uint64_t)item;
		char one[24];

		if (le32(entry) == 0) {
			continue;
		}
		snprintf(one, sizeof(one), " %u:%u@%#x", item, le32(entry + 8), le32(entry + 4));
		strlcat(line, one, sizeof(line));
		shown++;
	}
	say("socinfo: the legacy table has allocated:%s%s; the partition table has %u entries\n", line[0] ? line : " nothing", shown == 24 ? " ..." : "",
	    le32(smem_size - 0x1000 + 8));
	for (uint32_t i = 0; i < le32(smem_size - 0x1000 + 8) && i < 12; i++) {
		uint64_t entry = smem_size - 0x1000 + 32 + PTABLE_ENTRY * (uint64_t)i;

		say("socinfo:   partition %u: offset %#x size %#x hosts %#x/%#x, header magic there %#x\n", i, le32(entry), le32(entry + 4),
		    le16(entry + 12), le16(entry + 14), inside(le32(entry), 4) ? le32(le32(entry)) : 0);
	}
	say("socinfo: the first words of the region: %08x %08x %08x %08x %08x %08x %08x %08x\n", le32(0), le32(4), le32(8), le32(12),
	    le32(16), le32(20), le32(24), le32(28));
}

/* ---- what the structure holds ---------------------------------------------------------------------------- */

static struct {
	bool            valid;
	uint32_t        fmt, id, version, raw_id, raw_version, hw_platform, platform_version, hw_subtype, foundry, serial,
	    family, raw_family, raw_num;
	char            build[33];
} soc;

static const struct { uint32_t id; const char *name; } known[] = {
	{ 415, "SM8350" }, { 439, "SM8350" }, { 457, "SM8450" }, { 475, "SC7280 (SM7325, \"yupik\")" },
	{ 487, "SC7280" }, { 497, "QCM6490" }, { 498, "QCS6490" }, { 483, "SM6350" }, { 434, "SM6350" },
	{ 480, "SM7225" }, { 459, "SM7225" }, { 416, "SM8250" }, { 356, "SM8250" },
};

static const char *
soc_name(uint32_t id)
{
	for (unsigned i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
		if (known[i].id == id) {
			return known[i].name;
		}
	}
	return "an SoC this driver has no name for";
}

static void
parse(uint64_t at, uint64_t length)
{
	/*
	 * Offsets of the fields in struct socinfo (Linux socinfo.c). The structure grew by appending, so a field is
	 * read only when the item is long enough to hold it; one that is not there stays 0. (Linux does the same:
	 * it tests offsetof() against the item's size, not the format number.)
	 */
	#define FIELD(offset) (length >= (offset) + 4 ? le32(at + (offset)) : 0)

	soc.fmt = FIELD(0);
	soc.id = FIELD(4);
	soc.version = FIELD(8);
	for (unsigned i = 0; i < 32 && length >= 12 + i + 1; i++) {
		uint8_t c = b8(at + 12 + i);

		if (c == 0) {
			break;
		}
		soc.build[i] = (c >= 0x20 && c < 0x7f) ? (char)c : '?';
	}
	soc.raw_id = FIELD(44);
	soc.raw_version = FIELD(48);
	soc.hw_platform = FIELD(52);
	soc.platform_version = FIELD(56);
	soc.hw_subtype = FIELD(64);
	soc.foundry = FIELD(92);
	soc.serial = FIELD(96);
	soc.family = FIELD(108);
	soc.raw_family = FIELD(112);
	soc.raw_num = FIELD(116);
	#undef FIELD
}

static char soc_name_text[64] = "unknown";
static char soc_serial_text[16] = "";
static char soc_build_text[40] = "";
static int soc_id_value = -1;
static int soc_revision_value = -1;
static int soc_platform_value = -1;
static int soc_subtype_value = -1;
static int soc_foundry_value = -1;

SYSCTL_INT(_kern_hakios, OID_AUTO, soc_id, CTLFLAG_RD | CTLFLAG_LOCKED, &soc_id_value, 0,
    "the SoC's number in the firmware's table (475: SC7280/SM7325); -1: not read");
SYSCTL_STRING(_kern_hakios, OID_AUTO, soc_name, CTLFLAG_RD | CTLFLAG_LOCKED, soc_name_text, 0, "the SoC's name, as this kernel knows that number");
SYSCTL_INT(_kern_hakios, OID_AUTO, soc_revision, CTLFLAG_RD | CTLFLAG_LOCKED, &soc_revision_value, 0,
    "the chip revision, major << 16 | minor (0x10000 is 1.0)");
SYSCTL_INT(_kern_hakios, OID_AUTO, soc_platform, CTLFLAG_RD | CTLFLAG_LOCKED, &soc_platform_value, 0, "the board's hardware platform number");
SYSCTL_INT(_kern_hakios, OID_AUTO, soc_platform_subtype, CTLFLAG_RD | CTLFLAG_LOCKED, &soc_subtype_value, 0, "its subtype");
SYSCTL_INT(_kern_hakios, OID_AUTO, soc_foundry, CTLFLAG_RD | CTLFLAG_LOCKED, &soc_foundry_value, 0, "the foundry that made the die");
SYSCTL_STRING(_kern_hakios, OID_AUTO, soc_serial, CTLFLAG_RD | CTLFLAG_LOCKED, soc_serial_text, 0,
    "the chip's serial number (hex). It is this unit's, so it is not written to the log");
SYSCTL_STRING(_kern_hakios, OID_AUTO, soc_build, CTLFLAG_RD | CTLFLAG_LOCKED, soc_build_text, 0, "the firmware's build id");

__startup_func
static void
socinfo_init(void)
{
	uint32_t enabled = 0, pa = (uint32_t)SMEM_DEFAULT_PA, size = (uint32_t)SMEM_DEFAULT_SIZE;
	uint64_t offset, length;
	const char *how;

	if (!PE_parse_boot_argn("socinfo", &enabled, sizeof(enabled)) || !enabled) {
		return;
	}
	(void)PE_parse_boot_argn("soc_smem", &pa, sizeof(pa));
	(void)PE_parse_boot_argn("soc_smem_size", &size, sizeof(size));
	if (size < 0x2000 || size > 0x1000000) {
		say("socinfo: soc_smem_size=%#x is not a size SMEM can have; not read\n", size);
		return;
	}
	smem_size = size;
	smem = ml_io_map(pa, size);
	if (smem == 0) {
		say("socinfo: could not map %#x bytes at %#x\n", size, pa);
		return;
	}
	if (!smem_item(SMEM_HW_SW_BUILD_ID, &offset, &length, &how)) {
		say("socinfo: item %d is not in the shared memory at %#x (partition table magic %#x, legacy version word %#x)\n",
		    SMEM_HW_SW_BUILD_ID, pa, le32(smem_size - 0x1000), le32(4 * 4 * 4 + 4 * LEGACY_SBL_VERSION));
		list_items();
		return;
	}
	if (length < 56) {
		say("socinfo: item %d is %llu bytes, too short for a socinfo structure\n", SMEM_HW_SW_BUILD_ID, (unsigned long long)length);
		return;
	}
	parse(offset, length);
	soc.valid = true;
	soc_id_value = (int)soc.id;
	soc_revision_value = (int)soc.version;
	soc_platform_value = (int)soc.hw_platform;
	soc_subtype_value = (int)soc.hw_subtype;
	soc_foundry_value = (int)soc.foundry;
	strlcpy(soc_name_text, soc_name(soc.id), sizeof(soc_name_text));
	strlcpy(soc_build_text, soc.build, sizeof(soc_build_text));
	snprintf(soc_serial_text, sizeof(soc_serial_text), "%08x", soc.serial);
	say("socinfo: in %s: format %#x, SoC %u = %s, revision %u.%u, board platform %u (version %u, subtype %u), foundry %u,"
	    " %llu bytes, firmware build \"%s\"\n", how, soc.fmt, soc.id, soc_name_text, soc.version >> 16, soc.version & 0xffff,
	    soc.hw_platform, soc.platform_version, soc.hw_subtype, soc.foundry, (unsigned long long)length, soc.build);
	say("socinfo: raw id %u, raw version %#x; serial number ...%04x (all of it in kern.hakios.soc_serial)\n", soc.raw_id,
	    soc.raw_version, soc.serial & 0xffff);
}
STARTUP(EARLY_BOOT, STARTUP_RANK_LAST, socinfo_init);
