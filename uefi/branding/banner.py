#!/usr/bin/env python3
"""The pictures HakiOS's UEFI shows in place of Mu-Silicium's: the boot logo, the "no system to boot" screen and the USB mass
storage screen. Made here from a few shapes and a 5x7 pixel font, with nothing to install, so that anyone can change the text
or the colours and make them again.

The files are what Mu-Silicium's SiliciumPkg/Include/Resources/ holds (Extra.fdf.inc puts them into the firmware volume):
8-bit BMPs with a 256-colour palette, BootLogo.bmp and NoBootOS.bmp 300x300, MassStorage.bmp 500x500.

usage: uefi/branding/banner.py OUTDIR        (uefi/apply.sh runs it into the Mu-Silicium tree)
"""
import os
import struct
import sys

INK, RED, PAPER, GREY = (0, 0, 0), (0xE8, 0x17, 0x0F), (0xEF, 0xE9, 0xE6), (0x8A, 0x84, 0x82)
PALETTE = [INK, RED, PAPER, GREY]          # indices 0..3; the rest of the 256 are black

FONT = {                                   # 5x7, one string per row, "#" is a pixel
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ### "],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    " ": ["     "] * 7,
}


class Canvas:
    def __init__(self, size):
        self.size = size
        self.px = bytearray(size * size)      # palette indices, top row first

    def dot(self, x, y, c):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.px[y * self.size + x] = c

    def rect(self, x0, y0, w, h, c):
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                self.dot(x, y, c)

    def disc(self, cx, cy, r, c):
        for y in range(int(cy - r), int(cy + r) + 1):
            for x in range(int(cx - r), int(cx + r) + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.dot(x, y, c)

    def pill(self, x0, y0, w, h, c):
        """A rectangle with round ends (the emblem's eyes)."""
        r = w / 2
        self.rect(x0, int(y0 + r), w, int(h - 2 * r), c)
        self.disc(x0 + r - 0.5, y0 + r, r, c)
        self.disc(x0 + r - 0.5, y0 + h - r, r, c)

    def line(self, x0, y0, x1, y1, width, c):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(steps + 1):
            t = i / steps
            self.disc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width / 2, c)

    def text(self, s, y, scale, colours, spacing=1):
        """S centred at row Y; COLOURS gives each character's colour (a list as long as S, or one for all)."""
        width = len(s) * (5 + spacing) * scale - spacing * scale
        x = (self.size - width) // 2
        for i, ch in enumerate(s):
            c = colours[i] if isinstance(colours, list) else colours
            for row, bits in enumerate(FONT[ch]):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        self.rect(x + col * scale, y + row * scale, scale, scale, c)
            x += (5 + spacing) * scale

    def emblem(self, cx, top, unit):
        """The HakiOS emblem (the website's logo): a red head with two dark eyes and two ears, UNIT pixels to the logo's one."""
        red = 1
        u = unit
        self.line(cx - 9.5 * u, top + 9 * u, cx - 13 * u, top + 0 * u, 2.2 * u, red)     # ears
        self.line(cx + 9.5 * u, top + 9 * u, cx + 13 * u, top + 0 * u, 2.2 * u, red)
        self.disc(cx - 13 * u, top + 0 * u, 2.4 * u, red)
        self.disc(cx + 13 * u, top + 0 * u, 2.4 * u, red)
        self.disc(cx, top + 18.5 * u, 13 * u, red)                                        # head
        self.pill(int(cx - 6.5 * u), int(top + 14.5 * u), int(5 * u), int(7 * u), 0)    # eyes
        self.pill(int(cx + 1.5 * u), int(top + 14.5 * u), int(5 * u), int(7 * u), 0)

    def bmp(self, path):
        """An 8-bit BMP (BITMAPINFOHEADER, 256-colour palette, rows bottom-up, each a multiple of 4 bytes)."""
        stride = (self.size + 3) & ~3
        palette = b"".join(struct.pack("<BBBB", b, g, r, 0) for r, g, b in PALETTE + [INK] * (256 - len(PALETTE)))
        rows = b"".join(bytes(self.px[y * self.size:(y + 1) * self.size]) + b"\0" * (stride - self.size)
                        for y in reversed(range(self.size)))
        offset = 14 + 40 + len(palette)
        header = struct.pack("<2sIHHI", b"BM", offset + len(rows), 0, 0, offset)
        info = struct.pack("<IiiHHIIiiII", 40, self.size, self.size, 1, 8, 0, len(rows), 3779, 3779, 256, 256)
        with open(path, "wb") as f:
            f.write(header + info + palette + rows)


def wordmark(c, y, scale):
    c.text("HAKIOS", y, scale, [2, 2, 2, 2, 1, 1])     # HAKI in paper, OS in red: the website's wordmark


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)

    logo = Canvas(300)
    logo.emblem(150, 40, 4.4)
    wordmark(logo, 196, 6)
    logo.bmp(os.path.join(out, "BootLogo.bmp"))

    none = Canvas(300)
    none.emblem(150, 30, 3.2)
    wordmark(none, 150, 5)
    none.text("NO SYSTEM", 214, 3, 3)
    none.text("TO BOOT", 244, 3, 3)
    none.bmp(os.path.join(out, "NoBootOS.bmp"))

    storage = Canvas(500)
    storage.emblem(250, 60, 6)
    wordmark(storage, 300, 8)
    storage.text("USB MASS STORAGE", 400, 4, 3)
    storage.bmp(os.path.join(out, "MassStorage.bmp"))
    print("banner: BootLogo.bmp, NoBootOS.bmp, MassStorage.bmp written to %s" % out)


if __name__ == "__main__":
    main()
