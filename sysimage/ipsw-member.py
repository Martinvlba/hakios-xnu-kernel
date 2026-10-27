#!/usr/bin/env python3
"""Copy one member out of an IPSW the user has on their own disk.

usage: ipsw-member.py IPSW MEMBER OUTPUT

An IPSW is a ZIP64 archive; Python's zipfile reads it as it is. The member is streamed to OUTPUT
(inflated if it is compressed), so the 4.7 GB root filesystem image never has to fit in memory.
Nothing is downloaded: the project ships no file of Apple's and fetches none. The IPSW is the
user's, got by the user from wherever they get their firmware.
"""
import shutil
import sys
import zipfile


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__.split('\n\n')[1])
    ipsw, member, output = sys.argv[1:]
    try:
        archive = zipfile.ZipFile(ipsw)
    except (OSError, zipfile.BadZipFile) as error:
        sys.exit(f'ipsw-member.py: {ipsw}: {error}')
    with archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            sys.exit(f'ipsw-member.py: {member} is not in {ipsw} (is it the iOS 16.5 20F66 IPSW for iPhone10,3/iPhone10,6?)')
        with archive.open(info) as stream, open(output, 'wb') as out:
            shutil.copyfileobj(stream, out, 1 << 20)
    print(f'{member}: {info.file_size} bytes')


if __name__ == '__main__':
    main()
