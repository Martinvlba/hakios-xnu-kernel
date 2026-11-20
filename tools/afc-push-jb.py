#!/usr/bin/env python3
"""Push the package manager's prefix to the phone over AFC, for a boot whose payload has no room
for it (the 63 MB limit).  The prefix ends up in /var/mobile/Media/.jb, where init finds it
(src/init.c, mount_writable).

usage: afc-push-jb.py [JB_ROOT]    default JB_ROOT: ../jb/root/var/jb (tools/mk-jb.sh's output)
       afc-push-jb.py --tar [JB_ROOT]   the same as one file, Media/jb.tar, extracted by init at the next start
"""
import os, plistlib, socket, ssl, struct, sys, tempfile, time

MUX = '/var/run/usbmuxd'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mux_request(message):
    s = socket.socket(socket.AF_UNIX)
    s.connect(MUX)
    body = plistlib.dumps(message)
    s.sendall(struct.pack('<IIII', 16 + len(body), 1, 8, 1) + body)
    head = b''
    while len(head) < 16:
        head += s.recv(16 - len(head))
    n = struct.unpack('<I', head[:4])[0]
    data = b''
    while len(data) < n - 16:
        data += s.recv(n - 16 - len(data))
    return s, plistlib.loads(data)


def recv_all(s, n):
    data = b''
    while len(data) < n:
        part = s.recv(n - len(data))
        if not part:
            raise EOFError
        data += part
    return data


def send_plist(s, d):
    body = plistlib.dumps(d)
    s.sendall(struct.pack('>I', len(body)) + body)


def recv_plist(s):
    n = struct.unpack('>I', recv_all(s, 4))[0]
    return plistlib.loads(recv_all(s, n))


AFC_STATUS, AFC_DATA, AFC_MKDIR, AFC_OPEN, AFC_OPENED, AFC_WRITE, AFC_CLOSE, AFC_REMOVE = 1, 2, 9, 13, 14, 16, 20, 8
AFC_REMOVE_ALL = 22


class Afc:
    def __init__(self, s):
        self.s, self.number = s, 0

    def call(self, op, header=b'', payload=b''):
        self.number += 1
        self.s.sendall(b'CFA6LPAA' + struct.pack('<QQQQ', 40 + len(header) + len(payload),
                       40 + len(header), self.number, op) + header + payload)
        head = recv_all(self.s, 40)
        whole = struct.unpack('<Q', head[8:16])[0]
        data = recv_all(self.s, whole - 40)
        answer = struct.unpack('<Q', head[32:40])[0]
        if answer == AFC_STATUS:
            status = struct.unpack('<Q', data[:8])[0]
            if status != 0:
                raise OSError('AFC status %d for op %d' % (status, op))
        return answer, data

    def mkdir(self, path):
        try:
            self.call(AFC_MKDIR, path.encode() + b'\0')
        except OSError:
            pass

    def put(self, local, remote):
        _, data = self.call(AFC_OPEN, struct.pack('<Q', 3) + remote.encode() + b'\0')
        handle = data[:8]
        sent = 0
        with open(local, 'rb') as f:
            while True:
                part = f.read(1 << 16)
                if not part:
                    break
                self.call(AFC_WRITE, handle, part)
                sent += len(part)
        self.call(AFC_CLOSE, handle)
        return sent

    def put_bytes(self, content, remote):
        _, data = self.call(AFC_OPEN, struct.pack('<Q', 3) + remote.encode() + b'\0')
        handle = data[:8]
        self.call(AFC_WRITE, handle, content)
        self.call(AFC_CLOSE, handle)
        return len(content)

    def symlink(self, target, link):
        self.call(28, struct.pack('<Q', 2) + target.encode() + b'\0' + link.encode() + b'\0')

    def remove(self, path):
        try:
            self.call(AFC_REMOVE_ALL, path.encode() + b'\0')
        except OSError:
            pass


def connect_afc():
    # (11-20) the phone can drop out of usbmuxd's list for a moment while AFC reconnects: wait up to a minute for it
    for _ in range(30):
        _, r = mux_request({'MessageType': 'ListDevices'})
        devs = [d for d in r.get('DeviceList', []) if d.get('Properties', {}).get('ConnectionType') == 'USB']
        if devs:
            break
        time.sleep(2)
    if not devs:
        sys.exit('afc-push-jb: no device')
    dev_id = devs[0]['DeviceID']
    udid = devs[0]['Properties']['SerialNumber']
    s, answer = mux_request({'MessageType': 'ReadPairRecord', 'PairRecordID': udid})
    s.close()
    pair = plistlib.loads(answer['PairRecordData'])
    td = tempfile.TemporaryDirectory()
    cert = os.path.join(td.name, 'host.pem')
    with open(cert, 'wb') as f:
        f.write(pair['HostCertificate'] + b'\n' + pair['HostPrivateKey'])

    def conn(port):
        s, a = mux_request({'MessageType': 'Connect', 'DeviceID': dev_id, 'PortNumber': socket.htons(port)})
        if a.get('Number', 1) != 0:
            s.close()
            raise ConnectionError('port %d: %s' % (port, a))
        return s

    def do_tls(s):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            ctx.set_ciphers('ALL:@SECLEVEL=0')
        except ssl.SSLError:
            pass
        ctx.load_cert_chain(cert)
        return ctx.wrap_socket(s)

    s = conn(62078)
    send_plist(s, {'Label': 'afc-push-jb', 'Request': 'QueryType'})
    recv_plist(s)
    send_plist(s, {'Label': 'afc-push-jb', 'Request': 'StartSession',
                   'HostID': pair['HostID'], 'SystemBUID': pair['SystemBUID']})
    session = recv_plist(s)
    if session.get('EnableSessionSSL'):
        s = do_tls(s)
    send_plist(s, {'Label': 'afc-push-jb', 'Request': 'StartService', 'Service': 'com.apple.afc'})
    svc = recv_plist(s)
    s.close()
    if 'Error' in svc:
        sys.exit('afc-push-jb: lockdownd: ' + svc['Error'])
    return Afc(conn(svc['Port'])), td


def push_tree(reconnect_fn, local_root, remote_root):
    dirs, files, links, sent_bytes = 0, 0, 0, 0
    symlinks = []
    skipped = []
    t0 = time.time()
    afc = reconnect_fn()
    ops = 0

    for dirpath, dirnames, filenames in os.walk(local_root, followlinks=False):
        rel = os.path.relpath(dirpath, local_root)
        rdir = remote_root if rel == '.' else remote_root + '/' + rel
        afc.mkdir(rdir)
        dirs += 1
        ops += 1

        drop = []
        for name in sorted(dirnames):
            local_path = os.path.join(dirpath, name)
            if os.path.islink(local_path):
                target = os.readlink(local_path)
                link_rel = name if rel == '.' else rel + '/' + name
                symlinks.append(link_rel + '\t' + target)
                links += 1
                drop.append(name)
        for d in drop:
            dirnames.remove(d)

        for name in sorted(filenames):
            local_path = os.path.join(dirpath, name)
            remote = rdir + '/' + name
            if os.path.islink(local_path):
                target = os.readlink(local_path)
                link_rel = name if rel == '.' else rel + '/' + name
                symlinks.append(link_rel + '\t' + target)
                links += 1
            else:
                try:
                    sent_bytes += afc.put(local_path, remote)
                except (EOFError, OSError, ConnectionError) as e:
                    for attempt in range(5):
                        print('\n  reconnecting after %r on %s ...' % (e, remote))
                        time.sleep(2)
                        try:
                            afc = reconnect_fn()
                            sent_bytes += afc.put(local_path, remote)
                            break
                        except (EOFError, OSError, ConnectionError) as e2:
                            e = e2
                    else:
                        # (11-20: usr/libexec/sudo/sudo_noexec.so every time -- the phone's afcd drops the connection
                        # on it) left out, said at the end; nothing Sileo needs
                        skipped.append(remote)
                        afc = reconnect_fn()
                files += 1
                ops += 3
            if (files + links) % 200 == 0:
                print('\r  %d files, %d links, %d dirs, %d MB ...' % (files, links, dirs, sent_bytes >> 20),
                      end='', flush=True)
            if ops > 4000:
                try:
                    afc.s.close()
                except Exception:
                    pass
                afc = reconnect_fn()
                ops = 0

    if symlinks:
        manifest = ('\n'.join(symlinks) + '\n').encode()
        try:
            afc.put_bytes(manifest, remote_root + '/.symlinks')
        except (EOFError, OSError, ConnectionError):
            afc = reconnect_fn()
            afc.put_bytes(manifest, remote_root + '/.symlinks')
        print('\r  %d files, %d links (%d in .symlinks), %d dirs, %d MB in %.0f s' %
              (files, links, len(symlinks), dirs, sent_bytes >> 20, time.time() - t0))
    else:
        print('\r  %d files, %d dirs, %d MB in %.0f s' % (files, dirs, sent_bytes >> 20, time.time() - t0))
    for r in skipped:
        print('  LEFT OUT (would not go): %s' % r)
    return files + links


def remote_size(afc, path):
    # AFC_GET_FILE_INFO (10): "key\0value\0" pairs, st_size among them; -1 when it is not there
    try:
        _, data = afc.call(10, path.encode() + b'\0')
    except OSError:
        return -1
    parts = data.split(b'\0')
    for k, v in zip(parts[0::2], parts[1::2]):
        if k == b'st_size':
            return int(v)
    return -1


def push_tar(reconnect_fn, jb_root):
    """--tar (11-20): the prefix as ONE file, Media/jb.tar, which init extracts once into Media/.jb (src/init.c,
    extract_tar: ustar, its name prefix read too) -- 6925 files one by one stalled the USB link again and again. A drop
    is resumed: the phone asked how much arrived, the rest appended (AFC open mode 5)."""
    import tarfile
    tmp = tempfile.NamedTemporaryFile(suffix='.tar', delete=False)
    tmp.close()
    with tarfile.open(tmp.name, 'w', format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(os.listdir(jb_root)):
            t.add(os.path.join(jb_root, name), arcname=name, recursive=True)
    total = os.path.getsize(tmp.name)
    print('afc-push-jb: jb.tar of %d MB' % (total >> 20))
    afc = reconnect_fn()
    afc.remove('.jb')
    afc.remove('jb.tar')
    done, mode, t0 = 0, 3, time.time()
    while done < total:
        try:
            _, data = afc.call(AFC_OPEN, struct.pack('<Q', mode) + b'jb.tar\0')
            handle = data[:8]
            with open(tmp.name, 'rb') as f:
                f.seek(done)
                while True:
                    part = f.read(1 << 16)
                    if not part:
                        break
                    afc.call(AFC_WRITE, handle, part)
                    done += len(part)
                    if (done >> 20) % 8 == 0 and len(part) == 1 << 16 and (done & ((1 << 20) - 1)) == 0:
                        print('\r  %d of %d MB ...' % (done >> 20, total >> 20), end='', flush=True)
            afc.call(AFC_CLOSE, handle)
        except (EOFError, OSError, ConnectionError) as e:
            print('\n  resuming after %r at %d MB ...' % (e, done >> 20))
            time.sleep(3)
            afc = reconnect_fn()
            have = remote_size(afc, 'jb.tar')
            done = max(0, have)
            mode = 5 if have > 0 else 3
    print('\r  jb.tar: %d MB in %.0f s; on the phone: %d bytes' % (total >> 20, time.time() - t0, remote_size(afc, 'jb.tar')))
    os.unlink(tmp.name)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--tar':
        jb_root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'jb', 'root', 'var', 'jb')
        state = {}

        def reconnect_tar():
            afc, td = connect_afc()
            state['td'] = td
            return afc
        push_tar(reconnect_tar, jb_root)
        print('afc-push-jb: done; the next start extracts it (init) -- a minute or so more that once')
        return
    jb_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'jb', 'root', 'var', 'jb')
    if not os.path.isdir(jb_root):
        sys.exit('afc-push-jb: %s: not a directory (run tools/mk-jb.sh first)' % jb_root)

    local_count = sum(len(f) + len(d) for _, d, f in os.walk(jb_root))
    print('afc-push-jb: pushing %s (%d entries) to Media/.jb' % (jb_root, local_count))

    state = {}

    def reconnect():
        afc, td = connect_afc()
        state['td'] = td
        return afc

    afc = reconnect()
    afc.remove('.jb')
    push_tree(reconnect, jb_root, '.jb')
    print('afc-push-jb: done; reboot the phone for init to graft /var/jb from it')


if __name__ == '__main__':
    main()
