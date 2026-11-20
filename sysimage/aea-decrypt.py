#!/usr/bin/env python3
# Decrypt an Apple Encrypted Archive (AEA1, profile 1: symmetric key, HKDF-SHA256, AES-CTR, HMAC-SHA256) as the
# root filesystem and the OS cryptex of iOS 18+ IPSWs ship (*.dmg.aea). The archive's key is wrapped (HPKE, RFC
# 9180: DHKEM P-256 / HKDF-SHA256 / AES-256-GCM) in its own auth data; the private key that opens it is public for
# released builds, at the URL the auth data names (com.apple.wkms.fcs-key-url).
#
#   aea-decrypt.py IPSW MEMBER OUT [--pem FILE]   (reads MEMBER from inside the IPSW, a zip)
#   aea-decrypt.py FILE.aea OUT [--pem FILE]
#
# Without --pem the key is fetched from the URL in the file (nothing else is sent). Every cluster and segment is
# checked against its HMAC. Compression: none, LZFSE (needs /usr/bin/lzfse or the lzfse module) and zlib.
import sys, os, struct, json, base64, hmac, hashlib, zipfile, zlib, urllib.request, subprocess
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def hext(salt, ikm): return hmac.new(salt, ikm, hashlib.sha256).digest()
def hexp(prk, info, L):
    out = b''; t = b''; i = 1
    while len(out) < L:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest(); out += t; i += 1
    return out[:L]
def hkdf(ikm, salt, info, L): return hexp(hext(salt, ikm), info, L)
def ctr(key, iv, data):
    d = Cipher(algorithms.AES(key), modes.CTR(iv)).decryptor(); return d.update(data) + d.finalize()
def mac(key, *parts):
    m = hmac.new(key, digestmod=hashlib.sha256)
    for p in parts: m.update(p)
    return m.digest()

def hpke_open(sk, enc, ct):
    kem = b'KEM' + struct.pack('>H', 0x10)
    suite = b'HPKE' + struct.pack('>HHH', 0x10, 1, 2)
    lx = lambda s, salt, label, ikm: hext(salt, b'HPKE-v1' + s + label + ikm)
    le = lambda s, prk, label, info, L: hexp(prk, struct.pack('>H', L) + b'HPKE-v1' + s + label + info, L)
    dh = sk.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), enc))
    pkR = sk.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    shared = le(kem, lx(kem, b'', b'eae_prk', dh), b'shared_secret', enc + pkR, 32)
    ksc = b'\0' + lx(suite, b'', b'psk_id_hash', b'') + lx(suite, b'', b'info_hash', b'')
    secret = lx(suite, shared, b'secret', b'')
    return AESGCM(le(suite, secret, b'key', ksc, 32)).decrypt(le(suite, secret, b'base_nonce', ksc, 12), ct, b'')

def lzfse_dec(data, size):
    try:
        import lzfse; return lzfse.decompress(data)
    except ImportError:
        return subprocess.run(['lzfse', '-decode'], input=data, stdout=subprocess.PIPE, check=True).stdout

def main():
    a = sys.argv[1:]; pem = None
    if '--pem' in a: i = a.index('--pem'); pem = a[i + 1]; del a[i:i + 2]
    if len(a) == 3: f = zipfile.ZipFile(a[0]).open(a[1]); out = a[2]
    elif len(a) == 2: f = open(a[0], 'rb'); out = a[1]
    else: sys.exit(__doc__ or 'usage: aea-decrypt.py IPSW MEMBER OUT | FILE.aea OUT [--pem FILE]')
    pos = [0]
    def read(n):
        b = f.read(n)
        if len(b) != n: sys.exit('aea-decrypt: truncated archive')
        pos[0] += n; return b
    hdr = read(12)
    if hdr[:4] != b'AEA1': sys.exit('aea-decrypt: not an AEA1 archive')
    prof, adlen = struct.unpack('<II', hdr[4:12])
    if prof & 0xffffff != 1: sys.exit('aea-decrypt: profile %d (only 1, symmetric, is handled)' % (prof & 0xffffff))
    ad = read(adlen); kv = {}; i = 0
    while i < len(ad):
        l = struct.unpack('<I', ad[i:i + 4])[0]; k, v = ad[i + 4:i + l].split(b'\0', 1); kv[k] = v; i += l
    resp = json.loads(kv[b'com.apple.wkms.fcs-response'])
    pemdata = open(pem, 'rb').read() if pem else urllib.request.urlopen(kv[b'com.apple.wkms.fcs-key-url'].decode()).read()
    key = hpke_open(serialization.load_pem_private_key(pemdata, None),
                    base64.b64decode(resp['enc-request']), base64.b64decode(resp['wrapped-key']))
    salt = read(32)
    mk = hkdf(key, salt, b'AEA_AMK' + hdr[4:8], 32)
    rh_mac, rh_enc, cmac = read(32), read(48), read(32)
    rk = hkdf(mk, b'', b'AEA_RHEK', 80)
    if mac(rk[:32], cmac, ad, rh_enc, struct.pack('<Q', 32 + adlen)) != rh_mac: sys.exit('aea-decrypt: root header HMAC wrong (key?)')
    rh = ctr(rk[32:64], rk[64:80], rh_enc)
    fsize, esize, segsize, spc = struct.unpack('<QQII', rh[:24]); comp = chr(rh[24]); csum = rh[25]
    cbytes = {0: 0, 1: 8, 2: 32}[csum]
    print('aea-decrypt: %d bytes out, segments of %d, %d a cluster, compression %r' % (fsize, segsize, spc, comp), file=sys.stderr)
    esz = 8 + cbytes; hsz = esz * spc
    done = 0; cluster = 0
    with open(out + '.part', 'wb') as o:
        while done < fsize:
            ck = hkdf(mk, b'', b'AEA_CK' + struct.pack('<I', cluster), 32)
            m = hkdf(ck, b'', b'AEA_CHEK', 80)
            blob = read(hsz + 32 + 32 * spc); heads, macs = blob[:hsz], blob[hsz:]
            if mac(m[:32], macs, heads, struct.pack('<Q', len(macs))) != cmac: sys.exit('aea-decrypt: cluster %d HMAC wrong' % cluster)
            cmac = macs[:32]
            heads = ctr(m[32:64], m[64:80], heads)
            prk = hext(b'', ck)
            for s in range(spc):
                dsz, csz = struct.unpack('<II', heads[s * esz:s * esz + 8])
                if dsz == 0: continue
                data = read(csz)
                sk = hexp(prk, b'AEA_SK' + struct.pack('<I', s), 80)
                if mac(sk[:32], data, struct.pack('<Q', 0)) != macs[32 + 32 * s:64 + 32 * s]:
                    sys.exit('aea-decrypt: segment %d/%d HMAC wrong' % (cluster, s))
                data = ctr(sk[32:64], sk[64:80], data)
                if csz != dsz:
                    if comp == 'e': data = lzfse_dec(data, dsz)
                    elif comp == 'z': data = zlib.decompress(data, -15)
                    elif comp == '-': pass
                    else: sys.exit('aea-decrypt: compression %r not handled' % comp)
                if len(data) != dsz: sys.exit('aea-decrypt: segment %d/%d decoded to %d, not %d' % (cluster, s, len(data), dsz))
                o.write(data); done += dsz
            cluster += 1
            if cluster % 64 == 0: print('aea-decrypt: %d MB' % (done >> 20), file=sys.stderr)
    os.rename(out + '.part', out)
    print('aea-decrypt: done, %s (%d bytes)' % (out, done), file=sys.stderr)
main()
