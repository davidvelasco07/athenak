#!/usr/bin/env python3
"""Compare two AthenaK run dirs: .bin payloads after the parameter header (plus time/cycle),
history files bytewise, and stdout minus timing/warning lines. Exit 1 on any difference."""
import sys, os, glob, re
a, b = sys.argv[1], sys.argv[2]
ok = True
def payload(p):
    d = open(p, 'rb').read()
    head = d[:4096].decode('latin1')
    m = re.search(r'header offset=(\d+)', head); off = int(m.group(1))
    tc = re.findall(r'(time|cycle)=(\S+)', head)[:2]
    return tc, d[off:]
bins = sorted(os.path.relpath(p, a) for p in glob.glob(f'{a}/bin/*.bin'))
if not bins: print('NO BINS in', a); ok = False
for f in bins:
    q = f'{b}/{f}'
    if not os.path.exists(q): print('MISSING', q); ok = False; continue
    ta, pa = payload(f'{a}/{f}'); tb, pb = payload(q)
    if ta != tb: print('DIFF header time/cycle', f, ta, tb); ok = False
    if pa != pb:
        n = sum(x != y for x, y in zip(pa, pb)) + abs(len(pa)-len(pb))
        print(f'DIFF payload {f}: {n} bytes of {len(pa)}'); ok = False
for f in sorted(glob.glob(f'{a}/hst/*.hst')):
    q = f'{b}/hst/{os.path.basename(f)}'
    if not os.path.exists(q) or open(f,'rb').read() != open(q,'rb').read(): print('DIFF hst', f); ok = False
def so(p):
    return [l for l in open(p) if not re.search(r'elapsed=|cpu time|zone-cycles|### WARNING', l)]
if so(f'{a}/stdout.txt') != so(f'{b}/stdout.txt'): print('DIFF stdout'); ok = False
print(('IDENTICAL' if ok else 'DIFFERENT'), a, 'vs', b, f'({len(bins)} bins)')
sys.exit(0 if ok else 1)
