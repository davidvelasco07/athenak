#!/usr/bin/env python3
"""Reader for AthenaK .trk particle files (see src/outputs/track_prtcl.cpp).

The file is a sequence of dumps, each a 3-line text header followed by a contiguous
binary block of `nparticles` records of `nrec` float64s. Every column the species
carries is present -- for sinks that includes the MASS -- and the header names them,
so nothing here needs to know the particle type.

Usage
-----
    from read_trk import read_trk, sink_mass_history
    dumps = read_trk("trk/GMTF_sink.trk")        # list of dicts
    t, tags, mass = sink_mass_history(dumps)     # (ndump,), (nsink,), (ndump, nsink)

`sink_mass_history` keys particles by TAG across dumps, so a sink keeps its identity
(and therefore its colour in a plot) as others form, accrete and merge. Slots are NaN
before a sink exists and after it is absorbed by a merger.
"""
import re
import numpy as np

_HDR = re.compile(
    r"time=([\-0-9.eE+]+)\s+cycle=(\d+)\s+nranks=(\d+)\s+nparticles=(\d+)"
    r"\s+nidata=(\d+)\s+nrdata=(\d+)\s+nrec=(\d+)")


def read_trk(path):
    """Return a list of dumps, each {time, cycle, columns, data (npart, nrec)}."""
    raw = open(path, "rb").read()
    dumps, pos = [], 0
    while True:
        i = raw.find(b"# AthenaK particle data", pos)
        if i < 0:
            break
        # the header is exactly three '\n'-terminated lines
        j = i
        for _ in range(3):
            j = raw.find(b"\n", j) + 1
            if j == 0:
                return dumps
        head = raw[i:j].decode("ascii", "replace")
        m = _HDR.search(head)
        if m is None:
            pos = j
            continue
        t, cyc, _nr, npart, _nid, _nrd, nrec = (
            float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
            int(m.group(5)), int(m.group(6)), int(m.group(7)))
        cols = head.split("# columns:")[1].strip().split()
        nbytes = npart * nrec * 8
        blk = np.frombuffer(raw[j:j + nbytes], dtype=np.float64)
        if blk.size != npart * nrec:          # truncated final dump
            break
        dumps.append(dict(time=t, cycle=cyc, columns=cols,
                          data=blk.reshape(npart, nrec)))
        pos = j + nbytes
    return dumps


def column(dump, name):
    """Column `name` of one dump as a 1D array."""
    return dump["data"][:, dump["columns"].index(name)]


def sink_mass_history(dumps, key="tag", value="m"):
    """(time, tags, values) with values[i, j] the mass of tag[j] at time[i].

    Particles are keyed by tag so identity survives creation and merging; entries are
    NaN where that tag is absent from a dump.
    """
    if not dumps:
        return np.array([]), np.array([]), np.zeros((0, 0))
    tags = sorted({int(v) for d in dumps for v in column(d, key)})
    idx = {tg: n for n, tg in enumerate(tags)}
    out = np.full((len(dumps), len(tags)), np.nan)
    for i, d in enumerate(dumps):
        for tg, val in zip(column(d, key), column(d, value)):
            out[i, idx[int(tg)]] = val
    return (np.array([d["time"] for d in dumps]), np.array(tags), out)


if __name__ == "__main__":
    import sys
    ds = read_trk(sys.argv[1])
    print(f"{len(ds)} dumps; columns = {ds[0]['columns'] if ds else '-'}")
    for d in ds[:3] + ([] if len(ds) <= 6 else [None]) + ds[-3:]:
        if d is None:
            print("  ...")
            continue
        print(f"  t={d['time']:.5f} cycle={d['cycle']:5d} n={d['data'].shape[0]:4d} "
              f"M_tot={column(d, 'm').sum():.6e}" if d["data"].size else "  (empty)")
