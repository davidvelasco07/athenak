#!/usr/bin/env python3
"""Per-dump reader for AthenaK sink-particle histories.

Reads both formats into the same list of dumps:
  - particle_track files (file_type = particle_track, the current output; read with
    read_particle_track.py and regrouped by cycle, the tag column renamed "tag");
  - .trk files written by feature/sink-particles-only: a sequence of dumps, each a
    3-line text header followed by a contiguous binary block of `nparticles` records of
    `nrec` float64s, with the column names in the header.

Usage
-----
    from read_trk import read_trk, sink_mass_history
    dumps = read_trk("particle_track/GMTF_sink.particles.part_track")  # or a .trk file
    t, tags, mass = sink_mass_history(dumps)     # (ndump,), (nsink,), (ndump, nsink)

`sink_mass_history` keys particles by TAG across dumps, so a sink keeps its identity
(and therefore its colour in a plot) as others form, accrete and merge. Slots are NaN
before a sink exists and after it is absorbed by a merger.
"""
import re
import numpy as np

from read_particle_track import FILE_HEADER, read_particle_track

_HDR = re.compile(
    r"time=([\-0-9.eE+]+)\s+cycle=(\d+)\s+nranks=(\d+)\s+nparticles=(\d+)"
    r"\s+nidata=(\d+)\s+nrdata=(\d+)\s+nrec=(\d+)")


def _is_particle_track(path):
    with open(path, "rb") as f:
        head = f.read(FILE_HEADER.size)
    return len(head) == FILE_HEADER.size and head.startswith(b"ATHENAK_PARTICLE_TRACK")


def _particle_track_dumps(path):
    """particle_track records regrouped into one dump per output cycle"""
    t = read_particle_track(path)
    names = ["tag"] + [k for k in t if k not in ("time", "cycle", "ptag")]
    cols = [t["ptag"]] + [t[k] for k in names[1:]]
    data = np.column_stack([np.asarray(c, dtype=np.float64) for c in cols])
    dumps = []
    # blocks are appended in output order; a new dump starts where the cycle changes or
    # a tag repeats (the final output can write the last cycle a second time)
    starts, seen = [0], set()
    for i in range(len(t["cycle"])):
        if i > 0 and (t["cycle"][i] != t["cycle"][i-1] or int(t["ptag"][i]) in seen):
            starts.append(i)
            seen = set()
        seen.add(int(t["ptag"][i]))
    starts.append(len(t["cycle"]))
    for a, b in zip(starts[:-1], starts[1:]):
        if b > a:
            dumps.append(dict(time=float(t["time"][a]), cycle=int(t["cycle"][a]),
                              columns=names, data=data[a:b]))
    return dumps


def read_trk(path):
    """Return a list of dumps, each {time, cycle, columns, data (npart, nrec)}."""
    if _is_particle_track(path):
        return _particle_track_dumps(path)
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
        t, cyc = float(m.group(1)), int(m.group(2))
        npart, nrec = int(m.group(4)), int(m.group(7))
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
