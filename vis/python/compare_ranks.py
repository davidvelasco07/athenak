#!/usr/bin/env python3
"""Compare two AthenaK runs of the same problem that used different rank counts.

Usage: compare_ranks.py <runA> <runB> [var]

Blocks are matched between the runs by their LOGICAL LOCATION (level, lx1, lx2, lx3), not by
array order and not by resampling onto a uniform grid -- resampling is lossy and previously
produced a false "the fields differ" conclusion. Reports, per dump:
  * block-count and mesh-identity (a rank-count-dependent AMR criterion shows up here first,
    as differing block counts, BEFORE any field difference)
  * max |A - B| over the common blocks, absolute and relative to |field|max
Reads per-rank output (bin/rank_*/) as well as single-file output (bin/).
"""
import sys, glob, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bin_convert sits alongside
import bin_convert


def by_lloc(run_dir, idx, var):
    """{(lev,lx1,lx2,lx3): 3D block array} merged over every rank file for one dump."""
    out, t = {}, None
    dirs = sorted(glob.glob(os.path.join(run_dir, "bin", "rank_*"))) or \
           [os.path.join(run_dir, "bin")]
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"*.{var}.{idx}.bin")):
            fd = bin_convert.read_binary(f)
            v = fd["var_names"][0]
            t = fd["time"]
            for m in range(fd["n_mbs"]):
                ll = fd["mb_logical"][m]
                out[(int(ll[3]), int(ll[0]), int(ll[1]), int(ll[2]))] = fd["mb_data"][v][m]
    return t, out


def dumps(run_dir, var):
    dirs = sorted(glob.glob(os.path.join(run_dir, "bin", "rank_*"))) or \
           [os.path.join(run_dir, "bin")]
    idx = set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"*.{var}.*.bin")):
            idx.add(os.path.basename(f).split(".")[-2])
    return sorted(idx)


def main():
    A, B = sys.argv[1], sys.argv[2]
    var = sys.argv[3] if len(sys.argv) > 3 else "grav_phi"
    ia, ib = dumps(A, var), dumps(B, var)
    common_idx = [i for i in ia if i in ib]
    print(f"A={A}\nB={B}\nvar={var}   dumps: A={len(ia)} B={len(ib)} common={len(common_idx)}")
    worst_overall = 0.0
    mesh_mismatch = []
    for idx in common_idx:
        ta, ba = by_lloc(A, idx, var)
        tb, bb = by_lloc(B, idx, var)
        if ta is None or tb is None:
            continue
        ka, kb = set(ba), set(bb)
        common = ka & kb
        if not common:
            continue
        md = max(float(np.nanmax(np.abs(ba[k] - bb[k]))) for k in common)
        amp = float(np.nanmax(np.abs(np.concatenate([ba[k].ravel() for k in common]))))
        rel = md / amp if amp > 0 else 0.0
        worst_overall = max(worst_overall, rel)
        same = "MESH-OK " if (ka == kb) else "MESH-DIFF"
        if ka != kb:
            mesh_mismatch.append(idx)
        print(f"  dump{idx} t={ta:.5f}/{tb:.5f} {same} blocks A={len(ka)} B={len(kb)} "
              f"common={len(common)} onlyA={len(ka-kb)} onlyB={len(kb-ka)} | "
              f"max|dA-dB|={md:.4e} (|f|max={amp:.4f}, rel={rel:.3e})")
    print(f"\nWORST relative field difference over all dumps: {worst_overall:.3e}")
    if mesh_mismatch:
        print(f"MESH DIFFERED at dumps: {mesh_mismatch}  <-- rank-count-dependent refinement")
    else:
        print("Meshes identical at every compared dump.")
    if worst_overall == 0.0 and not mesh_mismatch:
        print("=> runs are BIT-IDENTICAL")


if __name__ == "__main__":
    main()
