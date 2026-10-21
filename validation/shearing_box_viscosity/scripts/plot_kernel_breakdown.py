#!/usr/bin/env python3
"""Per-kernel GPU time breakdown of the viscous MRI3D step from Nsight Systems kernel
summaries (nsys stats --report cuda_gpu_kern_sum --format csv), three schemes.

usage: plot_kernel_breakdown.py RUNDIR OUTDIR GPU
RUNDIR holds nsys_<scheme>/stats*.csv.  Kernels are grouped by their enclosing C++ method.
"""
import sys, os, csv, re, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmasher as cmr

R, OUT, GPU = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
schemes = [("plm_rk2", "PLM/rk2"), ("wenoz_rk3", "WENO-Z/rk3"), ("fv4mood_rk3", "FV4+MOOD/rk3")]


def _strip_balanced(s, open_c, close_c):
    """If s ends with a balanced open..close group, remove it."""
    if not s.endswith(close_c): return s
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == close_c: depth += 1
        elif s[i] == open_c:
            depth -= 1
            if depth == 0: return s[:i]
    return s


def group(name):
    """Enclosing Class::Method of the user lambda inside a Kokkos kernel (nvcc demangling).
    Kernels launched through the par_for helpers carry the helper's own lambda last and the
    user's method earlier, so take the first lambda whose owner is not a helper."""
    helpers = ("par_for", "par_for_outer", "par_for_inner", "ParallelFor", "ParallelReduce")
    owners = []
    for mm in re.finditer(r"::[\[{]lambda", name):
        head = name[:mm.start()].rstrip()
        head = _strip_balanced(head, "(", ")")
        head = _strip_balanced(head, "<", ">")
        m = re.search(r"((?:[A-Za-z_~]\w*::)*[A-Za-z_~]\w*)$", head)
        if m: owners.append(re.sub(r"^(hydro|mhd|parabolic|wellbalance)::", "", m.group(1)))
    for o in owners:
        if o.split("::")[-1] not in helpers and not o.startswith("Kokkos::"):
            return o
    if owners: return owners[0]
    m = re.search(r"Kokkos::Impl::(\w+)", name)
    return m.group(1) if m else name[:30]




data = {}
for s, lab in schemes:
    fs = glob.glob(f"{R}/nsys_{s}/stats*.csv")
    if not fs: continue
    rows = list(csv.DictReader(open(fs[0])))
    g = collections.defaultdict(float)
    for r in rows:
        try: g[group(r["Name"])] += float(r["Total Time (ns)"])*1e-9
        except (KeyError, ValueError): pass
    data[s] = g

allg = collections.Counter()
for g in data.values():
    T = sum(g.values())
    for k, v in g.items(): allg[k] = max(allg[k], v/T)
top = [k for k, _ in allg.most_common(14)]
cols = cmr.take_cmap_colors("cmr.chroma", len(top) + 1, cmap_range=(0.05, 0.95))
fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(data)); width = 0.6
for i, (s, lab) in enumerate([(s, l) for s, l in schemes if s in data]):
    g = data[s]; T = sum(g.values()); bottom = 0.0
    for j, k in enumerate(top):
        v = g.get(k, 0.0)/T*100
        ax.bar(i, v, width, bottom=bottom, color=cols[j], edgecolor="w", lw=0.4, label=k if i == 0 else None)
        if v > 4: ax.text(i, bottom + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=7)
        bottom += v
    rest = 100 - bottom
    ax.bar(i, rest, width, bottom=bottom, color="0.8", edgecolor="w", lw=0.4, label="other" if i == 0 else None)
    ax.text(i, 101, f"{T/20*1e3:.0f} ms / cycle", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([l for s, l in schemes if s in data]); ax.set_ylabel("share of GPU kernel time [%]"); ax.set_ylim(0, 108)
ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
ax.set_title(f"Viscous MRI3D 128×256×64 on one {GPU}: GPU kernel time by method (20 cycles, nsys)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/kernel_breakdown_mri3d_{GPU.lower()}.png", dpi=150)
for s, lab in schemes:
    if s not in data: continue
    g = data[s]; T = sum(g.values())
    print(f"== {lab}: {T/20*1e3:.1f} ms/cycle GPU kernel time")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1])[:10]: print(f"   {100*v/T:5.1f}%  {k}")
