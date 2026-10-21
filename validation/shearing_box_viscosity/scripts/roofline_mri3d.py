#!/usr/bin/env python3
"""Roofline of the viscous MRI3D step on one GH200 from Nsight Compute per-kernel metrics.

usage: roofline_mri3d.py RUNDIR OUTDIR
RUNDIR holds ncu_<scheme>/ncu.csv (ncu --csv --metrics ...) for the three schemes.
FLOP = dadd + dmul + 2 dfma (+ single-precision analogues), bytes = DRAM read + write,
time = gpu__time_duration.  Points: kernels grouped by the enclosing C++ method; marker
area = share of the step time; star = whole-step aggregate.
"""
import sys, os, csv, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmasher as cmr

R, OUT = sys.argv[1:3]
GPU = sys.argv[3] if len(sys.argv) > 3 else "A100"
os.makedirs(OUT, exist_ok=True)
# roofs (vendor peak numbers, stated on the figure)
ROOFS = {"A100": (1.555e12, 9.7e12, "A100 PCIe 40 GB: 1.56 TB/s HBM2e, 9.7 TFLOP/s FP64"),
         "GH200": (4.0e12, 34.0e12, "GH200: 4.0 TB/s HBM3, 34 TFLOP/s FP64")}
BW, PEAK_FP64, ROOF_LABEL = ROOFS[GPU]
schemes = [("plm_rk2", "PLM/rk2"), ("wenoz_rk3", "WENO-Z/rk3"), ("fv4mood_rk3", "FV4+MOOD/rk3")]
cols = cmr.take_cmap_colors("cmr.chroma", 7, cmap_range=(0.12, 0.88))
C = {"plm_rk2": cols[1], "wenoz_rk3": cols[3], "fv4mood_rk3": cols[5]}


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


def kernel_group(name):
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




def load(scheme):
    fn = f"{R}/ncu_{scheme}/ncu.csv"
    rows = [r for r in csv.reader(open(fn)) if r]
    # header row is the first row that starts with "ID"
    h = next(i for i, r in enumerate(rows) if r[0] == "ID")
    head = rows[h]; data = rows[h+1:]
    ci = {k: head.index(k) for k in ("ID", "Kernel Name", "Metric Name", "Metric Value")}
    per = collections.defaultdict(dict)
    for r in data:
        if len(r) <= max(ci.values()): continue
        try: v = float(r[ci["Metric Value"]].replace(",", ""))
        except ValueError: continue
        per[(r[ci["ID"]], r[ci["Kernel Name"]])][r[ci["Metric Name"]]] = v
    groups = collections.defaultdict(lambda: {"t": 0.0, "b": 0.0, "f": 0.0, "n": 0})
    for (kid, name), m in per.items():
        g = groups[kernel_group(name)]
        g["t"] += m.get("gpu__time_duration.sum", 0.0)*1e-9
        g["b"] += m.get("dram__bytes_read.sum", 0.0) + m.get("dram__bytes_write.sum", 0.0)
        f = (m.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", 0.0)
             + m.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", 0.0)
             + 2.0*m.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 0.0)
             + m.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)
             + m.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
             + 2.0*m.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0))
        g["f"] += f; g["n"] += 1
    return groups


fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharey=True)
ai = np.logspace(-2, 2.5, 200)
summary = {}
for ax, (s, lab) in zip(axes[:3], schemes):
    groups = load(s)
    T = sum(g["t"] for g in groups.values()); B = sum(g["b"] for g in groups.values()); F = sum(g["f"] for g in groups.values())
    summary[s] = (T, B, F)
    ax.loglog(ai, np.minimum(BW*ai, PEAK_FP64)/1e9, color="0.3", lw=1.2)
    ax.axvline(PEAK_FP64/BW, color="0.7", lw=0.6, ls=":")
    top = sorted(groups.items(), key=lambda kv: -kv[1]["t"])
    for name, g in top:
        if g["b"] <= 0 or g["t"] <= 0: continue
        x = g["f"]/g["b"]; y = g["f"]/g["t"]/1e9
        ax.scatter(x, y, s=20 + 1500*g["t"]/T, color=C[s], alpha=0.65, edgecolor="k", lw=0.4)
    for name, g in top[:7]:
        if g["b"] <= 0 or g["t"] <= 0: continue
        ax.annotate(f"{name.split('::')[-1]} {100*g['t']/T:.0f}%", (g["f"]/g["b"], g["f"]/g["t"]/1e9),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.scatter(F/B, F/T/1e9, marker="*", s=260, color=C[s], edgecolor="k", lw=0.8, zorder=5)
    ax.set_title(f"{lab}: step AI = {F/B:.2f} FLOP/B, {F/T/1e12:.2f} TFLOP/s, {B/T/1e12:.2f} TB/s", fontsize=9)
    ax.set_xlabel("arithmetic intensity [FLOP / DRAM byte]"); ax.grid(True, which="both", lw=0.3, alpha=0.4)
axes[0].set_ylabel("GFLOP/s (FP64 + FP32)")
ax = axes[3]
ax.loglog(ai, np.minimum(BW*ai, PEAK_FP64)/1e9, color="0.3", lw=1.2, label=ROOF_LABEL)
for s, lab in schemes:
    T, B, F = summary[s]
    ax.scatter(F/B, F/T/1e9, marker="*", s=300, color=C[s], edgecolor="k", lw=0.8, label=f"{lab} whole step")
ax.set_xlabel("arithmetic intensity [FLOP / DRAM byte]"); ax.set_title("whole-step aggregates", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(True, which="both", lw=0.3, alpha=0.4)
for a in axes: a.set_xlim(1e-2, 3e2); a.set_ylim(1e1, 1e5)
fig.suptitle(f"Viscous MRI3D 128×256×64 on one {GPU}: per-kernel roofline (marker area = share of profiled time; star = whole step)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/roofline_mri3d_{GPU.lower()}.png", dpi=150)
print(f"{'scheme':14s} {'time/step [s]':>14s} {'DRAM TB/s':>10s} {'TFLOP/s':>9s} {'AI':>6s}")
for s, lab in schemes:
    T, B, F = summary[s]; print(f"{s:14s} {T/3:14.4f} {B/T/1e12:10.2f} {F/T/1e12:9.2f} {F/B:6.2f}")
