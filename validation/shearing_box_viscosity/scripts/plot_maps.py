#!/usr/bin/env python3
"""Solution maps for the viscous shearing-box tests and the viscous Shu-Osher comparison.

usage: plot_maps.py MAPDIR G5DIR OUTDIR
MAPDIR: runs made by run_maps.sh (tab slices at x3=0); G5DIR: Shu-Osher runs with tab output.
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmasher as cmr

M, G5, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
cols = cmr.take_cmap_colors("cmr.chroma", 7, cmap_range=(0.12, 0.88))
C = {"plm": cols[1], "wenoz": cols[3], "fv4": cols[5]}
L = {"plm": "PLM/rk2", "wenoz": "WENO-Z/rk3", "fv4": "FV4+MOOD/rk3"}


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "vis", "python"))
import bin_convert  # noqa: E402


def read_bin(fn):
    """Return time and dict of stitched 2D arrays (ny, nx) from a 2D slice .bin dump."""
    fd = bin_convert.read_binary(fn)
    nx1, nx2 = fd["Nx1"], fd["Nx2"]
    x1 = fd["x1min"] + (np.arange(nx1) + 0.5)*(fd["x1max"] - fd["x1min"])/nx1
    x2 = fd["x2min"] + (np.arange(nx2) + 0.5)*(fd["x2max"] - fd["x2min"])/nx2
    out = {"x1": x1, "x2": x2}
    dx1 = (fd["x1max"] - fd["x1min"])/nx1; dx2 = (fd["x2max"] - fd["x2min"])/nx2
    for n in fd["var_names"]:
        a = np.full((nx2, nx1), np.nan)
        for m in range(fd["n_mbs"]):
            blk = np.asarray(fd["mb_data"][n][m])   # [nx3_out, nx2_out, nx1_out]
            if blk.shape[0] != 1:
                blk = blk[blk.shape[0]//2:blk.shape[0]//2 + 1]
            g = fd["mb_geometry"][m]                # block x1min,x1max,x2min,x2max,x3min,x3max
            i0 = int(round((g[0] - fd["x1min"])/dx1)); j0 = int(round((g[2] - fd["x2min"])/dx2))
            a[j0:j0 + blk.shape[1], i0:i0 + blk.shape[2]] = blk[0]
        out[n] = a
    return float(fd["time"]), out


def dumps(run):
    fs = sorted(glob.glob(f"{M}/{run}/bin/*.bin"))
    return [f for f in fs if ".rank_" not in f] or fs


def nearest(run, t_target):
    best = None
    for fn in dumps(run):
        t, _ = read_bin(fn)
        if best is None or abs(t - t_target) < abs(best[0] - t_target):
            best = (t, fn)
    return best[0], read_bin(best[1])[1]


# ---------------------------------------------------------------- 1. shwave maps
times = [0.0, 1.3, 2.6, 3.9]
runs = ["plm", "wenoz", "fv4"]
fig, ax = plt.subplots(len(runs) + 1, len(times), figsize=(3.0*len(times), 2.8*(len(runs) + 1)),
                       squeeze=False)
vmax = 1.0e-4*6.0
for r, run in enumerate(runs + ["wenoz_inv"]):
    rn = f"shw_{run}" if run != "wenoz_inv" else "shw_wenoz_inv"
    for c, tt in enumerate(times):
        t, d = nearest(rn, tt)
        a = ax[r, c]
        im = a.pcolormesh(d["x1"], d["x2"], d["velx"], cmap="cmr.chroma", vmin=-vmax, vmax=vmax,
                          shading="nearest")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(f"t = {t:.2f}", fontsize=10)
        if c == 0: a.set_ylabel(L[run] if run in L else "WENO-Z, inviscid", fontsize=9)
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6, label=r"$\delta v_x$")
fig.suptitle(r"Viscous vortical shwave, 128$^2$, $\nu=10^{-4}$: $\delta v_x(x,y)$ (last row inviscid)", fontsize=11)
fig.savefig(f"{OUT}/maps_shwave_vx.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. forced epicycle maps
fig, ax = plt.subplots(2, 4, figsize=(12.5, 5.6))
for c, run in enumerate(["plm", "wenoz", "fv4", "ref"]):
    t, d = nearest(f"epi_{run}", 2.0)
    im0 = ax[0, c].pcolormesh(d["x1"], d["x2"], d["dens"], cmap="cmr.chroma", vmin=0.9, vmax=1.1, shading="nearest")
    vmax2 = 4.0e-4
    im1 = ax[1, c].pcolormesh(d["x1"], d["x2"], d["vely"], cmap="cmr.chroma", vmin=-vmax2, vmax=vmax2, shading="nearest")
    for a in ax[:, c]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    ax[0, c].set_title((L[run] if run in L else "pre-change code, PLM/rk2") + f", t = {t:.1f}", fontsize=9)
ax[0, 0].set_ylabel(r"$\rho$"); ax[1, 0].set_ylabel(r"$v'_y$")
fig.colorbar(im0, ax=ax[0, :].tolist(), shrink=0.8); fig.colorbar(im1, ax=ax[1, :].tolist(), shrink=0.8)
fig.suptitle(r"Viscous forced epicycle, 64$^2$: density wave and the induced $v'_y = F t \cos kx$ (pre-change code: none)", fontsize=11)
fig.savefig(f"{OUT}/maps_epicycle.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. momentum kick map
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
for c, run in enumerate(["kick_phase1", "kick_final"]):
    t, d = nearest(run, 0.2)
    m2 = d["dens"]*d["vely"]
    vm = max(np.nanmax(np.abs(m2)), 1e-30)
    im = ax[c].pcolormesh(d["x1"], d["x2"], m2, cmap="cmr.chroma", vmin=-vm, vmax=vm, shading="nearest")
    ax[c].set_aspect("equal"); ax[c].axvline(0.0, color="w", lw=0.8, ls="--")
    ax[c].set_title(("Phase 1 ordering" if run == "kick_phase1" else "final code") + f": max|ρv'_y| = {vm:.1e}", fontsize=10)
    fig.colorbar(im, ax=ax[c], shrink=0.8)
fig.suptitle("Forced demotion of x < 0 (dashed line) on a uniform viscous box, t = 0.2: y-momentum", fontsize=11)
fig.savefig(f"{OUT}/maps_kick.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4. Shu-Osher profiles
def last_tab(run):
    fs = sorted(glob.glob(f"{run}/tab/*.tab")); return fs[-1]
def prof(fn):
    with open(fn) as f:
        head = f.readline(); names = f.readline().split()[1:]   # gid i x1v dens velx ...
    d = np.loadtxt(fn); t = float(re.search(r"time=(\S+)", head).group(1))
    return t, {n: d[:, names.index(n)] for n in names}
cases = [(f"{G5}/shu_nu0.0_true", "inviscid (ν = 0)", "0.5"),
         (f"{G5}/shu_nu0.01_cdtrue_nadEtrue", "ν = 0.01, full candidate", cols[5]),
         (f"{G5}/shu_nu0.01_cdfalse_nadEtrue", "ν = 0.01, inviscid candidate", cols[3]),
         (f"{M}/shu_nodetect", "ν = 0.01, no detection", cols[1])]
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for run, lab, c in cases:
    if not os.path.isdir(run): continue
    t, p = prof(last_tab(run)); x = p["x1v"]; pres = p["eint"]*p["dens"]*(1.4 - 1.0) if "eint" in p else None
    for a in (ax[0, 0], ax[0, 1]): a.plot(x, p["dens"], color=c, lw=1.0, label=f"{lab} (t = {t:.3f})")
    ax[1, 0].plot(x, p["velx"], color=c, lw=1.0)
    if pres is not None: ax[1, 1].plot(x, pres, color=c, lw=1.0)
t_, p_ = prof(last_tab(cases[1][0])); xs = p_["x1v"][np.argmax(np.abs(np.gradient(p_["dens"])))]
ax[0, 0].set_ylabel("density"); ax[0, 1].set_xlim(xs - 0.55, xs + 0.12); ax[0, 1].set_title("zoom: shock and post-shock oscillations")
ax[1, 0].set_ylabel("velocity"); ax[1, 1].set_ylabel("pressure (γ = 1.4)")
for a in ax.ravel(): a.set_xlabel("x")
ax[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Viscous Shu–Osher, FV4+MOOD, 400 cells: final profiles", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/profiles_shu_osher.png", dpi=150)
plt.close(fig)
print("figures written to", OUT)

# ---------------------------------------------------------------- 5. Shu-Osher density: viscosity and time step
G8 = os.path.join(os.path.dirname(G5.rstrip("/")), "g8")
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
panels = [("ν = 0.003, dt = min rule", f"{G5}/shu_nu0.003_cdtrue_nadEtrue", f"{G5}/shu_nu0.003_cdfalse_nadEtrue"),
          ("ν = 0.01, dt = min rule (energy floors fire)", f"{G5}/shu_nu0.01_cdtrue_nadEtrue", f"{G5}/shu_nu0.01_cdfalse_nadEtrue"),
          ("ν = 0.01, dt = harmonic rule", f"{G8}/h_nu0.01_cdtrue", f"{G8}/h_nu0.01_cdfalse"),
          ("ν = 0.03, dt = min rule", f"{G5}/shu_nu0.03_true", f"{G5}/shu_nu0.03_false")]
t0, p0 = prof(last_tab(f"{G5}/shu_nu0.0_true"))
for a, (title, full, inv) in zip(ax, panels):
    a.plot(p0["x1v"], p0["dens"], color="0.6", lw=1.0, label="inviscid, FV4+MOOD")
    for run, lab, c, ls in ((full, "full candidate", cols[5], "-"), (inv, "inviscid candidate", cols[3], "--")):
        if os.path.isdir(run):
            t, p = prof(last_tab(run)); a.plot(p["x1v"], p["dens"], color=c, lw=1.1, ls=ls, label=lab)
    a.set_title(title, fontsize=10); a.set_xlabel("x"); a.set_xlim(0.3, 1.0)
ax[0].set_ylabel("density"); ax[0].legend(fontsize=8, frameon=False)
fig.suptitle("Viscous Shu–Osher, FV4+MOOD, 400 cells, CFL 0.8, t = 0.47: candidate modes agree whenever no floors fire; at ν = 0.01 the min time-step rule lets demoted shock faces drive e < 0", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/profiles_shu_osher_nu.png", dpi=150)
plt.close(fig)
print("nu panel written")
