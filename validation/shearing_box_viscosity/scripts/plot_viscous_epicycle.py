#!/usr/bin/env python3
"""Figure for the viscous forced-epicycle gate (T5).

usage: plot_viscous_epicycle.py RUNDIR OUTDIR TESTDIR
RUNDIR holds ppmx_n32, ppmx_n64, fv4_n32, fv4_n64 and ref_ppmx_n32/64 (pre-change
binary),
each with viscepi.hydro.hst.  TESTDIR is tst/test_suite/sbox (linear-theory model).
"""
import sys, os, importlib.util, types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmasher as cmr

rundir, outdir, testdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
# load linear_theory() from the test without its pytest/testutils imports
for name in ("pytest", "test_suite", "test_suite.testutils", "athena_read"):
    sys.modules.setdefault(name, types.ModuleType(name))
spec = importlib.util.spec_from_file_location(
    "t5", os.path.join(testdir, "test_sbox_viscepicycle_mpicpu.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)


def load(name):
    d = np.loadtxt(os.path.join(rundir, name, "viscepi.hydro.hst"))
    return d[:, 0], d[:, 8]   # ideal gas: time dt mass m1 m2 m3 totE KE1 KE2(=8)


cols = cmr.take_cmap_colors("cmr.chroma", 7, cmap_range=(0.12, 0.88))
style = {
    "plm_n32": (cols[1], "--", "PLM/rk2, 32$^2$"),
    "plm_n64": (cols[1], "-", "PLM/rk2, 64$^2$"),
    "wenoz_n32": (cols[3], "--", "WENO-Z/rk3, 32$^2$"),
    "wenoz_n64": (cols[3], "-", "WENO-Z/rk3, 64$^2$"),
    "fv4_n32": (cols[5], "--", "FV4+MOOD/rk3, 32$^2$"),
    "fv4_n64": (cols[5], "-", "FV4+MOOD/rk3, 64$^2$"),
}
ref = {"ref_ppmx_n32": "pre-change code, 32$^2$",
       "ref_ppmx_n64": "pre-change code, 64$^2$"}

t0, _ = load("fv4_n64")
tt = np.linspace(0.0, t0[-1], 401)
key_th = t5.linear_theory(tt)
k = 2.0*np.pi*t5.nwx/t5.lx
F = t5.nu*t5.qshear*t5.omega0*t5.eps*k
key_geo = 0.25*t5.rho0*t5.volume*(F*tt)**2

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tt, key_geo*1e9, color="0.6", ls=":", lw=1.0,
           label=r"geostrophic $\rho_0 V (Ft)^2/4$")
ax[0].plot(tt, key_th*1e9, color="k", lw=1.2,
           label="linear theory (Coriolis+pressure+damping)")
for name, (c, ls, lab) in style.items():
    t, ke = load(name)
    ax[0].plot(t, ke*1e9, color=c, ls=ls, lw=1.4, label=lab)
for i, (name, lab) in enumerate(ref.items()):
    if os.path.isdir(os.path.join(rundir, name)):
        t, ke = load(name)
        ax[0].plot(t, ke*1e9, color="0.3", ls=("--", "-")[i], lw=1.0, marker="x", ms=3,
                   markevery=10, label=lab)
ax[0].set_xlabel(r"$t\,\Omega$"); ax[0].set_ylabel(r"$KE_y\ [10^{-9}]$")
ax[0].set_title("Viscous forced epicycle: $y$ kinetic energy")
ax[0].legend(fontsize=8, frameon=False)

for name, (c, ls, lab) in style.items():
    t, ke = load(name)
    th = t5.linear_theory(t)
    sel = t > 0
    ax[1].semilogy(t[sel], np.abs(ke[sel] - th[sel])/th[sel], color=c, ls=ls, lw=1.4,
                   label=lab)
ax[1].axhline(0.03, color="0.5", lw=0.8, ls=":")
ax[1].set_xlabel(r"$t\,\Omega$"); ax[1].set_ylabel(r"$|KE_y - $theory$|/$theory")
ax[1].set_title("Relative error vs linear theory (dotted: 3% gate)")
ax[1].set_ylim(1e-5, 10)
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "viscous_epicycle_key.png"), dpi=160)

print(f"{'run':14s} {'max rel err (t>=0.5)':>22s} {'KE_y(t_end)':>14s}")
for name in list(style) + [r for r in ref if os.path.isdir(os.path.join(rundir, r))]:
    t, ke = load(name)
    th = t5.linear_theory(t)
    sel = t >= 0.5
    err = (np.abs(ke[sel]-th[sel])/th[sel]).max()
    print(f"{name:14s} {err:22.4e} {ke[-1]:14.4e}")
print(f"{'theory':14s} {'':>22s} {th[-1]:14.4e}")
