#!/usr/bin/env python3
"""Figures for the viscous vortical shwave test (T2).

usage: plot_viscous_shwave.py RUNDIR OUTDIR
RUNDIR holds one subdirectory per run (ppmx_n32_visc, ..., fv4_n64_inv), each with
shwave.hydro.hst.  Produces amplitude-vs-time and error-vs-time figures.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmasher as cmr

rundir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
nu, amp, q, om = 1.0e-4, 1.0e-4, 1.5, 1.0
ky = 2.0*np.pi*2.0/0.5


def inviscid(t):
    s = q*om*t - 4.0
    return amp*17.0/(1.0 + s*s)


def viscous(t):
    s = q*om*t - 4.0
    return inviscid(t)*np.exp(-nu*ky*ky*(t + (s**3 + 64.0)/(3.0*q*om)))


def load(name):
    d = np.loadtxt(os.path.join(rundir, name, "shwave.hydro.hst"))
    return d[:, 0], np.sqrt(32.0*d[:, 6])   # cols: time dt mass m1 m2 m3 KE1(=6) (isothermal)


cols = cmr.take_cmap_colors("cmr.chroma", 7, cmap_range=(0.12, 0.88))
# scheme order: PLM, WENO-Z, then FV4 (warm end of the ramp)
style = {
    "plm_n64_visc": (cols[1], "--", "PLM/rk2, 64$^2$"),
    "plm_n128_visc": (cols[1], "-", "PLM/rk2, 128$^2$"),
    "wenoz_n64_visc": (cols[3], "--", "WENO-Z/rk3, 64$^2$"),
    "wenoz_n128_visc": (cols[3], "-", "WENO-Z/rk3, 128$^2$"),
    "fv4_n64_visc": (cols[5], "--", "FV4+MOOD/rk3, 64$^2$"),
    "fv4_n128_visc": (cols[5], "-", "FV4+MOOD/rk3, 128$^2$"),
}
nodetect = {"fv4_n64_visc_nodetect": (cols[5], "--", "FV4 no detection, 64$^2$"),
            "fv4_n128_visc_nodetect": (cols[5], "-", "FV4 no detection, 128$^2$")}
inv = {"plm_n128_inv": (cols[1], "PLM/rk2, 128$^2$, inviscid run"),
       "wenoz_n128_inv": (cols[3], "WENO-Z/rk3, 128$^2$, inviscid run"),
       "fv4_n128_inv": (cols[5], "FV4+MOOD/rk3, 128$^2$, inviscid run")}
tt = np.linspace(0, 5, 501)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tt, inviscid(tt)/amp, color="0.6", lw=1.0, ls=":", label="analytic, inviscid")
ax[0].plot(tt, viscous(tt)/amp, color="k", lw=1.2, label=r"analytic, $\nu=10^{-4}$")
for name, (c, ls, lab) in style.items():
    t, a = load(name)
    ax[0].plot(t, a/amp, color=c, ls=ls, lw=1.4, label=lab)
for name, (c, ls, lab) in nodetect.items():
    if os.path.isdir(os.path.join(rundir, name)):
        t, a = load(name)
        ax[0].plot(t, a/amp, color=c, ls=ls, lw=0.8, alpha=0.6, marker="x", ms=3,
                   markevery=4, label=lab)
for name, (c, lab) in inv.items():
    t, a = load(name)
    ax[0].plot(t[::5], a[::5]/amp, color=c, ls="none", marker="o", ms=3, mfc="none",
               label=lab)
ax[0].set_xlabel(r"$t\,\Omega$"); ax[0].set_ylabel(r"$\delta v_x / $amp")
ax[0].set_title("Vortical shwave amplitude")
ax[0].legend(fontsize=8, frameon=False)

for name, (c, ls, lab) in style.items():
    t, a = load(name)
    ax[1].semilogy(t, np.abs(a - viscous(t))/amp, color=c, ls=ls, lw=1.4, label=lab)
for name, (c, ls, lab) in nodetect.items():
    if os.path.isdir(os.path.join(rundir, name)):
        t, a = load(name)
        ax[1].semilogy(t, np.abs(a - viscous(t))/amp, color=c, ls=ls, lw=0.8, alpha=0.6,
                       marker="x", ms=3, markevery=4, label=lab)
for name, (c, lab) in inv.items():
    t, a = load(name)
    ax[1].semilogy(t, np.abs(a - inviscid(t))/amp, color=c, ls="none", marker="o",
                   ms=2.5, mfc="none", label=lab + " vs inviscid")
ax[1].set_xlabel(r"$t\,\Omega$"); ax[1].set_ylabel(r"$|\delta v_x - $analytic$|/$amp"); ax[1].set_ylim(1e-6, 30)
ax[1].set_title("Error vs analytic solution")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "viscous_shwave_amplitude_error.png"), dpi=160)

# mean L1 errors and convergence
print(f"{'run':18s} {'mean|err|/amp':>14s}")
for name in list(style) + [n for n in nodetect if os.path.isdir(os.path.join(rundir, n))]:
    t, a = load(name)
    print(f"{name:24s} {np.abs(a - viscous(t)).mean()/amp:14.4e}")
for name in inv:
    t, a = load(name)
    print(f"{name:24s} {np.abs(a - inviscid(t)).mean()/amp:14.4e}  (vs inviscid)")
