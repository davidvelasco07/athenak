"""
Viscous hydro shearing wave (shwave) test for the shearing box + viscosity.

Runs the incompressible (vortical) shwave of JG05 with a constant kinematic viscosity nu.
A single shearing-wave mode stays a single mode under viscosity, with k(t)^2 = kx(t)^2 +
ky^2 and kx(t) = kx0 + q Omega ky t, so the exact solution is the inviscid one times
    exp(-nu Phi(t)),   Phi(t) = Int_0^t k(t')^2 dt'.
For the stock parameters (nwx=-8, nwy=2, Lx=Ly=0.5, q=3/2, Omega=1) kx0/ky = -4 and
    dVx(t) = amp 17/(1 + (1.5 t - 4)^2) exp(-nu ky^2 [t + ((1.5 t - 4)^3 + 64)/4.5]).
This checks the viscous stress with the background shear in it (the stored velocity is
the perturbation, so sigma_xy needs the -q Omega term) and, with fv4+MOOD, that revised
faces keep the viscous flux.  Compares dVx from the history KE to the analytic solution
at two resolutions, like the inviscid test, but at 64^2 and 128^2: the initial
x-wavelength is 1/16, i.e. 4 cells at 32^2, where both schemes damp the wave numerically
within t < 0.2.
"""

# Modules
import pytest
from subprocess import Popen, PIPE
import test_suite.testutils as testutils
import athena_read
import numpy as np

_res = [64, 128]  # resolutions to test (32^2 has 4 cells per x-wavelength: unresolved)
input_file = "inputs/hydro_shwave.athinput"
nu = 1.0e-4

# (reconstruct, integrator, nghost, mood): max mean |error| at 64^2 and at 128^2.
# Absolute thresholds, not a convergence ratio: the initial condition is the
# incompressible mode, so with c_s = Omega = 1 the compressible solution differs from it
# at O(Omega/(c_s k)) ~ 2% of amp, a floor both schemes reach by 128^2 (measured
# 0.31 and 0.016 amp for WENO-Z, 0.35 and 0.017 amp for fv4).  A missing or wrong
# viscous stress gives O(1) errors (the inviscid run is off by 0.03-0.06 amp at 128^2
# vs the inviscid solution, but by >1 amp vs the viscous one).
# PLM/rk2 is not gated here: its numerical diffusion dominates nu = 1e-4 at these
# resolutions (mean error 2.0 amp at 64^2, 1.3 amp at 128^2, and 3.9 amp for the inviscid
# wave at 128^2), so it would only test that the run completes.
_cases = {
    ("wenoz", "rk3", 3, False): {64: 5.0e-5, 128: 4.0e-6},
    ("fv4", "rk3", 5, True): {64: 5.0e-5, 128: 4.0e-6},
}


def analytic(t, amp=1.0e-4, q=1.5, omega=1.0, ky=2.0*np.pi*2.0/0.5):
    """dVx(t) of the viscous vortical shwave (see module docstring)."""
    s = q*omega*t - 4.0
    inviscid = amp*17.0/(1.0 + s*s)
    phi = ky*ky*(t + (s**3 + 64.0)/(3.0*q*omega))
    return inviscid*np.exp(-nu*phi)


def compute_error(data):
    t = data["time"]
    return (np.abs(np.sqrt(32.0*data["1-KE"]) - analytic(t))).mean()


def arguments(res, recon, integ, nghost, mood):
    """Assemble arguments for run command"""
    return [
        "job/basename=shwave",
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=" + repr(res // 2),
        "mesh/nx2=" + repr(res),
        "meshblock/nx2=" + repr(res // 2),
        "mesh/nx3=4",
        "meshblock/nx3=4",
        "mesh/nghost=" + repr(nghost),
        "hydro/reconstruct=" + recon,
        "time/integrator=" + integ,
        "hydro/mood=" + ("true" if mood else "false"),
        "hydro/nu_iso=" + repr(nu),
    ]


def test_run():
    """Loop over schemes and resolutions; check convergence and absolute error."""
    try:
        for (recon, integ, nghost, mood), maxerr in _cases.items():
            for res in _res:
                results = testutils.mpi_run(
                    input_file, arguments(res, recon, integ, nghost, mood), threads=4)
                assert results, f"Viscous shwave run failed for {recon} at {res}."
                data = athena_read.hst("shwave.hydro.hst")
                err = compute_error(data)
                # delete history files so new data will be read properly
                Popen(["rm shwave.hydro.hst"], shell=True, stdout=PIPE).communicate()
                if err > maxerr[res]:
                    pytest.fail(
                        f"Viscous shwave ({recon}/{integ}, {res}^2) error too large, "
                        f"error: {err:g} threshold: {maxerr[res]:g}"
                    )
    finally:
        testutils.cleanup()
