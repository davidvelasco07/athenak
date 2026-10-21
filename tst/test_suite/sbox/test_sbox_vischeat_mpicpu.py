"""
Shearing-box viscous-heating exact gate.

A uniform ideal gas at rest in the shearing frame (v' = 0) with constant kinematic
viscosity nu. The viscous stress of the background shear, sigma_xy = -rho nu q Omega, is
uniform, so every flux divergence vanishes and the momentum must stay exactly zero, while
the viscous work of the background shear heats the gas at exactly
    dE/dt = rho nu (q Omega)^2 * Volume      (= (9/4) rho nu Omega^2 V for q = 3/2).
Both RK2 and RK3 integrate a constant source exactly, so the only error is the round-off
of the RK stage averaging (~1e-13 of E after ~1e3 steps, i.e. ~1e-10 of dE); the gate is
set at 1e-9 of dE, nine orders below what a missing heating term gives (1.0).
Runs PLM/rk2, WENO-Z/rk3 and the fallback scheme (fv4 + MOOD / rk3) on 4 ranks.
"""

# Modules
import pytest
from subprocess import Popen, PIPE
import test_suite.testutils as testutils
import athena_read
import numpy as np

input_file = "inputs/hydro_sbox_vischeat.athinput"

# (reconstruct, integrator, nghost, mood, extra arguments).  The last two cases force
# MOOD to demote every cell with x < 0 at every revision (mood_debug_demote_x1max), so
# half the faces are rewritten by the revise pass; the gate then also checks that the
# revised faces keep their viscous flux, with the diffusion fluxes added after the
# fallback (mood_candidate_diffusion=false, default) and inside the MOOD candidate (true).
_cases = [
    ("plm", "rk2", 3, False, []),
    ("wenoz", "rk3", 3, False, []),
    ("fv4", "rk3", 5, True, []),
    ("fv4", "rk3", 5, True, ["hydro/mood_debug_demote_x1max=0.0", "time/tlim=0.2",
                             "hydro/mood_candidate_diffusion=true"]),
    ("fv4", "rk3", 5, True, ["hydro/mood_debug_demote_x1max=0.0", "time/tlim=0.2",
                             "hydro/mood_candidate_diffusion=false"]),
]

# problem parameters (must match the input file)
d0, nu, qshear, omega0 = 1.0, 0.01, 1.5, 1.0
volume = 0.5**3
heating_rate = d0*nu*(qshear*omega0)**2*volume
rtol_energy = 1.0e-9


def arguments(recon, integ, nghost, mood, extra):
    """Assemble arguments for run command"""
    args = [
        "job/basename=vischeat",
        "hydro/reconstruct=" + recon,
        "time/integrator=" + integ,
        "mesh/nghost=" + repr(nghost),
    ]
    if mood:
        args += ["hydro/mood=true"]
    return args + extra


def check_gate(data):
    """Momentum exactly zero; total energy grows at exactly the analytic rate."""
    t = data["time"]
    for key in ("1-mom", "2-mom", "3-mom"):
        maxmom = np.abs(data[key]).max()
        if maxmom != 0.0:
            pytest.fail(f"{key} is not exactly zero: {maxmom:g}")
    de = data["tot-E"] - data["tot-E"][0]
    de_exact = heating_rate*(t - t[0])
    err = np.abs(de[1:] - de_exact[1:]).max()/np.abs(de_exact[-1])
    return err


def test_run():
    """Loop over schemes and run the gate."""
    try:
        for recon, integ, nghost, mood, extra in _cases:
            results = testutils.mpi_run(
                input_file, arguments(recon, integ, nghost, mood, extra), threads=4)
            assert results, f"Viscous heating test run failed for {recon}/{integ}."
            data = athena_read.hst("vischeat.hydro.hst")
            err = check_gate(data)
            # delete history files so new data will be read properly
            Popen(["rm vischeat.hydro.hst"], shell=True, stdout=PIPE).communicate()
            if err > rtol_energy:
                pytest.fail(
                    f"Shearing-box viscous heating off the analytic rate for "
                    f"{recon}/{integ} {extra}: rel. error {err:g} "
                    f"threshold: {rtol_energy:g}"
                )
    finally:
        testutils.cleanup()
