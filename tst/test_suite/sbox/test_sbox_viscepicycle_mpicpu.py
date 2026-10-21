"""
Shearing-box viscous forced epicycle: a gate for the background-shear term of the stress.

Hydro and MHD store the velocity perturbation v' = v - v_K, so the viscous stress must
carry the background shear explicitly: sigma_xy = rho nu (dv'_y/dx + dv'_x/dy - q Omega).
On a uniform-pressure state with rho = rho0 (1 + eps sin kx) and v' = 0 (an exact
hydrodynamic steady state), the -q Omega part of the stress has a divergence and forces
    dv'_y/dt = F(x) = -nu q Omega d(ln rho)/dx ~ -nu q Omega eps k cos kx.
For small eps the linear response of the Fourier amplitudes (v'_x, v'_y, drho/rho0) =
(a, b, d) x (cos kx, cos kx, sin kx) obeys
    da/dt = 2 Omega b - cs^2 k d - (4/3) nu k^2 a
    db/dt = -(2-q) Omega a + F - nu k^2 b
    dd/dt = k a
With cs k >> Omega (here cs k ~ 16 Omega) the response is a secular v_y = F t in
geostrophic balance with a small acoustic-epicyclic oscillation at omega^2 = kappa^2 +
cs^2 k^2, not a free epicycle.  The test integrates this system and compares
KE_y(t) = (1/4) rho0 V b(t)^2 from the history output.  Corrections are O(eps^2) from the
non-uniform background and O(v'/cs) nonlinearity, so eps = 0.1 and a 3% tolerance.
Without the -q Omega term F = 0 and KE_y stays at round-off: this is the negative control
(the pre-change code passes the heating gate's companion test, the viscous shwave, because
at uniform density the term is a constant flux with zero divergence).
Runs PLM/rk2, WENO-Z/rk3 and fv4+MOOD/rk3 at 32^2 and 64^2 on 4 ranks.
"""

# Modules
import pytest
from subprocess import Popen, PIPE
import test_suite.testutils as testutils
import athena_read
import numpy as np

input_file = "inputs/hydro_sbox_viscepicycle.athinput"
_res = [32, 64]

# (reconstruct, integrator, nghost, mood): max relative error of KE_y(t) over t >= 0.5
_cases = {
    ("plm", "rk2", 3, False): 0.03,
    ("wenoz", "rk3", 3, False): 0.03,
    ("fv4", "rk3", 5, True): 0.03,
}

# problem parameters (must match the input file)
rho0, p0, gamma = 1.0, 1.0, 5.0/3.0
eps, nwx, lx = 0.1, 1, 0.5
nu, qshear, omega0 = 1.0e-4, 1.5, 1.0
volume = 0.5**3


def linear_theory(t_out):
    """RK4 integration of the linear response; returns KE_y(t_out)."""
    k = 2.0*np.pi*nwx/lx
    cs2 = gamma*p0/rho0
    F = nu*qshear*omega0*eps*k
    nk2 = nu*k*k

    def rhs(y):
        a, b, d = y
        return np.array([2.0*omega0*b - cs2*k*d - (4.0/3.0)*nk2*a,
                         -(2.0 - qshear)*omega0*a + F - nk2*b,
                         k*a])

    dt = 1.0e-4
    y = np.zeros(3)
    t = 0.0
    key = np.zeros_like(t_out)
    for n, tn in enumerate(t_out):
        while t < tn - 0.5*dt:
            k1 = rhs(y)
            k2 = rhs(y + 0.5*dt*k1)
            k3 = rhs(y + 0.5*dt*k2)
            k4 = rhs(y + dt*k3)
            y = y + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
            t += dt
        key[n] = 0.25*rho0*volume*y[1]**2
    return key


def compute_error(data):
    t = data["time"]
    key = linear_theory(t)
    sel = t >= 0.5
    return (np.abs(data["2-KE"][sel] - key[sel])/key[sel]).max()


def arguments(res, recon, integ, nghost, mood):
    """Assemble arguments for run command"""
    return [
        "job/basename=viscepi",
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=" + repr(res // 2),
        "mesh/nx2=" + repr(res),
        "meshblock/nx2=" + repr(res // 2),
        "mesh/nghost=" + repr(nghost),
        "hydro/reconstruct=" + recon,
        "time/integrator=" + integ,
        "hydro/mood=" + ("true" if mood else "false"),
    ]


def test_run():
    """Loop over schemes and resolutions; compare KE_y(t) to linear theory."""
    try:
        for (recon, integ, nghost, mood), maxerr in _cases.items():
            for res in _res:
                results = testutils.mpi_run(
                    input_file, arguments(res, recon, integ, nghost, mood), threads=4)
                assert results, f"Viscous epicycle run failed for {recon} at {res}."
                data = athena_read.hst("viscepi.hydro.hst")
                err = compute_error(data)
                # delete history files so new data will be read properly
                Popen(["rm viscepi.hydro.hst"], shell=True, stdout=PIPE).communicate()
                if err > maxerr:
                    pytest.fail(
                        f"Viscous forced epicycle ({recon}/{integ}, {res}^2) KE_y off "
                        f"linear theory, max rel. error: {err:g} threshold: {maxerr:g}"
                    )
    finally:
        testutils.cleanup()
