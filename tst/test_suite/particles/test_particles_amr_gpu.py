"""Particle ownership across AMR regrids with a moving refined region."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


def _check_particle_amr(basename, mpi=False):
    history = Path(f"{basename}.user.hst")
    try:
        history.unlink(missing_ok=True)
        flags = [f"job/basename={basename}"]
        if mpi:
            ran = testutils.mpi_run("inputs/particle_amr.athinput", flags, threads=2)
        else:
            ran = testutils.run("inputs/particle_amr.athinput", flags)
        assert ran, "particle AMR run failed"

        if not history.exists():
            pytest.fail(f"particle AMR history was not written: {history}")
        data = athena_read.hst(str(history))

        # 64 initial MeshBlocks with eight particles each, tags 0..511
        np.testing.assert_array_equal(data["npart"], 512.0)
        np.testing.assert_array_equal(data["tag_sum"], 511.0*512.0/2.0)
        np.testing.assert_array_equal(data["owner_err"], 0.0)
        # the check is only meaningful if the mesh really changed, more than once
        if data["mesh_seq"][-1] < 5:
            pytest.fail(f"too few mesh updates: {data['mesh_seq'][-1]:g}")
        if np.max(data["nmb"]) <= data["nmb"][0]:
            pytest.fail("the mesh never refined")
    finally:
        history.unlink(missing_ok=True)


def test_particle_amr_gpu():
    """Particles keep correct owners while blocks refine, derefine and renumber."""
    _check_particle_amr("particle_amr_gpu")
