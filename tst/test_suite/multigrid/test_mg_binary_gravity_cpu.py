"""
Multigrid binary gravity tests for CPU.

Tests MG defect convergence and L2 accuracy for two compact spheres
on uniform and SMR meshes.

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

import pytest

from test_suite.multigrid.mg_utils import (
    assert_solver_convergence, run_athenak,
    parse_final_defect, assert_defect_consistency, cleanup,
    weighted_threshold,
)
threshold = 1E-9

NCELLS_SMR = 64**3  # SMR test mesh at its effective finest resolution, 64^3

_GRAVITY_FLAGS = [
    "time/nlim=1",
    "gravity/niteration=-1",
    "gravity/npresmooth=2",
    "gravity/npostsmooth=2",
    "gravity/full_multigrid=true",
]

_stored_defects = {}


# ---------------------------------------------------------------------------
# Defect convergence per V-cycle (uniform, SMR)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("rg", [4, 8])
def test_binary_gravity_uniform_cpu(res, rg):
    """Binary gravity MG defect convergence on uniform mesh."""
    try:
        mb = res // rg
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
                f"meshblock/nx1={mb}", f"meshblock/nx2={mb}",
                f"meshblock/nx3={mb}",
                "mesh_refinement/refinement=none",
                "gravity/show_defect=2",
                f"gravity/threshold={weighted_threshold(threshold, res**3)}",
            ] + _GRAVITY_FLAGS,
        )
        assert results[0], "Binary gravity uniform CPU run failed"
        assert_solver_convergence(
            results[1],
            weighted_threshold(threshold, res**3),
            max_iterations=10,
            max_avg_ratio=0.0625,
            label="binary_gravity_uniform_cpu: ",
        )
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["uniform"] = d
    finally:
        cleanup()


@pytest.mark.parametrize("res", [16])
@pytest.mark.parametrize("rg", [4])
def test_binary_gravity_smr_cpu(res, rg):
    """Binary gravity with 2-level SMR (16^3 base, effective finest = 64^3)."""
    try:
        mb = res // rg
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
                f"meshblock/nx1={mb}", f"meshblock/nx2={mb}",
                f"meshblock/nx3={mb}",
                "refined_region3/level=2",
                "refined_region4/level=2",
                "gravity/show_defect=2",
                f"gravity/threshold={weighted_threshold(threshold, NCELLS_SMR)}",
            ] + _GRAVITY_FLAGS,
        )
        assert results[0], "Binary gravity SMR CPU run failed"
        # The volume-weighted norm counts the coarse cells by their volume: this mesh
        # needs one V-cycle more than with the old norm (14) and averages 0.127.
        assert_solver_convergence(results[1], weighted_threshold(threshold, NCELLS_SMR),
                                  max_iterations=14,
                                  max_avg_ratio=0.13,
                                  label="binary_gravity_smr_cpu: ")
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["smr"] = d
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Defect consistency: uniform vs SMR at same effective resolution
# ---------------------------------------------------------------------------

def test_binary_gravity_defect_consistency_cpu():
    """Final defect should be comparable between uniform 64^3 and 2-level SMR."""
    if "uniform" not in _stored_defects or "smr" not in _stored_defects:
        pytest.skip("Prerequisite tests did not run or failed")
    # A threshold-stopped solve lands anywhere within one V-cycle's reduction (about
    # 16x) below the target, so the two final defects need not agree closely; 0.5
    # orders as in the MPI test (0.11 with the old norm, 0.25 with the volume-weighted
    # one, both runs below the same target).
    assert_defect_consistency(
        [_stored_defects["uniform"], _stored_defects["smr"]],
        max_spread=0.5,
        label="binary_gravity_defect_consistency_cpu: ",
    )
