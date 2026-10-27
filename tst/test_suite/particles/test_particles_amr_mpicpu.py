"""Two-rank particle ownership across AMR regrids and load balancing."""

from test_suite.particles.test_particles_amr_gpu import _check_particle_amr


def test_particle_amr_mpicpu():
    """Particles follow their MeshBlocks when load balancing moves them between ranks."""
    _check_particle_amr("particle_amr_mpicpu", mpi=True)
