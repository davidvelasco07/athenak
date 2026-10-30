# GR MOOD trial velocity regression

The GR NAD candidate must be the primitive velocity recovered from the same trial conserved state, candidate magnetic field and metric used by PAD, rather than S_i/D. The optional GR C2P scratch view preserves the existing interface and immutable trial/reference states. The change leaves NR/SR, predictor, tolerance and SED paths unchanged.

Run `python3 validation/gr_mood_velocity/run_unit.py --build build_torus --output /tmp/gr_mood_velocity` using an existing serial Kokkos build. The runner extracts the actual detector context from current source and checks GR P2C/C2P round trips and velocity component/magnitude acceptance and rejection for 288 states (flat and Kerr-Schild metrics, two adiabatic indices, several densities, fields and velocities). Assertions stay enabled. SED is disabled to isolate quantity consistency; this is not a full solver or long-time test.

Production provenance: the equivalent isolated fix was validated in twelve four-cycle GPU controls and a completed base128 L4 torus through t=10000. Full evidence is in the shared `~/athenak_runs/torus_gr_detector_fix_20261029` and `torus_grfix128_20261029` topics. The frozen production binary is b4427900ae390d16cfd21022124cc3d95b812ffdb5852e50c8aab2bcb25b8a76; it does not include later branch features. No curvature/plateau detector is included.
