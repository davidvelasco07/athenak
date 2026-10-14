Reproduce from AthenaK commit 2d251bb09d67fe76ac97d419041c33dc906db2ca and its Kokkos source.
Requires Python 3.12+, NumPy, CMake, C++17 compiler, Open MPI, and local MPI sockets.

python3 reproduce.py --repo /path/to/athenak --kokkos /path/to/kokkos --destination /path/to/new/experiment

The destination must not exist. The source repository and Kokkos source are read only.
The script builds with Kokkos bounds checks and runs the 72-case suite.
See halo-test-report.txt for the result and its limits.
