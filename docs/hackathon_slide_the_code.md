# Hackathon slide: The code (AthenaK)

Copy the **Slide** section into PowerPoint / Google Slides / Keynote.

---

## Slide

### The code — **AthenaK**

**Exascale-ready astrophysics simulation**

- **Written in C++ and CUDA** — Modern C++ solvers with **Kokkos** for performance portability across CPUs and GPUs
- **Uses MPI for distributed computing** — Block-based AMR and domain decomposition across many nodes
- **Uses HDF5 for I/O** — Large datasets via HDF5 (athdf) in the analysis stack, plus VTK, tabular, and restart formats

---

## Speaker notes (optional)

| Bullet | One-liner |
|--------|-----------|
| C++ & CUDA | Rewrite of Athena++ using Kokkos; one codebase for CPU and GPU backends. |
| MPI | Mesh blocks distributed over MPI ranks with communication in the AMR / boundary layer. |
| HDF5 | Post-processing uses HDF5 (athdf); runtime also supports VTK, tab, binary restart. |
