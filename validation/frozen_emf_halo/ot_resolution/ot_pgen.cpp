// Validation entry point: use the repository's unmodified Orszag-Tang initial data.
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  OrszagTang(pin, restart);
}
