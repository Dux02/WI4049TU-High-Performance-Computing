#!/bin/bash
set -euo pipefail

# Run from the script's directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

module load intel/oneapi_2026.1
module load mpi/latest

make clean
make

echo "Build successful: pde_solver.x"
