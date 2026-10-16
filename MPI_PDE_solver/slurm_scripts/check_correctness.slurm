#!/bin/bash
#SBATCH --partition=compute-p2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:10:00
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=check_correctness_%j.out
#SBATCH --error=check_correctness_%j.err

set -euo pipefail

module load intel/oneapi_2026.1
module load mpi/latest
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

cd "$SLURM_SUBMIT_DIR"
solver="$PWD/pde_solver.x"
results="$PWD/check_correctness_${SLURM_JOB_ID}"
mkdir -p "$results"

# Include uneven local block sizes.
for grid in "64 65" "66 68"; do
    read -r nx ny <<< "$grid"

    for processes in 1 2 4; do
        run_dir="$results/${nx}x${ny}_p${processes}"
        mkdir -p "$run_dir"

        echo "=== Grid=${nx}x${ny} Processes=$processes ==="

        (
            cd "$run_dir"
            srun --mpi=pmi2 --ntasks="$processes" --cpu-bind=cores \
                "$solver" -x "$nx" -y "$ny" \
                -b 1.8 -r 100000 -t 1e-10 > run.out
        )

        grep -E 'Status:|Iterations:|Final Change:|Execution Time:|Wall Flux Balance|Global Energy Balance|L2 Norm|H1 Semi-Norm' \
            "$run_dir/run.out"
    done
done
