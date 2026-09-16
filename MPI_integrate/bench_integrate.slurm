#!/bin/bash
#SBATCH --partition="compute-p2"
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --output="bench_integrate.dat"
#SBATCH --time=00:05:00
#SBATCH --account="Education-EEMCS-Courses-WI4049TU"

module load 2026 cpu openmpi

# ============================================================
# MPI numerical integration scaling benchmark
#
# Runs integrate.x for several:
#   - numbers of MPI processes (NP)
#   - global problem sizes (N)
#
# Only the numerical SUMMARY lines are written to stdout,
# resulting in a CSV-compatible bench_integrate.dat file.
#
# OpenMPI warnings (for example about unavailable CUDA
# libraries on CPU nodes) are redirected to a separate file.
# ============================================================

NPROC="1 2 4 6 8 10 12 14 16"
NS="10000 1000000"

# Print one CSV header.
echo "nproc,global_n,v1_ms,v2_ms,v3_ms"

# Run the complete scaling experiment.
for N in ${NS}; do
    for NP in ${NPROC}; do

        # Run with NP MPI processes.
        #
        # stderr is redirected so that OpenMPI warnings do not
        # contaminate the benchmark data.
        #
        # grep selects only actual numerical SUMMARY lines.
        # This excludes the header printed by integrate.x.
        #
        # sed removes the "SUMMARY " prefix.
        srun --ntasks=${NP} ./integrate.x ${N} 2>>bench_integrate.err \
            | grep "^SUMMARY *[0-9]" \
            | sed 's/^SUMMARY //'

    done
done
