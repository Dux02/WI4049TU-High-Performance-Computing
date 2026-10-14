#!/bin/bash
#SBATCH --partition=compute-p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:30:00
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=benchmark_single_%j.out
#SBATCH --error=benchmark_single_%j.err

set -euo pipefail

module load intel/oneapi_2026.1
module load mpi/latest

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

cd "$SLURM_SUBMIT_DIR"

hostname
lscpu

for REP in 1 2 3; do
    echo "Repeat=$REP"

    srun --mpi=pmi2 --cpu-bind=cores \
        ./pde_solver.x \
        -x 4096 -y 4097 \
        -b 1.8 -r 1000 -t 0 -q
done
