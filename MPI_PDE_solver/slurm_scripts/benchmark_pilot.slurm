#!/bin/bash
#SBATCH --partition=compute-p2
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --account=Education-EEMCS-Courses-WI4049TU
#SBATCH --output=benchmark_pilot_%j.out
#SBATCH --error=benchmark_pilot_%j.err

set -euo pipefail

module load intel/oneapi_2026.1
module load mpi/latest
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

results="results/pilot_${SLURM_JOB_ID}"
mkdir -p "$results"

hostname
lscpu

echo "processes,nx,ny,beta,iterations,seconds,gflops" \
    > "$results/timings.csv"

for processes in 1 2 4 8 16 32 64; do
    echo "=== Overlap pilot: P=$processes ==="

    srun --mpi=pmi2 --ntasks="$processes" --cpu-bind=cores \
        ./bin/sor_overlap \
        -x 10000 -y 10000 -b 1.8 -r 20 -t 0 -q \
        > "$results/p${processes}.out"

    grep '^BENCH,' "$results/p${processes}.out" \
        | cut -d, -f2- >> "$results/timings.csv"

    grep -E '^Iterations:|^Execution Time:|^Performance:' \
        "$results/p${processes}.out"
done

echo "Pilot complete: $results/timings.csv"
