#!/bin/bash
#SBATCH --partition="compute-p2"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3GB
#SBATCH --output="node_speed.out"
#SBATCH --time=00:10:00
#SBATCH --account="Education-EEMCS-Courses-WI4049TU"
set -e
set -o pipefail


# Likwid is currently not in the software stack or node image, so it's a bit tricky to load:
module load 2026 cpu
module load compilers gcc/13.4.0
module use /projects/unsupported/spack2026-cpu/share/spack/lmod/linux-almalinux9-x86_64/gcc/13.4.0/
module load likwid


# print the name of the compute node we ended up on:
echo "This is node " `hostname`

# TODO: measure the Daxpy performance

echo "-------------------------------------------------------------"
echo "Benchmark the speed of computing y = a*x + y, x, y in R^n    "
echo "-------------------------------------------------------------"


echo "1. Core Rmax (data resides in L1 cache"
echo "-------------------------------------------------------------"

likwid-bench -t daxpy_avx_fma -w S0:10kiB:1 | grep -e "MFlops/s"

echo "2. Core Bandwidth (data resides in RAM)"
echo "-------------------------------------------------------------"
likwid-bench -t daxpy_avx_fma -w S0:2GiB:1 | grep -e "MFlops/s" -e "MByte/s"

echo "2. Socket Scaling (data resides in RAM)"
echo "-------------------------------------------------------------"
for P in 1 2 4 8 12 16 20 24 28 32; do
  echo "Cores = $P"
  likwid-bench -t daxpy_avx_fma -w S0:2GiB:$P | grep -e "MFlops/s" -e "MByte/s"
done;
