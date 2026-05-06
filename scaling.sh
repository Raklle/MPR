#!/bin/bash -l
#SBATCH --job-name=bucket_scaling
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=plgmpr26-cpu

module purge
module load gcc/12.2.0

if [ ! -x ./bucket_sort_atomic ] || [ bucket_sort_atomc.cpp -nt ./bucket_sort_atomic ]; then
    echo ">>> Kompilacja..." >&2
    g++ -O2 -fopenmp -std=c++17 -Wall bucket_sort_atomic.cpp -o bucket_sort_atomic || exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

N=100000000
N_BUCKETS=10000
THREAD_LIST="1 2 4 8 12 16 24 32 48"
REPEATS=3

CSV_FILE="scaling_${SLURM_JOB_ID}.csv"

echo "threads,N,N_BUCKETS,t_a,t_b,t_c,t_d,t_e,sorted,repeat" > "$CSV_FILE"

for T in $THREAD_LIST; do
    if [ "$T" -le "$SLURM_CPUS_PER_TASK" ]; then
        export OMP_NUM_THREADS=$T
        for R in $(seq 1 $REPEATS); do
            OUTPUT=$(./bucket_sort_atomic $N $N_BUCKETS)

            echo "$OUTPUT" | grep "^RESULT," | sed 's/^RESULT,//' | \
                awk -v r=$R '{print $0","r}' >> "$CSV_FILE"
        done
    fi
done
