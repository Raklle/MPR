#!/bin/bash -l
#SBATCH --job-name=bucket_scaling
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=plgmpr26-cpu

module purge
module load gcc/12.2.0

# ===================== KOMPILACJA =====================

if [ ! -x ./bucket_sort_1 ] || [ bucket_sort_1.cpp -nt ./bucket_sort_1 ]; then
    echo ">>> Kompilacja bucket_sort_1..." >&2

    g++ -O2 -fopenmp -std=c++17 -Wall \
        bucket_sort_1.cpp \
        -o bucket_sort_1 || exit 1
fi

if [ ! -x ./bucket_sort_1 ] || [ bucket_sort_1.cpp -nt ./bucket_sort_1 ]; then
    echo ">>> Kompilacja bucket_sort_1..." >&2

    g++ -O2 -fopenmp -std=c++17 -Wall \
        bucket_sort_1.cpp \
        -o bucket_sort_1 || exit 1
fi

# ===================== OPENMP =====================

export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ===================== PARAMETRY =====================

N_BUCKETS=10000

# rozmiary problemu
SIZE_LIST="
1000000
5000000
10000000
25000000
50000000
100000000
200000000
"

REPEATS=3

CSV_FILE="scaling_${SLURM_JOB_ID}.csv"

echo "algorithm,threads,N,N_BUCKETS,t_a,t_b,t_c,t_d,t_e,sorted,repeat" > "$CSV_FILE"

# ===================== TESTY =====================

for N in $SIZE_LIST; do

    echo "====================================="
    echo "N = $N"
    echo "====================================="

    for R in $(seq 1 $REPEATS); do

        echo ">>> ALG1 | repeat $R"

        OUTPUT=$(./bucket_sort_1 $N $N_BUCKETS)

        echo "$OUTPUT" \
            | grep "^RESULT," \
            | sed 's/^RESULT,//' \
            | awk -v alg="ALG1" -v r=$R \
                '{print alg","$0","r}' \
            >> "$CSV_FILE"


        echo ">>> ATOMIC | repeat $R"

        OUTPUT=$(./bucket_sort_1 $N $N_BUCKETS)

        echo "$OUTPUT" \
            | grep "^RESULT," \
            | sed 's/^RESULT,//' \
            | awk -v alg="ATOMIC" -v r=$R \
                '{print alg","$0","r}' \
            >> "$CSV_FILE"

    done
done

echo ">>> DONE"
echo "Wyniki zapisane do: $CSV_FILE"