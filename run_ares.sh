#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-testing
#SBATCH --account=plgmpr26-cpu

module add .plgrid plgrid/tools/openmpi

SRC="pi_ares.c"
PROG="run_pi"
OUT="results_${SLURM_JOB_ID}.csv"

mpicc -O3 "$SRC" -o "$PROG"

echo "scaling,size_name,processes,total_points,used_points,pi,time_s" > "$OUT"

NAMES=("SMALL" "MEDIUM" "LARGE")
VALS=(10000000 450000000 20000000000)

for i in {0..2}; do
    NAME=${NAMES[$i]}
    N_BASE=${VALS[$i]}

    for p in {1..12}; do

        RES_S=$(mpirun -np $p ./"$PROG" "$N_BASE" | tail -n 1)
        echo "strong,$NAME,$RES_S" >> "$OUT"

        if [ "$NAME" = "LARGE" ]; then
            N_WEAK_BASE=$((22000000000 / 12))
        else
            N_WEAK_BASE=$N_BASE
        fi

        N_TOTAL=$((N_WEAK_BASE * p))
        RES_W=$(mpirun -np $p ./"$PROG" "$N_TOTAL" | tail -n 1)
        echo "weak,$NAME,$RES_W" >> "$OUT"
    done
done