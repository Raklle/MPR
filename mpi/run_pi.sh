#!/bin/bash

set -euo pipefail

MACHINEFILE="./MPR/allnodes"
SRC="./MPR/mpi_pi.c"
PROG="./MPR/mpi_pi"
OUTDIR="./MPR/results_vcluster"

REPEATS=2
MIN_P=1
MAX_P=12

# Lepsze rozmiary na testy niż wcześniej
STRONG_POINTS=100000000        # 1e8
WEAK_POINTS_PER_PROC=10000000  # 1e7 na proces

mkdir -p "$OUTDIR"

OUTFILE="$OUTDIR/vcluster_pi_results.csv"

echo "[INFO] Compiling $SRC ..." >&2
mpicc -O3 -o "$PROG" "$SRC"
echo "[INFO] Compilation done." >&2

echo "run,scaling,processes,total_points,used_points,pi,time_s" > "$OUTFILE"

echo "[INFO] Starting tests on vCluster..." >&2
echo "[INFO] REPEATS=$REPEATS, P=$MIN_P..$MAX_P" >&2
echo "[INFO] STRONG_POINTS=$STRONG_POINTS" >&2
echo "[INFO] WEAK_POINTS_PER_PROC=$WEAK_POINTS_PER_PROC" >&2

for RUN in $(seq 1 $REPEATS); do
    echo "[INFO] ===== RUN $RUN / $REPEATS =====" >&2

    for P in $(seq $MIN_P $MAX_P); do
        # STRONG
        echo "[INFO] Strong: run=$RUN np=$P total=$STRONG_POINTS" >&2
        LINE=$(mpiexec -machinefile "$MACHINEFILE" -np "$P" "$PROG" "$STRONG_POINTS")
        echo "$RUN,strong,$LINE" >> "$OUTFILE"

        # WEAK
        TOTAL_WEAK=$((WEAK_POINTS_PER_PROC * P))
        echo "[INFO] Weak:   run=$RUN np=$P total=$TOTAL_WEAK" >&2
        LINE=$(mpiexec -machinefile "$MACHINEFILE" -np "$P" "$PROG" "$TOTAL_WEAK")
        echo "$RUN,weak,$LINE" >> "$OUTFILE"
    done
done

echo "[INFO] Done." >&2
echo "[INFO] Results saved to: $OUTFILE" >&2