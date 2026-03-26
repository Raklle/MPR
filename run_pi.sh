#!/bin/bash

set -euo pipefail

MACHINEFILE="./MPR/allnodes"
PROG="./MPR/mpi_pi"
SRC="./MPR/mpi_pi.c"
OUTDIR="./MPR/results_vcluster"

REPEATS=5
MIN_P=1
MAX_P=12

# Strong scaling: stały całkowity problem
STRONG_POINTS=10000000

# Weak scaling: stały problem na proces
WEAK_POINTS_PER_PROC=1000000

mkdir -p "$OUTDIR"

# Kompilacja
echo "[INFO] Compiling $SRC ..." >&2
mpicc -O3 -o "$PROG" "$SRC"
echo "[INFO] Compilation done." >&2

STRONG_OUT="$OUTDIR/strong_c.csv"
WEAK_OUT="$OUTDIR/weak_c.csv"

echo "run,processes,total_points,used_points,pi,time_s" > "$STRONG_OUT"
echo "run,processes,total_points,used_points,pi,time_s" > "$WEAK_OUT"

echo "[INFO] Starting vCluster experiments..." >&2
echo "[INFO] REPEATS=$REPEATS" >&2
echo "[INFO] PROCESSES=$MIN_P..$MAX_P" >&2
echo "[INFO] STRONG_POINTS=$STRONG_POINTS" >&2
echo "[INFO] WEAK_POINTS_PER_PROC=$WEAK_POINTS_PER_PROC" >&2

# -----------------------------
# STRONG SCALING
# -----------------------------
echo "[INFO] Strong scaling..." >&2

for RUN in $(seq 1 $REPEATS); do
    echo "[INFO] Strong: run $RUN / $REPEATS" >&2

    for P in $(seq $MIN_P $MAX_P); do
        echo "[INFO] Strong: np=$P total_points=$STRONG_POINTS" >&2

        LINE=$(mpiexec -machinefile "$MACHINEFILE" -np "$P" "$PROG" "$STRONG_POINTS")
        echo "$RUN,$LINE" >> "$STRONG_OUT"
    done
done

# -----------------------------
# WEAK SCALING
# -----------------------------
echo "[INFO] Weak scaling..." >&2

for RUN in $(seq 1 $REPEATS); do
    echo "[INFO] Weak: run $RUN / $REPEATS" >&2

    for P in $(seq $MIN_P $MAX_P); do
        TOTAL_POINTS=$((WEAK_POINTS_PER_PROC * P))

        echo "[INFO] Weak: np=$P total_points=$TOTAL_POINTS" >&2

        LINE=$(mpiexec -machinefile "$MACHINEFILE" -np "$P" "$PROG" "$TOTAL_POINTS")
        echo "$RUN,$LINE" >> "$WEAK_OUT"
    done
done

echo "[INFO] Done." >&2
echo "[INFO] Results saved to:" >&2
echo "  $STRONG_OUT" >&2
echo "  $WEAK_OUT" >&2