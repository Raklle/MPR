#!/bin/bash

set -e

MACHINEFILE="./MPR/allnodes"
BENCH="./MPR/pingpong.py"
OUTDIR="./MPR/results"

REPEATS=10
MIN_SIZE=1
MAX_SIZE=1048576      # 1 MB
ITER=10000
WARMUP=100

mkdir -p "$OUTDIR/ssend"
mkdir -p "$OUTDIR/rsend"

echo "[INFO] Starting measurements..."
echo "[INFO] repeats=$REPEATS min=$MIN_SIZE max=$MAX_SIZE iter=$ITER warmup=$WARMUP"

for MODE in ssend rsend; do
    echo "[INFO] MODE = $MODE"

    for RUN in $(seq 1 $REPEATS); do
        OUTFILE="$OUTDIR/$MODE/run_${RUN}.csv"

        echo "[INFO] Run $RUN / $REPEATS -> $OUTFILE"

        mpiexec -machinefile "$MACHINEFILE" -np 2 "$BENCH" "$MODE" "$MIN_SIZE" "$MAX_SIZE" "$ITER" "$WARMUP" > "$OUTFILE"

        echo "[INFO] Finished $MODE run $RUN"
    done
done

echo "[INFO] All measurements done."