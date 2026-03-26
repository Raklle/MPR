#!/bin/bash

set -e

MACHINEFILE="./MPR/allnodes"
BENCH="./MPR/pingpong.py"
OUTDIR="./MPR/results"

REPEATS=5
MIN_SIZE=1
MAX_SIZE=1
ITER=1000000
WARMUP=100

mkdir -p "$OUTDIR/send"
mkdir -p "$OUTDIR/ssend"

echo "[INFO] Starting measurements..." >&2
echo "[INFO] repeats=$REPEATS min=$MIN_SIZE max=$MAX_SIZE iter=$ITER warmup=$WARMUP" >&2

for MODE in send ssend; do
    echo "[INFO] MODE = $MODE" >&2

    for RUN in $(seq 1 $REPEATS); do
        OUTFILE="$OUTDIR/$MODE/run_${RUN}.csv"

        echo "[INFO] Run $RUN / $REPEATS -> $OUTFILE" >&2

        mpiexec -machinefile "$MACHINEFILE" -np 2 "$BENCH" "$MODE" "$MIN_SIZE" "$MAX_SIZE" "$ITER" "$WARMUP" > "$OUTFILE"

        echo "[INFO] Finished $MODE run $RUN" >&2
    done
done

echo "[INFO] All measurements done." >&2