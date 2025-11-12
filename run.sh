#!/bin/bash
set -e

OUTPUT_FILE="atax_timings.csv"
echo "Kernel,Time(s)" > $OUTPUT_FILE

MODES=("SEQUENTIAL" "PARALLEL" "PARALLEL_NORACE" "REDUCTION" "COLLAPSE" "OPTIMIZED" "OPTIMIZED_TILING" "TARGET")

for mode in "${MODES[@]}"
do
    echo "========================"
    echo "Running mode: $mode"
    echo "========================"

    # Compila
    make EXT_CFLAGS="-D$mode -DEXTRALARGE_DATASET -DPOLYBENCH_TIME" clean all || { echo "Compilation failed for $mode"; exit 1; }

    # Cattura il tempo
    TIME_OUTPUT=$(./atax_acc)

    # Salva nel CSV
    echo "$mode,$TIME_OUTPUT" >> $OUTPUT_FILE

    echo "Time for $mode: $TIME_OUTPUT s"
done

echo "========================"
echo "All done. Results saved in $OUTPUT_FILE"
