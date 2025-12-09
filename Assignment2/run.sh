#!/bin/bash
set -e

OUTPUT_FILE="atax_timings.csv"
echo "Kernel,Dataset,Time(s)" > "$OUTPUT_FILE"

MODES=("BASE" "PINNED" "UVM" "CONST" "TILING" "TRANSPOSE" "STREAMS" "OPT_STREAMS")
DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET" "EXTRALARGE_DATASET")

for mode in "${MODES[@]}"
do
    echo "========================"
    echo "Running mode: $mode"
    echo "========================"

    for dataset in "${DATASETS[@]}"
    do
        echo "------------------------"
        echo "Using dataset: $dataset"
        echo "------------------------"

        make clean
        make EXERCISE=atax.cu EXT_CFLAGS="-D${mode} -D${dataset} -DPOLYBENCH_TIME" all || { echo "Compilation failed for $mode $dataset"; exit 1; }

        # Esegui e cattura solo il tempo (es. "GPU kernels elapsed time: 1.234 ms")
        TIME_OUTPUT=$(./atax_cuda 2>&1 | grep "GPU kernels elapsed time" | awk '{print $5}')

        # Converti da ms a s (dividi per 1000)
        TIME_SECONDS=$(echo "$TIME_OUTPUT" | awk '{print $1/1000}')

        # Aggiungi riga al CSV
        echo "$mode,$dataset,$TIME_SECONDS" >> "$OUTPUT_FILE"

        echo "Time for mode=$mode, dataset=$dataset → $TIME_SECONDS s"
    done
done
