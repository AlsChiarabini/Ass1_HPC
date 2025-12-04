#!/bin/bash
set -e

OUTPUT_FILE="atax_timings.csv"
echo "Kernel,Dataset,Time(s)" > "$OUTPUT_FILE"

MODES=("SEQUENTIAL" "PARALLEL" "PARALLEL_NORACE" "REDUCTION" "COLLAPSE" "OPTIMIZED" "OPTIMIZED_TILING" "TARGET")
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
        make EXT_CFLAGS="-D${mode} -D${dataset} -DPOLYBENCH_TIME" all || { echo "Compilation failed for $mode $dataset"; exit 1; }

        # Esegui e cattura solo il tempo (es. "Time in seconds = 1.234")
        TIME_OUTPUT=$(./atax_acc | tail -n 1)

        # Aggiungi riga al CSV
        echo "$mode,$dataset,$TIME_OUTPUT" >> "$OUTPUT_FILE"

        echo "Time for mode=$mode, dataset=$dataset → $TIME_OUTPUT s"
    done
done

echo "========================"
echo "Computing averages..."
echo "========================"

# Calcola la media per ogni mode (ignora l'intestazione)
{
    echo ""
    echo "Kernel,Average_Time(s)"
    awk -F',' 'NR>1 {sum[$1]+=$3; count[$1]++} END {for (m in sum) print m","sum[m]/count[m]}' "$OUTPUT_FILE" | sort
} >> "$OUTPUT_FILE"

echo "========================"
echo "All done. Results saved in $OUTPUT_FILE"
