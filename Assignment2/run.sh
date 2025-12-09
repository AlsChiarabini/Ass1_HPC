#!/bin/bash
set -e

OUTPUT_FILE="atax_timings.csv"
echo "Kernel,Dataset,Time(ms)" > "$OUTPUT_FILE"

# Mappa MODE → FILE
declare -A MODE_FILES=(
    ["BASE"]="atax_base.cu"
    ["BASE_PINNED"]="atax_base_pinned.cu"
    ["BASE_UVM"]="atax_base_uvm.cu"
    ["BASE_CONST"]="atax_base_const.cu"
    ["OPT_PINNED"]="atax_opt_pinned.cu"
    ["OPT_UVM"]="atax_opt_uvm.cu"
    ["OPT_CONST"]="atax_opt_const.cu"
    ["OPT_TILING"]="atax_opt_tiling.cu"
    ["OPT_TRANSPOSE"]="atax_opt_trasposta.cu"
    ["OPT_STREAMS"]="atax_opt_streams.cu"
)

DATASETS=("MINI_DATASET" "SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET" "EXTRALARGE_DATASET")

for mode in "${!MODE_FILES[@]}"
do
    FILE="${MODE_FILES[$mode]}"
    
    echo "========================"
    echo "Running mode: $mode ($FILE)"
    echo "========================"

    for dataset in "${DATASETS[@]}"
    do
        echo "------------------------"
        echo "Using dataset: $dataset"
        echo "------------------------"

        make clean
        make EXERCISE="$FILE" EXT_CFLAGS="-D${dataset} -DPOLYBENCH_TIME" all || { 
            echo "Compilation failed for $mode $dataset"; 
            continue
        }

        # Esegui e cattura il tempo
        TIME_OUTPUT=$(./atax_cuda 2>&1 | grep "GPU kernels elapsed time" | awk '{print $5}')

        # Se non trova tempo, segna N/A
        if [ -z "$TIME_OUTPUT" ]; then
            TIME_OUTPUT="N/A"
            echo "No timing found (possible crash/OOM)"
        fi

        # Aggiungi al CSV (già in ms)
        echo "$mode,$dataset,$TIME_OUTPUT" >> "$OUTPUT_FILE"

        echo "Time for mode=$mode, dataset=$dataset → $TIME_OUTPUT ms"
    done
done

echo ""
echo "Benchmark completato! Risultati in: $OUTPUT_FILE"
