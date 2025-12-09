#!/bin/bash

# Dataset disponibili
DATASETS=("MINI" "SMALL" "STANDARD" "LARGE" "EXTRALARGE")

# File CUDA da testare
FILES=(
"atax_base_const.cu"
"atax_base_pinned.cu"
"atax_base_streams.cu"
"atax_base_uvm.cu"
"atax_opt_const.cu"
"atax_opt_pinned.cu"
"atax_opt_streams.cu"
"atax_opt_tiling.cu"
"atax_opt_trasposta.cu"
"atax_opt_uvm.cu"
)

# CSV Output
OUT="risultati_atax.csv"
echo "file,dataset,time_ms" > $OUT

# Backup del Makefile
cp Makefile Makefile.bak

for FILE in "${FILES[@]}"; do

    echo ">> Imposto EXERCISE=$FILE"
    # Modifica EXERCISE nel Makefile
    sed -i "s/^EXERCISE *= *.*/EXERCISE = $FILE/" Makefile

    for DS in "${DATASETS[@]}"; do
        
        DATAFLAG=$(echo "$DS" | tr '[:upper:]' '[:lower:]')"_DATASET"

        echo "   -> Compilo con -D$DATAFLAG"

        # Clean + build
        make clean >/dev/null 2>&1
        make EXT_CFLAGS="-D$DATAFLAG" >/dev/null 2>&1

        # Run
        OUTPUT=$(make run 2>/dev/null)

        # Estrai tempo Polybench (xx.xx ms)
        TIME=$(echo "$OUTPUT" | grep -Eo '[0-9]+\.[0-9]+ ms' | grep -Eo '[0-9]+\.[0-9]+')

        # Se non trovato
        if [ -z "$TIME" ]; then
            TIME="N/A"
        fi

        echo "$FILE,$DS,$TIME" | tee -a $OUT
    done

done

# Ripristina Makefile originale
mv Makefile.bak Makefile

echo "=== COMPLETATO ==="
echo "Risultati salvati in $OUT"
