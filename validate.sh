#!/usr/bin/env bash

TEMPI_FILE="tempi.txt"

declare -a exercises=(
	"atax.c"
	"atax_tiling.c"
	"atax_target.c"
	"atax_for.c"
)

declare -a datasets=(
	"MINI_DATASET"
	"SMALL_DATASET"
	"STANDARD_DATASET"
	"LARGE_DATASET"
	"EXTRALARGE_DATASET"
)

for exercise in "${exercises[@]}"; do
	echo "Running ${exercise}" >&2
	{
		echo "${exercise}"
		echo "$(printf '%0.s-' {1..40})"
	} >>"${TEMPI_FILE}"

	for dataset in "${datasets[@]}"; do
		dataset_flag=""
		if [[ "${dataset}" != "STANDARD_DATASET" ]]; then
			dataset_flag="-D${dataset}"
		fi

		ext_flags="-DPOLYBENCH_TIME"
		if [[ -n "${dataset_flag}" ]]; then
			ext_flags+=" ${dataset_flag}"
		fi

		echo "  ${exercise} :: ${dataset}" >&2
		output=$(make EXERCISE="${exercise}" EXT_CFLAGS="${ext_flags}" clean all run)

		{
			echo "${dataset}:"
			echo "${output}"
			echo
		} >>"${TEMPI_FILE}"
	done
done
