#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <model_path> <gpu_model: 3060|4090> <ngl>"
	echo "Example: $0 models/qwen.gguf 3060 21"
}

if [ "$#" -ne 3 ]; then
	usage
	exit 1
fi

MODEL_PATH="$1"
GPU_MODEL="$2"
NGL="$3"

if [ ! -f "$MODEL_PATH" ]; then
	echo "error: model file not found: $MODEL_PATH"
	exit 1
fi

case "$GPU_MODEL" in
	3060|4090)
		;;
	*)
		echo "error: gpu_model must be 3060 or 4090"
		usage
		exit 1
		;;
esac

if ! [[ "$NGL" =~ ^[0-9]+$ ]]; then
	echo "error: ngl must be a non-negative integer"
	exit 1
fi

mkdir -p logs/bench

model_filename="$(basename "$MODEL_PATH")"
model_name="${model_filename%.*}"

prefill_lengths=(512 1024 2048 4096)
prefill_repeats=(3 3 3 3)
pipo_modes=(0 1)

thread=4
batch_sizes=(512 512 512 512)
micro_batch_sizes=(512 512 512 512)
prefill_delays=(0 10 10 60)
delay_args=(--delay 0)

if [ "$GPU_MODEL" = "4090" ]; then
	thread=8
	batch_sizes=(512 1024 2048 2048)
	micro_batch_sizes=(512 1024 2048 2048)
	delay_args=()
fi

for pipo_mode in "${pipo_modes[@]}"; do
	if [ "$pipo_mode" = "0" ]; then
		mode_name="base"
	else
		mode_name="pipo"
	fi

	for idx in "${!prefill_lengths[@]}"; do
		prefill_len="${prefill_lengths[$idx]}"
		prefill_repeat="${prefill_repeats[$idx]}"
		prefill_delay="${prefill_delays[$idx]}"
		batch_size="${batch_sizes[$idx]}"
		micro_batch_size="${micro_batch_sizes[$idx]}"
		output_path="logs/bench/p_${mode_name}_${prefill_len}_${model_name}.json"

		if [ "$GPU_MODEL" = "3060" ]; then
			delay_args=(--delay "$prefill_delay")
		fi

		./build-release/bin/llama-bench -m "$MODEL_PATH" -o json -oe md\
			-r "$prefill_repeat" "${delay_args[@]}" \
			-b "$batch_size" -ub "$micro_batch_size" -t "$thread" -ngl "$NGL" -p "$prefill_len" \
			--pipo "$pipo_mode" -config examples/pipo-alg/alg_cfg/perf.json \
			-n 0 \
			> "$output_path"
	done
done
