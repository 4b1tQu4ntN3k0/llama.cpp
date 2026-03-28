

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

decode_lengths=(32 128 512 1024 2048)
decode_delays=(0 10 10 30 300)
decode_repeats=(3 3 1 1 1)
pipo_modes=(0 1)
decode_offloads=(1 1 1 1 1)

thread=4
bench_threads=4
if [ "$GPU_MODEL" = "4090" ]; then
    bench_threads=8
    decode_delays=(0 0 0 0 0)
    decode_offloads=(0 0 0 0 0)
fi

if [ "$GPU_MODEL" = "4090" ]; then
    for pipo_mode in "${pipo_modes[@]}"; do
        if [ "$pipo_mode" = "0" ]; then
            mode_name="base"
        else
            mode_name="pipo"
        fi

        output_path="logs/bench/d_${mode_name}_${model_name}.json"

        ./build-release/bin/llama-bench -m "$MODEL_PATH" -o json -oe md \
            -r "$(IFS=,; echo "${decode_repeats[*]}")" \
            --delay "$(IFS=,; echo "${decode_delays[*]}")" \
            -b 512 -ub 512 -t "$bench_threads" -ngl "$NGL" -p 0 \
            --pipo "$pipo_mode" -config examples/pipo-alg/alg_cfg/perf.json \
            -d 2048 -n "$(IFS=,; echo "${decode_lengths[*]}")" -do 0 \
            > "$output_path"
    done

    exit 0
fi

for pipo_mode in "${pipo_modes[@]}"; do
    if [ "$pipo_mode" = "0" ]; then
        mode_name="base"
    else
        mode_name="pipo"
    fi

    for idx in "${!decode_lengths[@]}"; do
        decode_len="${decode_lengths[$idx]}"
        decode_delay="${decode_delays[$idx]}"
        decode_repeat="${decode_repeats[$idx]}"
        decode_offload="${decode_offloads[$idx]}"
        output_path="logs/bench/d_${mode_name}_${decode_len}_${model_name}.json"

        ./build-release/bin/llama-bench -m "$MODEL_PATH" -o json \
            -r "$decode_repeat" --delay "$decode_delay" \
            -b 512 -ub 512 -t "$bench_threads" -ngl "$NGL" -p 0 \
            --pipo "$pipo_mode" -config examples/pipo-alg/alg_cfg/perf.json \
            -d 2048 -n "$decode_len" -do "$decode_offload" \
            > "$output_path"
    done
done