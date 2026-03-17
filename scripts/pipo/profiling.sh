#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$ROOT_DIR/build-release/bin/llama-pipo-perf"

DEFAULT_MODEL="/mnt/win_E/llm_model/Qwen3-14B-Q4_K_M.gguf"
DEFAULT_CFG="$ROOT_DIR/examples/pipo-alg/alg-cfg/14B.json"

MODEL="$DEFAULT_MODEL"
CFG="$DEFAULT_CFG"

usage() {
	cat <<EOF
Usage: $(basename "$0") [-m model_path] [-c config_path] [-- extra_args]

Options:
  -m <path>    GGUF model path (default: $DEFAULT_MODEL)
  -c <path>    Config json path (default: $DEFAULT_CFG)
  -h           Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") -m /path/to/model.gguf -c /path/to/14B.json
  $(basename "$0") -m /path/to/model.gguf -- -ngl 99
EOF
}

while getopts ":m:c:h" opt; do
	case "$opt" in
		m) MODEL="$OPTARG" ;;
		c) CFG="$OPTARG" ;;
		h)
			usage
			exit 0
			;;
		:) echo "Error: Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
		\?) echo "Error: Invalid option -$OPTARG" >&2; usage; exit 1 ;;
	esac
done

shift $((OPTIND - 1))

if [ ! -x "$BINARY" ]; then
	echo "Error: binary not found or not executable: $BINARY" >&2
	exit 1
fi

exec "$BINARY" -m "$MODEL" -c "$CFG" "$@"