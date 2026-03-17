#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$ROOT_DIR/build-release/bin/llama-pipo-override"

DEFAULT_PROFILE="$ROOT_DIR/examples/pipo-alg/alg-cfg/pipo_profile.json"
DEFAULT_CFG_DIR="$ROOT_DIR/examples/pipo-alg/alg-cfg/"
DEFAULT_RATIO="0.77"

PROFILE="$DEFAULT_PROFILE"
CFG_DIR="$DEFAULT_CFG_DIR"
RATIO="$DEFAULT_RATIO"
ENABLE_MOE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [-p profile_json] [-c cfg_dir] [-r ratio] [--moe] [-- extra_args]

Options:
  -p <path>    Profile json path (default: $DEFAULT_PROFILE)
  -c <path>    Static cfg directory (default: $DEFAULT_CFG_DIR)
  -r <value>   Ratio value (default: $DEFAULT_RATIO)
  --moe        Enable MoE mode
  -h           Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") -p examples/pipo-alg/pipo_profile.json -c examples/pipo-alg/alg-cfg/ -r 0.77
  $(basename "$0") -r 0.8 --moe
  $(basename "$0") -r 0.8 -- --verbose
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p)
            PROFILE="$2"
            shift 2
            ;;
        -c)
            CFG_DIR="$2"
            shift 2
            ;;
        -r)
            RATIO="$2"
            shift 2
            ;;
        --moe|-moe)
            ENABLE_MOE="--moe"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -x "$BINARY" ]; then
    echo "Error: binary not found or not executable: $BINARY" >&2
    exit 1
fi

exec "$BINARY" "$PROFILE" -c "$CFG_DIR" -r "$RATIO" $ENABLE_MOE "$@"
