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

usage() {
    cat <<EOF
Usage: $(basename "$0") [-p profile_json] [-c cfg_dir] [-r ratio] [-- extra_args]

Options:
  -p <path>    Profile json path (default: $DEFAULT_PROFILE)
  -c <path>    Static cfg directory (default: $DEFAULT_CFG_DIR)
  -r <value>   Ratio value (default: $DEFAULT_RATIO)
  -h           Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") -p examples/pipo-alg/pipo_profile.json -c examples/pipo-alg/alg-cfg/ -r 0.77
  $(basename "$0") -r 0.8 -- --verbose
EOF
}

while getopts ":p:c:r:h" opt; do
    case "$opt" in
        p) PROFILE="$OPTARG" ;;
        c) CFG_DIR="$OPTARG" ;;
        r) RATIO="$OPTARG" ;;
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

exec "$BINARY" "$PROFILE" -c "$CFG_DIR" -r "$RATIO" "$@"
