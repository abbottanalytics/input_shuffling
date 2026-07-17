#!/bin/bash
# Run input shuffling v2. Usage:
#   ./run_input_shuffling_v2.sh [config.yaml]
#
# Lives in the input_shuffling_v2 repo; a symlink in Analytics Projects/
# points here. Works from any cwd (and through the symlink): it resolves
# its real location and runs from the Analytics Projects directory, so
# config and data paths stay relative to Analytics Projects as before.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_DIR="$(dirname "$SCRIPT_PATH")"           # .../input_shuffling_v2
PROJECTS_DIR="$(dirname "$REPO_DIR")"          # .../Analytics Projects

CONFIG=${1:-input_shuffling_v2/configs/kddcup1998_target_b.yaml}
#CONFIG=${1:-input_shuffling_v2/configs/titanic_survived.yaml}
#CONFIG=${1:-input_shuffling_v2/configs/kddcup1998_target_b_fast.yaml}

cd "$PROJECTS_DIR" || exit 1
PYTHONPATH=input_shuffling_v2/src:. \
python3 -m input_shuffling_v2.in_shuff_cli --config "$CONFIG"
