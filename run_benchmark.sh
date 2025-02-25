#!/bin/bash

# run_benchmark.sh
if [ $# -eq 0 ]; then
    # choose default env to use
    VENV="your_env"
else
    VENV="$1"
    shift  # Remove the first argument (venv name)
fi

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if virtual environment exists
if [ ! -d "${SCRIPT_DIR}/virt_envs/${VENV}" ]; then
    echo "Error: Virtual environment '${VENV}' not found in virt_envs/"
    exit 1
fi

# Activate the virtual environment and run the evaluation
source "${SCRIPT_DIR}/virt_envs/${VENV}/bin/activate"
python "${SCRIPT_DIR}/run_eval.py" "$@"
deactivate