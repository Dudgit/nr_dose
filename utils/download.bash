#!/usr/bin/env bash
set -euo pipefail

# Resolve the script directory so we can bind and call the script by absolute path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow the HF token to be provided via env or inline when running this script.
# Example: HF_TOKEN=your_token bash download.bash

# Bind the repository directory into the container at /workspace so the
# container can reliably access this script regardless of container root.
singularity exec \
	--env HF_TOKEN="${HF_TOKEN:-}" \
	--bind "${SCRIPT_DIR}:/workspace" \
	--bind "${SCRIPT_DIR}/data:/data" \
	utils/dwd.sif \
	python /workspace/download_dataset.py --output-dir /data

# EOF