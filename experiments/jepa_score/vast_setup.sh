#!/bin/bash
# Setup script for Vast.ai instances.
# Installs dependencies and prepares the experiment environment.
# Usage: bash vast_setup.sh

set -e

echo "=== JEPA-SCORE Experiment Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Install dependencies
pip install --quiet torch torchvision numpy scikit-learn scipy

# Create working directory
mkdir -p /workspace/jepa_score
cd /workspace/jepa_score

echo "=== Setup complete ==="
