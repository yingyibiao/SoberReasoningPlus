#!/usr/bin/env bash
set -euo pipefail

# Simplified runner that launches the Python entry point a single time.
# All sweeps over decoding parameters are now handled inside ``main.py`` so
# that the vLLM model is initialised only once.

source "/code-fsx/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate soberplus

MODEL="Qwen/Qwen3-1.7B"
TP=8
DTYPE="bfloat16"
TASKS="aime24,aime25,amc23,minerva,olympiadbench,math_500"

TEMPS=(0.0 0.6 0.8 0.9 1.0)
TOP_PS=(0.6 0.8 0.9 0.95 1.0)
NUM_RUNS=1

python main.py \
    --model "$MODEL" \
    --tensor_parallel_size $TP \
    --dtype "$DTYPE" \
    --tasks "$TASKS" \
    --temperatures "${TEMPS[@]}" \
    --top_ps "${TOP_PS[@]}" \
    --num_runs $NUM_RUNS

echo "All evaluations finished."

