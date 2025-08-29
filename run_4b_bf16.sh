#!/bin/bash

source "/code-fsx/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate soberplus

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/checkpoints-fsx/yibiaoy-sandbox/HF"

unset VLLM_ATTENTION_BACKEND

LOCAL_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus"
OUTPUT_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus/outputs"

# Define prompts directly in the shell script
export SYSTEM_PROMPT="You are a helpful assistant."
export MATH_QUERY_TEMPLATE="Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: \$\\boxed{{ANSWER}}\$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
"

MODELS=(
    Qwen/Qwen3-4B
)

DTYPES=("bfloat16")

MAX_NUM_SEQUENCES=(256)
MAX_NUM_BATCHED_TOKENS=(262144)

MAX_MODEL_LENGTHS=(34816)
MAX_TOKENS_LIST=(32768)

TEMPS_NON_ZERO=(1.0 0.9 0.8 0.6)
TOP_PS=(1.0 0.95 0.9 0.8 0.7)

# Define how many seeds to run for each task
# Format: "task_string:num_seeds"
TASK_NUM_SEEDS_CONFIG=(
    "custom|aime24|0|0:32"
    "custom|aime25|0|0:32"
    "custom|amc23|0|0:32"
    "custom|math_500|0|0:5"
    "custom|minerva|0|0:5"
    "custom|olympiadbench|0|0:5"
)

# The outer loops for model configuration remain the same
for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do
cd $LOCAL_DIR

set -x

echo "Running test for TEMP=0, TOP_P=1.0 ..."
python main_diff.py \
    --model $MODEL \
    --task_num_seeds "${TASK_NUM_SEEDS_CONFIG[@]}" \
    --temperature "0" \
    --top_p "1.0" \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens $MAX_TOKENS \
    --max_model_length $MAX_MODEL_LENGTH \
    --custom_tasks_directory lighteval_tasks.py \
    --system_prompt "$SYSTEM_PROMPT" \
    --use_chat_template \
    --dtype $DTYPE \
    --max_num_seqs $MAX_NUM_SEQUENCES \
    --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
    --tensor_parallel_size 8 \
    --pipeline_parallel_size 1 \
    --data_parallel_size 1

# 运行 2: 处理所有非零温度和所有 top_p 的组合
if [ ${#TEMPS_NON_ZERO[@]} -gt 0 ]; then
    echo "Running tests for non-zero temperatures with all TOP_P values..."
    python main_diff.py \
        --model $MODEL \
        --task_num_seeds "${TASK_NUM_SEEDS_CONFIG[@]}" \
        --temperature "${TEMPS_NON_ZERO[@]}" \
        --top_p "${TOP_PS[@]}" \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --system_prompt "$SYSTEM_PROMPT" \
        --use_chat_template \
        --dtype $DTYPE \
        --max_num_seqs $MAX_NUM_SEQUENCES \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --tensor_parallel_size 8 \
        --pipeline_parallel_size 1 \
        --data_parallel_size 1
fi

set +x

done
done
done
done

echo "All tests completed."
