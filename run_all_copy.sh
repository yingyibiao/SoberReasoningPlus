source "/code-fsx/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate soberplus

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=32
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VLLM_TORCH_COMPILE=0
export VLLM_CUDA_GRAPH=0
export HF_HOME="/checkpoints-fsx/yibiaoy-sandbox/HF"
LOCAL_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus"
OUTPUT_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus/output"

unset VLLM_ATTENTION_BACKEND
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_ATTENTION_BACKEND=FLASHINFER
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=1

# Set path for prompts file and load SYSTEM_PROMPT
export PROMPTS_PATH="$LOCAL_DIR/prompts.json"
SYSTEM_PROMPT=$(python -c "import json; print(json.load(open('$PROMPTS_PATH'))['SYSTEM_PROMPT'])")

MODELS=(
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    # Qwen/Qwen3-1.7B
    # Qwen/Qwen3-4B
    # Qwen/Qwen3-8B
    # Qwen/Qwen3-14B
)

MAX_NUM_SEQUENCES=(128)


MAX_NUM_BATCHED_TOKENS=(
    # 262144
    131072
    # 65536
    # 32768
    # 16384
)

TOP_PS=(
    # 0.8 
    # 0.9 
    # 0.95 
    # 0.98 
    1.0
)


TEMPS=(
    # 0.0 
    # 0.2 
    # 0.4 
    # 0.6 
    # 0.8 
    1.0
)

DTYPES=(
    "bfloat16" 
    # "float16" 
    # "float32"
)

MAX_MODEL_LENGTHS=(
    34816
)

MAX_TOKENS_LIST=(
    32768
)

for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
cd $LOCAL_DIR

set -x

NUM_RUNS=1

TASKS=(
    # "custom|aime24|0|0"
    "custom|math_500|0|0"
    # "custom|amc23|0|0"
    # "custom|aime25|0|0"
    # "custom|gpqa:diamond|0|0"
    # "custom|minerva|0|0"
    # "custom|olympiadbench|0|0"
)
for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
for TASK in "${TASKS[@]}"; do
    python main.py \
        --model $MODEL \
        --task $TASK \
        --temperature $TEMP \
        --top_p $TOP_P \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --system_prompt "$SYSTEM_PROMPT" \
        --use_chat_template \
        --dtype $DTYPE \
        --max_num_seqs $MAX_NUM_SEQUENCES \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --tensor_parallel_size 4 \
        --pipeline_parallel_size 1 \
        --data_parallel_size 1
done
done

# Analysis every experiments
echo "Starting analysis..."

MODEL_NAME_WITH_SLASH="${MODELS[0]}"
MODEL_NAME_WITH_UNDERSCORE=$(echo "$MODEL_NAME_WITH_SLASH" | sed 's/\//_/g')

# Run analysis scripts
python $LOCAL_DIR/analyze_results.py \
    --base_dir "$OUTPUT_DIR" \
    --model_name_pattern "$MODEL_NAME_WITH_UNDERSCORE" \
    --tokenizer_name "$MODEL_NAME_WITH_SLASH"

# Convert parquet to csv
echo "Converting Parquet files to CSV..."
find "$OUTPUT_DIR" -type f -name "*.parquet" | while read -r parquet_file; do
    python $LOCAL_DIR/convert_parquet_to_csv.py "$parquet_file"
done

echo "Analysis complete."

done
done
done
done
done
done

echo "Generating summary analysis..."

RESULTS_PATH="$OUTPUT_DIR/$MODEL_NAME_WITH_UNDERSCORE/all_experiments_results.json"
ANALYSIS_OUTPUT_PATH="$OUTPUT_DIR/$MODEL_NAME_WITH_UNDERSCORE/analysis_results.json"

python $LOCAL_DIR/analyze_summary.py \
    --results-path "$RESULTS_PATH" \
    --output-path "$ANALYSIS_OUTPUT_PATH"
