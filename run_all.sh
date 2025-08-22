source "/code-fsx/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate soberplus

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=32
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/checkpoints-fsx/yibiaoy-sandbox/HF"

LOCAL_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus"
OUTPUT_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus/output"
LOG_ROOT="$OUTPUT_DIR/logs"

unset VLLM_ATTENTION_BACKEND
# export VLLM_ATTENTION_BACKEND=XFORMERS

# —— SYSTEM_PROMPT ——
export PROMPTS_PATH="$LOCAL_DIR/prompts.json"
SYSTEM_PROMPT=$(
    python - <<'PY'
import json, os
p=os.environ["PROMPTS_PATH"]
print(json.load(open(p))["SYSTEM_PROMPT"])
PY
)

# =========================
# ===== Sweep 配置区 =====
# =========================
MODELS=(
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    # Qwen/Qwen3-1.7B
    # Qwen/Qwen3-4B
    # Qwen/Qwen3-8B
    # Qwen/Qwen3-14B
)

MAX_NUM_SEQUENCES=(
    128
    # 64
    # 32
)

MAX_NUM_BATCHED_TOKENS=(
    # 262144
    131072
    # 65536
    # 32768
    # 16384
)

TOP_PS=(
    0.8
    # 0.9
    # 0.95
    # 0.98
    # 1.0
)

TEMPS=(
    0.0
    0.2
    0.4
    0.6
    0.8
    1.0
)

DTYPES=(
    "bfloat16"
    # "float16"
    # "float32"
)

MAX_MODEL_LENGTHS=(
    34816
    18432
    10240
)

MAX_TOKENS_LIST=(
    32768
    16384
    8192
)

# —— 任务与 seeds 映射 ——
declare -A TASK_SEEDS=(
    ["aime24"]=32
    ["aime25"]=32
    ["amc23"]=32
    ["math_500"]=2
    ["minerva"]=10
    ["olympiadbench"]=10
    # ["gpqa:diamond"]=10
)

# 本次要跑的任务（默认全开；按需注释）
TASK_NAMES=(
    # "aime24"
    # "aime25"
    # "amc23"
    "math_500"
    # "minerva"
    # "olympiadbench"
    # "gpqa:diamond"
)

# —— RUN 会话与环境日志 ——
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
MASTER_LOG="$RUN_DIR/master.log"

{
    echo "==== RUN_ID: $RUN_ID ===="
    echo "Start: $(date -Is)"
    echo "HOST: $(hostname)"
    echo "CWD: $(pwd)"
    echo "--- GPU Info ---"
    nvidia-smi || true
    echo "--- Key Python pkgs ---"
    pip freeze | grep -E -i 'vllm|flash|triton|xformers|torch|cuda' || true
    echo "--- Env snapshot (filtered) ---"
    env | grep -E -i 'NCCL|OMP|VLLM|CUDA|HF_HOME' || true
} | tee -a "$MASTER_LOG"

# —— 计划概览 ——
{
    echo "[计划任务与 seeds 数]"
    for t in "${TASK_NAMES[@]}"; do
        if [[ -z ${TASK_SEEDS[$t]+x} ]]; then
            echo "  ! 未在 TASK_SEEDS 中找到任务：$t  -> 请先在映射里配置 seeds 数" >&2
            exit 1
        fi
        echo "  - ${t}: ${TASK_SEEDS[$t]} seeds"
    done
    echo
} | tee -a "$MASTER_LOG"

trap 'echo "End: $(date -Is)" | tee -a "$MASTER_LOG"' EXIT

# =========================
# ===== 主训练循环  =====
# =========================
for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
    for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for DTYPE in "${DTYPES[@]}"; do
                for TOP_P in "${TOP_PS[@]}"; do
                    for TEMP in "${TEMPS[@]}"; do
                        cd "$LOCAL_DIR"

                        MODEL_UNDERSCORE="${MODEL//\//_}"

                        for TASK_NAME in "${TASK_NAMES[@]}"; do
                            # seeds 映射检查（再次兜底）
                            if [[ -z ${TASK_SEEDS[$TASK_NAME]+x} ]]; then
                                echo "ERROR: 任务 $TASK_NAME 未配置 seeds" | tee -a "$MASTER_LOG"
                                exit 1
                            fi
                            NUM_RUNS="${TASK_SEEDS[$TASK_NAME]}"
                            TASK="custom|${TASK_NAME}|0|0"

                            echo ">>> Running: model=$MODEL task=$TASK seeds=$NUM_RUNS temp=$TEMP top_p=$TOP_P max_tokens=$MAX_TOKENS dtype=$DTYPE max_model_length=$MAX_MODEL_LENGTH" |
                                tee -a "$MASTER_LOG"

                            for ((RUN = 0; RUN < NUM_RUNS; RUN++)); do
                                SEED=$RUN

                                # —— 每个实验的专属输出与日志目录 ——
                                EXP_DIR="$OUTPUT_DIR/$MODEL_UNDERSCORE/$TASK_NAME/seed_${SEED}/temp_${TEMP}__top_p_${TOP_P}__maxlen_${MAX_TOKENS}"
                                mkdir -p "$EXP_DIR"
                                RUN_LOG="$EXP_DIR/run.log"
                                ARGS_JSON="$EXP_DIR/args.json"

                                # 先把参数导出为环境变量，Python 里统一读取
                                export ARG_MODEL="$MODEL"
                                export ARG_TASK="$TASK"
                                export ARG_TEMPERATURE="$TEMP"
                                export ARG_TOP_P="$TOP_P"
                                export ARG_OUTPUT_DIR="$OUTPUT_DIR"
                                export ARG_MAX_NEW_TOKENS="$MAX_TOKENS"
                                export ARG_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH"
                                export ARG_DTYPE="$DTYPE"
                                export ARG_MAX_NUM_SEQS="${MAX_NUM_SEQUENCES[0]}"
                                export ARG_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS[0]}"
                                export ARG_TP="4"
                                export ARG_PP="1"
                                export ARG_DP="1"
                                export ARG_SEED="$SEED"

                                # 记录参数 JSON（安全：使用 env，避免转义/展开问题）
                                python - <<'PY' >"$ARGS_JSON"
import json, os
def geti(k, d=None):
    v=os.environ.get(k)
    if v is None: return d
    try: return int(v)
    except: 
        try: return float(v)
        except: return v
cfg = {
  "model": os.environ.get("ARG_MODEL",""),
  "task": os.environ.get("ARG_TASK",""),
  "temperature": geti("ARG_TEMPERATURE"),
  "top_p": geti("ARG_TOP_P"),
  "output_dir": os.environ.get("ARG_OUTPUT_DIR",""),
  "max_new_tokens": geti("ARG_MAX_NEW_TOKENS"),
  "max_model_length": geti("ARG_MAX_MODEL_LENGTH"),
  "dtype": os.environ.get("ARG_DTYPE",""),
  "max_num_seqs": geti("ARG_MAX_NUM_SEQS"),
  "max_num_batched_tokens": geti("ARG_MAX_NUM_BATCHED_TOKENS"),
  "tp": geti("ARG_TP"),
  "pp": geti("ARG_PP"),
  "dp": geti("ARG_DP"),
  "seed": geti("ARG_SEED"),
  "system_prompt": "<<<omitted>>>"
}
print(json.dumps(cfg, indent=2))
PY

                                # 真正执行 + tee 记录日志
                                {
                                    echo "---- $(date -Is) RUN START ----"
                                    set -x
                                    python main.py \
                                        --model "$MODEL" \
                                        --task "$TASK" \
                                        --temperature "$TEMP" \
                                        --top_p "$TOP_P" \
                                        --output_dir "$OUTPUT_DIR" \
                                        --max_new_tokens "$MAX_TOKENS" \
                                        --max_model_length "$MAX_MODEL_LENGTH" \
                                        --custom_tasks_directory lighteval_tasks.py \
                                        --system_prompt "$SYSTEM_PROMPT" \
                                        --use_chat_template \
                                        --dtype "$DTYPE" \
                                        --max_num_seqs "${MAX_NUM_SEQUENCES[0]}" \
                                        --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS[0]}" \
                                        --tensor_parallel_size 4 \
                                        --pipeline_parallel_size 1 \
                                        --data_parallel_size 1 \
                                        --seed "$SEED"
                                    set +x
                                    echo "---- $(date -Is) RUN END ----"
                                } 2>&1 | tee -a "$RUN_LOG" | tee -a "$MASTER_LOG" >/dev/null

                            done # seed
                        done     # task

                        # —— 单次 sweep 分析（针对当前 $MODEL）——
                        ANALYSIS_LOG="$RUN_DIR/analysis_${MODEL_UNDERSCORE}.log"
                        {
                            echo "=== Analysis for $MODEL_UNDERSCORE @ $(date -Is) ==="
                            python "$LOCAL_DIR/analyze_results.py" \
                                --base_dir "$OUTPUT_DIR" \
                                --model_name_pattern "$MODEL_UNDERSCORE" \
                                --tokenizer_name "$MODEL"
                            echo "--- Converting Parquet to CSV ---"
                            find "$OUTPUT_DIR" -type f -name "*.parquet" | while read -r parquet_file; do
                                python "$LOCAL_DIR/convert_parquet_to_csv.py" "$parquet_file"
                            done
                            echo "=== Analysis done for $MODEL_UNDERSCORE ==="
                        } 2>&1 | tee -a "$ANALYSIS_LOG" | tee -a "$MASTER_LOG" >/dev/null

                    done
                done
            done
        done
    done
done

# =========================
# ===== 最终 Summary  =====
# =========================
SUMMARY_LOG="$RUN_DIR/summary.log"
{
    echo "=== Summary (all models) @ $(date -Is) ==="
    for MODEL in "${MODELS[@]}"; do
        MODEL_UNDERSCORE="${MODEL//\//_}"
        RESULTS_PATH="$OUTPUT_DIR/$MODEL_UNDERSCORE/all_experiments_results.json"
        ANALYSIS_OUTPUT_PATH="$OUTPUT_DIR/$MODEL_UNDERSCORE/analysis_results.json"

        if [[ -f $RESULTS_PATH ]]; then
            python "$LOCAL_DIR/analyze_summary.py" \
                --results-path "$RESULTS_PATH" \
                --output-path "$ANALYSIS_OUTPUT_PATH"
            echo "  ✓ $MODEL -> $ANALYSIS_OUTPUT_PATH"
        else
            echo "  ✗ $MODEL -> 缺少结果文件：$RESULTS_PATH"
        fi
    done
    echo "=== Summary done ==="
} 2>&1 | tee -a "$SUMMARY_LOG" | tee -a "$MASTER_LOG" >/dev/null
