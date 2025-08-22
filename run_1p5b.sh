#!/usr/bin/env bash
# =========================
# Clean & Robust Sweep Runner (eval-only, save real system prompt)
# =========================
set -euo pipefail
IFS=$'\n\t'

# -------------------------
# conda environment setup
# -------------------------
source "/code-fsx/yibiaoy-sandbox/miniconda3/etc/profile.d/conda.sh"
conda activate soberplus

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/checkpoints-fsx/yibiaoy-sandbox/HF"

unset VLLM_ATTENTION_BACKEND  # 后端默认 unset，按 dtype 动态设置

# -------------------------
# 目录配置
# -------------------------
readonly LOCAL_DIR="/code-fsx/yibiaoy-sandbox/SoberReasoningPlus"
readonly OUTPUT_DIR="$LOCAL_DIR/outputs"
readonly LOG_ROOT="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

# -------------------------
# System Prompt（从 JSON 中读取）
# -------------------------
export PROMPTS_PATH="$LOCAL_DIR/prompts.json"
SYSTEM_PROMPT="$(
python - <<'PY'
import json, os
p = os.environ.get("PROMPTS_PATH")
try:
    with open(p, "r") as f:
        j = json.load(f)
    sp = j.get("SYSTEM_PROMPT", None)
    if not isinstance(sp, str) or len(sp.strip()) == 0:
        print("", end="")
    else:
        print(sp)
except Exception:
    print("", end="")
PY
)"
if [[ -z "${SYSTEM_PROMPT}" ]]; then
  echo "[FATAL] SYSTEM_PROMPT 读取失败：请检查 $PROMPTS_PATH 中的 SYSTEM_PROMPT 键。" >&2
  exit 1
fi

# -------------------------
# Sweep 配置
# -------------------------
MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

TOP_PS=(
  0.6
  0.8
  0.9
  0.95
  1.0
)

TEMPS=(
  0.0
  0.6
  0.7
  0.8
  0.9
  1.0
)

DTYPES=(
  "bfloat16"
  "float32"   # 如果要跑 FP32，打开本行，会自动设 VLLM_ATTENTION_BACKEND=XFORMERS
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

readonly MAX_NUM_SEQS=128
readonly MAX_NUM_BATCHED_TOKENS=131072

TP=4
PP=1
DP=1

# GPU 数量校验
NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 0)
if (( NUM_GPU == 0 )); then
  echo "[FATAL] 未检测到 GPU，请检查环境。" >&2
  exit 1
fi
if (( TP > NUM_GPU )); then
  echo "[WARN] 要求 TP=$TP 但仅有 $NUM_GPU 张 GPU，自动下调为 TP=$NUM_GPU"
  TP=$NUM_GPU
fi

# -------------------------
# 任务与 seeds
# -------------------------
declare -A TASK_SEEDS=(
  ["aime24"]=32
  ["aime25"]=32
  ["amc23"]=32
  ["minerva"]=10
  ["olympiadbench"]=10
  ["math_500"]=10
)

TASK_NAMES=(
  "aime24"
  "aime25"
  "amc23"
  "math_500"
  "minerva"
  "olympiadbench"
  # "gpqa:diamond"
)

# -------------------------
# 日志与工具函数
# -------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_ID"
MASTER_LOG="$RUN_DIR/master.log"
mkdir -p "$RUN_DIR"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

dump_env_snapshot() {
  {
    echo "==== RUN_ID: $RUN_ID ===="
    echo "Start: $(date -Is)"
    echo "HOST: $(hostname)"
    echo "CWD: $(pwd)"
    echo "--- GPU Info ---"
    nvidia-smi || true
    echo "--- Key Python pkgs ---"
    pip freeze | grep -Ei '^(torch|triton|vllm|xformers|flash-attn|flashinfer|cuda)=' || true
    echo "--- Env snapshot (filtered) ---"
    env | grep -E -i 'NCCL|OMP|VLLM|CUDA|HF_HOME' || true
  } | tee -a "$MASTER_LOG"
}

trap 'echo "End: $(date -Is)" | tee -a "$MASTER_LOG"' EXIT

# 根据 dtype 设置后端
set_attention_backend() {
  local dtype="$1"
  if [[ "$dtype" == "float32" ]]; then
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    log "[INFO] DTYPE=float32 → 使用 VLLM_ATTENTION_BACKEND=XFORMERS"
  else
    unset VLLM_ATTENTION_BACKEND
  fi
}

# -------------------------
# 参数快照写入（每个实验一个 args.json）
# -------------------------
write_args_json() {
  local path="$1"
  python - <<'PY' >"$path"
import os, json

def num(v, default=None):
    if v is None: return default
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return v

cfg = {
    "model": os.environ.get("ARG_MODEL",""),
    "task": os.environ.get("ARG_TASK",""),
    "temperature": num(os.environ.get("ARG_TEMPERATURE")),
    "top_p": num(os.environ.get("ARG_TOP_P")),
    "output_dir": os.environ.get("ARG_OUTPUT_DIR",""),
    "max_new_tokens": num(os.environ.get("ARG_MAX_NEW_TOKENS")),
    "max_model_length": num(os.environ.get("ARG_MAX_MODEL_LENGTH")),
    "dtype": os.environ.get("ARG_DTYPE",""),
    "max_num_seqs": num(os.environ.get("ARG_MAX_NUM_SEQS")),
    "max_num_batched_tokens": num(os.environ.get("ARG_MAX_NUM_BATCHED_TOKENS")),
    "tp": num(os.environ.get("ARG_TP")),
    "pp": num(os.environ.get("ARG_PP")),
    "dp": num(os.environ.get("ARG_DP")),
    "seed": num(os.environ.get("ARG_SEED")),
    # 默认写真实的 system prompt
    "system_prompt": os.environ.get("SYSTEM_PROMPT", "")
}
print(json.dumps(cfg, indent=2, ensure_ascii=False))
PY
}

# -------------------------
# 主运行函数：执行单个实验
# -------------------------
run_one() {
  local model="$1" task_name="$2" seed="$3" temp="$4" top_p="$5" max_tokens="$6" max_model_length="$7" dtype="$8"

  local task="custom|${task_name}|0|0"
  local model_us="${model//\//_}"

  local exp_dir="$OUTPUT_DIR/$model_us/$task_name/seed_${seed}/temp_${temp}__top_p_${top_p}__maxlen_${max_tokens}"
  mkdir -p "$exp_dir"

  local run_log="$exp_dir/run.log"
  local args_json="$exp_dir/args.json"

  export ARG_MODEL="$model"
  export ARG_TASK="$task"
  export ARG_TEMPERATURE="$temp"
  export ARG_TOP_P="$top_p"
  export ARG_OUTPUT_DIR="$OUTPUT_DIR"
  export ARG_MAX_NEW_TOKENS="$max_tokens"
  export ARG_MAX_MODEL_LENGTH="$max_model_length"
  export ARG_DTYPE="$dtype"
  export ARG_MAX_NUM_SEQS="$MAX_NUM_SEQS"
  export ARG_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
  export ARG_TP="$TP"
  export ARG_PP="$PP"
  export ARG_DP="$DP"
  export ARG_SEED="$seed"
  export SYSTEM_PROMPT="$SYSTEM_PROMPT"

  set_attention_backend "$dtype"

  write_args_json "$args_json"

  {
    echo "---- $(date -Is) RUN START ----"
    set -x
    python "$LOCAL_DIR/main.py" \
      --model "$model" \
      --task "$task" \
      --temperature "$temp" \
      --top_p "$top_p" \
      --output_dir "$OUTPUT_DIR" \
      --max_new_tokens "$max_tokens" \
      --max_model_length "$max_model_length" \
      --custom_tasks_directory "$LOCAL_DIR/lighteval_tasks.py" \
      --system_prompt "$SYSTEM_PROMPT" \
      --use_chat_template \
      --dtype "$dtype" \
      --max_num_seqs "$MAX_NUM_SEQS" \
      --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
      --tensor_parallel_size "$TP" \
      --pipeline_parallel_size "$PP" \
      --data_parallel_size "$DP" \
      --seed "$seed"
    set +x
    echo "---- $(date -Is) RUN END ----"
  } 2>&1 | tee -a "$run_log" | tee -a "$MASTER_LOG" >/dev/null
}

# -------------------------
# ========= 执行 Sweep（评测）=========
# -------------------------
dump_env_snapshot

# 校验：MAX_MODEL_LENGTHS 与 MAX_TOKENS_LIST 必须一一对应
if (( ${#MAX_MODEL_LENGTHS[@]} != ${#MAX_TOKENS_LIST[@]} )); then
  echo "[FATAL] MAX_MODEL_LENGTHS(${#MAX_MODEL_LENGTHS[@]}) 与 MAX_TOKENS_LIST(${#MAX_TOKENS_LIST[@]}) 长度不一致，无法一一对应。" >&2
  exit 1
fi

# 额外校验：自定义任务文件存在且含 TASKS_TABLE（轻量正则）
CUSTOM_TASK_FILE="$LOCAL_DIR/lighteval_tasks.py"
if [[ ! -f "$CUSTOM_TASK_FILE" ]]; then
  echo "[FATAL] 未找到自定义任务文件：$CUSTOM_TASK_FILE" >&2
  exit 1
fi
if ! grep -q "TASKS_TABLE" "$CUSTOM_TASK_FILE"; then
  echo "[FATAL] $CUSTOM_TASK_FILE 中未发现 TASKS_TABLE 定义（lighteval 需要）。" >&2
  exit 1
fi

cd "$LOCAL_DIR"

# 逐索引配对 max_model_length 与 max_tokens（确保一一对应）
for i in "${!MAX_MODEL_LENGTHS[@]}"; do
  max_model_length="${MAX_MODEL_LENGTHS[$i]}"
  max_tokens="${MAX_TOKENS_LIST[$i]}"

  for model in "${MODELS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
      for top_p in "${TOP_PS[@]}"; do
        for temp in "${TEMPS[@]}"; do
          log ">>> RunPlan: model=$model task_list=${TASK_NAMES[*]} seeds=VAR temp=$temp top_p=$top_p max_tokens=$max_tokens dtype=$dtype max_model_length=$max_model_length"
          for task_name in "${TASK_NAMES[@]}"; do
            if [[ -z ${TASK_SEEDS[$task_name]+x} ]]; then
              log "[FATAL] 任务 $task_name 未配置 seeds"
              exit 1
            fi
            num_runs="${TASK_SEEDS[$task_name]}"
            for ((seed=0; seed<num_runs; seed++)); do
              run_one "$model" "$task_name" "$seed" "$temp" "$top_p" "$max_tokens" "$max_model_length" "$dtype"
            done
          done
        done
      done
    done
  done
done

log ">>> Sweep 完成，所有实验已执行。"
