#!/bin/bash
# End-to-end A/B for fused_linear (design v2: Q7 acceptance A/B and the PR 7
# 500-step loss-curve gate), DFlash or DSpark.
#
# Arms share data, seed and hyper-parameters; only the loss path differs:
#   eager    --loss-implementation eager             (reference, may OOM at 152k)
#   fused    --loss-implementation fused             (today: bf16 logits)
#   fl       --loss-implementation fused_linear      (input tiles; full vocab:
#                                                     shared-head delta form)
#   fl_fp16  ... --fused-linear-logits float16
#   fl_fp32  ... --fused-linear-logits float32       (precision ceiling)
# Precision matters (#788) if acceptance orders fl_fp32 ~ fl_fp16 >= fl > fused,
# growing with training length; full vocabulary only (a draft vocabulary
# keeps distinct heads and today's logit numerics for `fl`).
#
# Two modes:
#   MAX_STEPS=500 (curves): train only, compare train.log losses / accuracies
#   MAX_STEPS=    (acceptance): full EPOCHS, then serve and evaluate each arm
#
# Needs vLLM, guidellm (scripts/evaluate/requirements.txt) and 2+ GPUs
# (VLLM_GPUS serve hidden states, TRAIN_GPUS train).
# Usage (from the speculators repo root):
#   MODEL_TYPE=dflash ARMS="fused fl fl_fp16" bash scripts/fused_linear_ab/ab_fused_linear.sh
#   MODEL_TYPE=dspark MAX_STEPS=500 ARMS="eager fused fl" bash scripts/fused_linear_ab/ab_fused_linear.sh
set -euo pipefail

# ============ Configuration ============
MODEL_TYPE="${MODEL_TYPE:-dflash}"   # dflash | dspark
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-hf:inference-optimization/speculators-ci-datasets:tutorial_regen}"
OUT="${OUT:-./output/ab_fused_linear_${MODEL_TYPE}}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
EPOCHS="${EPOCHS:-10}"              # #788's gap grew with training
MAX_STEPS="${MAX_STEPS:-}"           # set (e.g. 500) for the loss-curve gate
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
LOSS_FN="${LOSS_FN:-tv}"             # cancelling-gradient loss: precision matters
WEIGHTING="${WEIGHTING:-fixed-exp-decay}"  # dpace requires LOSS_FN=ce
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-}"   # empty = full vocabulary (shared head)
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-3072}"
NUM_LAYERS="${NUM_LAYERS:-5}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"
MARKOV_RANK="${MARKOV_RANK:-256}"    # dspark
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
ARMS="${ARMS:-fused fl fl_fp16}"
EVAL_DATASET="${EVAL_DATASET:-RedHatAI/speculator_benchmarks}"
EVAL_SUBSETS="${EVAL_SUBSETS:-HumanEval,qa}"
VLLM_PORT=8000
EVAL_PORT=8108
VLLM_GPUS="${VLLM_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
# =======================================

arm_args() {
    case "$1" in
        eager) echo "--loss-implementation eager" ;;
        fused) echo "--loss-implementation fused" ;;
        fl) echo "--loss-implementation fused_linear" ;;
        fl_fp16) echo "--loss-implementation fused_linear --fused-linear-logits float16" ;;
        fl_fp32) echo "--loss-implementation fused_linear --fused-linear-logits float32" ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

model_args=(--speculator-type "$MODEL_TYPE")
if [[ "$MODEL_TYPE" == dspark ]]; then
    model_args+=(--markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE"
        --enable-confidence-head --confidence-head-with-markov)
fi
vocab_args=()
prep_args=()
if [[ -n "$DRAFT_VOCAB_SIZE" ]]; then
    vocab_args=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
    prep_args=(--no-skip-token-freq)
fi
step_args=()
[[ -n "$MAX_STEPS" ]] && step_args=(--max-steps "$MAX_STEPS")

wait_ready() {
    until curl -sf "http://localhost:$1/health" > /dev/null 2>&1; do sleep 2; done
}

echo "=== Hidden-state server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
wait_ready "$VLLM_PORT"

echo "=== Prepare data (shared by all arms) ==="
speculators prepare-data --model "$MODEL" --data "$DATASET" --output "$OUT/data" \
    --max-samples "$MAX_SAMPLES" --seq-length "$SEQ_LENGTH" "${prep_args[@]}" \
    --render-endpoint "http://localhost:${VLLM_PORT}"

for arm in $ARMS; do
    echo "=== Train arm: $arm ==="
    mkdir -p "$OUT/$arm"
    # shellcheck disable=SC2046
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
        -m speculators.train \
        --verifier-name-or-path "$MODEL" --data-path "$OUT/data" \
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
        --save-path "$OUT/$arm/checkpoints" "${vocab_args[@]}" "${model_args[@]}" \
        --epochs "$EPOCHS" "${step_args[@]}" --lr "$LR" --seed "$SEED" \
        --total-seq-len "$SEQ_LENGTH" --block-size "$BLOCK_SIZE" \
        --max-anchors "$MAX_ANCHORS" --num-layers "$NUM_LAYERS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --loss-fn "$LOSS_FN" --per-position-loss-weight "$WEIGHTING" \
        $(arm_args "$arm") --on-missing generate \
        2>&1 | tee "$OUT/$arm/train.log"
done
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

if [[ -n "$MAX_STEPS" ]]; then
    echo "Loss-curve mode: compare loss / per-position accuracy in $OUT/*/train.log."
    exit 0
fi

for arm in $ARMS; do
    ckpt=$(ls -d "$OUT/$arm/checkpoints"/*/ | sort -V | tail -1)
    echo "=== Evaluate arm: $arm ($ckpt) ==="
    vllm serve "$ckpt" --port "$EVAL_PORT" &
    EVAL_PID=$!
    wait_ready "$EVAL_PORT"
    python scripts/evaluate/evaluate.py --target "http://localhost:${EVAL_PORT}/v1" \
        --dataset "$EVAL_DATASET" \
        --data-column-mapper "kind=generative_column_mapper,column_mappings.text_column=prompt" \
        throughput --subsets "$EVAL_SUBSETS" --max-requests 80 \
        2>&1 | tee "$OUT/$arm/eval.log"
    kill "$EVAL_PID" 2>/dev/null || true
    wait "$EVAL_PID" 2>/dev/null || true
done
echo "Compare acceptance length / per-position acceptance in $OUT/*/eval.log."
