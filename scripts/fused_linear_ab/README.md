# fused_linear validation runs (throw-away branch)

This branch is the full stack (#1389 → #1390 → #1424 → #1425) plus the scripts below. It is for running the PR 7 gates of the design (v3) on a multi-GPU machine; nothing here is meant to be merged.

## 0. Setup

- Your usual speculators online-training environment (vLLM with hidden-state extraction, `scripts/evaluate/requirements.txt` for step 4), on this branch: `pip install -e .`
- Access to `Qwen/Qwen3-8B` (or set `MODEL=...` and a matching `TARGET_LAYER_IDS`).
- Run everything from the repo root. Set `HF_HUB_OFFLINE=1` if the machine cannot reach the Hub (models must then be cached).

Quick sanity check of the build (GPU, ~2 min):

```bash
pytest -q -p no:cacheprovider tests/unit/models/test_fused_linear_shared_head.py \
  tests/unit/models/test_dspark_fused_linear.py tests/unit/models/test_dflash_fused_linear.py
```

## 1. FSDP2 check (2 GPUs, ~2 min)

One training step of a tiny DFlash and DSpark model sharded as the trainer does; `fused_linear` must match `fused` (loss, every gradient after reduce-scatter, `markov_w2` included) and take the shared-head path.

```bash
torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py
torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py --replicate-frozen
torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py --compile
```

Expected: `PASS dflash` and `PASS dspark` for each. Send the output either way.

## 2. Smoke run of the A/B pipeline (~minutes)

Checks data preparation, the hidden-state server and both loss paths before the long runs. GPUs: `VLLM_GPUS` serve hidden states, `TRAIN_GPUS` / `NUM_TRAIN_GPUS` train (defaults `0,1` and `2,3` / `2`).

```bash
MODEL_TYPE=dflash MAX_STEPS=20 MAX_SAMPLES=200 ARMS="fused fl" \
  OUT=./output/ab_smoke bash scripts/fused_linear_ab/ab_fused_linear.sh
```

Note the step time printed in `output/ab_smoke/*/train.log` to size the runs below.

## 3. Loss curves, 500 steps (PR 7 gate)

Training only. Same data, seed and hyper-parameters per arm; compare loss and per-position accuracy.

```bash
# DFlash, TV loss (cancelling gradient: where precision shows)
MODEL_TYPE=dflash MAX_STEPS=500 LOSS_FN=tv ARMS="fused fl fl_fp16" \
  bash scripts/fused_linear_ab/ab_fused_linear.sh
# DSpark, its example loss
MODEL_TYPE=dspark MAX_STEPS=500 LOSS_FN='{"ce": 0.1, "tv": 0.9}' ARMS="fused fl fl_fp16" \
  bash scripts/fused_linear_ab/ab_fused_linear.sh
```

The design also asks for an `eager` arm. It materializes fp32 softmaxes over V = 152k and may OOM; run it separately so an OOM does not stop the others: `ARMS=eager` with the same variables.

## 4. Acceptance A/B (design Q7; decides the default tiles)

Full training, then each arm is served and evaluated. Long: start after step 3 looks sane.

```bash
# DFlash, TV loss: does precision raise acceptance?
MODEL_TYPE=dflash LOSS_FN=tv EPOCHS=10 ARMS="fused fl fl_fp16 fl_fp32" \
  bash scripts/fused_linear_ab/ab_fused_linear.sh
# DFlash default config (CE + D-PACE): only the tile dtype differs (CE has no shared-head form)
MODEL_TYPE=dflash LOSS_FN=ce WEIGHTING=dpace EPOCHS=10 ARMS="fused fl_fp16" \
  OUT=./output/ab_fused_linear_dflash_ce bash scripts/fused_linear_ab/ab_fused_linear.sh
```

If time is short: fewer epochs (`EPOCHS=3`) still shows the trend (#788's gap grew with training), and `fl_fp32` can be dropped.

Arms: `fused` = today (bf16 logits); `fl` = `fused_linear`, `input` tiles (full vocab: shared-head delta form, KL / TV gradient error 2–9 % vs 16–87 %); `fl_fp16` = fp16 tiles (~0.4 %); `fl_fp32` = fp32 tiles (~0.2 %, slowest).

## What to send back

- Step 1 output.
- For steps 2–4: every `$OUT/<arm>/train.log` and `$OUT/<arm>/eval.log`, plus `$OUT/<arm>/checkpoints/*/run.yaml` of one arm (the resolved config).
- GPU model, count and driver / torch / vLLM versions.
