#!/usr/bin/env bash
set -euo pipefail
# Qwen coupler-only (no LoRA) at default processor pixel bounds, to be paired
# against the existing Qwen2_5_VL_3B_BaseMatched run. Both arms use the same
# processor settings, so this is a matched adapter-only contrast mirroring the
# Gemma configuration. Inference only.

RELEASE_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the directory containing code/}"
EVAL_ROOT="$RELEASE_ROOT/code/VLMEvalKit"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID=qwen_coupler_only_matched_20260826
OUTPUT_ROOT="$RELEASE_ROOT/outputs/$RUN_ID"
LOG_ROOT="$OUTPUT_ROOT/logs"

export PYTHONPATH="$EVAL_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LMUData="${LMUData:?set LMUData to the EASI benchmark data directory}"
mkdir -p "$LOG_ROOT"

{
  echo "run_id=$RUN_ID"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=ICVAQwen2_5_VL_3B (coupler only, lora_path=None)"
  echo "pair_against=outputs/qwen3b_base_matched_authoritative (Qwen2_5_VL_3B_BaseMatched)"
  echo "processor_bounds=defaults on both arms"
  sha256sum "$RELEASE_ROOT/code/model_gca_v3.py" \
            "$RELEASE_ROOT/code/VLMEvalKit/vlmeval/config.py" \
            "$RELEASE_ROOT/code/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py"
} > "$OUTPUT_ROOT/RUN_MANIFEST.txt"

run_eval() {
  local bench="$1"
  local done_file="$OUTPUT_ROOT/$bench.done"
  [[ -f "$done_file" ]] && { echo "[skip] $bench"; return 0; }
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $bench"
  cd "$EVAL_ROOT"
  "$PYTHON_BIN" run.py --data "$bench" --model ICVAQwen2_5_VL_3B \
    --work-dir "$OUTPUT_ROOT/$bench" --judge exact_matching \
    > "$LOG_ROOT/$bench.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$done_file"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done  $bench"
}

run_eval MMSIBench_wo_circular
run_eval BLINK
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_ROOT/ALL_DONE"
echo "ALL DONE"
