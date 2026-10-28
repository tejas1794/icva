#!/usr/bin/env bash
set -euo pipefail

# Gemma-4 checkpoint-40k matched suite, inference only, no retraining.
# For each benchmark the coupler-enabled and coupler-bypassed arms are run
# back to back so a complete matched pair is available as early as possible.
# Benchmarks are ordered by observed cost (cheapest first).

RELEASE_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the directory containing code/}"
EVAL_ROOT="$RELEASE_ROOT/code/VLMEvalKit"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID=gemma40k_matched_suite_20260826
OUTPUT_ROOT="$RELEASE_ROOT/outputs/$RUN_ID"
LOG_ROOT="$OUTPUT_ROOT/logs"

export PYTHONPATH="$EVAL_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LMUData="${LMUData:?set LMUData to the EASI benchmark data directory}"
mkdir -p "$LOG_ROOT"

{
  echo "run_id=$RUN_ID"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "release_commit=$(git -C "$RELEASE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "purpose=authoritative Gemma 40k rows + matched coupler bypass control"
  echo "already_complete_elsewhere=BLINK,MMSIBench_wo_circular"
  echo "--- worktree status ---"
  git -C "$RELEASE_ROOT" status --short 2>/dev/null || true
  echo "--- implementation hashes ---"
  sha256sum \
    "$RELEASE_ROOT/code/model_gca_v3.py" \
    "$RELEASE_ROOT/code/VLMEvalKit/vlmeval/config.py" \
    "$RELEASE_ROOT/code/VLMEvalKit/vlmeval/vlm/gemma.py"
} > "$OUTPUT_ROOT/RUN_MANIFEST.txt"

run_eval() {
  local model_name="$1" output_label="$2" bench="$3"
  local done_file="$OUTPUT_ROOT/$output_label.done"
  local log_file="$LOG_ROOT/$output_label.log"
  if [[ -f "$done_file" ]]; then
    echo "[skip] $output_label already complete"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $output_label"
  cd "$EVAL_ROOT"
  "$PYTHON_BIN" run.py --data "$bench" --model "$model_name" \
    --work-dir "$OUTPUT_ROOT/$output_label" --judge exact_matching \
    > "$log_file" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$done_file"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done  $output_label"
}

for bench in MindCubeBench_tiny_raw_qa EmbSpatialBench ViewSpatialBench; do
  run_eval ICVAGemma4_40k_with_LoRA            "on_${bench}"  "$bench"
  run_eval ICVAGemma4_40k_co_trained_LoRA_only "off_${bench}" "$bench"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_ROOT/ALL_DONE"
echo "ALL DONE"
