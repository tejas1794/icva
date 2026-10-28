#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the directory containing code/}"
EVAL_ROOT="$RELEASE_ROOT/code/VLMEvalKit"
VENV_PYTHON="${PYTHON_BIN:-python}"

export PYTHONPATH="$EVAL_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LMUData="${LMUData:?set LMUData to the EASI benchmark data directory}"

cd "$EVAL_ROOT"
exec "$VENV_PYTHON" run.py \
  --data BLINK MMSIBench_wo_circular \
  --model ICVAGemma4_40k_with_LoRA \
  --work-dir "$RELEASE_ROOT/outputs/gemma40k_full_authoritative" \
  --judge exact_matching
