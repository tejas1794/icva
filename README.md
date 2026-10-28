# I-CVA: Identity-Initialized Cross-View Adaptation for Compact Multimodal Models

Code and per-question predictions for the paper *I-CVA: Identity-Initialized
Cross-View Adaptation for Compact Multimodal Models* (Journal of Imaging,
under review).

I-CVA is a recurrent gated cross-attention module placed between the vision
encoder and the language model of a frozen multimodal backbone. Its output is
added through a residual whose scale is initialized to zero, so training starts
from exactly the pretrained model, and inputs with fewer than two images bypass
it. The paper evaluates it on Gemma-4-E4B-it and Qwen2.5-VL-3B-Instruct against
the pretrained models and an independently trained LoRA baseline.

## Contents

```
verify.py                     recomputes the paper's headline numbers from logs/
code/
  model_gca_v3.py             the coupler (Sections 3.2-3.3)
  train_v3.py                 coupler training over a frozen backbone
  train_lora_only.py          the LoRA baseline
  measure_latency.py          inference memory and latency measurement
  vlmeval_hooks/
    qwen_coupler_hook.py      where the coupler attaches on Qwen2.5-VL
    gemma_coupler_hook.py     where the coupler attaches on Gemma-4
latency/                      paired latency and peak-memory measurement
analysis/
  paired_analysis.py          paired bootstrap, McNemar and sign tests
  suite_matched_pairs.py      matched coupler-on/off comparison across benchmarks
  sat_answer_prior.py         training-corpus answer distribution audit
  eval_sat_v2.py              training-distribution accuracy harness
scripts/                      evaluation driver scripts
logs/                         per-question predictions, one directory per condition
```

## Verifying the reported numbers

No GPU and no model weights are required:

```bash
pip install pandas numpy openpyxl
python verify.py
```

This recomputes, directly from `logs/`, the LoRA regression (Section 4.1), the
convex-gate ablation (4.2), the three predeclared BLINK populations (4.3), the
avoided damage on Qwen (4.4), the matched inference-path control (4.6) and the
refinement-depth sweep (4.7). Rows marked `OK` match the value reported in the
paper.

Point estimates and win/loss counts reproduce exactly. Bootstrap interval
endpoints may differ from the paper by one resampling lattice step: the paper's
intervals come from `analysis/paired_analysis.py`, which draws from a single RNG
stream shared across all contrasts, and `verify.py` does not replicate that draw
order. On small populations the effect is visible (at n=133 a percentile moves
in steps of 1/133 = 0.75 points), so the Gemma target interval reads
[-3.01,+8.27] here against [-3.76,+8.27] in the paper. Run
`analysis/paired_analysis.py` to reproduce the published endpoints exactly.

## Per-question predictions

Each condition directory holds the evaluator's outputs per benchmark:

- `*_acc.csv`: per-category accuracy
- `*_exact_matching_result.xlsx` / `*_extract_matching.xlsx`: one row per
  question, with `index`, `category`, the question and options, the model's
  `prediction`, the ground-truth `answer` and `hit`

Conditions are joined by `index`, never by row position. All differences in the
paper are paired at the item level over the intersection of two conditions'
indices. Benchmark images are not redistributed; all benchmarks are available
through the EASI evaluation suite.

| Directory | Condition |
|---|---|
| `qwen_base`, `gemma_base` | pretrained model |
| `qwen_lora_only`, `gemma_lora_only` | independently trained rank-16 LoRA, no coupler |
| `gemma_lora_only_step5000` | Gemma LoRA at 5,000 steps (Section 4.1.1) |
| `qwen_icva`, `gemma_icva` | I-CVA: coupler over a frozen backbone |
| `qwen_cotrained_on` / `_off` | same checkpoint, coupler enabled / bypassed (Section 4.6) |
| `gemma_matched` | Gemma matched suite on MindCube, EmbSpatialBench and ViewSpatialBench |
| `qwen_ksweep` | refinement depth K in {1, 2, 5} at fixed weights (Section 4.7) |
| `gemma_gate` | convex-gate ablation (Section 4.2) |

On Gemma, the I-CVA results for MindCube, EmbSpatialBench and ViewSpatialBench
are the `*_with_LoRA_*` files in `gemma_matched`. They come from the same
checkpoint as `gemma_icva`; the co-trained LoRA is inert on Gemma (Section 3.4.1).

## Training

Training runs on a single 24 GB GPU:

```bash
python code/train_v3.py --backbone qwen --steps 10000        # coupler
python code/train_lora_only.py --backbone qwen --steps 10000 # LoRA baseline
```

Weights & Biases logging defaults to offline; set `WANDB_MODE=online` and
`WANDB_PROJECT` to enable it.

## Evaluation

Evaluation uses VLMEvalKit under the EASI benchmark suite. The coupler-specific
evaluator code is in `code/vlmeval_hooks/`: it attaches the coupler to the
visual token stream and leaves the input unchanged when fewer than two images
are present (the single-image bypass of Section 3.2.4). The driver scripts in
`scripts/` read `PROJECT_ROOT`, `LMUData` and `PYTHON_BIN` from the environment.

## Model weights

Model weights are not included in this repository (a single coupler checkpoint
is 357-469 MB). They are available from the authors on request.

## Notes

- Training runs are single-seed; the item-level bootstrap quantifies evaluation
  uncertainty at a fixed checkpoint, not training-seed variation.
- The training stream is not pinned to a dataset revision. A cached
  reconstruction contains 172,384 rows, of which 20,660 are multi-image eligible.
- The convex-gate condition (`logs/gemma_gate`) ships with per-question
  predictions; its checkpoint was not retained.

## License

MIT License; see `LICENSE`.
