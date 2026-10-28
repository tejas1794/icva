# Latency and memory

`qwen_paired_ab_latency.json` -- produced by `measure_latency_paired.py`.

Qwen2.5-VL-3B-Instruct, single RTX 3090, bfloat16, generation capped at 5 new
tokens so the measurement isolates vision encoding and prefill, the only stage
on which the coupler executes.

One model is loaded once and the coupler hook is toggled between passes, so the
two arms share weights, decoding path, and thermal state. Each sequence length
gets 25 paired repetitions with the arm order alternating every repetition;
`paired_delta_ms` is the mean of the within-repetition differences and `se_ms`
its standard error.

| N images | refinements | coupler off | coupler on | paired delta | overhead |
|---|---|---|---|---|---|
|  2 |  1 |  355.1 ms |  358.9 ms |  +3.8 +/- 3.8 ms | +1.1% (n.s.) |
|  4 |  3 |  519.8 ms |  524.2 ms |  +4.4 +/- 3.6 ms | +0.8% (n.s.) |
| 16 | 15 | 1555.2 ms | 1600.5 ms | +45.3 +/- 13.9 ms | +2.9% |
| 32 | 31 | 2982.2 ms | 3056.6 ms | +74.4 +/- 30.8 ms | +2.5% |

At two and four images the difference is not resolved from zero. Where it is
resolved the cost is 2.4--3.0 ms per adjacent-pair refinement, linear in the
number of pairs.

Peak inference memory (deterministic, reproduced across independent runs):
7.107 -> 7.204 GB at 2 images, 8.676 -> 8.774 GB at 32. The increase is
+0.098 GB at every sequence length, which is the coupler's 52.46M bfloat16
parameters; refinement activations add nothing measurable.
