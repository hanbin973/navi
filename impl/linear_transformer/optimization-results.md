# Attention optimization probe results

GPU verification job **66885286**: 35 tests passed in 102.16 seconds.
Performance job **66885309**: all five modes finished, on one NVIDIA A40
(node gl1503), with a fresh process for each mode. JAX 0.12.0.1.
Raw artifacts: `runs/linear-opt-probe-66885286/<backend>/probe.json`.

## Protocol

Baseline job 66880252: 504 contexts, 8 queries, n=512, kernel rank 128,
batch and microbatch 32; other architecture/training settings preserved.
Load identical trained baseline model leaves, create fresh Adam state, and retain
the original 10,000-step warmup-cosine schedule. Use identical random keys,
episodes, and horizons across modes. Warm up 20 updates, then time five blocks
of 20 updates with full device synchronization. Exclude warmup/compilation
from step timings. Record device allocator training peak before evaluation,
including compilation and initialization, rather than total reserved GPU memory.
Evaluate the same first 16 held-out episodes at rollout prefix 32 afterward.

The cuDNN control also starts from linear-trained weights, so its NLL is not a
comparison with a trained softmax model. This probe measures a short continuation,
not independent training quality or long-horizon evaluation. There is one sequence
of five blocks per mode; small timing differences should not be overinterpreted.

## Measurements

| Backend | ms/update | Peak training GiB | Held-out NLL |
|---|---:|---:|---:|
| linear | 272.72 | 13.77 | 0.889330 |
| linear-direct | 275.69 | 13.84 | 0.889305 |
| linear-bf16 | 258.34 | 12.69 | 0.889510 |
| linear-direct-bf16 | 254.67 | 12.75 | 0.889374 |
| cudnn | 222.53 | 9.87 | timing control only |

## Interpretation

BF16 alone reduces update time by 5.27% and peak memory by 7.87% relative to the original linear path.

Direct contraction alone is 1.09% slower and uses 0.51% more peak memory; it did not produce the expected end-to-end benefit. It remains opt-in. This ablation does not establish whether the decoder was insignificant or compiler lowering erased the theoretical gain.

Combined mode reduces update time by 6.62% and memory by 7.43% versus original linear. Its additional time reduction over BF16 alone is 1.42%, with slightly higher memory; that small difference needs repetition before treating it as reliable.

Even the fastest linear mode remains 14.45% slower and uses 29.16% more peak memory than the cuDNN timing control. The two implemented changes do not close the gap. Mixed modes still retain FP32 surrounding projections, feature-map computation, normalization, and their autodiff state. Further attribution would require profiling rather than another unmeasured optimization.

Use `--attention-backend linear-bf16` for the measured memory improvement, or `linear-direct-bf16` for the fastest result in this probe. Existing `linear` semantics and the harness baseline remain unchanged. No full retraining was performed.

Repository Ruff, changed-file formatting, and `git diff --check` passed. Direct review of the implementation and probe found no remaining actionable in-scope issues.
