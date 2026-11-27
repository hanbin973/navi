# Linear Transformer experiment

## Objective and scope

Replace softmax attention in the harness baseline with deterministic normalized
ELU+1 kernel attention. This user-requested experiment includes implementation
and one baseline training submission. Existing softmax defaults remain unchanged.
The baseline is `harness/latest_baseline.json` (job 66844920), not the subsequent
512-context query sweep.

## Design

Reuse the existing Q/K/V/output projections, heads, normalization, feed-forward
layers, residual scaling, initialization keys, and parameter shapes. Add an
opt-in `--attention-backend linear` to the refine-iteration runner and dispatch
both `AttentionTokenMixer` and shared `_Attention` through one kernel helper.
This selector denotes an attention algorithm here, rather than a softmax engine.

For phi(x)=ELU(x)+1, form S=sum_j phi(k_j) v_j^T and z=sum_j phi(k_j),
then return phi(q_i)^T S / max(phi(q_i)^T z, 1e-6).
Use FP32 features, summaries, and division. No token-by-token score matrix is
formed. The state mixer is unmasked self-attention. The baseline query decoder
processes each query independently and supplies a per-head log key bias;
incorporate exp(bias-max(bias)) into key features. This preserves learned pair
biases without a quadratic intermediate. Reject general query-dependent bias
and pairwise/causal masks; key-only masks and empty/fully masked memory are
supported, with zero output for empty memory.

This is a different kernel, not a softmax approximation. Compared with random
feature attention it needs no feature-rank choice or extra random state. Compared
with changing only the state mixer it tests linear attention throughout the
baseline. Decoder memory remains query-dependent, so this does not promise
constant-time decoding across all queries. FP32 accumulation differs from the
baseline cuDNN bfloat16 attention; record that difference in run metadata.

The local derivation and primary-paper links are in
`impl/transformer/linear-attention.md`. No new numerical dependency is needed.
Checkpoint leaf layouts remain unchanged; the existing saved backend selector
reconstructs the appropriate algorithm during replay.

## Work and acceptance criteria

1. Implement the helper, dispatch, and CLI choice without altering softmax paths.
2. Test JIT outputs and gradients against explicit pairwise evaluation of the
   same ELU+1 kernel, including key bias, masking, and batching; exercise empty
   memory, rejected pairwise masks, state-mixer permutation symmetry, and the
   complete refine-iteration objective/replay path.
3. Run focused tests on an allocated compute node, plus Ruff and diff checks.
4. Submit one training job with the exact baseline CLI settings except linear
   attention and a two-hour time limit (carried over from this session).
   Preserve seed 0, 504 contexts, 8 queries, n=512, width 32, state width 16,
   4 heads, identity-tanh draw/row bases, tanh state read, attention mixer depth
   1, query depth 2, running-mean integrator, batch/microbatch 32, 10,000 steps,
   learning rate 0.005, 128 test episodes, t_max=32, eval multiplier 16, k=32,
   128 features, alpha=1, and all other baseline options. Request one GPU,
   2 CPUs, 32 GB host RAM, account jonth0, using the common launcher.
5. Compare held-out NLL and Pearson curves, training/evaluation time, and peak
   GPU memory with the baseline after completion; do not claim improvement
   or success from submission alone.

## Progress

- Implemented the common ELU+1 kernel and both dispatch paths; added CLI selection.
- GPU verification job 66880196: 21 tests passed in 59.58 seconds, covering
  pairwise output/gradient parity, batching, masking, permutation symmetry,
  checkpoint roundtrip, and existing softmax regression tests.
- GPU job 66880238: final rollout-loss/gradient check passed (1 test, 19.79 seconds).
- Repository Ruff check passed. All changed Python files pass Ruff formatting.
  Whole-repository formatting reports an existing unrelated issue at
  `impl/optax_integrator/integrators.py:105`; left unchanged.
- `git diff --check` and submission shell syntax check passed.
- Direct diff review found no actionable in-scope issues. Numerical evidence
  does not establish training quality or speedup; compare completed run artifacts.
- Reproducible baseline submission is in `impl/linear_transformer/submit.sh`.
  Training job **66880252** submitted with the recorded baseline settings and
  two-hour limit. Output: `runs/refine-iteration-66880252/`; log:
  `slurm-66880252.out`. Training completion and quality comparison are pending.

## Performance optimization ablation (2026-11-27)

User authorized direct single-query contraction and mixed-precision attention
operands, plus short probes. Keep `linear` unchanged for checkpoint replay.
Add `linear-direct`, `linear-bf16`, and `linear-direct-bf16` backend selectors.
Direct contraction applies only at query length one; state self-attention keeps
associative summaries. BF16 modes cast dot operands, including intermediate
summary/scores, to BF16 with explicit FP32 accumulation; feature maps, log-key
weights, key sums, and normalization stay FP32. Model parameters stay FP32.

Verification: single- and multi-query outputs and input/bias gradients against
an independent explicit kernel, mapped execution, empty/fully masked memory,
end-to-end rollout gradients, saved-backend checkpoint reconstruction, and
existing softmax tests. GPU tests in progress; investigate any mismatch first.

Probe runner: `experiments/refine_iteration/probe_linear_attention.py`.
One allocation runs the four linear modes and a cuDNN timing control sequentially
in fresh processes on the same GPU. Snapshot baseline metadata, load identical
trained baseline weights, create fresh Adam state, retain the 10,000-step LR
schedule, warm up 20 updates, and time five blocks of 20 updates. Seeds,
episodes, horizons, batch/microbatch 32, and model architecture match across
modes. Timings synchronize all returned arrays and exclude warm-up compilation.
Record training peak memory before evaluation. Evaluate 16 held-out episodes at
prefix 32 afterward. These are continuation/speed probes, not full retraining
or quality proof; the cuDNN control starts from linear-trained weights.

Status: implementation complete. GPU test job 66885140 passed 31 checks and
found one overly strict BF16 batching assertion (maximum absolute discrepancy
0.00025618; FP32-reference output and gradient checks passed). The batching
assertion now uses BF16-aware tolerance. Final verification job 66885286 passed all 35 tests in 102.16 seconds.
Performance ablation job: 66885309, all five variants finished after successful verification, with
20-minute limit on spgpu. Artifacts: `runs/linear-opt-probe-66885286/`.
Ruff, changed-file formatting, and diff whitespace checks passed.
The harness baseline is not changed by this ablation.

Final measurements, limitations, and recommendations are in
`impl/linear_transformer/optimization-results.md`. Both optimization controls are
implemented and verified; the harness baseline remains unchanged.

User selected `linear-direct-bf16` as the default for new refine-iteration runs.
Structured runs using this backend default to full-batch episode microbatching,
matching the probes. Existing baseline metadata and saved backend choices stay
unchanged.
