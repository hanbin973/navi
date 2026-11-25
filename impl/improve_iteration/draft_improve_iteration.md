# Separate state and draw mixing in the fixed-point equation

This plan tracks implementation and verification on `refine-iteration`.
Commands are relative to the repository root.

## Reference behavior

On `refine-iteration` at `8d5575f`, [MixerEquation](../../navi/fixed_point.py)
encodes the latest iterate and observations once per block. Its per-draw
path concatenates each aligned draw value with that row's context and runs a
gated embedding, pre-norm mixer blocks, and a head. Under the block's
`vmap`, reverse-mode differentiation can retain these nonlinear activations
for each of the `K` draws. Even the mean-token setting runs the channel
network and layer norms for every draw.

On Git `main` at `22fbd6d`, `JointDrawConditioner` forms a joint score
by taking a mean over rows and multiplies that score by each row's draw
value. `AxialAttentionEquation` adds a deterministic context projection
to the mean of the projected draw contributions. Use that row-mean
structure inside the current mixer equation. Factorize the joint scorer
into context coefficients and scalar draw features. Keep the
`refine-iteration` state encoder and its latest-iterate read semantics.

## Proposed equation

Compose the draw path from a context coefficient network, a scalar draw
basis, and a row pool. The basis and row pool are independently configurable.
The equations below use the default four-feature basis and uniform row mean.

Let `N` denote context rows, `W` the context width, `R` basis functions,
`D` state channels, and `K` joint draws per block. The indices `i` and `j`
identify context rows, `k` identifies a draw, `r` identifies a basis
function, and `d` identifies a state channel.

```text
C[N,W]     = state_encoder(read(masked_latest), observations)
det[N,D]   = state_head(C)
A[N,R,D]   = coefficient_head(C).reshape(N,R,D)
B[k,j,r]   = basis_r(X[k,j])
s[k,:,:]   = row_pool(A, B[k])
s[k,r,d]   = mean_j(A[j,r,d] * B[k,j,r])  # Uniform mean pool
q[k,d]     = sum_r(s[k,r,d])
f[k,i,d]   = X[k,i] * q[k,d]
estimate  = det + mean_k(f[k])
```

| Variable | Meaning |
| --- | --- |
| `masked_latest[N,D]` | Latest committed iterate, or zeros if its history slot is invalid. |
| `read(masked_latest)[N,D]` | Iterate features after the read map, which is the identity in the current experiment. |
| `observations[N,1]` | Observed value for each context row, aligned with the iterate and draws. |
| `X[K,N]` | Joint prior draws. `X[k,i]` is draw `k`'s value at context row `i`. |
| `C[N,W]` | Context features, encoded once per block from the read iterate and observations. |
| `det[N,D]` | Deterministic contribution to the update estimate. |
| `A[N,R,D]` | Learned context coefficients. `A[j,r,d]` weights basis function `r` at row `j` for state channel `d`. |
| `B[K,N,R]` | Draw-only features. `B[k,j,r]` evaluates basis function `r` at `X[k,j]`. |
| `s[K,R,D]` | Pooled scores for each draw, basis function, and state channel. The default pool averages the coefficient-weighted features over rows `j`. |
| `q[K,D]` | Per-draw scores in state space after summation over basis functions. |
| `f[K,N,D]` | Conceptual stack of single-draw state contributions. |
| `estimate[N,D]` | Deterministic term plus the mean over joint draws, passed to the integrator. |

The previous model used one context coefficient per row and basis function.
A shared projection then sent each basis contribution along a fixed direction
in state space. The new head learns that direction from the context for each
row and basis function. For example, the identity feature can give a row
positive weight in one state channel and negative weight in another, and the
relative weights can change across rows. Even with `R=1`, the pooled score
can point in different directions across draws because the row coefficient
vectors can differ. The full formulation contains the previous uniform-mean
model as the special case `A[j,r,d] = A_old[j,r] * P[r,d]`. The deterministic
head keeps its bias.

The default basis is `[1, x, tanh(x), tanh(x + 1) - tanh(1)]` in the draw
dtype. The first two features supply a context-only score and a
context-dependent linear draw score. The remaining features make the score
nonlinear in the draw.

### Block reduction and memory

The mean pool computes all scores with a contraction over context rows.
Sum the resulting basis scores before pairing each joint draw with its
score in state space.

```text
S[K,R,D] = einsum("knr,nrd->krd", B, A) / N
Q[K,D]   = S.sum(axis=1)
mean_per_draw[N,D] = (X.T @ Q) / K
```

The final contraction pairs each row's draw value with the score of the
same joint draw and works for any nonempty block, including `K=1`.
This preserves cross-row dependence within each draw. The widened
coefficient head and its reverse-mode activations stay outside the draw
axis. The default block reducer contracts directly, avoiding explicit
`[K,N,R,D]` products and `[K,N,D]` contributions. Custom pools have their
own memory costs.

The coefficient buffer grows from `N*R` to `N*R*D` values, and the default
pool's score buffer grows from `K*R` to `K*R*D`. The basis buffer remains
`K*N*R`. At `N=504, K=32, R=4, D=16` in float32, the coefficient buffer
grows from about 8 KiB to 126 KiB per episode per step. The score buffer
grows from 0.5 KiB to 8 KiB, and the basis buffer is 252 KiB. An explicit
`[K,N,R,D]` product would be about 3.94 MiB at these dimensions. If the
rollout retains every coefficient buffer for a 16-episode microbatch and
32 steps, the coefficient buffers alone account for about 63 MiB. Actual
residuals and peak memory depend on JAX transformations and fusion.

The row-pooling contraction now takes `O(K*N*R*D)` work, up from
`O(K*N*R)`, and the final block contraction takes `O(K*N*D)`. The
coefficient head's output layer is wider and is still evaluated only once
per row per step. At hidden width 64, `R=4` and `D=16`, widening that layer
and removing the shared projection adds 3,836 trainable scalars.

For an isolated float32 CPU equation at `N=504, D=16, W=32`, compare the
original pre-refactor mixer at `8d5575f` with the current per-channel
mixer at `R=4`. Both used the same inputs and initialization seed, and
one mean-pooling state block. The original also used one mean-pooling
draw block. The table reports compiled temporary bytes for a scalar
squared loss and for its value and parameter gradient.

| Draws | Original forward bytes | Current forward bytes | Original value and gradient bytes | Current value and gradient bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 516,224 | 459,776 | 3,278,304 | 2,145,248 |
| 8 | 3,678,208 | 516,224 | 17,934,560 | 2,276,320 |
| 32 | 14,712,832 | 709,760 | 76,237,280 | 2,663,392 |

At 32 draws the current value-and-gradient estimate is about 96.5% lower
than the original, from 76.24 MB to 2.66 MB. The intermediate separable
model with a shared projection used 2.40 MB in the same setup. The
original mixer replicated gated embedding and mixer activations across
draws. The current mixer computes the widened coefficients once per context
row and keeps the draw-dependent work in the basis and contractions.
These are single-update, single-episode CPU compiler estimates. GPU peak
memory during differentiated training rollouts remains unmeasured.

## 1. Establish the baseline and compose the two paths

Implementation complete. Use the original mean-token draw configuration as the
performance baseline from commit `8d5575f`. Record one
representative training configuration before changing wiring, including
`N`, `K`, update horizon, episode microbatch size, dtype, and attention
backend. Use the same configuration for the memory comparison.

The configuration recorded by run 66809703 uses 504 context rows, 32 draws
per block, a 32-block training horizon, and an episode microbatch of 16.
State width is 16, token width is 32, state and draw depths are 1, and the
attention backend is cuDNN. The update boundary uses float32, with bfloat16
compute for cuDNN attention.

In [fixed_point.py](../../navi/fixed_point.py), use abstract Equinox
interfaces for the state and draw paths. Concrete modules own parameters and
compose existing channel mixers and state `MixerBlock` objects, following the
[repository Equinox pattern](../../harness/pattern/equinox.md).

| Object | Contract |
| --- | --- |
| `AbstractStatePath` | Encode masked history and observations to `C[N,W]`, and map `C` to the deterministic `[N,D]` term. The concrete state path owns the read, embedding, blocks, and head. |
| `AbstractDrawBasis` | Map one scalar draw value to `[R]` with a declared basis width. The default concrete basis supplies the four fixed features above. Other fixed or learned scalar feature maps can fill this role. |
| `AbstractDrawPool` | Map aligned coefficients `[N,R,D]` and basis values `[N,R]` to a row-permutation-invariant score `[R,D]` for one draw. Supply a block method producing `[K,R,D]` with the same semantics as mapping the single-draw method. The default concrete pool implements a uniform mean using a contraction. |
| `AbstractDrawPath` | Prepare a draw-free payload from `C`, evaluate one aligned draw using that payload, and reduce a block with the same semantics as averaging single draws. The concrete basis path owns the coefficient network and receives the basis and pool as constructor components. |
| `MixerEquation` | Compose the paths through `encode`, `deterministic`, and `per_draw`. Return a small `NamedTuple` encoding containing state features and the prepared draw payload. Keep this encoding local to the block. |

Use `eqx.AbstractVar` for required widths and `__check_init__` for
cross-component shape checks. Keep learned arrays as PyTree leaves. Preserve
the latest-slot mask before applying the read map and the existing float32
update boundary. Keep `FixedPointUpdate`, the integrator, and the recurrent
carry contract intact.

Set the coefficient network output width to `R*D`.
Each pool operates on rows within one draw and must preserve its score under
a joint permutation of the coefficient and basis rows. For example, a
context-weighted mean can fold normalized row weights into the prepared
coefficients. A nonlinear row pool can compute scores per draw while keeping
the context network outside the draw axis. Benchmark each pool's memory
behavior independently. Once a pool supplies `S[K,R,D]`, the draw path sums
the basis axis and applies the same `X.T @ Q` reduction.

Keep `AbstractEquation.__call__` as the shared encoding and addition logic.
Give it an abstract `mean_per_draw(context, draws)` hook and delegate the
mixer equation's hook to the basis draw path. This lets it expose a single-draw
reference while calling the optimized reducer during normal updates. Search
for all `AbstractEquation` subclasses when adding the hook.

## 2. Implement and verify separable draw reduction

Implementation complete. Depends on the path interfaces in task 1.

Implement the single-draw equation and the block contraction in the same
concrete draw path, sharing basis evaluation and the row-mean score. Prepare
the coefficients once per update step, before any operation over draws.
Keep gradients through the coefficient network, the state encoder, and
previous iterates. Keep the draw path to basis evaluation, row pooling,
basis summation, and multiplication.

Add focused tests in [test_mixer_equation.py](../../tests/test_mixer_equation.py).
The existing axial tests import an equation absent from this branch at
`8d5575f`, so the mixer tests use their own module.
Compare the optimized estimate with an explicit mean of single-draw terms
and check reverse-mode gradients through parameters, history, observations,
and draw values against the explicit reference. Use nonidentical draws with
nonzero cross-row interactions, `K=1` and `K>1`, and a masked latest slot.
Verify draw-order invariance, simultaneous row-permutation equivariance,
and behavior through JIT and a batched, differentiated multi-step rollout.
Exercise one-shot versus split execution over the same sequence of blocks
and compare state, step count, and draw count.

Run `JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_mixer_equation.py`.
The numerical parity target is the explicit average of the **new** single-draw
function, within float32 tolerances.

## 3. Wire the experiment and version its checkpoints

Implementation complete. Depends on verified equation behavior in task 2.

[The runner](../../experiments/refine_iteration/run.py) uses the basis path
with the existing state depth, state token mixer, predictor initialization,
and gain configuration. The model builder accepts a basis and pool. The
runner records the selected basis and the mean pool in version-7 checkpoint
metadata. Use `--draw-basis identity` for the single-basis variant.
[Replay](../../experiments/refine_iteration/replay_correlation.py) validates
the version, basis, and pool before loading leaves. Run historical
version-5 workloads from commit `8d5575f`.

Exercise both built-in basis choices through the builder and CLI and
verify version-7 serialization round trips and rejection of version 6.
Document the experiment options in
[the repository map](../../harness/repository-map.md).

All eight mixer tests passed after widening the coefficient head, including
gradient parity, independent state-channel contributions, and version-7
serialization and validation. A three-step CPU training run replayed its
checkpoint with zero NLL difference. Ruff lint, format, and whitespace
checks passed.

## 4. Verify memory savings and model quality

GPU verification pending confirmation of Slurm resources and run budget.
The comparison depends on task 3.

Run matched GPU forward and value-and-gradient comparisons for an isolated
equation update and for the training rollout. Follow
[the cluster harness](../../harness/cluster.md) and confirm resources and
environment before submitting. Use the same GPU type, backend, shapes, seed,
dtype, draw inputs, and episode microbatch size. Include `K=1` and sweep
larger `K` at fixed `N` and rollout length. Record compiled memory estimates,
measured peak device memory where available, and synchronized step time
after warm-up. Inspect the differentiated trace to verify the context
coefficient network runs outside the draw axis. Require lower backward memory
at the representative multi-draw workload and lower memory growth with `K`
for the isolated update. If the legacy workload runs out of memory, record
the failure and also compare smaller matched workloads that both complete.

Train on matched data and compare held-out prefix NLL and Pearson correlation
at the same draw counts. Basis width controls the number of scalar draw
features available to each state channel. Record the memory and quality tradeoff before treating the
basis model as the experiment baseline. A larger basis requires a measured
quality benefit and a repeat of the memory comparison.

Finish with `uv run --locked ruff check .`,
`uv run --locked ruff format --check .`,
`JAX_PLATFORMS=cpu uv run --locked pytest -q`, and `git diff --check`.
Record actual results and any remaining unverified acceptance criteria here.

### Original implementation verification

These checks predate the removal of the legacy path and the introduction of
per-channel coefficients.

- The CPU runner completed three matched training steps and prefix evaluation
  for each architecture on a small structured workload. This exercises the
  training path with two draws per update. It provides a smoke check for
  finite losses and metrics. Both layouts replayed with zero NLL difference.
- The ten focused mixer tests passed using the project uv environment.
  Repository-wide Ruff lint and format checks passed. The full pytest command
  hit a pre-existing collection error because `tests/test_fixed_point.py`
  imports the absent `AxialAttentionEquation`. All 72 tests collected with
  that file excluded passed.
- An in-memory fixture built from the original version-5 runner and equation
  at `8d5575f` produced exactly the same serialized model bytes as the new
  legacy runner with the same seed and XLA settings. Loading the original
  bytes also reproduced the update drift exactly.
- A CPU compiler check isolated a float32 equation evaluation and gradient
  with `N=64`, `D=16`, `W=32`, one mean mixer block on each legacy path,
  and the same state path on the candidate. The table reports
  `memory_analysis().temp_size_in_bytes` for the compiled forward scalar loss
  and compiled value-and-parameter-gradient. GPU peak memory and full-rollout
  savings are pending.

| Draws per block | Legacy forward bytes | Basis forward bytes | Legacy value and gradient bytes | Basis value and gradient bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 61,568 | 49,280 | 318,720 | 211,200 |
| 8 | 463,872 | 57,472 | 2,219,264 | 211,200 |
| 32 | 1,843,200 | 82,048 | 8,542,208 | 260,352 |
