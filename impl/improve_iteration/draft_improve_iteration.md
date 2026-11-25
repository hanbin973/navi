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
basis, a row pool, and an output projection. The basis and row pool are
independently configurable. The equations below instantiate the default
combination of a four-feature basis and uniform row-mean pooling.

Let `N` be the number of context rows, `D` the state width, `W` the state
encoder width, `R` the number of scalar basis functions, and `K` the number
of joint draws in the block. The indices `i` and `j` identify context rows,
`k` identifies a draw, and `r` identifies a basis function. The slice `:`
selects all channels on its axis.

```text
C[N,W]   = state_encoder(read(masked_latest), observations)
d[N,D]   = state_head(C)
A[N,R]   = coefficient_head(C)
B[k,i,r] = basis_r(X[k,i])
s[k,:]   = row_pool(A, B[k])
         = mean_j(A[j,:] * B[k,j,:])  # Uniform mean pool
f[k,i,:] = (X[k,i] * s[k,:]) @ P[R,D]
estimate = d + mean_k(f[k])
```

| Variable | Meaning |
| --- | --- |
| `masked_latest[N,D]` | Latest committed iterate, or zeros if its history slot is invalid. |
| `read(masked_latest)[N,D]` | Iterate features after the state read map, which is the identity in the current experiment. |
| `observations[N,1]` | Observed value for each context row, aligned with the iterate and draws. |
| `X[K,N]` | Joint prior draws. Row `k` of this array is one draw over all context rows, and `X[k,i]` is its value at context row `i`. |
| `C[N,W]` | Encoded context features, computed once per block from the read iterate and observations. |
| `d[N,D]` | Deterministic contribution to the update estimate. |
| `A[N,R]` | Learned, context-dependent coefficients. Entry `A[j,r]` weights basis function `r` at context row `j`. |
| `B[K,N,R]` | Draw-only basis values. Entry `B[k,i,r]` evaluates scalar basis function `r` at `X[k,i]`. |
| `s[K,R]` | Pooled score for each draw and basis channel. The default pool averages coefficient-weighted basis values over context rows `j`. |
| `P[R,D]` | Learned projection from basis channels to state channels, shared across rows and draws. |
| `f[K,N,D]` | Conceptual per-draw contribution. Entry `f[k,i,:]` is the state-channel update term for row `i` from draw `k`. |
| `estimate[N,D]` | Deterministic term plus the mean contribution over draws `k`, passed to the integrator. |

Use a per-row context network for `coefficient_head`, with a gated channel
mixer as the initial implementation. Start with a fixed scalar basis
`[1, x, tanh(x), tanh(x + 1) - tanh(1)]`, evaluated in the draw dtype.
The first two features provide a context-only score and a context-dependent
linear draw term, respectively. The remaining features let the score respond
nonlinearly to the draw. Use a shared bias-free projection `P` for the
draw contribution and keep the deterministic head's bias. This specifies a
new parameterization for training. It approximates the joint score used in
`main` with a finite separable basis.

The uniform mean pool supports a block implementation as two contractions.

```text
S[K,R] = einsum("knr,nr->kr", B, A) / N
M[N,R] = (X.T @ S) / K
mean_per_draw = M @ P
```

`S[K,R]` stacks the same pooled scores as `s`. Each entry of `M[N,R]`
is the mean over draws of `X[k,i] * S[k,r]`. Projecting `M` gives
`mean_per_draw[N,D]`, the draw contribution averaged over the block.

The multiplication by each row's draw value stays paired with the pooled
score from that same joint draw. That pairing retains within-draw
cross-row dependence. Averaging `X` and `S` independently would change
the mean-field map. The derivation works for `K=1` and any nonempty block.
The optimized block path can carry `B[K,N,R]` and `S[K,R]`, with
context-network intermediates sized independently of `K`. It avoids staging
a `[K,N,D]` contribution tensor and a broadcasted `[K,N,R,D]` product.
Profile the compiled reverse pass to determine the actual memory reduction.

## 1. Establish the baseline and compose the two paths

Implementation complete. Use the original mean-token draw configuration as the
performance baseline and retain it for checkpoint replay. Record one
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
| `AbstractDrawPool` | Map aligned coefficients and basis values, each `[N,R]`, to a row-permutation-invariant score `[R]` for one draw. Supply a block method producing `[K,R]` with the same semantics as mapping the single-draw method. The default concrete pool implements a uniform mean using a contraction. |
| `AbstractDrawPath` | Prepare a draw-free payload from `C`, evaluate one aligned draw using that payload, and reduce a block with the same semantics as averaging single draws. The concrete basis path owns the coefficient network and projection and receives the basis and pool as constructor components. |
| `MixerEquation` | Compose the paths through `encode`, `deterministic`, and `per_draw`. Return a small `NamedTuple` encoding containing state features and the prepared draw payload. Keep this encoding local to the block. |

Use `eqx.AbstractVar` for required widths and `__check_init__` for
cross-component shape checks. Keep learned arrays as PyTree leaves. Preserve
the latest-slot mask before applying the read map and the existing float32
update boundary. Keep `FixedPointUpdate`, the integrator, and the recurrent
carry contract intact.

Size the coefficient network and projection from the selected basis width.
Each pool operates on rows within one draw and must preserve its score under
a joint permutation of the coefficient and basis rows. For example, a
context-weighted mean can fold normalized row weights into the prepared
coefficients. A nonlinear row pool can compute scores per draw while keeping
the context network outside the draw axis. Benchmark each pool's memory
behavior independently. Once a pool supplies `S[K,R]`, the draw path can use
the same `X.T @ S` reduction because the final projection is linear.

Keep `AbstractEquation.__call__` as the shared encoding and addition logic.
Give it an abstract `mean_per_draw(context, draws)` hook. Implement the
existing mapped mean for the legacy equation and delegate the new equation's
hook to the basis draw path. This lets the new equation expose a single-draw
reference while calling the optimized reducer during normal updates. Search
for all `AbstractEquation` subclasses when adding the hook.

## 2. Implement and verify separable draw reduction

Implementation complete. Depends on the path interfaces in task 1.

Implement the single-draw equation and the block contraction in the same
concrete draw path, sharing basis evaluation and the row-mean score. Prepare
the coefficients once per update step, before any operation over draws.
Keep gradients through the coefficient network, the state encoder, and
previous iterates. Keep the draw path to basis evaluation, row pooling,
multiplication, and projection.

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

## 3. Wire the experiment and preserve replay

Implementation complete. Depends on verified equation behavior in task 2.

Use the basis path for new runs in
[refine_iteration/run.py](../../experiments/refine_iteration/run.py), with
an explicit legacy architecture option for paired runs. Retain the legacy
builder and its parameter tree and initialization for version-5 checkpoints.
Keep state depth, state token mixer, predictor initialization, and gain
configuration wired as before. Expose basis and pool selection through the
model builder, with the concrete defaults above. The runner also offers
`--draw-basis identity` for the rank-one variant. Treat draw depth and draw
token mixer as legacy options and reject explicit uses with the basis
architecture, including a clear error for draw attention. Record the selected
architecture, basis, and pool in run metadata. Use version 6 for basis runs
and keep version 5 for legacy runs. Dispatch
on version and architecture in
[replay_correlation.py](../../experiments/refine_iteration/replay_correlation.py)
before deserializing leaves.

Exercise builder and CLI construction for both architectures, a short
training step through the actual runner path, and serialization round trips
for both layouts. Verify version-5 replay with an existing compatible
checkpoint or a fixture produced by the version-5 builder. Check that
unsupported layouts fail before loading leaves. Update
[the repository map](../../harness/repository-map.md) with the model roles and
experiment options.

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
at the same draw counts. The draw contribution lies in the span of the
`R` projection rows, so basis width also controls the capacity of the
stochastic term. Record the memory and quality tradeoff before treating the
basis model as the experiment baseline. A larger basis requires a measured
quality benefit and a repeat of the memory comparison.

Finish with `uv run --locked ruff check .`,
`uv run --locked ruff format --check .`,
`JAX_PLATFORMS=cpu uv run --locked pytest -q`, and `git diff --check`.
Record actual results and any remaining unverified acceptance criteria here.

### Verification record

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
