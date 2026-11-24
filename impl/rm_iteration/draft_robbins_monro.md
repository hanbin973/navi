# Robbins–Monro iteration

[navi/fixed_point.py](../../navi/fixed_point.py) defines a learned fixed-point
iteration over prior draws. It follows the carry and module composition in
[bayes_update.py](../../navi/bayes_update.py) and the
[Equinox abstract/final pattern](../../harness/pattern/equinox.md).

## Update convention

The carry stores a pre-activation iterate `a` with shape `[N, D]`. Each
block supplies `K` joint draws over the same `N` context rows. The equation
encodes the committed history once to build a draw-independent
context `c`. With `n_t` denoting the cumulative draw count after the block,
the update is

```text
estimate = deterministic(c) + mean_k(per_draw(c, draw_k))
drift   = estimate - a
a_next  = a + gain(n_t, K) * drift
```

Conditional on the committed history, if each draw has the prior as its
marginal distribution, the estimate has the same conditional expectation
for every block size. Nonlinearities can act on the state or on individual
draw terms. The shared block reduction and update are affine in the
per-draw terms. The snapshot exposes `a_next` directly. The predictor's
readout owns any output bounding, such as applying `tanh` before decoding.
The equation's `encode` hook owns any transform it needs for its own use of
the committed history.

## State and abstract interfaces

### SolutionCarry

`SolutionCarry` replaces `BayesCarry` and retains the four fields in a
`NamedTuple`.

| Field | Contract |
| --- | --- |
| `history` | Float32 `[H, N, D]` pre-activation iterates in right-aligned history slots, newest last. |
| `valid` | Boolean `[H]` mask of initialized slots. |
| `step` | Scalar int32 count of committed draw blocks. |
| `draw_count` | Scalar int32 count of individual joint draws consumed. |

### AbstractEquation

`AbstractEquation` replaces `AbstractProposal`. It declares positive
integer `state_width` and `history_capacity` as `eqx.AbstractVar[int]`.
Concrete equations implement three hooks.

| Hook | Contract |
| --- | --- |
| `encode(history, valid, observations)` | Build a draw-independent `EquationContext` PyTree from the committed iterates `[H, N, D]`, the `[H]` validity mask, and `[N, 1]` observations. |
| `deterministic(context)` | Return the draw-free part of the estimate as `[N, D]`. |
| `per_draw(context, draw)` | Map one joint draw `[N]` to a contribution `[N, D]`. This can mix context rows within the draw. |

The shared `__call__(history, valid, draws, observations)` accepts a draw-major
block `[K, N]`. It encodes the committed history once, maps `per_draw` over the
block with `jax.vmap`, and returns the deterministic term plus the mean over
the draw axis. Permuting the draws preserves the estimate up to roundoff.

### AbstractDrawConditioner

`AbstractDrawConditioner.__call__(draw, prefix_features)` takes one float32 draw
`[N]` and prefix features `[N, W]` and returns one term `[N, W]`, where
`W = output_width`. It may be nonlinear in the draw and must be equivariant
to a common row permutation of both inputs. The positive integer
`output_width` is an `eqx.AbstractVar[int]`. An equation can use this module
inside `per_draw`.

### AbstractIntegrator

Concrete integrators implement `gain(draw_count, block_size)`. Its inputs
are the cumulative draw count including the current block and the static
block size. Its output is a scalar gain in `(0, 1]`. The interface specifies
a first-block gain of one to discard the zero initialization.

The shared `__call__(latest, estimate, draw_count, block_size)` computes the
drift and applies the gain. It returns `(next_iterate, drift)`, both shaped
`[N, D]`.

## Gain schedules

- `RunningMeanIntegrator` uses `K / n_t`. This weights block updates by their
  draw counts and gives the `1 / t` update at committed block index `t` for
  constant block size.
- `PolynomialGainIntegrator` uses `min(1, scale * K * n_t**(-exponent))`.
  The exponent defaults to `0.67` and must be in `(0.5, 1]`. The scale
  defaults to `1.0` and must be positive. The first block uses gain one,
  including when the scheduled formula would return a smaller value.

## Rollout

`FixedPointUpdate` composes the equation and integrator. Initialization creates
a zero history with only the latest slot marked valid and both counters
set to zero. It requires at least one context row.

- `step` accepts a single draw `[N]` or a nonempty block `[K, N]`. It
  checks the shapes of the block, observations, and equation estimate and
  converts those floating-point inputs to float32. It commits one iterate,
  advances the block count by one and draw count by `K`, and returns the
  new carry and drift.
- `drift` evaluates the same update against a fixed carry without a commit.
- `snapshot` returns a `ContextSnapshot` with observations, the latest
  pre-activation iterate as `embeddings`, and the cumulative draw count.
- `advance_chunk` scans over `[T, N]` single draws or `[T, K, N]` blocks
  and returns the final carry.

## Restructuring the concrete Bayesian updater

The port reuses the axial encoder, both pooling MLPs, and the
output projection from [bayes_update.py](../../navi/bayes_update.py). The
running estimate lives in pre-activation space. For this port, the predictor
readout applies elementwise `tanh` to the snapshot embeddings before
decoding. The axial port independently bounds its history inputs
inside `encode` to retain the existing encoder's bounded input convention.

### MeanPooling → JointDrawConditioner

Keep the existing `scorer`, `prefix_scorer`, and their widths and
initialization. For one draw `x[N]` and prefix features `p[N, W]`, compute
one pooled contribution as follows.

```text
row_score[i]    = scorer(concat(x[i], p[i])) + x[i] * prefix_scorer(p[i])
joint_score    = mean_i(row_score[i])                 # [W]
draw_feature   = x[:, None] * joint_score[None, :]     # [N, W]
```

The row mean summarizes the joint draw and preserves its alignment across
context rows. The shared equation averages the resulting contributions
over draws. That average recovers the original `MeanPooling` output for
identical inputs and weights, up to roundoff.

### AxialAttentionProposal → AxialAttentionEquation

Keep the existing history and observation projections, causal temporal
masks, context-row attention, and axial layer configuration. Make the
concrete equation implement the three hooks of `AbstractEquation`.

- `encode(history, valid, observations)` applies elementwise `tanh` to
  the committed history for this port and runs the existing axial prefix
  encoder to produce `p[N, W]`. The validity mask continues to exclude
  uninitialized slots. This transform is local to the concrete encoder.
- `deterministic(p)` applies the prefix half of the output projection
  and adds its bias.
- `per_draw(p, x)` calls the draw conditioner and applies the
  draw half of the output projection.

Keep one affine projection with weight `[D, 2W]` and bias `[D]`, using
its two weight slices in the separate hooks. This preserves the weight
layout and accounts for the bias exactly once. For fixed encoder inputs
and matching weights, the resulting estimate matches the raw output of
the existing proposal up to roundoff. Prefix attention runs once per
block. Only the single-draw computations are mapped over draws.

### TanhRunningMeanIntegrator → RunningMeanIntegrator

Use the existing fixed-point `RunningMeanIntegrator` for the running-mean
configuration. It applies `K / n_t` to the pre-activation drift. With
constant block size this is the old `1 / t` gain. The readout applies
`tanh` to the committed pre-activation iterate, so multi-block trajectories
generally differ from the original running mean of bounded candidates.
Any nonlinear transform intended to act on each draw before averaging
belongs in `per_draw`.

Compose the port as
`fixed_point.FixedPointUpdate(AxialAttentionEquation(...), RunningMeanIntegrator())`.
Construct the bar readout with `bound_embeddings=True` to bound the raw
snapshot embeddings before context encoding.
The snapshot still carries the predictor's observations, embeddings, and
draw count. The pre-activation carry and drift have new semantics, so
resume fixed-point rollouts from a carry produced by the new updater.

## Interface checks for implementation

- Compare single-draw and multi-draw estimates against an explicit reference
  for the deterministic term plus the average of nonlinear per-draw terms.
  Check draw permutation invariance and gradients against the same reference.
- Verify the drift sign, the gain's draw-count convention, and the separation
  between the raw iterate exposed in snapshots and the bounded input to
  the predictor readout.
- Compare whole and resumed scans and check carry shapes, dtypes, and counters.
- Compare the ported block estimate to the original raw proposal at
  matched weights and encoder inputs, including the output projection
  bias. Check that the first running-mean snapshot exposes the raw
  estimate and that the readout applies `tanh` before decoding.

## Implementation verification

The concrete conditioner, axial equation, and readout option are implemented.
The four fixed-point CPU tests passed, along with five existing tests for
Bayesian integration, causal axial attention, and cached predictor context.
Ruff lint and format checks passed for the changed Python files.
