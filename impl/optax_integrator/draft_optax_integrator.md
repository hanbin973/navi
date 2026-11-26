# Optax-backed integrators for the fixed-point rollout

This note specifies a change to `navi/fixed_point.py`, a new
`navi/integrator.py` module, and the
`experiments/fixed_point/` runner so that any optax transformation can serve
as the stochastic-approximation integrator. A tested prototype accompanies
this note as `integrators.py`. Place both under `impl/optax_integrator/`
(note as `draft_optax_integrator.md`, prototype as
`prototype_integrators.py`). The prototype is a reference, not an import
target; port its contents into the package as described below.

Follow `AGENTS.md`: use the project uv environment, run relevant checks,
and update `harness/repository-map.md` for added or moved components.

## Goal

Let experiments swap integrators (plain Robbins–Monro, heavy-ball momentum,
Adam preconditioning, schedule-free SGD, and any future optax transform)
through a factory registry and CLI flag, without new integrator classes per
optimizer. Existing behavior and model checkpoints must be preserved.

## Core idea

The integrator step is `a <- a + gain * (estimate - a)`. This equals SGD on
the pseudo-gradient `g = a - estimate` with learning rate `gain`. Any optax
`GradientTransformation` applied to `g` therefore defines an integrator.
`Phi(a) - a` is generally not a gradient field, so non-SGD optimizers are
heuristics without Robbins–Monro guarantees; document this in docstrings.

## Required design

### 1. Separate module and stateful interface (`navi/integrator.py`)

Move `AbstractIntegrator`, `RunningMeanIntegrator`, and
`PolynomialGainIntegrator` from `fixed_point.py` into `navi/integrator.py`.
Place `AbstractGainIntegrator`, draw schedules, `OptaxIntegrator`, and all
integrator factories in this same singular module. Keep equations, carries,
snapshots, and `FixedPointUpdate` in `fixed_point.py`. The dependency runs
from `fixed_point.py` to `integrator.py`, without a reverse import. Update
callers, experiment imports, tests, and the repository map. Preserve existing
integrator names as imports in `fixed_point.py` for caller compatibility.
Verify behavior and checkpoint compatibility across the extraction.


Replace the current `AbstractIntegrator` with the prototype's interface:

| Member | Contract |
| --- | --- |
| `init(latest) -> state` | Optimizer state PyTree for an `[N, D]` iterate. Default `()`. |
| `_advance(latest, estimate, draw_count, block_size, state) -> (next, state)` | Abstract. One step for blocks after the first. |
| `__call__(latest, estimate, draw_count, block_size, state) -> (next, drift, state)` | Shared. Computes `drift = estimate - latest`. On the first block (`draw_count == block_size`) returns `estimate` and `init(estimate)`; otherwise the `_advance` result. Selection uses `jnp.where` and `jax.tree.map`. |
| `readout(latest, state) -> Array` | What the predictor sees. Default identity. |

Add `AbstractGainIntegrator(AbstractIntegrator)` with abstract
`gain(draw_count, block_size)` and stateless `_advance`. Port
`RunningMeanIntegrator` and `PolynomialGainIntegrator` onto it. The
first-block override in `PolynomialGainIntegrator.gain` becomes redundant
because the base class now handles the first block; keep its `__check_init__`
validation.

The first-block reset matters for stateful optimizers. Without it, momentum
and Adam ingest the spurious first pseudo-gradient `0 - estimate`.

Out of scope: the separate legacy `AbstractIntegrator` and
`TanhRunningMeanIntegrator` in `navi/bayes_update.py`, which `navi/__init__.py`
exports, and `PooledRunningMeanIntegrator` in `experiments/modular_bayes/`.
Do not change them.

### 2. Carry and update wiring

- Add `integrator_state: Any = ()` as the last field of `SolutionCarry`. The
  default keeps positional construction elsewhere working; grep for
  `SolutionCarry(` to confirm.
- `FixedPointUpdate.initial_state` sets the field to
  `self.integrator.init(zeros[-1])`, where `zeros` is the history buffer.
- `FixedPointUpdate._integrate` passes `carry.integrator_state` and returns
  the new state; `step` stores it in the new carry; `drift` discards it.
- `FixedPointUpdate.snapshot` returns
  `self.integrator.readout(carry.history[-1], carry.integrator_state)` as
  `embeddings`. `history` keeps the iterate at which `Phi` is evaluated
  (the y sequence for schedule-free).
- `navi/averaging.py` keeps working unchanged because it reads
  `solution.history[-1]`. Leave `AveragedUpdate` in place.

### 3. Optax bridge (`navi/integrator.py`)

Port from the prototype:

- `DrawSchedule = Callable[[draw_count, block_size], step]`, with
  `running_mean_schedule()` (`K / n_t`) and
  `polynomial_schedule(exponent=0.67, scale=1.0)`
  (`min(1, scale * K * n_t**-exponent)`).
- `scale_by_draw_schedule(schedule)`, an
  `optax.GradientTransformationExtraArgs` whose update reads the
  `draw_count` and `block_size` keyword arguments and multiplies updates by
  `-schedule(...)`. This keeps gains defined on the cumulative draw count,
  so variable block sizes and resumed rollouts match current semantics.
  Optimizers that use optax's internal step counter, such as
  `optax.adam(1e-2)`, also work but count blocks, not draws.
- `OptaxIntegrator(AbstractIntegrator)` with fields
  `transform: optax.GradientTransformation = eqx.field(static=True)` and
  `readout_fn: Callable | None = eqx.field(static=True, default=None)`.
  `_advance` wraps the transform with `optax.with_extra_args_support`, calls
  `update(latest - estimate, state, latest, draw_count=..., block_size=...)`,
  and applies the updates with `optax.apply_updates`.
- Factories: `sgd(schedule=None, momentum=None)`,
  `adam(schedule=None, b1, b2, eps)`, and
  `schedule_free(learning_rate, b1)`. The last uses
  `optax.contrib.schedule_free_sgd` with
  `readout_fn=optax.contrib.schedule_free_eval_params`.

Static transform fields add no array leaves, so existing `models.eqx`
checkpoints must still deserialize. Verify this explicitly.

### 4. Experiment runner (`experiments/fixed_point/run.py`)

`INTEGRATORS` currently maps names to classes called with no arguments.
Change the entries to zero-argument factories built from parsed arguments.
Keep the existing names and add new ones:

| `--integrator` | Construction |
| --- | --- |
| `running-mean` (default) | `RunningMeanIntegrator()` (unchanged numerics) |
| `polynomial` | `PolynomialGainIntegrator(exponent, scale)` |
| `averaging` | as today: polynomial wrapped in `AveragedUpdate(UniformAverager())` |
| `sgd-momentum` | `sgd(polynomial_schedule(exponent, scale), momentum=...)` |
| `adam` | `adam(polynomial_schedule(exponent, scale))` |
| `schedule-free` | `schedule_free(learning_rate, b1)` |

Add flags `--gain-exponent` (0.67), `--gain-scale` (1.0), `--momentum` (0.9),
`--sf-learning-rate` (0.5), and `--sf-b1` (0.9). Record all of them in
`run.json`. Check that `replay_correlation.py`, which validates
`args.integrator in INTEGRATORS`, accepts the new names and rebuilds the
same integrator from the recorded arguments. Decide whether
`checkpoint_version` needs a bump. Model leaves are unchanged, but resumed
carries gain a field. Document the decision.

## Tests (`tests/test_integrators.py`, plus updates to `tests/test_fixed_point.py`)

1. **Exact parity.** `sgd()` matches `RunningMeanIntegrator` with variable
   block sizes (for example `[4, 4, 8, 2, 4, 4, 16, 1]`): max difference 0.
   `sgd(polynomial_schedule())` matches `PolynomialGainIntegrator` at fixed
   `K` within `2e-7`. The prototype test achieved 0.0 and 1.2e-7.
2. **Full-rollout parity.** `FixedPointUpdate` with the ported
   `RunningMeanIntegrator` reproduces the pre-change rollout on an existing
   `test_fixed_point.py` fixture.
3. **First-block reset.** After block one, every integrator returns exactly
   the estimate, and momentum and Adam states equal `init(estimate)`.
4. **Resume.** A rollout split by `advance_chunk` and resumed from the carry
   equals the unsplit rollout for `sgd`, `adam`, and `schedule-free`. This
   checks that `integrator_state` is carried.
5. **Convergence toy (BLUP root).** Use N=30, K=8 draws per block, 3,000
   blocks, `xi ~ N(0, C)` with `C = L L^T`, and
   `Phi(a) = a - eta * (mean_k xi xi^T a + s2 * a - y)` with `eta = 0.05` and
   `s2 = 1`. Every integrator except running mean reaches a relative error
   of at most 0.05 against `alpha = (C + s2 I)^{-1} y`. The prototype gave
   about 0.015. Running mean gave about 0.59 here because a `1/t` gain needs
   `eta * lambda_min` large enough; do not assert on it. Mark the test slow
   if it exceeds the suite budget.
6. **Differentiability.** `jax.grad` through a jitted `lax.scan` rollout
   with respect to `y` and to equation parameters is finite for all
   integrators.
7. **Checkpoint compatibility.** A model serialized before the change
   deserializes into the new class tree with
   `eqx.tree_deserialise_leaves`.
8. **Readout.** For `schedule-free`, `snapshot().embeddings` equals
   `schedule_free_eval_params(state, history[-1])` and differs from
   `history[-1]` after a few blocks.

## Verification record to fill in

- Unit tests above pass under uv; list commands and results.
- One short GPU training smoke run per new integrator through the Slurm
  harness (`harness/cluster.md`), reporting prefix NLL against the
  `running-mean` baseline at the same configuration as
  `harness/latest_baseline.json`.
- Peak memory for `adam` versus `running-mean` at `N=504, K=32`, horizon 32,
  microbatch 16. Expect extra rollout residuals of about 2x the `[N, D]`
  iterate per step for Adam and 1x for momentum and schedule-free.

## Notes and non-goals

- Gain-schedule parameters are static closures and so not trainable. A
  follow-up could make them `eqx` fields passed through the extra arguments
  to meta-learn the schedule jointly with `Phi`. Do not do this now.
- Keep the old "gain in (0, 1]" wording only on `AbstractGainIntegrator`;
  optax integrators have no such bound.
- The first-block selection evaluates both branches. Keep `_advance` free of
  NaN at `latest = 0` so gradients through the unused branch stay finite.


## Implementation progress (2026-11-26)

- Implemented `navi/integrator.py`: stateful interface, scalar gain integrators,
  draw-count schedules, and Optax SGD/momentum, Adam, and schedule-free factories.
- Wired optimizer state through `SolutionCarry`, initial state, steps, resumed
  scans, and integrator-specific snapshot readout. Existing fixed-point imports
  retain compatibility exports. Legacy Bayesian integrators remain untouched.
- Repository-driven deviation: the draft's axial `experiments/fixed_point`
  runner and `tests/test_fixed_point.py` import `AxialAttentionEquation`, which
  this branch removed before this change. The active runner is
  `experiments/refine_iteration`, matching `harness/latest_baseline.json`.
  Integrator flags, metadata, and replay support therefore live there; historical
  runner/tests are unchanged. Tests use the current mixer and a small equation.
- Keep checkpoint version 9: persisted artifacts contain model array leaves,
  not rollout carries. The module extraction and static transforms do not add
  model arrays. Replay reconstructs the integrator from metadata; absent new
  settings use original defaults. Live stateful rollout carries must include
  optimizer state; old four-field carries only default correctly for stateless
  integrators. Carries are not persisted by this runner.
- Retained the polynomial gain's first-block override for direct `gain` callers;
  the shared integrator interface independently guarantees first-block reset.
- CPU Slurm job 66849398: `JAX_PLATFORMS=cpu uv run --locked --no-sync python
  -m pytest -q tests/test_integrators.py tests/test_mixer_equation.py`:
  **38 passed in 46.06s**. Covers variable-block exact SGD parity, polynomial
  parity, first-block reset, resumed state, finite JIT gradients, schedule-free
  readout, BLUP convergence, runner checkpoint round trips, and CLI validation.
- Changed-file Ruff lint/format and scoped `git diff --check` pass.
- GPU smoke/memory job 66849400 and actual baseline checkpoint verification
  completed successfully (details below). GPU comparisons use three training steps and eight evaluation
  episodes with baseline model dimensions, horizon 32, N=504, K=32, batch 32,
  microbatch 16. They are execution checks, not optimizer-quality conclusions.


### Final CPU checks and review

- Final CPU Slurm job 66849444: **44 passed in 39.60s** using the focused
  test command above. Added six zero-drift gradient cases after review.
  Adam uses `eps_root=eps**2` in addition to its denominator epsilon to
  keep derivatives finite at zero; custom Optax transforms must likewise
  support differentiating their unused first-block branch.
- Existing baseline checkpoint `runs/refine-iteration-66844920/models.eqx`
  loaded successfully into the new model template. CPU job 66849438 used
  XLA attention (same array leaves) in place of the recorded cuDNN backend;
  a three-block rollout matched legacy running-mean arithmetic exactly.
- Direct review covered the new module, carry/readout wiring, runner/replay,
  tests, and compatibility. No unresolved correctness findings remain.
- Historical fixed-point tests cannot collect on this branch because they
  import the already-removed axial equation. They were not rewritten to
  obscure that pre-existing mismatch. The full historical suite was not run.

Usage with the active runner:

```sh
JAX_PLATFORMS=cuda sbatch --account=jonth0 --gpus=1 --cpus-per-task=2 \
  --mem=32G --time=00:30:00 harness/slurm/python.sbatch \
  -m experiments.refine_iteration.run --structured --steps 3 \
  --integrator adam --gain-exponent 0.67 --gain-scale 0.1
JAX_PLATFORMS=cuda sbatch --account=jonth0 --gpus=1 --cpus-per-task=2 \
  --mem=32G --time=00:30:00 harness/slurm/python.sbatch \
  -m experiments.refine_iteration.run --structured --steps 3 \
  --integrator schedule-free --sf-learning-rate 0.5 --sf-b1 0.9
```

These short smoke submissions use the Slurm launcher; review resource
requests for longer runs as described in `harness/cluster.md`.


### GPU verification (completed)

Slurm job **66849400** completed with exit code 0 in 8m04s. Each integrator
ran in a fresh Python process. Reproducible drivers are `verify_gpu.py` and
`verify_legacy_checkpoint.py` in this directory; invoke them as
`-m impl.optax_integrator.verify_gpu` (GPU) and
`-m impl.optax_integrator.verify_legacy_checkpoint` (CPU), respectively,
through the Slurm launcher from the repository root. Artifacts, exact `run.json` settings, prefix
summaries, and resource measurements are in
`runs/optax-integrator-66849400/{running-mean,sgd-momentum,adam,schedule-free}/`.
All runs used the same short schedule and held-out episodes. Prefix NLLs:

| Integrator | NLL at block 1 | NLL at block 32 | Peak device bytes |
| --- | ---: | ---: | ---: |
| running-mean | 3.866507 | 3.833894 | 5300376320 |
| sgd-momentum | 3.461285 | 2.974950 | 5300892160 |
| adam | 3.588814 | 3.629664 | 5386572800 |
| schedule-free | 4.005150 | 3.966334 | 5300898048 |

Adam used 86,196,480 additional peak device bytes (about 82.2 MiB, 1.6%)
versus running mean. These are measured peaks for the complete training and
evaluation process, rather than isolated optimizer-state storage. Three
training updates do not establish optimizer quality or convergence on the
training task. The test suite separately verifies convergence on the BLUP toy.

Implementation and the requested verification are complete for the active
runner, with the historical-runner deviation documented above. No commit or
push was performed.
