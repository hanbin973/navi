# Implementation note: realistic heritability prior and standardized inputs

## Goal

Make training and evaluation match practice in two ways:

1. **Heritability prior.** Fix the total variance and draw heritability from a realistic range, mostly below 0.5.
2. **Standardization.** Treat the phenotype intercept as unknown and standardize observations and targets by the context observations' mean and standard deviation.

Both changes must keep the model, the draws, and the oracle consistent with one another. Section 5 lists the invariants that tests must check.

## Background

**Current heritability prior.** `simulate_trait` in `navi/simulation.py` draws the genetic variance $g$ and the residual variance $r$ independently as $2^{U(-2, 2)}$, using `LOG2_VARIANCE_BOUNDS = (-2.0, 2.0)`. Heritability $h^2 = g/(g+r)$ then has median 0.5 and 10–90% range 0.18–0.82. Half of all episodes have noise ratio $\lambda = r/g < 1$, where the system $(\Sigma_P + \lambda I)$ is worst conditioned.

| $\lambda$ | $h^2$ | Condition number, $n = 512$ | Condition number, $n = 1016$ |
|---|---|---|---|
| 1/16 | 0.94 | 118 | 234 |
| 1 | 0.50 | 55 | 101 |
| 16 | 0.06 | 7 | 11 |

The current total variance $g + r$ also varies by a factor of 16 across episodes, which the model must infer before anything else.

**Current intercept.** Observations have no intercept: $y = F_C z + \varepsilon$ with $z \sim \mathcal{N}(0, g I_m)$, where $F_C$ is the context factor (`selected_factor(factors, context)`) and $m$ is the spectral rank.

**Expected effect.** With $h^2 \leq 0.5$ every episode has $\lambda \geq 1$, so conditioning and the draw budget per rank direction should matter much less. The identity-to-oracle range also shrinks, so comparisons must use the fraction of that range closed.

## 1. Heritability prior

**Model.** Fix $g + r = 1$ and draw $h^2 \sim U(h^2_{\min}, h^2_{\max})$. Then $g = h^2$ and $r = 1 - h^2$.

**Changes**

- `navi/simulation.py`
  - Add a variance-prior option, either as a field passed to `simulate_trait` or as a module constant set by the runner:
    - `log2-uniform`: the current behavior, kept for comparability;
    - `h2-uniform`: the new prior, with bounds `H2_BOUNDS = (0.05, 0.5)` by default.
  - In `simulate_trait`, replace the `variances = 2 ** uniform(...)` line with a call to a helper `sample_variances(key, prior, dtype)` returning `(g, r)` for either prior.
- `experiments/refine_iteration/run.py`
  - Add flags `--variance-prior {log2-uniform, h2-uniform}` (default `h2-uniform` once validated), `--h2-min` and `--h2-max`.
  - Record all three in `run.json`.
  - Update `structured_calibration_targets` to draw $g$ from the active prior. See Section 3 for the simpler replacement.
- `experiments/refine_iteration/baselines.py`
  - Replace `_variance_grid` with a prior-aware grid.
  - For `h2-uniform`: a one-dimensional grid of `VARIANCE_GRID_SIZE**2` midpoints over $[h^2_{\min}, h^2_{\max}]$, with `genetic = h2` and `residual = 1 - h2`. Keep the same number of grid points so the cost is unchanged.
  - For `log2-uniform`: the current two-dimensional grid.

## 2. Unknown intercept

**Model.**

$$
y = \mu \mathbf{1} + F_C z + \varepsilon, \qquad z \sim \mathcal{N}(0, g I_m), \qquad \varepsilon \sim \mathcal{N}(0, r I_N),
$$

with a flat prior on $\mu$. Let $C = I_N - \frac{1}{N} \mathbf{1} \mathbf{1}^\top$ be the centering matrix over the $N$ context rows. Inference with a flat intercept prior is equivalent to working with the centered data $C y$ in the $(N - 1)$-dimensional subspace orthogonal to $\mathbf{1}$. The relevant context factor is the centered factor

$$
\tilde{F}_C = C F_C ,
$$

that is, $F_C$ with each column's mean over the context rows subtracted.

**Targets.** Query targets stay the genetic values $f_* = F_* z$, where $F_*$ is the query factor. The intercept affects observations only.

**Changes, all using the same centered factor $\tilde{F}_C$**

| Location | Change |
|---|---|
| `sample_structured_episode` in `run.py` | Compute `context = selected_factor(...)`, then `centered = context - context.mean(axis=0, keepdims=True)`. Pass `centered` everywhere a context factor is used below. Keep the uncentered factor only for simulating $y$ |
| Draws: `structured_draws`, `sample_updater_draws` | Use `centered` as `context_factor`. Each draw then has mean zero over the context rows and covariance $\tilde{F}_C \tilde{F}_C^\top = C K_{CC} C$ |
| Predictor kernel features: `kernel_features(context, queries)` | Pass `centered` for the context argument, so the cross features are $F_* \tilde{F}_C^\top$. The query diagonal $\lVert F_{*,q} \rVert^2$ is unchanged |
| Oracle: `_spectral_posterior` in `baselines.py` | See below |

**Oracle with a flat intercept prior.** In `_spectral_posterior(context_factor, query_factor, observations, genetic, residual)`, pass the centered factor $\tilde{F}_C$ and the centered observations $C y$, and make one more change:

- `gram`, `rhs`, `projected_y` and `y_squared` need no code change once they receive centered inputs;
- in `logdet`, replace `observations.size` with `observations.size - 1`, because the centered data lie in an $(N - 1)$-dimensional subspace.

This gives the restricted (REML) marginal likelihood up to a constant that does not depend on $(g, r)$. The posterior means and variances of the query genetic values follow from the existing formulas. Apply the same centering to the identity-kernel plug-in baseline in `paired_bar_scores`: center the observations and use $N - 1$ wherever the observation count enters.

## 3. Standardization

**Definition.** With context observations $y \in \mathbb{R}^N$:

$$
\bar{y} = \frac{1}{N} \sum_j y_j, \qquad
s = \sqrt{\frac{1}{N - 1} \sum_j (y_j - \bar{y})^2}, \qquad
y^{\mathrm{std}} = \frac{y - \bar{y}}{s}, \qquad
f_*^{\mathrm{std}} = \frac{f_*}{s}.
$$

Targets are scaled but not shifted, because the genetic values have prior mean zero and the intercept belongs to the observations.

**Why it is exact.** $s$ and $\bar{y}$ are functions of the observed data, so standardization is an invertible change of units given $y$. Any posterior for $f_*$ maps to a posterior for $f_*^{\mathrm{std}}$ by dividing its mean by $s$ and its variance by $s^2$.

**Changes**

| Location | Change |
|---|---|
| `sample_structured_episode` | Compute $\bar{y}$ and $s$ from the simulated context observations. Store `y_std` as the model's observation input and `targets / s` as the training targets. Store `s` on the episode |
| Draws | No scaling. The draws describe the kernel, which is unit-free relative to the observation scale |
| Oracle and identity baseline | Compute the posteriors in original units with centered inputs (Section 2). Then divide each query's mean by $s$ and its variance by $s^2$ before `_normal_bar_log_masses`. Score against `targets / s` |
| Bin edges | Calibrate on standardized targets. Simplest: replace `structured_calibration_targets` with a function that simulates full episodes through `sample_structured_episode` under the active prior and returns their standardized targets. The current function draws targets without observations, so it cannot compute $s$ |
| Flag | `--standardize {on, off}`, default `on` once validated, recorded in `run.json`. With `off`, use the raw centered observations and raw targets |

All NLLs, model and oracle alike, are then reported in standardized units. The change of units adds the same $\log s$ to every method for a given episode, so gaps between methods are unaffected.

## 4. Logging

Add these columns to `episodes.csv`, one value per episode, repeated across prefixes:

| Column | Meaning |
|---|---|
| `genetic_variance` | $g$ |
| `residual_variance` | $r$ |
| `h2` | $g / (g + r)$ |
| `scale` | $s$ |

Add the fraction of the identity-to-oracle range closed to `summary.json` for each prefix:

$$
\mathrm{closed}_t = \frac{\mathrm{identity\_nll}_t - \mathrm{nll}_t}{\mathrm{identity\_nll}_t - \mathrm{oracle\_nll}_t} .
$$

## 5. Invariants and tests

Add these to a new `tests/test_standardization.py`, plus extensions of the existing simulation tests.

1. **Centered draws.** Every draw has mean zero over the context rows, to float tolerance. Over many draws, the empirical covariance approaches $C K_{CC} C$.
2. **Oracle matches a dense reference.** For small $N$ (say 12) and rank (say 4), compare `_spectral_posterior` with centered inputs against a dense computation with an explicit intercept prior $\mu \sim \mathcal{N}(0, \tau^2)$ with large $\tau$ (e.g. $10^4$):
   - marginal likelihood, up to a constant independent of $(g, r)$, so compare differences across grid points;
   - query posterior means and variances.
3. **Scale invariance.** Multiplying the raw observations and targets by any $a > 0$ and adding any constant to the observations leaves the standardized inputs, standardized targets, and standardized oracle NLL unchanged.
4. **Shift invariance of features and draws.** Kernel features and draws depend only on the centered factor, so they are unchanged by the intercept.
5. **Prior bounds.** Under `h2-uniform`, every sampled episode has $g + r = 1$ and $h^2 \in [h^2_{\min}, h^2_{\max}]$. The oracle grid covers the same interval.
6. **Backward compatibility.** With `--variance-prior log2-uniform --standardize off` and the intercept centering disabled, every output matches the current code exactly. Gate the centering behind the `--standardize` flag or a separate `--intercept {on, off}` flag.

## 6. Experiments after implementation

Use the current baseline configuration (`harness/latest_baseline.json`), changing only the new flags.

| Run | $n$ | $K$ | Prior | Standardize |
|---|---|---|---|---|
| A | 512 | 32 | h2-uniform [0.05, 0.5] | on |
| B | 1016 | 32 | h2-uniform [0.05, 0.5] | on |
| C | 1016 | 64 | h2-uniform [0.05, 0.5] | on |

**Report**

- the fraction of the range closed at each prefix, compared with the matching runs under the old prior;
- the gap and the fraction closed, stratified by $h^2$ into, for example, five quantile bins;
- whether the size and rank effects seen under the old prior (B versus A, and C versus B) shrink.

**Expected outcome.** Smaller effects from conditioning and rank. A smaller benefit from increasing $K$. Lower absolute NLL in standardized units.

## Out of scope

- Model architecture changes.
- Non-Gaussian priors.
- Joint scoring across queries; the oracle stays the product of per-query marginal predictive distributions.


## Implementation progress (2026-11-27)

- Implemented both priors with legacy random-key mapping preserved. Only the
  online structured refine runner defaults to `h2-uniform` and standardization.
- `--standardize on` jointly enables flat-intercept centering and scaling. With
  `off`, observations, factors, draws, targets, and calibration retain their
  legacy semantics. Other runners and offline dataset formats are unchanged.
- Model features, draws, and restricted-likelihood Gaussian references use the
  same centered factor. Reference posteriors use original observation units,
  then transform means and variances to target units. Episode scale and variance
  metadata are recorded, with overall and five-group h2 summaries per prefix.
- Checkpoint layout remains version 9; replay uses recorded settings and defaults
  missing settings to the legacy behavior. The architecture is unchanged.
- A zero intercept is simulated as a representative of the shift-invariant
  observation model; no arbitrary proper intercept prior is introduced.
- Unit-invariance qualification: changing observation and target units by `a`
  requires changing prior variances by `a**2`. A fixed-total-variance prior is
  expressed in original trait units and is not invariant if its variances are
  held fixed while observations are rescaled. The tests check the consistent
  change of units against a dense diffuse-intercept reference.
- Direct review covered changed code, replay callers, tests, and CLI semantics.
  Six numerical/CLI tests passed locally; the initial Slurm verification passed
  62 relevant regressions. Two subsequent summary tests also passed. Ruff lint
  passes repository-wide and changed Python files are formatted. Repository-wide
  format checking finds a pre-existing issue in
  `impl/optax_integrator/integrators.py`, left untouched.
- Verification job `66887501` passed regressions but its smoke wrapper failed
  to import the project when launched by file path. Job `66887526` reruns via
  `-m impl.std_experiment.verify`, including GPU training and checkpoint replay.
- User authorized the baseline scheduler resources (2 CPUs, 32 GB) with a
  four-hour limit and all three GPU partitions. `python -m impl.std_experiment.submit --dry-run` prints
  reproducible A/B/C commands from `harness/latest_baseline.json`; invoking it
  without `--dry-run` submits jobs and records IDs in `submissions.json`.

The second verification job passed all 64 relevant regression tests. Its cuDNN
training/evaluation smoke produced all expected artifacts and checkpoint replay
matched saved NLL exactly (maximum difference zero). A/B/C were submitted as `66887532`,
`66887533`, and `66887534`, each with `afterok:66887526`. Exact commands are in
`submissions.json`. Each run requests one GPU across all three partitions,
2 CPUs, 32 GB, and four hours. B/C use population 1016 and context count 1008,
following the plan's n column and retaining eight queries.

Remaining: inspect final GPU verification and completed A/B/C runs before
reporting experimental comparisons. Matching old-prior comparisons
and scientific conclusions remain pending full run results.


Final verification: job `66887526` completed with exit code 0 after 64 passing
regressions, cuDNN training/evaluation, and exact checkpoint replay. Ruff lint,
format checks on changed Python files, and `git diff --check` pass.

The first A/B/C submissions (66887532–66887534) exited during CLI parsing because
baseline metadata `lr_schedule` is an internal default rather than an exposed
flag. The helper now validates commands against the actual runner parser and
omits that flag; the warmup-cosine schedule is unchanged. Failed submission
commands are retained in `submissions.failed-cli.json`.

Replacement A/B/C jobs: A `66887547`, B `66887548`, C `66887549`.
These have the same four-hour resources and successful verification dependency.
Full training results and old-prior comparisons are still pending. Next action:
inspect their final Slurm state and logs, then compare per-prefix and h2-group
summaries. Do not treat submission as experimental success.
