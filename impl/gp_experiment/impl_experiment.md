# Gaussian feature-effect experiment implementation plan

Source design is [draft_experiment.md](draft_experiment.md). The target is
the Navi repository root. The implementation request authorizes building
and verifying the runner. See the execution record below for task status.

The deliverable is a repository-local runner under
`experiments/gp_experiment/` that trains one Navi model with fixed synthetic
context rows and fresh query rows each episode, selects a checkpoint on
independent validation episodes, and compares
the selected model with true-parameter BLUP on independent test episodes.
Use the existing Navi APIs and locked dependencies. The Git repository holds
the experiment source and tests. Pass an approved output directory for run
artifacts. Update the repository map when adding the runner.

## 1. Prepare the panel and episode generator

Add `experiments/gp_experiment/data.py`. Inspect `navi/helper.py` and
`tests/test_navi_helper.py` for the query-by-context Gram feature orientation.

- Generate one FP32 context matrix X_C with independent standard-normal
  entries and fixed IDs 0 through 1599. Draw a fresh FP32 X_Q of shape
  (400, 2000) each episode from the same distribution. For reduced CPU
  checks use n - n_context query rows.
- Compute u = (sum(X_Q ** 2, axis=1) / p)[:, None] and
  kappa = (X_Q @ X_C.T / p)[..., None] per episode. The full-run shapes
  are (400, 1) and (400, 1600, 1).
- Return the query rows, latent query targets, observed context, and
  both true variances in one episode draw. Split the episode key into
  five keys for query rows, both variance draws, effects, and noise.
  Sample each variance as 2 ** uniform(-2, 2). Draw beta with standard
  deviation sqrt(sigma_g2 / p) and context noise with standard deviation
  sqrt(sigma_e2). Use the same beta in g_Q = X_Q @ beta and
  y_C = X_C @ beta + noise.
- Derive separate design, initialization, training, validation, test, and
  bootstrap keys from a recorded master seed. Fold each episode index into
  its stream key. Keep validation and test episode indices fixed across
  checkpoint comparisons and reruns.

Verify in `tests/test_gp_experiment.py` that context rows and IDs stay
fixed, repeated keys reproduce draws, and distinct episode keys change the
query rows and `kappa`. Compare the query features with entries of the
Gram matrix of the stacked context and query rows. Reconstruct a draw from
its five keys to verify the variance scaling and shared feature effects.

## 2. Implement oracle scoring

Add `experiments/gp_experiment/evaluation.py`. This depends on the episode
contract in task 1 and can proceed alongside task 3 once that contract is set.

- Form `K_CC` once in host NumPy float64 from the fixed `X_C`. For each
  test episode, compute `K_QC = X_Q @ X_C.T / p` in float64 from that
  episode's stored query rows. Solve
  `(K_CC + (sigma_e2 / sigma_g2) * I) @ alpha = y_C` by Cholesky and return
  `K_QC @ alpha`. Use NumPy linear solves with the Cholesky factors. Check
  finiteness and a relative residual `norm(A @ alpha - y_C) /
  max(norm(y_C), tiny)` of at most `1e-9`. Include the episode index and
  variances in a solve failure. Use the same query rows and observed context
  for both predictors.
- Store one result row per episode with the episode index, sampled variances,
  true heritability, each predictor's latent MSE, their paired MSE difference
  (Navi minus BLUP), and their mean squared prediction disagreement.
  Record each predictor's Pearson correlation with the latent query
  targets within each episode. In the summary, report the mean over
  defined correlations and the number of episodes contributing to it.
- Aggregate 100 test episodes by default. Resample the episode-level
  differences 2,000 times using the bootstrap stream and report the 2.5th
  and 97.5th percentiles of the bootstrap means. Report the mean paired
  difference and count in four heritability bins with edges 0, 0.25, 0.5,
  0.75, and 1. Use left-closed bins, include 1 in the last bin, and report
  a null mean for empty bins.

Verify the oracle against feature-space ridge with penalty
`p * sigma_e2 / sigma_g2` on small designs at several variance ratios,
including a rank-deficient design. Test result aggregation with known
paired errors, a fixed bootstrap key, and empty bins. The oracle observes
the true variances, so the paired error gap can reflect both that extra
information and Navi's approximation error.

## 3. Implement training and the runner

Add `experiments/gp_experiment/run.py` and run it as a module from the
repository root. This depends on task 1. Wire in task 2 for the final test pass.

- Configure `Navi` with single-channel context, query, and pair features,
  `num_ids=n_context`, width 32, four heads, and one query layer by default.
  Fit 32 bars from 8,192 simulated training-prior targets using the training
  key at index 0. Save the edges with the model settings. Train the default
  head on query negative log-density and use its predictive mean for test
  point metrics. Model inputs are context phenotypes, IDs, the Gram
  diagonal, and cross-Gram features multiplied by $\sqrt p$.
- Compile each interval between validations with `eqx.filter_jit` and
  scan the optimizer updates on device. Each update uses one fresh episode,
  `eqx.filter_value_and_grad`, and Optax Adam at `1e-3`. In the same call,
  evaluate the fixed validation episodes sequentially on device. Fetch the
  interval's losses, finiteness flags, and mean validation loss together.
  Validate dimensions and counts at startup and report the first nonfinite
  loss or gradient by update index.
- Expose `--seed`, `--n`, `--p`, `--n-context`, `--steps`,
  `--validate-every`, `--validation-episodes`, `--test-episodes`, and
  `--output-dir`. Require the step count, validation cadence, and output
  directory. Default the master seed to 0 and validation to 32 episodes.
  Record validation negative log-density at step 0 using the same validation episodes as
  later checks. Validate at the requested cadence and after the final update.
  Select the checkpoint from the post-update validations. Serialize
  the checkpoint with the lowest mean validation negative log-density using Equinox
  leaf serialization, keeping the earlier checkpoint on ties. Reload it
  into the same model structure for the final test pass.
- Write `best.eqx`, `run.json`, `episodes.csv`, and `summary.json`.
  Record the resolved settings, seed, device, wall-clock time, validation
  history, pre-update training negative log-density for each update, and selected step in
  `run.json`. Require a fresh run directory
  to prevent artifacts from different runs being mixed.

Verify on CPU that small compiled updates have finite gradients and change
trainable parameters. Reload the selected checkpoint and verify that its
predictions match the saved model. Check that a fresh test draw is passed
to both predictors with identical query rows and observed context, and
that the final result has one row per test episode.
Use a reduced-size end-to-end smoke run from the repository root with a
fresh temporary output directory.

```sh
GP_RUN_DIR="$(mktemp -d)/run"
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_gp_experiment.py
JAX_PLATFORMS=cpu uv run --locked python -m experiments.gp_experiment.run \
  --n 16 --p 12 --n-context 12 --steps 2 --validate-every 1 \
  --validation-episodes 2 --test-episodes 3 --output-dir "$GP_RUN_DIR"
JAX_PLATFORMS=cpu uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
git diff --check
```

## 4. Verify full-size execution on Slurm

After the CPU checks pass, prepare the locked uv environment and review
[cluster prerequisites](../../harness/cluster.md). Follow the existing GPU
template for submission from the repository root and use
`JAX_PLATFORMS=cuda uv run --locked --no-sync python -m experiments.gp_experiment.run`
in the allocation. Confirm time, memory, CPU requests, GPU compatibility,
and output location before submission.

Run a one-update pilot at the full panel size with one validation episode
and one test episode. Confirm the CUDA device, successful forward and
backward execution, finite metrics, and usable checkpoint and evaluation
files. Use pilot timing and memory to set and record the main run's update
budget and validation cadence. Run the main comparison with 100 independent
test episodes and capture its completed job status, run directory, selected
step, and metrics in this plan. Completion requires valid paired results
and the reported uncertainty interval. There is no required MSE gap size.

## 5. Evaluate the kernel-conditioned Navi decoder

**State.** The runner changes passed the CPU smoke check, and the
full-panel GPU run completed. The runner uses the
[Navi library decoder milestone](../navi/impl_navi_jax_equinox.md#5-add-kernel-conditioned-attention-logits).
The existing runner already passes scalar observed phenotypes and aligned
Gram features through the public Navi API. Start a fresh model and optimizer
in a fresh output directory.

In `experiments/gp_experiment/run.py`, evaluate the selected checkpoint's
held-out negative log-density with the existing bar loss on the same test
episodes used for paired point scoring. Record the test loss as `navi_nll` in `episodes.csv` and its mean in
`summary.json`. Keep reporting Navi and
oracle BLUP MSE and Pearson correlation, the paired MSE gap and its
bootstrap interval, and the validation negative log-density history from
step zero. Extend the runner test in `tests/test_gp_experiment.py` to
check that held-out loss is finite and matches the loss recomputed from
the selected checkpoint on the replayed test episodes.

Run the focused CPU test and the reduced-size smoke command from task 3.
Then use the Slurm workflow in task 4 for the new decoder, including a
full-panel pilot that exercises forward and backward execution with
the kernel-conditioned bias. Record the revision, command, seed, resolved
settings, backend, and completed job status with the metrics. Completion
requires finite metrics and the paired evaluation artifacts.

Generate the cross-kernel feature as $x_*^\top x_i/\sqrt p$ and
pass it to Navi for both the pair MLP and the logit bias. Record
$\sqrt p$ as `pair_kernel_scale` in `run.json`. The self-kernel
feature and oracle covariance use the original $1/p$ normalization.

### Full-panel GPU result before the initialization change

Slurm job `66364149` completed with exit code `0:0` in 3 minutes
52 seconds on one NVIDIA A40 with two CPUs, 16 GiB of requested host
memory, and a 15-minute limit. The run used seed 1, depth three, 5,000
updates, 32 validation episodes every 500 steps, and 100 test episodes.
All four artifacts were written to
`/home/hblee/scratch1/gp_experiment/kernel-bias-66364149`.

Validation negative log-density was 1.6452 at initialization, 1.5855 at
the selected step 3,500, and 1.5961 at step 5,000. Held-out negative
log-density was 1.5060. Held-out latent MSE was 1.2943 for Navi and
0.7850 for oracle BLUP. The paired MSE gap was 0.5093 with a 95%
bootstrap interval of [0.4010, 0.6262]. Mean within-episode Pearson
correlations were 0.0368 for Navi and 0.5527 for BLUP.

### GPU result with the query read gain

The runner records the $\sqrt n$ gain in `run.json` as
`query_read_scale`. The isolated trial (Slurm job `66365561`) used a
gain of 40 with 1,600 context individuals and retained the $\sqrt p$
kernel-input scale. It completed 5,000 steps in 3 minutes 49 seconds
and selected step 3,500 at validation negative log-density 1.4186.
Across 100 held-out episodes, Navi achieved negative log-density
1.3565, latent MSE 1.0226, and mean within-episode Pearson correlation
0.4988. Oracle BLUP achieved latent MSE 0.7850 and correlation 0.5527.
The paired MSE gap was 0.2377 with a 95% bootstrap interval of
[0.1994, 0.2771].

### GPU result with both kernel paths scaled

Slurm job `66391566` completed 5,000 steps in 3 minutes 50 seconds with
seed 1 and depth three. The runner supplied cross-kernel features
scaled by $\sqrt{2000}$ to both the pair MLP and the logit projection.
The query read gain was $\sqrt{1600}=40$. Validation selected step
5,000 at negative log-density 1.4226.

Across 100 held-out episodes, Navi achieved negative log-density
1.3540, latent MSE 0.9609, and mean within-episode Pearson correlation
0.4982. Oracle BLUP achieved latent MSE 0.7850 and correlation 0.5527.
The paired MSE gap was 0.1759 with a 95% bootstrap interval of
[0.1416, 0.2127].

## Prior results

The previous decoder showed weak learning in the Gaussian experiment.
At depth three, held-out latent MSE was 1.294 for Navi and 0.785 for
oracle BLUP, with mean within-episode Pearson correlations of 0.001
and 0.553, respectively.
