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

- Configure `Navi` with single-channel context, query, pair, and output
  features, `num_ids=n_context`, width 32, four heads, and one query layer.
  Call `Navi` directly with the fixed context IDs and the query features
  computed from the same episode draw as the targets. Remove the singleton
  output axis before comparing with `g_Q`. Model inputs are context
  phenotypes, IDs, and unscaled Gram features.
- Build one `eqx.filter_jit` training step using
  `eqx.filter_value_and_grad` and Optax Adam at `1e-3`. Update trainable
  arrays with `eqx.apply_updates`. Use one fresh episode per update,
  validate dimensions and counts at startup, and fail with the update
  index if loss or gradients are nonfinite.
- Expose `--seed`, `--n`, `--p`, `--n-context`, `--steps`,
  `--validate-every`, `--validation-episodes`, `--test-episodes`, and
  `--output-dir`. Require the step count, validation cadence, and output
  directory. Default the master seed to 0 and validation to 32 episodes.
  Record validation MSE at step 0 using the same validation episodes as
  later checks. Validate at the requested cadence and after the final update.
  Select the checkpoint from the post-update validations. Serialize
  the checkpoint with the lowest mean validation latent MSE using Equinox
  leaf serialization, keeping the earlier checkpoint on ties. Reload it
  into the same model structure for the final test pass.
- Write `best.eqx`, `run.json`, `episodes.csv`, and `summary.json`.
  Record the resolved settings, seed, device, wall-clock time, validation
  history, pre-update training MSE for each update, and selected step in
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

## Execution record

- Tasks 1–3 implemented in `experiments/gp_experiment/`. The runner uses
  the existing project dependencies and Navi public model. The tests cover
  fresh query features, shared effects, the ridge equivalence, compiled
  updates, checkpoint selection, and paired evaluation from replayed keys.
- The reduced CPU CLI run used 16 individuals, 12 features, 12 context rows,
  two updates, two validation episodes, and three test episodes. It wrote
  all four artifacts with finite scores. This verifies the execution path
  at reduced size.
- The project uv environment's executables passed all 38 CPU tests,
  including seven focused experiment tests. Ruff lint, format checks,
  and `git diff --check` passed.
- Task 4 pilot submitted as Slurm job `66355590` on `spgpu` with a
  15-minute time limit, one GPU, one node, two CPUs, and 16 GB of memory.
  The requested output directory is `/home/hblee/scratch1/gp_experiment`.
  The pilot writes `slurm-66355590.out` and `pilot-66355590/` there.
  Slurm reported `COMPLETED` with exit code `0:0`, elapsed time 51 seconds,
  and peak host RSS about 2.4 GiB. The runner logged 46.3 seconds on an
  NVIDIA A40 and produced all four artifacts. The single test episode had
  latent MSE 3.819 for Navi and 1.443 for BLUP.
- Main comparison submitted as Slurm job `66355668` on `spgpu` with a
  12-hour limit and the pilot's resource settings. Uses seed 1, 10,000
  updates, validation every 500 updates on 32 episodes, and final
  evaluation on 100 held-out episodes. The log is `slurm-66355668.out`
  and the run directory is `main-66355668/` under the scratch output path.
  Slurm reported `COMPLETED` with exit code `0:0`, elapsed time 2 minutes
  46 seconds, and peak host RSS about 2.5 GiB. The run recorded all 20
  validation points and selected update 9,000 with validation MSE 1.41089.
  All four artifacts are present, including 100 paired test rows. Test
  latent MSE was 1.29400 for Navi and 0.78497 for oracle BLUP. The mean
  paired difference was 0.50904, with a 95% episode-bootstrap interval of
  [0.40080, 0.62580]. Mean squared prediction disagreement was 0.50425.
