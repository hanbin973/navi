# Offline simulation with structured kernels

This document specifies the Navi simulator and dataset-backed axial attention
experiment. The initial configuration uses population size $n=512$, rank
$m=128$, and 8 queries. The generator defaults to using the remaining 504
rows as context. `--alpha` sets one positive exponent per dataset and
defaults to 1.0.

[`experiments/modular_bayes/run.py`](../../experiments/modular_bayes/run.py)
draws an independent Gaussian feature panel for each episode. Its Gram
matrix has no prescribed spectrum. The axial runner's legacy path imports
its sampler and draw generator and runs them inside training and evaluation.
Both online runners default to 64 context rows and 8 queries. The dataset
path controls the spectrum,
makes population and context sizes configurable, and reuses a dataset across
training runs.

## 1. Define the spectral kernel and simulator

Implemented in `navi/simulation.py`, reusing `QueryContextKernel`, `DatasetSimulator`,
`TrainingEpisode`, and the ordered selection convention from
[`navi/helper.py`](../../navi/helper.py). Use one factor representation with
JAX array leaves `U` of shape `(n, m)` and `eigenvalues` of shape `(m,)`.
Both adapters use this representation. The initial basis distribution is
proposed as an independent random orthonormal basis per episode, preserving
the current benchmark's variation in kernel orientation across episodes.

For integer $1 \le m \le n$ and finite $\alpha > 0$, draw an $n\times m$
standard Gaussian matrix and take its reduced QR. Multiply each column of
$Q$ by the sign of the corresponding diagonal of $R$ so the convention is
independent of QR sign choices. Treat a zero or nonfinite diagonal as a
failed basis construction. Set

$$
\lambda_k = n\,\frac{k^{-\alpha}}{\sum_{j=1}^m j^{-\alpha}},
\qquad k=1,\ldots,m, \qquad
K=U\operatorname{diag}(\lambda)U^\top.
$$

$U^\top U=I_m$ gives rank $m$ and $\operatorname{tr}K=n$. Here $n$ counts
all population rows, including context and query rows. Trace normalization
sets the population mean of the diagonal to one. Individual diagonals can
vary. Compute the normalized spectrum in host float64 and validate that
every eigenvalue is finite and in the normal positive float32 range.
Use float32 for the initial artifact and JAX compute contract, with
dtype-aware tolerances for orthonormality and trace. Fail before writing an
episode if the requested spectrum loses rank at storage precision.

Let $F=U\operatorname{diag}(\sqrt\lambda)$, implemented by column scaling.
For ordered, disjoint context and query indices $C$ and $Q$, with counts
$c$ and $q$, use selected rows $F_C$ and $F_Q$.

- Draw one $z\sim\mathcal N(0,I_m)$ for both selections. Given genetic
  variance $g$ and residual variance $r$, emit latent targets
  $f_Q=\sqrt g F_Q z$ and observations
  $y_C=\sqrt g F_C z+\sqrt r\,\epsilon$, with independent
  $\epsilon\sim\mathcal N(0,I_c)$. Keep the existing independent
  log-uniform variance priors $g=2^{a_g}$ and $r=2^{a_r}$ with
  $a_g,a_r\sim\operatorname{Uniform}[-2,2)$ for the benchmark recipe.
- Independently sample updater inputs $F_C z_t$ using fresh standard-normal
  vectors. Their covariance is $K_{CC}$. Keep rollout layout `(T, k, c)`
  for $T$ update steps and $k$ draws per step. Generate these once offline
  in bounded chunks, including the longest requested evaluation prefix.
- For the predictor, return $\operatorname{diag}(K_{QQ})$ as `(q, 1)` and
  $K_{QC}=F_QF_C^\top$ as `(q, c, 1)`. Compute the diagonal by row-wise
  sums of squares. These are raw kernel entries. The legacy feature-panel
  runner applies a different scaling to its cross features, so record the
  raw convention in the dataset version and use it consistently across splits.

Host validation covers configuration, shapes, dtype and row indices. The
offline recipe selects a deterministic random partition for each episode,
with $c+q\le n$ and both counts positive. Gather and sample in pure JAX
functions with explicit PRNG keys, fixed shapes per compiled batch, and no
filesystem access. Derive keys from the dataset seed, split and episode ID,
then use separate streams for basis, selections, variances, trait, noise
and updater draws. Give updater draws absolute indices so changing shard or
generation chunk boundaries leaves their random keys unchanged. Different
compiled batch shapes can change float32 rounding.

**Verification.** In `tests/test_simulation.py`, check the eigenvalue ratios,
orthonormality, rank and trace against a tiny dense reference, including
`m=1` and `m=n`. Use a fixed basis with noncontiguous selections to check
feature values, order and reconstruction of the joint draw from a shared
latent vector. Check the context/query cross-covariance, noisy context
variance and updater covariance against the specified kernel blocks with a
seeded small-case sampling test. Check repeatability, key separation,
chunk-invariant draw generation, and eager versus JIT/batched behavior.

## 2. Generate and persist complete episodes

The host-side dataset reader/writer lives in `navi/simulation_data.py` and the
generation entry point in `experiments/attention_proposal/generate_data.py`.
Generation assigns independent train, validation and test split IDs and
writes a versioned JSON manifest and bounded shards of numeric `.npy`
arrays. The manifest records the dataset ID, schema and generator
versions, seed, basis recipe, feature convention, $n,m,\alpha,c,q$, split
episode counts, shard ranges, dtypes, variance recipe, draw horizons/block
size, and the JAX version and backend. Compute bar bin edges from a bounded,
deterministically selected subset of saved training targets. Record the
calibration sample count and check that the edges are strictly increasing
in float32, as required by `navi/distribution.py`. For prefix drift
diagnostics, persist one independent 32-step probe draw block per held-out
episode under a separate stream and reuse it at each checkpoint, matching
the current `score_prefix` convention.

Each shard stores `U`, `eigenvalues`, context and query indices,
observations, targets, and the realized updater draws, with an episode
axis. Record per-episode variances for provenance. The kernel arrays cost
$O(nm+m)$ per episode and represent the full population kernel. The
rectangular $F$ is a temporary computation. Store no Gram matrices, and
persist draws as arrays so reuse incurs no stochastic data generation.
Per-episode storage also includes $Tkc$ draw values, which can exceed the
factor size. Estimate storage for each split before generation. Generate
one bounded shard at a time and publish the completed directory with its
manifest by same-filesystem rename. Refuse to overwrite existing datasets.

Read numeric arrays with NumPy memory mapping and `allow_pickle=False`.
Validate the manifest, requested split, shard coverage, shapes, dtypes,
and horizon coverage before handing a dataset to the runner. Validate
array values and index bounds when each shard or batch is loaded. Validate
orthonormality during generation to avoid running QR or its verification
on every training load. Gather a bounded batch into host arrays, derive
features deterministically from selected factor rows, then transfer the
batch to JAX. Read only the requested draw slice when serving a rollout
chunk or evaluation checkpoint.

**Verification.** In `tests/test_simulation_data.py`, round-trip a tiny
multi-shard dataset and compare all loaded arrays, features and bin edges
with the generated inputs. Exercise incomplete and incompatible artifacts,
out-of-range episode indices, split isolation, and requests beyond the
stored horizon. Check that the schema has only factors for the kernel,
with an `m < n` fixture that would expose an accidental dense Gram array.

## 3. Train and evaluate from the saved dataset

The `--dataset` argument selects stored episodes in
[`experiments/attention_proposal/run.py`](../../experiments/attention_proposal/run.py).
After step 2, use a host iterator to supply stored episode batches to the
compiled train chunk. Stack a bounded number of batches along the optimizer
step axis so the scan consumes distinct batches. Each batch carries
observations, query features, pair features, targets and draws. Feed these
to the existing `rollout_loss`.
Use a seeded training shuffle over episode IDs with a recorded epoch and
batch policy. Keep horizon sampling as training control. Initialize the
bar predictor from the saved edges. Use stored held-out episodes and draw
slices for prefix evaluation. Record the dataset ID, loaded split names,
and consumption settings in `run.json`. Keep dataset mode opt-in for
historical replay, and leave the modular runner's dense exact-kernel
baseline outside the first integration.

**Verification.** In `tests/test_structured_training.py`, run a tiny saved
dataset through an optimizer update and held-out prefix evaluation with
finite outputs. Compare a fixed-horizon loss from the loaded batch with
the same loss from its pre-write arrays. Make the online episode, prior-bin,
and kernel-draw helpers raise in the dataset-backed test so accidental
resimulation fails the test. Check that the historical entry point still
works without the dataset argument.

## 4. Verify scaling and document the workflow

Run the following from the repository root after implementation.

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

Run a compute-node pilot using the launch guidance in
[`harness/cluster.md`](../../harness/cluster.md). With $m,q,T,k$ held fixed,
compare the existing context scale ($c=64$) with a larger context, adjusting
$n$ alongside $c$. Record generation time, artifact size, batch-load time,
device memory and training step time. Confirm the dataset path can train
at the larger context and reuse the same artifact on a second run. At fixed
$m$, factor storage grows linearly in $n$ and reduced QR costs $O(nm^2)$.
Full rank still costs $O(n^2)$ to store $U$. Query features cost $O(qcm)$
to assemble and $O(qc)$ to hold. The axial proposal's context attention
has quadratic work in $c$, so offline low-rank simulation alone gives no
guarantee about the largest trainable context.

## Online structured runs

Use `--structured` to generate a new factorized kernel, trait and updater
draws for every episode. The defaults use $n=512$, $m=128$, 504 context rows,
8 queries, and $\alpha=1.0$. At 10,000 optimizer updates with batch size 32,
the runner generates 320,000 training episodes. Run from the repository root
on an allocated compute node.

```sh
uv run --locked --no-sync python -m experiments.attention_proposal.run \
  --structured --alpha 1.0 --steps 10000 --batch-size 32 \
  --output-dir runs/spectral-online-a1p0
```

Structured runs on CUDA default to cuDNN attention and process all episodes
in an optimizer batch in parallel. Use `--episode-microbatch-size` to limit
the number of concurrent episodes if memory is constrained. The runner
accumulates microbatch gradients on the device and still takes one optimizer
step for the full batch. Structured runs using XLA default to one episode
at a time and accept the same override. `--attention-backend` can select
either implementation explicitly.

Training metrics return to the host in chunks of 100 updates. Held-out
episodes and diagnostic draws use separate keys. `--n`, `--m`, `--alpha`,
`--context-count`, and `--queries` control the simulation. The `--dataset`
path below remains available for replay.

Online structured calibration uses 1,024 independent prior episodes, up
from 128. With eight queries this yields 8,192 scalar calibration targets.
It samples the same Haar-basis target law by drawing the coefficient norm
and a uniform population direction, which avoids QR during calibration.

## Paired Gaussian evaluation

Each held-out episode is scored once with two Gaussian references. Both
use the run's bar edges, its context observations, and its latent query
targets. The oracle conditions on the selected factors. It diagonalizes
$F_C^\top F_C$ once and uses the rotated query factors to obtain the
posterior marginal means and variances in $O(m+qm)$ per variance point.
It averages Gaussian bar probabilities over a $32\times32$ midpoint grid
in the independent log-base-2 variance priors, weighting each point by
the context marginal likelihood. Integrating the bars costs an additional
$O(qB)$ per point for $B$ bars.

The identity reference fits $K=I$ using the same variance prior. Its
likelihood depends on the sum of genetic and residual variances. The
plug-in uses the posterior mean genetic variance for a zero-mean latent
query Gaussian. The oracle instead integrates variance uncertainty, so
the identity-to-oracle gap includes the effect of the plug-in.

The reference columns in `episodes.csv` repeat across prefixes for each
episode. `nll_above_oracle` is model NLL minus oracle NLL, and
`identity_minus_oracle` is identity NLL minus oracle NLL. Both gaps have
paired episode bootstrap intervals in `summary.json`. `prefix_nll.png`
shows both reference levels. The oracle floor is an expected bar NLL,
so an individual episode's realized gap can be negative.

Each run also writes `prefix_pearson.csv` and `prefix_pearson.png`.
At each prefix, Pearson correlation pools the paired posterior means
and true targets across all held-out episodes and queries. The CSV
records update steps and total joint draws. Both references use the
means of their bar-projected distributions. The figures show the oracle
reference, and the CSV also records the identity reference.
The latent identity Gaussian itself has a constant zero mean, so its
latent Pearson correlation is undefined. For completed online
structured runs, `python -m experiments.attention_proposal.replay_correlation
--output PATH RUN_DIR [RUN_DIR ...]` regenerates the figures from saved
checkpoints and compares the runs by steps and by total draws.

## Offline generation and replay

Run these commands from the repository root on an allocated compute node
with the locked uv environment. This example uses a short horizon and two
episodes per held-out split. Adjust `--alpha` and use a new dataset directory
for each exponent. Reuse the dataset path and a new run directory to train
again.

```sh
uv run --locked --no-sync python -m experiments.attention_proposal.generate_data \
  --output-dir runs/spectral-a1p0-data --n 512 --m 128 --alpha 1.0 \
  --k 4 --train-horizon 4 --eval-horizon 8 \
  --train-episodes 8 --validation-episodes 2 --test-episodes 2

uv run --locked --no-sync python -m experiments.attention_proposal.run \
  --dataset runs/spectral-a1p0-data --output-dir runs/spectral-a1p0-run \
  --t-max 4 --eval-multiplier 2 --steps 6 --batch-size 1
```

Select `--eval-split validation` for tuning and `--eval-split test` for
held-out reporting. The runner reads dimensions and bin edges from the
manifest. Its requested evaluation horizon is `--t-max` multiplied by
`--eval-multiplier` and must fit the stored split. `--data-chunk-steps`
sets the number of optimizer batches loaded together and defaults to one.
`run.json` records the dataset ID and consumption settings. `resources.json`
records training, evaluation and batch-loading times and GPU peak bytes when
provided by the backend. The generator logs its storage estimate by split
before writing data. At the default episode counts and rollout lengths,
the estimated array payload is about 5.8 GiB.

## Verification record

The paired reference implementation passed all 62 CPU tests, including
dense Gaussian checks for the posterior, grid integration and identity
plug-in. Slurm job `66669400` verified 8,192 calibration targets and both
reference scores on an NVIDIA A40 at $n=512$, $m=128$, and $\alpha=1.5$.
The local results are under
`runs/structured-kernel-pilot-20261123/oracle-n512-c504-verified`.

Slurm job `66669335` selected cuDNN automatically on an NVIDIA A40. It
completed three optimizer updates with all 32 episodes batched together,
at $n=512$, 504 context rows, $m=128$, and the 32-block training horizon.
Training and held-out metrics were finite. JAX reported 22.5 GiB of peak
device memory in use. The local artifacts are under
`runs/structured-kernel-pilot-20261123/cudnn-batch32-n512`.

The online path passed all 55 CPU tests. Slurm job `66669304` ran online
generation on an NVIDIA A40 at $n=512$, $m=128$, and $\alpha=1.5$. Three
optimizer updates at batch size two and two held-out evaluations through
four rollout blocks completed with finite metrics. The local artifacts
are under `runs/structured-kernel-pilot-20261123/online-n512-c504`.

### Offline pilot

The full CPU suite passed all 54 tests. Ruff lint and format checks passed
for `navi`, `experiments`, `tests` and `harness`. Both repository-wide Ruff
checks flag the existing `runs/bayes-update-66428137/pearson/replay.py`.

Slurm job `66669147` completed on an NVIDIA A40. All three runs completed
six optimizer updates and finite held-out evaluation at prefixes 1, 2, 4
and 8. The two $n=512$ runs used different model seeds and the same dataset
ID. Both scales used $m=128$, $\alpha=1.5$, 8 queries, 4 draws per block,
8 training episodes and 2 episodes in each held-out split.

| Population | Context | Run | Artifact (MiB) | Generation (s) | Training per update (s) | Batch loading per update (ms) | Peak device (MiB) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 64 | Initial | 1.71 | 9.56 | 7.00 | 8.78 | 148.35 |
| 512 | 504 | Initial | 4.54 | 9.62 | 8.99 | 9.04 | 148.69 |
| 512 | 504 | Replay | 4.54 | Reused | 8.98 | 7.24 | 148.69 |

Generation time includes process startup. Training averages include JIT
compilation. These runs verify the pipeline at the initial population size
with batch size one and a short rollout. The local pilot summary is in
`runs/structured-kernel-pilot-20261123/pilot-summary.json`.
