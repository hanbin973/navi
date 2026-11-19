# Implementation plan for approximate Bayesian updates with Navi

Train a permutation-equivariant update to fold one kernel draw at a time into
context state, and use Navi to read out a predictive density from that state.
Train both networks end to end with the same PFN loss at selected prefixes.
The first experiment measures whether the learned predictor continues to
improve after it has processed more draws than it saw during training.

The plan targets the Navi repository root. This consolidation covers the
design, implementation, and experiment verification. Use the project uv
environment for local checks and the repository Slurm harness for substantial
training.

## 1. Bayesian target

Let $D=(y,\kappa)$, where $y\in\mathbb R^n$ contains context phenotypes
and $\kappa$ includes both query features and query–context features. Let
$K$ be the kernel on those ordered context tokens. Conditional on $K$,
the draws $\xi^{(s)}\in\mathbb R^n$ are i.i.d. with law
$p(\xi\mid K)$, and the entire draw stream is conditionally independent
of $(f,f_*,y,\kappa)$. The trait prior can be any joint episode law that
satisfies this factorization. For recovery of the full kernel, require the
infinite draw stream to determine $K$ almost surely.

Write $q_0(f_*)=p(f_*\mid D)$. For a prefix of length $t$, Bayes'
rule gives

$$
\begin{aligned}
q_t(f_*) &= p(f_*\mid D,\xi^{(1:t)})\\
&= q_{t-1}(f_*)\,
\frac{p(\xi^{(t)}\mid f_*,D,\xi^{(1:t-1)})}
     {p(\xi^{(t)}\mid D,\xi^{(1:t-1)})}.
\end{aligned}
$$

The ratio uses predictive draw laws that integrate over the remaining
uncertainty in $K$. For each fixed measurable event $A$, $q_t(A)$
is a bounded conditional-expectation martingale. Its almost-sure limit is
$p(f_*\in A\mid D,\xi^{(1:\infty)})=p(f_*\in A\mid D,K)$, by
recoverability and conditional independence. This gives convergence of
posterior probabilities for fixed events, including the PFN bars.
Pointwise convergence of densities would require additional regularity.

## 2. Model and reuse

Use the tensor contract below for a single episode. The query count is $m$.

| Value | Shape |
| --- | --- |
| Context phenotypes $y$, supplied to Navi as a column | $(n,1)$ |
| Query features and query–context pair features | $(m,d_u)$, $(m,n,d_\kappa)$ |
| Draws and recurrent state | $(T,n)$, $(n,d_b)$ |
| Query targets and bar logits | $(m,)$, $(m,B)$ |

Initialize $b_0=0$. Compose a new Equinox update module with the Navi
readout. Concatenate $[b_{t-1,i},\xi_i^{(t)},y_i]$ per row, apply a shared
input projection, mix rows with a dense context block, and project back to
$d_b$. Reuse the existing attention and MLP primitives in
[`navi/model.py`](../../navi/model.py). Cross-token mixing lets the update
use information about joint draws. Keep the same context order for $y$,
every draw, state rows, and the context axis of query pair features.
Use shared row operations and attention without positional or ID features.

Seed the readout with $W_c y_i + W_b b_{t,i}$, using Navi's existing
context projection for $W_c$ and a new bias-free state projection for
$W_b$. Factor the layerwise encoder stack out of `Navi._encode` so either
input path can start from projected tokens. Keep the existing ID-based
entry points and their parameter layout intact. Use shared internal functions that operate on the encoder and decoder
components so the new readout can own a state projection in place of the
ID table while the existing Navi module retains its fields. Reuse the layerwise context states, pair encoders,
kernel bias, decoder, and query head. A context encoding can serve multiple
query batches at a given $t$. Recompute it when the state changes.
The head predicts one marginal density per query, following the existing
Navi contract.

Reuse [`FullSupportBarDistribution`](../../navi/distribution.py) for the
sampled-target log loss. Use one set of bar edges calibrated on training
prior targets for all learned curves. Within this fixed bar family, the
population-optimal logits describe conditional bar probabilities. The
within-bar density and tail shapes are fixed by the family.

## 3. Rollout and training

For random $1\le T\le T_{\max}$, score powers of two up to $T$ and
include the terminal step once. Use nonnegative weights that sum to one
within the episode, initially uniform over the selected steps. Sample
$T$ independently of the episode and draws. Compute

$$
\mathcal L(\phi)=\mathbb E\left[
\sum_{t\in\mathcal T(T)} w_t
\left(-\log q_\phi(f_*\mid D,b_t)\right)\right],
\qquad b_t=U_\phi(b_{t-1},\xi^{(t)},y).
$$

Average the marginal losses over queries before reducing over time and
episodes. For fixed state mapping and step, unrestricted log loss recovers
$p(f_*\mid D,b_t)$. That is the prefix posterior when the state is
predictively sufficient, meaning
$f_*\perp\xi^{(1:t)}\mid(D,b_t)$. State width is a capacity choice.
Sufficiency also depends on the update family and learning. With shared
readout parameters across steps, the state must retain enough information
about the amount of evidence to distinguish prefixes with different
uncertainty. Include a broadcast deterministic prefix-count feature in the
readout's state input, for example $1/(1+t)$, when using the same readout
across steps. This feature tends to zero at the infinite horizon.

Use a fixed-shape `jax.lax.scan` over $T_{\max}$ with a sampled horizon
mask for state advances and a score mask for the loss. Keep the carry shape
and FP32 dtype fixed. Use scalar scores so the scan need not emit all
logits or states. Use scalar conditional scoring for single episodes and
profile before batching sparse scoring, since mapping the predicate can
turn a conditional into selection. Apply filtered JIT and
`eqx.filter_value_and_grad` around the rollout loss so late losses
differentiate through earlier updates. Reuse the optimizer and finite
gradient checks from the [GP runner](../../experiments/gp_experiment/run.py)
as the training pattern. Add rematerialization if backpropagation memory
limits the intended training horizon.

## 4. Damped update and long-horizon inference

Use the residual update in the first experiment so training and evaluation
share the same recurrence.

$$
b_t=b_{t-1}+\gamma_t F_\phi(b_{t-1},\xi^{(t)},y),
\qquad \gamma_t=\eta(t+t_0)^{-\alpha},
\quad \eta>0,\ t_0\ge 0,\ \tfrac12<\alpha\le 1.
$$

This schedule decays to zero, has divergent sum, and has square-summable
steps. Convergence to a deterministic function of $(K,y)$ also requires
stochastic approximation conditions. In particular, the conditional mean
drift $h(b;K,y)=\mathbb E[F_\phi(b,\xi,y)\mid K]$ needs an attracting
root with a basin covering the reachable iterates, along with stability of
the iterates and control of the update noise. Multiple basins can produce
history-dependent limits. A learned stable fixed point can still lose
predictive information about $K$. PFN loss near the end of training
encourages a useful late readout, and the long-horizon test measures whether
that behavior persists. Log state norms and drift estimates at late
checkpoints to distinguish stability from small increments caused by
$\gamma_t$ alone.

Stream evaluation in fixed-size chunks, retaining $b_t$ and the global
step so the schedule and draw keys continue across chunks. Decode only at
requested checkpoints. Derive each draw key from the episode draw key and
its absolute index to make the prefix reproducible across chunk sizes.

## 5. First experiment

1. **Simulate variable kernels.** Reuse `draw_episode` and `query_features`
   from [GP data](../../experiments/gp_experiment/data.py). Sample a fresh
   standardized context feature matrix $X\in\mathbb R^{n\times p}$ per
   episode, with fixed $n,p$ and the existing GP feature scaling. Set
   $K=XX^\top/p$. Generate draws with
   $\xi^{(s)}=Xz_s/\sqrt p$, using fresh standard-normal $z_s$ from a
   stream independent of the panel, trait and horizon draws. Conditional on
   $X$ their law is $\mathcal N(0,K)$, which depends on $X$ only
   through $K$. The sample second moment recovers $K$ almost surely,
   including for rank-deficient panels. Hold out episode keys and panels.
   A fixed context panel would let the learned weights absorb that one
   kernel and weaken the test of information gained from the draw stream.

2. **Add a direct-kernel density baseline.** Train a Navi readout on the same
   episode law and PFN targets with exact $K$ supplied to its context
   encoder. For a first implementation, use a learned per-head attention
   bias derived from $K_{ij}$ and pass $K_{ii}$ as a row feature.
   Keep the decoder, bars, episode splits and training budget comparable.
   Expose the full kernel without ordering-dependent row vectors.
   Also report a $D$-only readout using phenotype tokens without IDs.
   The existing `oracle_predict` in
   [GP evaluation](../../experiments/gp_experiment/evaluation.py) returns
   a mean given true variance parameters. It can contextualize MSE.
   A PFN loss baseline needs a predictive density with the same information
   contract as the exact-kernel model.

3. **Test continuation of a frozen model.** Train with random
   $T\le T_{\max}$, then evaluate one uninterrupted draw prefix per
   held-out episode on a logarithmic grid through at least
   $16T_{\max}$, including $T_{\max}$. Plot mean query NLL against
   $t$, mark the training boundary, and show both learned baseline
   losses as horizontal references. Use paired bootstrap intervals over
   episodes for changes from $T_{\max}$ and gaps to the direct-kernel
   baseline. Record the seeds, effective horizon and schedule, per-episode
   scores, and late-state diagnostics. Evidence for extrapolation is an
   improvement beyond $T_{\max}$ and a narrowing gap toward the
   direct-kernel baseline, with uncertainty reported. Expected Bayes log
   risk decreases with more information under finite-risk conditions.
   The learned curve can plateau, fluctuate, or degrade, and the learned
   direct-kernel score is a reference rather than a theoretical bound.
   If the two baselines are indistinguishable, the episode design has not
   established measurable room for draw-based improvement.

## 6. Implementation tasks

### 1. State-conditioned readout and equivariant update

**Implemented.** Share projected-token encoder and decoder internals in
`navi/model.py`. Add the state projection and Equinox update/readout
modules in `navi/bayes_update.py`. Keep the existing ID-based Navi
signatures and parameter layout. This task can proceed alongside the
episode sampler in task 3.

Verify existing model and PFN parity tests. In a focused test, jointly
permute phenotype rows, draw coordinates, state rows and the context
axis of query pair features. Check that the update permutes the state
and that the marginal query logits agree within FP32 tolerance.

### 2. Prefix training and streaming continuation

**Implemented.** Build on task 1. Implement the damped step, sparse scored
training scan, and chunked inference using one step implementation.
Keep the absolute step in the streaming carry and use the same prefix
feature and schedule throughout.

Compare a short scan with explicit steps and a manual weighted loss.
Cover $T=1$, a power-of-two terminal step, and a non-power-of-two terminal
step. Verify that steps beyond the episode horizon leave state and loss
unchanged. Check filtered JIT and batched versus individual scores. On a
nondegenerate fixture, differentiate a late-only loss through the initial
state and verify finite, nonzero gradients for the update and readout.

### 3. Varying-panel experiment and baselines

**Implemented.** Add the sampler and runner under `experiments/bayes_update/`.
The sampler can be developed independently. The learned runs require
tasks 1 and 2. Use independent keys for panels, trait simulation, horizons
and draws. Log the effective configuration and emit paired per-episode
scores at each checkpoint. Train the exact-kernel and phenotype-only
readouts on the same episode law.

Check the empirical draw second moment against the Gram on a small
fixed panel with a sampling tolerance. Verify the direct-kernel baseline
is invariant under joint permutation of both kernel axes and the context
inputs. On the same held-out episode, compare one-shot and chunked
evaluation at common checkpoints. Run a tiny CPU experiment with finite
scores for all three curves and check that evaluation leaves model
parameters unchanged.

### 4. Held-out extrapolation experiment

**Submitted.** Slurm accepted job `66428137`. Evaluation results are
pending. Save the plot, paired scores, uncertainty
intervals and state diagnostics described in §5. Report the
post-horizon change and the gap to the direct-kernel baseline. An
inconclusive result still answers the experiment only to the precision
supported by the paired scores.

## 7. Verification and execution record

Run from the repository root with the project uv environment.

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

Use [development.md](../../harness/development.md) for local checks and
[cluster.md](../../harness/cluster.md) to confirm site settings before the
full run. Update the repository map when the implementation modules land.
As tasks are completed, record commands, outcomes, deviations, and any
checks that remain blocked here.

### Local verification

- All 50 CPU pytest tests passed.

- Ruff lint and format checks passed on tracked Python files and the new
  implementation and test files. The Slurm script passed `bash -n`.
- A CPU smoke run with pre-norm and three query layers trained all three
  readouts for two updates and scored two held-out episodes through 48
  draws with a training horizon of three. It wrote the model checkpoint,
  paired scores, summary, and plot.
- The experiment uses nonzero attention initialization for every readout
  so predictive loss can reach the updater from the first training step.
  Existing ID-based Navi initialization is preserved.

### GPU configuration

The submission script requests one GPU on `spgpu`, two CPUs, 32 GiB of
host memory, and a two-hour limit. The runner uses pre-norm and three
query layers. It defaults to 5,000 updates, 64 context tokens, 16 features,
eight queries, a maximum training horizon of 32, and 128 held-out episodes
evaluated through 512 draws. Results are written to
`runs/bayes-update-<jobid>/`.
