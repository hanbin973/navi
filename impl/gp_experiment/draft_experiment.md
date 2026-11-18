# Gaussian feature-effect experiment

Measure how closely Navi predicts latent genetic values compared with the
best linear unbiased predictor (BLUP) under a known Gaussian feature-effect
prior. Train one Navi model across a distribution of genetic and residual
variances, inferring their effect on prediction from the observed context.
Each episode uses all 2,000 individuals and all 2,000 features, with
1,600 observed context phenotypes and 400 query targets.

## Data and splits

- Generate one context design matrix `X_C` of shape `(1600, 2000)` with
  independent standard-normal entries. Keep it and the context IDs fixed
  across training, validation, and testing. Draw a fresh query design
  matrix `X_Q` of shape `(400, 2000)` from the same feature distribution
  in each episode. Together the two blocks define that episode's 2,000
  individuals.
- For each episode, independently draw `log2(sigma_g²)` and
  `log2(sigma_e²)` uniformly from `[-2, 2]`. This proposed starting range
  gives each variance support from 0.25 to 4, varying both total scale and
  signal-to-noise ratio. Use the same hyperprior for training, validation,
  and testing.
- Given the episode variances, draw independent effects
  `beta[j] ~ Normal(0, sigma_g² / p)`, with the second argument denoting
  variance. Use the same effects for `g_C = X_C @ beta` and
  `g_Q = X_Q @ beta`. Draw independent
  residuals `eps[c] ~ Normal(0, sigma_e²)` for the context individuals.
- Each episode supplies fresh context observations `y_C = g_C + eps_C`
  and prediction targets `g_Q` for its newly sampled query individuals.
- Use separate PRNG streams for the design, model initialization, training,
  validation, and final evaluation. Within each episode, split the key for
  query features, both variance draws, feature effects, and residuals. Fix validation and
  test episode seeds before training and reserve the test episodes for final
  evaluation.

## Train Navi

Precompute `K_CC = X_C @ X_C.T / p`. In each episode, compute
`K_QC = X_Q @ X_C.T / p` and the query diagonal
`diag_Q = sum(X_Q ** 2, axis=1) / p` from the same query rows used to
generate `g_Q`. Use an experiment-local simulator to draw the episode
jointly and retain its sampled variances and query rows for evaluation.

Use `Navi` configured for a single
context value, one query feature, one pair feature, 32 bar logits, and
1,600 IDs. The model receives `y_C[:, None]`, the context IDs, the query diagonal
of `K` with shape `(400, 1)`, and `K[Q, C]` (the query-by-context block)
with shape `(400, 1600, 1)`. Keep the Gram features unscaled by the sampled
variances so Navi learns to predict with the variances latent. Recompute
the query diagonal and `kappa = K_QC[..., None]` for each episode.
Set the bar edges from simulated training-prior target quantiles. Train
with mean query negative log-density against `g_Q` and use the predictive
mean for point metrics.

Use Optax Adam with an initial learning rate of `1e-3`. Start with width 32,
four attention heads, one query layer, and one episode per optimizer step.
Check a full-size forward and backward step on an allocated GPU before
setting the training budget. Record the update budget and validation cadence
before the main run, and retain the checkpoint with the lowest mean
validation negative log-density.

## Evaluate against oracle BLUP

For each held-out episode, use the same `C`, `Q`, and `y_C` for both
predictors. With that episode's sampled variance parameters and known zero
mean, compute

```text
A = sigma_g² * K_CC + sigma_e² * I_1600
alpha = solve(A, y_C)
g_hat_Q = sigma_g² * K_QC @ alpha
```

Use a Cholesky solve in float64 for the oracle and check the solve residual
and finiteness. Positive residual variance makes `A` positive definite even
when `K_CC` is singular. The oracle uses the true variance parameters
directly, so the experiment needs no Gaussian MLE fitter.

Start with 100 held-out test episodes. Report the mean latent MSE for each
predictor, the paired difference in latent MSE (Navi minus BLUP) with a 95%
bootstrap interval over episodes, and the mean squared disagreement between
their predictions. Also summarize the paired error gap by true heritability
`sigma_g² / (sigma_g² + sigma_e²)`, with episode counts per bin.
Oracle BLUP is the conditional mean given the observed context and the true
variances. Navi's squared-error target is the conditional mean given its
inputs, which averages over the posterior uncertainty in those variances.
The measured gap can therefore include both the value of knowing the true
variances and Navi's approximation error. The finite-sample gap can have
either sign.

## Implementation and run checks

Keep the future runner and experiment configuration in `experiments/gp_experiment/`
at the repository root, outside the `navi` library. Before the full run,
verify on small synthetic episodes that context IDs and rows remain fixed,
distinct episode keys produce fresh query rows and `kappa`, and predictions
have the same shape as the latent targets, the training loss and gradients
are finite, and the context-space BLUP agrees with the equivalent
feature-space ridge prediction at several variance ratios. Verify that
Navi's inputs contain no direct variance metadata and that the oracle uses
the same draw that generated the episode. Use the project uv environment and follow
the [cluster harness](../../harness/cluster.md) for the full-size run. Record
the seeds, model and training settings, runtime, device, and evaluation
metrics.
