# Navi helper draft

Navi training needs two generators: a query–context kernel generator for model inputs and a dataset simulator for training examples. Both generators receive the same ordered context and query selections. Deployment uses the kernel generator with measured context phenotypes.

The interfaces follow the [Navi model draft](../navi/model_draft.md) and the [PFN generalization of BLUP](../navi/pfn-generalization-of-blup.md).

## Abstract interfaces

Use two abstract Equinox modules with one abstract method each. Concrete subclasses own the prepared arrays and configuration for a kernel family or simulation prior. For the fixed-cohort draft, context and query selections can be ordered index arrays into prepared individual data. Stable context IDs travel with the context selection.

### QueryContextKernel

`features(context, queries) -> (u, kappa)`

For `n` context individuals and `m` queries, return `u[m, d_u]` and `kappa[m, n, d_kappa]`. The query axis follows the query selection. The context axis follows the context selection and aligns with the context phenotypes and IDs.

For a BLUP specialization with `M` kernels, `u[j, k]` contains kernel `k` evaluated at query `j` against itself. `kappa[j, i, k]` contains kernel `k` evaluated between query `j` and context individual `i`. Both outputs use the same kernel channel order.

### DatasetSimulator

`sample(key, context, queries) -> (f_query, y_context)`

Sample one trait jointly across the selected individuals. Return latent query targets `f_query[m]` and observed context phenotypes `y_context[n]` in selection order. The simulator samples prior parameters and latent values, then applies the observation model to the context latent values. The explicit JAX key identifies the draw. Context and query values share the same sampled prior parameters and latent realization.

## Episode assembly

During training, an episode assembly function converts scalar context phenotypes to `x[n, 1]` and calls `Navi(x, ids, u, kappa)`. The training loss scores the model predictions against `f_query`. During deployment, measured context phenotypes supply `x` and the kernel generator supplies `u` and `kappa`.

## Specialization order

1. **Fixed cohort and reserve training queries.** Prepare kernel and simulator data once. Cache query and pair features when the ordered selections are fixed and the outputs fit in memory. Sample fresh traits with independent keys.
2. **Held-out cohort training queries.** Select fixed-size context and query index arrays before calling the generators. Preserve stable IDs and gather or compute the selected kernel block. Generate query targets and context phenotypes from one joint draw.
3. **Larger cohorts and incoming queries.** Reuse prepared context data and compute pair features in query blocks. Each call supplies a dense pair tensor for the current block. Simulator subclasses can use prior-specific representations to reduce the cost of joint draws.

## First implementation checks

Start with a small Gaussian simulator with observation noise and a matching kernel generator. Check the joint latent covariance and the observed context covariance against the Gaussian reference. Use the analytic BLUP posterior as a reference for the resulting prediction task. Verify that permutations of the selections produce matching output permutations, cached features agree with direct computation, and fixed-shape calls work under `eqx.filter_jit` and batched sampling.
