# Navi helper implementation plan

The [helper draft](helper_draft.md) defines two independently reusable Equinox interfaces. Add the implementation to `navi/helper.py` and place focused CPU tests in `tests/test_navi_helper.py`. The first milestone covers fixed-cohort Gaussian simulation, reserve training queries, and held-out cohort training queries. Implementation of the CPU reference milestone is authorized.

[Navi](../../navi/model.py) already accepts `x`, stable context IDs, `u`, and dense pair features. The existing `encode_context` and `decode_queries` methods also support repeated query reads. Helper code can use both entry points with the existing model signatures.

## 1. Define abstract interfaces and a shared index convention

**Implemented.** Add `QueryContextKernel` and `DatasetSimulator` as `eqx.Module` subclasses with abstract methods:

- `features(context, queries) -> (u, kappa)` returns `u[m, d_u]` and `kappa[m, n, d_kappa]`.
- `sample(key, context, queries) -> (f_query, y_context)` returns vectors of lengths `m` and `n`, respectively.

Both selections are ordered one-dimensional integer arrays into the same prepared population. Keep stable model IDs separate from population row indices. Specify a nonempty context and query selection for the first implementation. Validate prepared shapes and index inputs at the host boundary before entering the compiled training loop. Keep prepared arrays as PyTree leaves and keep random key splitting explicit.

Verify the interface through the concrete implementations in task 2 and the episode tests in task 3.

## 2. Implement a small Gaussian reference pair

**Implemented.** Depends on task 1. Add concrete implementations to `navi/helper.py`.

- Store a stack of prepared population Gram matrices. Gather diagonal entries for `u` and ordered query–context blocks for `kappa`. Use the same kernel channel order in both outputs.
- Use fixed covariance weights and independent Gaussian observation noise for the reference prior. Prepare a factor of the weighted latent covariance across the full population, including reserve training queries. Each call draws one population latent vector, selects the query targets and context latent values, and adds noise to the selected context values using a separate subkey. Prepare a Cholesky factor on the host. The reference prior requires a symmetric positive-definite weighted covariance. Keep dense population storage and factorization confined to the reference implementation.

In `tests/test_navi_helper.py`, compare both kernel outputs against explicit indexing with reordered selections. Verify same-key reproducibility and compare empirical moments on a small fixture with the analytic latent query covariance, observed context covariance, and query–context cross-covariance. Use fixed seeds and tolerances based on sampling error. Check fixed-shape execution under `eqx.filter_jit` and compare mapped draws against calls with the same individual keys. For permutations, compare kernel outputs exactly or within floating-point tolerance and check simulation moments in selection order. Independent observation noise can change its row assignment for a reordered same-key draw.

## 3. Assemble and exercise training episodes

**Implemented.** Depends on tasks 1 and 2. Add an episode assembly function in `navi/helper.py`. Use the same selections for both generators, gather stable context IDs in context order, and pass `y_context[:, None]`, IDs, `u`, and `kappa` to Navi. Return predictions and latent query targets so the caller can apply the loss. Let the caller retain prepared features for fixed selections and reuse the kernel generator with measured phenotypes at deployment.

Test fixed reserve training queries and a held-out split using a tiny real Navi model. Compare helper predictions against the direct model call and verify target order and ID alignment with deliberately reordered selections. Compare cached and freshly generated features and predictions. Check the Gaussian conditional mean and covariance from the fixture against the analytic BLUP reference as a check on the prediction task. Learned model accuracy requires a separate training experiment.

## 4. Verify and record the milestone

**Verified.** Depends on tasks 2 and 3. Update the [repository map](../../harness/repository-map.md) when the helper module and tests are added. Run the behavioral and static checks from the repository root:

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_helper.py tests/test_navi_model.py
JAX_PLATFORMS=cpu uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
git diff --check
```

The focused tests cover alignment, joint simulation, and compiled execution. The full suite checks compatibility with existing model and PFNs behavior. Record actual outcomes and any unavailable checks. Follow the [cluster guidance](../../harness/cluster.md) before scheduling substantial computation.

## Verification record

The full CPU suite passed all 31 tests, including the focused helper and model
tests. Ruff lint and format checks passed. Whitespace and local Markdown links
were checked. The verification runs used the existing locked uv environment
with `--no-sync`. GPU and large-cohort performance checks remain outside the
CPU reference milestone.

## Follow-on specialization

After the CPU reference milestone, add an incoming-query implementation that evaluates kernel blocks against prepared context data. Compare concatenated query-block predictions with the dense reference and reuse encoded context across blocks. Add faster prior-specific samplers when a concrete prior and cohort size are available. Set throughput and memory targets from that workload before claiming a performance improvement.
