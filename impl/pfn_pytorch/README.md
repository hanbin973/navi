# PFNs PyTorch reference

The [PFNs submodule](PFNs/) is pinned at
[f04d3cad](https://github.com/SamuelGabriel/PFNs/tree/f04d3cad1d4d464f1039ec04d55b8f004b60494a).
For the model equations, see [PFN design notes](../pfn/README.md).

## Module structure

| Layer | Upstream modules | Responsibility |
| --- | --- | --- |
| Experiment entry points | [run_training_cli.py](PFNs/pfns/run_training_cli.py), [example_config.py](PFNs/example_config.py) | Load or construct the experiment configuration. The example wires a prior, encoders, regression criterion, optimizer, and batch shape. |
| Episode generation | [priors/prior.py](PFNs/pfns/priors/prior.py), [priors/utils.py](PFNs/pfns/priors/utils.py), [priors/data_loading.py](PFNs/pfns/priors/data_loading.py), [batch_shape_sampler.py](PFNs/pfns/batch_shape_sampler.py) | Construct samplers, compose prior wrappers, and choose episode dimensions and the context/query split. Standard and importance-sampling loaders implement different sampling workflows. |
| Input encoding | [model/encoders.py](PFNs/pfns/model/encoders.py) | Compose preprocessing and projection steps for inputs, targets, and optional style metadata. |
| Model assembly | [model/transformer_config.py](PFNs/pfns/model/transformer_config.py), [model/transformer.py](PFNs/pfns/model/transformer.py) | Build the model, group columns, embed observed targets, assemble tokens, run the layer stack, and decode query representations. |
| Transformer blocks | [model/layer.py](PFNs/pfns/model/layer.py), [model/multi_head_attention.py](PFNs/pfns/model/multi_head_attention.py), [model/mlp.py](PFNs/pfns/model/mlp.py), [model/layer_norm.py](PFNs/pfns/model/layer_norm.py) | Implement axis-specific attention, projections and K/V caching, feed-forward layers, and normalization. |
| Predictive distributions | [model/bar_distribution.py](PFNs/pfns/model/bar_distribution.py), [model/criterions.py](PFNs/pfns/model/criterions.py) | Interpret output logits and compute the regression or classification objective. |
| Training and applications | [train.py](PFNs/pfns/train.py), [optimizer/](PFNs/pfns/optimizer/), [scripts/](PFNs/pfns/scripts/) | Run optimization and checkpointing. Application adapters provide tabular prediction and Bayesian optimization workflows. |

## Interfaces worth reusing

**Episode sampler.** Implement a sampler returning the `Batch` contract or
supply one of the existing `get_batch` functions. `AdhocPriorConfig`
accepts callables or named modules and composes them through
`get_batch_sequence`. Keep labels available to the model (`y`) distinct
from the loss targets (`target_y`). When integrating with a separate
training loop, check axis order explicitly. Legacy prior code can use
sequence-first tensors, and the model normalizes input layout according
to `batch_first`.

**Encoders.** `EncoderConfig` builds composable `SequentialEncoder`
pipelines. Reuse the full pipeline when matching upstream preprocessing,
or use individual steps for missing-value indicators, normalization,
variable feature counts, and linear projection. Steps use keyed tensor
state and may depend on the context split or fitted statistics. When
porting them, keep context statistics and categorical metadata explicit
and preserve the channel order expected by the projection.

**Backbone.** Use `TransformerConfig.create_model()` for the standard
configuration or instantiate `TableTransformer` for custom encoders and
heads. In the ordinary supervised path, pass context features and
context labels as `x` and `y`, with query features as `test_x`. The
model infers the split from the length of `y`. With feature attention
enabled, the model appends a target token to each row's feature-group
tokens and reads the last slot on query rows. The row-only path instead
adds feature and target embeddings in a single slot. It requires the
feature encoder to produce one group. This checkout's row-only path uses
its own target encoding, including the handling of unknown targets.

**Custom decoder.** Supply `decoder_dict` when constructing
`TableTransformer`, or call with `only_return_standard_out=False`
and attach a downstream head to the returned `test_embeddings`.
For ordinary supervised tasks these embeddings have shape
`[batch, query_rows, embedding_width]`. Preserve the class order or
regression bin order expected by the criterion.

**Individual blocks.** `PerFeatureLayer` is the first extraction boundary
for a custom backbone that already has embedded tokens. Work with a grid
`[batch, rows, feature_slots, embedding_width]` and pass the context
length for attention across rows. If extracting `MultiHeadAttention`
directly, also account for the layer's axis reshaping, context/query
key selection, and optional caching. For a JAX port, implement the
uncached path first and compare its outputs and gradients before adding
cache or memory-saving branches.

**Regression distribution.** `BarDistribution` and
`FullSupportBarDistribution` can be paired with an independent backbone.
Keep the ordered bin edges and output width together. Reuse the same
distribution for the loss and for statistics such as quantiles. The
full-support variant's tail behavior is part of that contract. The
[distribution tests](PFNs/tests/model/test_bar_distribution.py)
are useful reference cases for a port.

For a small custom model, the shortest path is to reuse the backbone and
distribution, choose a new sampler, and keep the optimizer loop local.
For an Equinox implementation, use the PyTorch checkout as a reference
for layout, masking, and numerical comparisons. The upstream
[Apache 2.0 license](PFNs/LICENSE) covers the vendored source.
