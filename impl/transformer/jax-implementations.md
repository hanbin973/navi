# JAX linear-attention implementation survey

| Upstream | Implementation and intended use | Maintenance evidence and integration scope |
| --- | --- | --- |
| [MaxText Qwen3](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/models/qwen3.py) | Flax NNX Gated DeltaNet with JAX chunkwise and recurrent paths. A reference for model training and sharding. | [Release notes](https://github.com/AI-Hypercomputer/maxtext/releases) document Gated DeltaNet context parallelism. Its model layer takes MaxText configuration and mesh conventions. |
| [SGL-JAX linear attention backends](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/attention_backend.md) | Gated DeltaNet and other hybrid recurrent inference paths. The [Kimi Linear guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/autoregressive/Moonshotai/Kimi-Linear.md) shows a model deployment. | Its [GDN prefill issue](https://github.com/sgl-project/sglang-jax/issues/1416) tracks kernel development. The implementation is coupled to serving metadata and recurrent state management, and training support requires separate confirmation. |
| [jax-linear-attention](https://github.com/jla-org/jax-linear-attention) | Specialist Pallas Mosaic GPU kernels for several recurrent linear-attention operators, including Gated DeltaNet variants, with documented forward/backward coverage for individual kernels. | The repository documents per-operator coverage and Hopper benchmarks. Treat this as a hardware-specific candidate and verify its commit, test coverage, and JAX compatibility before adoption. |

Start with the plain JAX operator for a numerical reference. Compare
each candidate on the same shapes, masks, initial state, and dtypes. For
a recurrent operator, check one-shot versus chunked execution, including
the returned state and boundaries between packed examples. For training,
compare gradients under JIT as well as forward outputs. For timing,
warm up compilation, synchronize outputs, and report device, package
versions, dtype, forward/backward mode, and peak device memory. Project
GPU checks are subject to the [cluster guidance](../../harness/cluster.md).
