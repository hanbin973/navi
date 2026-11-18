# Fused dense softmax attention in JAX

Dense attention computes

$$
Y=\operatorname{softmax}\!\left(QK^\top/\sqrt{d}+B+M\right)V,
$$

where $B$ is an optional score bias, $M$ assigns $-\infty$ to disallowed
positions, and softmax normalizes over keys. FlashAttention tiles the
computation and maintains online softmax statistics instead of storing the
full pairwise score and probability matrices in device memory. Its dense
arithmetic still scales quadratically in sequence length. See
[FlashAttention](https://arxiv.org/abs/2205.14135).

## JAX cuDNN backend

Pass **already projected** query, key, and value arrays to
JAX, with layout $Q:[B,T,H_q,D]$ and $K,V:[B,S,H_{kv},D]$.
The output has the query shape. The attention primitive applies no output
projection or attention dropout.

~~~python
import jax


@jax.jit
def causal_attention(q, k, v):
    return jax.nn.dot_product_attention(
        q, k, v, is_causal=True, implementation="cudnn"
    )
~~~

This example is for causal self-attention with equal query and key
lengths. The cuDNN path requires a compatible NVIDIA GPU and JAX CUDA
environment. Use FP16 or BF16 inputs for an initial compatibility check,
then check the installed JAX/cuDNN versions for supported head dimensions,
dtypes, sequence lengths, masks, and gradients. JAX documents that explicit
cuDNN selection can raise for unsupported shapes or dtypes. Compile a
separate reference using explicit XLA selection and compare outputs and
training gradients with tolerances appropriate to the dtype. Backend choice
should be static at the JIT boundary. See
[JAX API and shape contract](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html).

A boolean mask uses true for allowed entries and broadcasts to
$[B,H_q,T,S]$. For PFN context-only keys, a rectangular call can use all
context and query rows as queries while supplying only context keys and
values. Its allowed edges match the design when both row types attend to
all valid context rows. For padded episodes, check sequence-length support
and exclude padded query predictions from the loss. Compare fused and XLA
results for the actual PFN shapes and padding patterns.

## Grouped-query and multi-query attention

MHA uses $H_q=H_{kv}$. GQA shares one K/V head across each group of
$H_q/H_{kv}$ query heads, which requires divisibility. MQA sets $H_{kv}=1$.
The nominal K/V cache contains $2SH_{kv}D$ scalars per batch element and
layer. Query heads still compute their own attention weights. Moving from
MHA to GQA changes the projection shapes and checkpoint contract. See
[Grouped-Query Attention](https://arxiv.org/abs/2305.13245).
