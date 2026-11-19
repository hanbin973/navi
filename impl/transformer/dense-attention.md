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

## JAX API contract

Pass projected arrays as $Q:[B,T,H_q,D]$ and $K,V:[B,S,H_{kv},D]$.
The batch axis may be omitted. Keys and values share a shape, and
$H_q$ must be divisible by $H_{kv}$. The output has the query shape.
With `return_residual=True`, JAX also returns the softmax log-sum-exp
with shape $[B,T,H_q]$ or $[T,H_q]$. Projection and dropout belong to
the caller.

The API accepts a logits scale (default $1/\sqrt D$), causal attention,
four-dimensional broadcastable additive bias and boolean masks, per-batch
int32 query and key/value lengths, and a local window. Bias and mask
broadcast over $[B,H_q,T,S]$, and true mask entries participate. A
window integer is symmetric, and a pair specifies left and right spans.
Backend selection is static at the JIT boundary. The current default
`implementation=None` resolves to XLA.
[JAX API documentation](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)

| Capability | XLA | cuDNN |
| --- | --- | --- |
| Execution | Uses JAX's XLA lowering. Navi uses FP32 on CPU and CUDA. | Uses fused attention on a compatible NVIDIA CUDA stack. Explicit selection raises when a configuration is unsupported. |
| Shapes and dtypes | Supports square and rectangular dense attention, including a single query. Device-specific dot algorithms can limit low-precision execution. | The public API specifies a backend-dependent subset. In the installed JAX FP16/BF16 path on A40, the head width must be a multiple of 8 and at most 128. Test forward and backward independently. |
| Causality and windows | Applies the causal, padding, and local-window masks to logits. Accepts symmetric or asymmetric local windows. | Can skip causal regions in the kernel. The installed wrapper rejects a noncausal local window with a nonzero right span. Causal windows use the left span. |
| Bias, masks, and padding | Accepts additive bias, boolean masks, and per-batch sequence lengths under the API shape contract. | Accepts these options subject to kernel constraints. The installed wrapper converts boolean masks to additive bias and selects a padding mask when sequence lengths are supplied. |

## Navi CUDA checks

The dense Navi paths are noncausal with full visibility over their
supplied source. The context path uses $T=S$. Each independent query
read uses $T=1$ and $S$ equal to the context length. The trial uses
$[B,T,H_q,D]$ projections, two heads, and no mask or bias.

Slurm job `66351514` ran on an A40 with driver 595.71.05, JAX and
CUDA 13 plugin 0.12.0.1, and cuDNN 9.28.1.4. BF16 cuDNN results
were compared with BF16 and FP32 XLA. Training checks included
gradients with respect to Q, K, and V.

| Layout $(T,S,D)$ | XLA FP32 and BF16 | cuDNN BF16 forward | cuDNN BF16 training |
| --- | --- | --- | --- |
| $(16,16,32)$ | Passed | Passed | Passed |
| $(1,16,32)$ or $(1,15,32)$ | Passed | Passed | Rejected sequence lengths |
| $(3,3,2)$ or $(1,3,2)$ | Passed | Rejected head width | Rejected head width |
| $(1,16,32)$ with query duplicated inside the cuDNN call | Unmodified reference | Passed | Passed |

In the installed JAX cuDNN training check, absent bias is passed as an
empty array and the shape guard treats it as present. The resulting
even-length requirement applied to both sequence axes in these unmasked
calls. Earlier job `66351267` also found a GPU dot-precision error in
FP16 XLA for the single-query cases it tested.

The duplicated-query probe feeds two identical query rows to cuDNN and
keeps only the first output. Since the read is noncausal and both rows
see the same keys and values independently, this preserves the logical
single-query result and gradients. It adds a physical query row and was
verified here only with 16 keys and head width 32. Navi uses explicit
XLA for arbitrary context lengths.

## Grouped-query and multi-query attention

MHA uses $H_q=H_{kv}$. GQA shares one K/V head across each group of
$H_q/H_{kv}$ query heads, which requires divisibility. MQA sets $H_{kv}=1$.
The nominal K/V cache contains $2SH_{kv}D$ scalars per batch element and
layer. Query heads still compute their own attention weights. Moving from
MHA to GQA changes the projection shapes and checkpoint contract. See
[Grouped-Query Attention](https://arxiv.org/abs/2305.13245).
