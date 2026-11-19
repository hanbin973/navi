# Linear attention and recurrent memory

## Feature-map attention

Let $\phi(q_i),\phi(k_j)\in\mathbb R^r$ be nonnegative feature vectors
and $v_j\in\mathbb R^{d_v}$. A normalized kernel attention computes

$$
S=\sum_j\phi(k_j)v_j^\top,\qquad z=\sum_j\phi(k_j),\qquad
o_i=\frac{S^\top\phi(q_i)}{z^\top\phi(q_i)}.
$$

Associativity allows every query to reuse $S$ and $z$. The attention
work for $T$ queries and $N$ keys is $O((T+N)rd_v)$ after computing the
features. For causal attention, keep prefix sums in a JAX scan and update
the state before reading the output if self-attention is inclusive.
Performer uses random features to approximate the softmax kernel.
A deterministic feature map such as ELU plus one defines its own kernel.
See [Linear Transformers](https://arxiv.org/abs/2006.16236) and
[Performer](https://arxiv.org/abs/2009.14794).

For a PFN experiment, build the sums solely from valid context rows and
read them for both context and query rows. The sums are permutation
invariant in exact arithmetic and do not mix query rows. Use an explicit
empty-context policy and a documented denominator floor if needed.
Accumulate sums in FP32 when using low-precision projections, then
compare against explicit pairwise evaluation of the same kernel.
A general pairwise mask may prevent reuse of a single summary.

## Gated delta-rule attention

Let $S_t\in\mathbb R^{d_k\times d_v}$ be the recurrent state after
token $t$. A scalar-gated delta update can be written as

$$
\widehat S_t=\alpha_t S_{t-1},\qquad
S_t=\widehat S_t+\beta_t k_t(v_t-\widehat S_t^\top k_t)^\top,\qquad
o_t=S_t^\top q_t.
$$

The decay $\alpha_t$ discounts older state. The update gate $\beta_t$
controls how much the current key corrects the value retrieved from
that state. Kimi Delta Attention extends the decay to key channels.
A single recurrent step costs $O(d_k d_v)$ per head and keeps fixed-size
state. Chunkwise implementations trade dense work within chunks for
parallelism, and training can still retain activations for backward.
These order-dependent updates fit sequence models and need an explicit
ordering contract before use on unordered tabular rows. Hybrid models
combine recurrent layers with softmax-attention layers. See
[Gated Delta Networks](https://arxiv.org/abs/2412.06464) and
[Kimi Linear](https://arxiv.org/abs/2510.26692).
