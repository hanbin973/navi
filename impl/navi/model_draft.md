# Navi model draft

Navi encodes a fixed set of identified context items once per episode.
Each query reads the context through its own attention memory, built by
combining context states with query–context feature vectors. The network
takes numeric features and defaults to a PFNs bar-distribution head for
scalar regression. The caller supplies features, training targets, and
fixed bin edges on the target scale.

The architecture has two paths. A row-wise Transformer contextualizes
observations from the fixed set of items. A query stack uses those states
and pair features to produce an output independently for each query.

## Input and output

Shapes below omit a leading batch dimension for independent episodes.
Let $n$ be the number of context items, $m$ the number of queries, and
$d$ the model width.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Context observations $X$ | $n\times d_x$ | One observed feature vector $x_i$ per context item |
| Context IDs | $n$ | Stable indices into a learned table for the fixed set of context items |
| Query features $U$ | $m\times d_u$ | One feature vector $u_j$ per query |
| Pair features $\kappa$ | $m\times n\times d_\kappa$ | One generic feature vector $\kappa_{ji}$ for query $j$ and context item $i$ |
| Output $Z$ | $m\times K$ | One logit per bar and query for the default head |

The context axis of $\kappa$ follows the same order as $X$ and the context
IDs. Its query axis follows $U$. IDs identify items independently of their
row positions. Training targets stay outside the forward inputs. Each
episode requires at least one valid context item.

## Context encoder

Project each context observation and add the corresponding learned ID
embedding. Apply $L_c$ row self-attention blocks to the resulting tokens.

$$
h_i^{(0)} = E_x(x_i) + e_{\operatorname{id}(i)}, \qquad
H^{(\ell)} = C_\ell(H^{(\ell-1)}), \quad 1\leq \ell\leq L_c.
$$

Reuse PFNs `PerFeatureLayer` with feature attention disabled and one slot
per row. Its input layout is $[B,n,1,d]$. Treat all valid rows as context
and omit row-position encodings. Each block applies row self-attention,
adds its residual, and normalizes across the embedding coordinates. It
then applies a token-wise feed-forward network, adds its residual, and
applies a second normalization across the embedding coordinates. For the initial implementation, use batches with a common context
length and run the uncached upstream layer path. Navi retains the resulting
hidden states itself. See the [pinned PFNs layer](../pfn_pytorch/PFNs/pfns/model/layer.py).

## Query path

Project the query feature vector to initialize its state. At each query
layer, construct a separate memory for each query. Query layer $\ell$ reads $H^{(\ell-1)}$, so the
first query layer reads the projected observations and ID embeddings
before any context self-attention. With $L_q$ query layers, this schedule
uses $L_c=L_q-1$ context blocks.

$$
q_j^{(0)} = E_u(u_j),\qquad
p_{ji}^{(\ell)} = g_\ell\!\left([h_i^{(\ell-1)};\kappa_{ji}]\right),
\quad 1\leq \ell\leq L_q.
$$

The bracket denotes concatenation. Start with a learned MLP for
$g_\ell:\mathbb R^{d+d_\kappa}\to\mathbb R^d$. These pair representations
supply both keys and values. Apply the existing PFNs attention and
feed-forward modules with separate post-normalization stages.

$$
\begin{aligned}
a_j^{(\ell)} &=
  \operatorname{LN}_{\ell,1}\!\left(
    q_j^{(\ell-1)} +
    \operatorname{MHA}_{\ell}(q_j^{(\ell-1)},P_j^{(\ell)},P_j^{(\ell)})
  \right),\\
q_j^{(\ell)} &=
  \operatorname{LN}_{\ell,2}\!\left(a_j^{(\ell)} +
    \operatorname{FFN}_{\ell}(a_j^{(\ell)})\right),\\
z_j &= D(q_j^{(L_q)}).
\end{aligned}
$$

$P_j^{(\ell)}$ stacks the $n$ pair vectors for query $j$. In the upstream
attention API, the query state is the input and the pair tensor is the
separate source for both K and V. Flatten episode and query axes into a
batch of independent reads. Each read has one query token and $n$ memory
tokens. Use upstream residual addition or explicit addition once per
sublayer.

## Default output head and training loss

Use the PFNs decoder with two linear layers and GELU between them, shared
across queries. It maps each final query state to $K$ logits.
Pass $K+1$ finite, strictly increasing reference edges $b_0,\ldots,b_K$
as the `bin_edges` argument to `Navi`, with $K\ge 2$. Estimate the edges once, for example from
training-prior target quantiles, and reuse them in training and inference.
Use `model.distribution.loss(logits, targets)` for training and
`model.distribution.mean(logits)` for point predictions. An explicit
custom `head` takes an `out_size` and uses a caller-defined objective.

Follow the pinned PFNs [full-support bar distribution](../pfn_pytorch/PFNs/pfns/model/bar_distribution.py).
Softmax gives the bar masses $\pi_k$. For each interior bar the
conditional density $f_k(y)$ is uniform with width $b_k-b_{k-1}$.
The first and last bars are outward-facing half-normal densities anchored
at $b_1$ and $b_{K-1}$. Set each tail scale to its reference width divided
by the median of a unit half-normal, so half of each tail bar's mass lies
within its reference interval. Use the upstream bucket convention, which
assigns an exact interior edge to the bar on its left.

For a nonempty episode with finite scalar targets $y_j$, minimize the
mean negative log-density over queries:

$$
\mathcal L=-\frac{1}{m}\sum_{j=1}^{m}
 \left[\log\pi_{k(y_j)}(z_j)+\log f_{k(y_j)}(y_j)\right].
$$

Use log-softmax and gather the selected bar. Compute both the loss and
distribution mean from the same edges and tail scales. The mean weights
interior midpoints and half-normal means by their respective bar masses.
Point-metric comparisons such as BLUP use this distribution mean.

## Reuse and execution

| Component | Implementation boundary |
| --- | --- |
| Observation and query projections | Reuse linear or MLP projections where their input layout fits |
| Context Transformer | Reuse row-only `PerFeatureLayer` |
| Query block primitives | Reuse `MultiHeadAttention`, `MLP`, and `LayerNorm` |
| Query-specific memory | Add the pair encoder and the wrapper that aligns, batches, and reads the pair features |
| Decoder and training | Share the fixed PFNs bar distribution between the default loss and inference. Supply task-specific episode sampling and supervision |

The wrapper owns context ID lookup, layerwise state capture, and query
assembly. Context states depend on the context observations, IDs, and
current model parameters. They can serve many queries during inference.
Pair representations and projected keys and values depend on the query's
pair features, so compute them per query. Chunk the query axis if needed
to bound the temporary $Bmn d$ pair activations per layer. At fixed width,
dense context attention uses $O(n^2)$ score work per context block, and
query attention uses $O(mn)$ score work per query block.

## Verification

- Compare context outputs with the upstream row-only layer on identical
  tokens, including the residual and normalization order.
- Compare batched query attention with a per-query loop using identical
  parameters. Check outputs and training gradients.
- Compare cached and freshly encoded context, and verify that changing
  query inputs leaves the context states unchanged.
- Jointly permute observations, IDs, and the context axis of $\kappa$.
  Confirm that query outputs are invariant within numerical tolerance.
- Check that a single query gives the same output alone or in a chunk,
  and that reordering queries reorders outputs correspondingly.
- On CPU, compare bar losses and means with a reference for unequal widths,
  exact edges, and both tails. Compare batched and per-query losses and
  parameter gradients under JIT.
