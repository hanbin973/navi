# Navi model draft

Navi encodes the observed phenotypes of identified context individuals
once per trait. Each test individual reads those context states through
cross-attention conditioned on its kernel values with the context
individuals. The default output head is a PFNs bar distribution for
scalar regression.

## Input and output

Shapes below omit a leading batch dimension for independent traits.
Let $n$ be the number of context individuals, $m$ the number of test
individuals, and $d_{mathrm{model}}$ the model width.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Context phenotypes $y$ | $n$ | One observed scalar $y_i$ per context individual |
| Context IDs | $n$ | Stable indices into the learned individual embedding table |
| Query features $U$ | $m\times d_u$ | One feature vector $u_*$ per test individual |
| Kernel values $\tilde\kappa$ | $m\times n\times d_\kappa$ | One vector $\tilde\kappa_{*i}$ per test–context pair |
| Output $Z$ | $m\times K$ | One logit per bar and test individual for the default head |

The context axis of $\tilde\kappa$ follows the order of $y$ and the
context IDs. Its query axis follows $U$. IDs identify individuals
independently of row position. Test phenotypes used for supervision stay
outside the forward inputs. Each trait requires at least one valid
context individual.

## Context encoder

Run the context encoder once per trait. Project each observed phenotype
and add the learned embedding $e_i$ for individual $i$:

$$
h_i^{(0)} = W_c\,y_i + c + e_i.
$$

Starting from these tokens, apply $L-1$ standard Transformer blocks.
Each block applies self-attention across context tokens and a token-wise
MLP, with a residual and layer normalization after each sublayer:

$$
t \leftarrow \mathrm{LN}\big(t + \mathrm{Attn}(t,t)\big),
\qquad
t \leftarrow \mathrm{LN}\big(t + \mathrm{MLP}(t)\big).
$$

Retain the initial states and every block output,
$h^{(0)},\ldots,h^{(L-1)}$. Context tokens attend exclusively to other
tokens in the context sequence, including themselves.

## Query path

Run the query path independently for each test individual $*$. Initialize
its state from its query features:

$$
q^{(0)} = W_u\,u_* + c_u.
$$

At each layer $\ell=1,\ldots,L$, build a pair representation for every
context individual using that layer's two-layer ReLU MLP. In parallel,
apply a learned linear map without intercept to the same kernel vector
to obtain one logit bias per attention head:

$$
p_{*i} = \mathrm{MLP}_\ell\big([h_i^{(\ell-1)};\ \tilde\kappa_{*i}]\big),
\qquad
\beta_{*i} = B_\ell\,\tilde\kappa_{*i} \in \mathbb{R}^{H}.
$$

The brackets denote concatenation. For each of the $H$ heads, project
the current query state $q=q^{(\ell-1)}$ to a query and the pair
representations to keys and values. For head dimension $d$, add the
bias to the scaled dot-product logit before taking the softmax over
context individuals. The projections below belong to layer $\ell$:

$$
a_i^{(h)} = \operatorname{softmax}_i\!\left(
\frac{1}{\sqrt{d}}\left\langle W_Q^{(h)}q,\ W_K^{(h)}p_{*i}\right\rangle
+ \beta_{*i,h}\right),
\qquad
\mathrm{read} = \sum_{h=1}^{H} W_{\mathrm{out}}^{(h)}
\sum_{i=1}^{n} a_i^{(h)} W_V^{(h)}p_{*i}.
$$

Apply the same residual and post-normalization order as on the context
side, using the cross-attention read in the first sublayer:

$$
q \leftarrow \mathrm{LN}(q+\mathrm{read}),
\qquad
q \leftarrow \mathrm{LN}\big(q+\mathrm{MLP}(q)\big).
$$

The result is $q^{(\ell)}$. The output head reads $q^{(L)}$. Each
attention read has one query token and $n$ pair tokens. Queries never
attend to each other.

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
| Phenotype and query projections | Use affine projections and add the learned individual embedding on the context side |
| Context Transformer | Reuse row-only `PerFeatureLayer` |
| Query block primitives | Reuse the attention, MLP, and normalization primitives where they support the specified equations |
| Query-specific memory | Build pair vectors and per-head logit biases from the aligned kernel vectors |
| Decoder and training | Share the fixed PFNs bar distribution between the default loss and inference. Supply task-specific episode sampling and supervision |

The wrapper owns context ID lookup, layerwise state capture, and query
assembly. Context states depend on the observed phenotypes, IDs, and
current model parameters. They can serve many queries during inference.
Pair representations and projected keys and values depend on the query's
kernel values, so compute them per query. Chunk the query axis if needed
to bound the temporary $Bmn d_{\mathrm{model}}$ pair activations per layer. At fixed width,
dense context attention uses $O(n^2)$ score work per context block, and
query attention uses $O(mn)$ score work per query block.

## Verification

- Compare context outputs with the upstream row-only layer on identical
  tokens, including the residual and normalization order.
- Compare batched query attention with a per-query loop using identical
  parameters. Check outputs and training gradients.
- Compare cached and freshly encoded context, and verify that changing
  query inputs leaves the context states unchanged.
- Compare attention with an explicit scaled dot-product reference that
  adds the kernel-derived bias before softmax.
- Jointly permute phenotypes, IDs, and the context axis of $\tilde\kappa$.
  Confirm that query outputs are invariant within numerical tolerance.
- Check that a single query gives the same output alone or in a chunk,
  and that reordering queries reorders outputs correspondingly.
- On CPU, compare bar losses and means with a reference for unequal widths,
  exact edges, and both tails. Compare batched and per-query losses and
  parameter gradients under JIT.
