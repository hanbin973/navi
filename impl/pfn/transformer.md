# Transformer design

## Original PFN and row-token TabPFN

Let $n$ be the number of labeled rows, $m$ the number of query rows,
$p$ the input width after preprocessing, and $d$ the embedding width.
Stack context rows first and query rows second. Each row contributes one
$d$-dimensional token. The feature encoder $e_x:\mathbb R^p\to\mathbb R^d$
is shared across all rows. The target encoder $e_y:\mathcal Y\to\mathbb R^d$
maps the task's label representation into the same space.

$$
h_r^{(0)}=
\begin{cases}
e_x(x_r)+e_y(y_r), & 1\le r\le n,\\
e_x(x_j^*), & r=n+j,\quad 1\le j\le m.
\end{cases}
$$

The initial tensor $H^{(0)}$ has shape $(n+m)\times d$, with an optional
leading batch dimension. The target embedding is added to the context row
at the same index. Query rows receive only the feature embedding in the
reference TabPFN implementation. See
[TabPFN row encoder](https://huggingface.co/spaces/TabPFN/TabPFNEvaluation/blob/main/TabPFN/transformer.py).

The row-token design omits row positional embeddings to preserve the
set structure. In each block, scaled dot-product attention uses an additive
mask $M$ with zero for allowed pairs and $-\infty$ for blocked pairs.

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

Context tokens attend to context tokens. Each query attends to the context
independently of other queries. In an attention residual branch, head
outputs are concatenated and projected back to width $d$ before the
elementwise addition. The residual at row $r$ always comes from the
sublayer input at row $r$, so a query carries its own feature information
even when its attention values come entirely from context rows.

### Row-token block order

The original TabPFN reference layer defaults to post-normalization. With
dropout suppressed for clarity, one block maps $H$ to $H'$ as follows.

$$
\begin{aligned}
U &= \operatorname{LN}_1\!\left(H+\operatorname{MHA}(H;M)\right),\\
H' &= \operatorname{LN}_2\!\left(U+\operatorname{FFN}(U)\right).
\end{aligned}
$$

Each LayerNorm computes mean and variance over the $d$ embedding
coordinates of one token. The FFN applies a shared
$d\to d_{\mathrm{ff}}\to d$ transformation independently to every token.
When residual dropout is enabled, it acts on the sublayer output before
addition. The reference layer also exposes pre-normalization, which
changes the equations to
$U=H+\operatorname{MHA}(\operatorname{LN}_1(H);M)$ and
$H'=U+\operatorname{FFN}(\operatorname{LN}_2(U))$.
See [TabPFN reference layer](https://huggingface.co/spaces/TabPFN/TabPFNEvaluation/blob/main/TabPFN/layer.py).

A shared head maps each final query representation to class or bar logits.
A context-independent-of-queries attention mask allows
reuse of context keys and values. For $n$ context rows and $m$ queries, allowed
attention blocks contain $n^2+mn$ pairs. A dense masked implementation can
still allocate $(n+m)^2$ scores. See [original PFN](https://arxiv.org/abs/2112.10510)
and [original TabPFN](https://arxiv.org/abs/2207.01848).

## TabPFN v2

TabPFN v2 represents the table as a grid of embeddings. Let $R=n+m$ and
let $G$ be the number of encoded feature groups including the target slot.
The hidden state has shape $R\times G\times d$ for embedding width $d$.

### Feature and target tokens

Let $s$ be the number of features per group after padding. For row $r$
and feature group $g<G$, let $u_{r,g}\in\mathbb R^s$ hold preprocessed
values and let $a_{r,g}\in\mathbb R^s$ encode missing or nonfinite entries.
For the target slot, let $\tilde y_r$ be the preprocessed target value
and let $a_r^y$ be its missingness indicator. The v2 reference encoder
uses separate projections for features and targets.

$$
H^{(0)}_{r,g}=
\begin{cases}
W_x[u_{r,g};a_{r,g}]+c_g, & 1\le g<G,\\
W_y[\tilde y_r;a_r^y]+b_y, & g=G.
\end{cases}
$$

Here $[u;a]$ denotes concatenation and $c_g$ is a feature-group embedding
shared across rows. The query target starts as missing, so it uses an
imputed value derived from context statistics and a missing-target
indicator. The head reads only the final target slot $H^{(L)}_{n+j,G}$
for query $j$. See [TabPFN v2 reference encoder and decoder](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2.py).

### TabPFN v2 block order

A v2 block first mixes feature and target slots within each row, then
mixes rows independently at each slot, then applies the token-wise FFN.
Each sublayer returns a tensor of the same shape as its input. Its
residual addition pairs matching row and slot indices. The reference
implementation uses a separate, non-affine LayerNorm after each sum,
over the embedding width $d$ of each token.

$$
\begin{aligned}
U &= \operatorname{LN}_{\mathrm{feat}}(H + A_{\mathrm{feat}}(H)),\\
V &= \operatorname{LN}_{\mathrm{row}}(U + A_{\mathrm{row}}(U)),\\
H' &= \operatorname{LN}_{\mathrm{ff}}(V + \operatorname{FFN}(V)).
\end{aligned}
$$

$A_{\mathrm{feat}}$ attends across all slots within a row, including the
target slot. $A_{\mathrm{row}}$ uses context rows as keys and values for
both context and query rows at the same slot. For example, within-row
attention can put information about a query's features into its target
slot before across-row attention compares it with context target slots.
See [TabPFN v2 reference block](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2.py).

The two axes have separate attention operations. With independent query
rows, the context and context-to-query score blocks scale as
$G(n^2+mn)$ across rows, and dense within-row attention scales as $RG^2$.
These counts omit batch size, head count and embedding width. See
[TabPFN v2](https://doi.org/10.1038/s41586-024-08328-6).
