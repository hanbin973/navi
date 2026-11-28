# Linear transformer with draws as positional encodings

## 1. Purpose

This note collects the design for the draw path of the fixed-point equation, rewritten as a standard linear transformer. The goals are:

- a generic, maintainable shape built from standard attention blocks;
- a cheap variant, kept as an option, whose memory matches the current basis path without rematerialization;
- the two design principles enforced by the structure, not by hand-written contractions.

## 2. Setting and notation

Everything below happens within one step of the fixed-point iteration.

| Symbol | Meaning | Count |
|---|---|---|
| $k$ | draw index | $K$ |
| $i, j$ | context row (individual) | $N$ |
| $\xi[k,j]$ | scalar value of draw $k$ at row $j$, sampled from the prior $P$ | |
| $a$ | the current iterate, shape $N \times D$ | |
| $y$ | the observations | |
| $c_j$ | row $j$'s content token: the state-path output computed from $\mathrm{read}(a_j)$ and $y_j$, once per step | width $W$ |
| $\phi_r$ | fixed scalar basis functions of a draw value | $r = 1, \dots, R$ |
| $m$ | attention channel index (heads times head width) | $M$ |
| $e$ | value channel index | $E$ |
| $d$ | state channel index | $D$ |
| $\Sigma_P$ | covariance of the draws, $\mathbb{E}_P\left[\xi \xi^\top\right]$ | |

The step estimate is

$$
\widehat{\Phi}(a) = \mathrm{det}(c) + \frac{1}{K} \sum_{k=1}^{K} f[k, :, :],
$$

where $\mathrm{det}(c)$ is the draw-free term from the state head and $f[k,i,d]$ is draw $k$'s contribution.

## 3. Framing: draws are positions

In a standard transformer, tokens carry content and positional encodings carry the only information about how tokens relate. Here the same split holds:

- **Content.** Each context row's token $c_j$ carries the state and the observation, but nothing about the kernel.
- **Positions.** Each draw places every row at a random coordinate $\xi[k,j]$. The dependence between rows lives in how these coordinates co-vary across draws.

So $\xi[k,j]$ is row $j$'s position, resampled for every draw, and a fixed basis $\phi(\xi) = \left[\phi_1(\xi), \dots, \phi_R(\xi)\right]$ is the positional encoding. Attention learns to use the relations between rows only through these random positions.

The closest prior art is stochastic positional encoding for linear transformers (Liutkus et al., ICML 2021), which samples random positional codes whose cross-covariance implements a chosen attention kernel in expectation. Here the prior supplies that covariance.

## 4. Design principles

| Principle | Requirement |
|---|---|
| Fuse before mixing | A row's draw value must combine with its content multiplicatively before rows are mixed, so the state chooses what the prior is asked |
| Mix along rows, never along draws | Rows interact only within a single draw; draws are combined only through the final mean |

Together these make the step estimate a mean of single-draw terms, so it is unbiased for the mean-field map and its expectation does not depend on $K$.

## 5. Generic shape

Draws form the batch axis and context rows form the sequence axis. Each block is linear attention over rows followed by a per-token MLP, with residual connections.

**Content projections**, computed once per step:

$$
A_i = W_q \, c_i, \qquad B_j = W_k \, c_j, \qquad V_j = W_v \, c_j .
$$

**Positional modulation** of queries and keys only, in the style of rotary encodings:

$$
q[k,i,m] = A_{i,m} \, \psi_m\left(\xi[k,i]\right), \qquad
\kappa[k,j,m] = B_{j,m} \, \psi_m\left(\xi[k,j]\right),
$$

where $\psi_m$ is a fixed encoding channel: either one basis function $\phi_{r(m)}$ or a learned linear combination $\sum_r P_{m,r} \phi_r$.

**Attention over the rows of one draw**, with kernel $\mathrm{sim}$:

$$
o[k,i,e] = \frac{1}{N} \sum_{j=1}^{N} \mathrm{sim}\left(q[k,i,:], \kappa[k,j,:]\right) \, V_{j,e} .
$$

**Further blocks.** Optional additional blocks act on the per-draw tokens $o[k,i,:]$ with the same attention and MLP structure, still with draws as the batch axis.

**Head and mean.** A per-token head maps to $D$ channels, giving $f[k,i,:]$, and the step estimate averages over $k$.

The values never contain draws, so draw randomness enters only through the attention pattern.

## 6. Kernel choices

For linear attention the kernel is $\mathrm{sim}(q, \kappa) = \varphi(q)^\top \varphi(\kappa)$ for a feature map $\varphi$. Two independent choices apply.

| Choice | Meaning |
|---|---|
| Normalized | divide by $\sum_j \mathrm{sim}(q_i, \kappa_j)$, so the weights for each row sum to one |
| Unnormalized | use the fixed scale $1/N$ |
| Positive | $\varphi \geq 0$, for example $\varphi(x) = \mathrm{elu}(x) + 1$ |
| Signed | $\varphi$ can be negative, for example the identity |

Normalization requires positive weights, so signed kernels go with no normalization.

**Draw side: signed and unnormalized, with the identity feature map.**

- Signed, because the kernel information is the covariance of draw values, which has either sign and appears through cancellation across draws. Positive feature maps add a mean term $\mathbb{E}\left[\varphi(\xi_i)\right] \mathbb{E}\left[\varphi(\xi_j)\right]$ that swamps the covariance.
- Unnormalized, because a per-draw denominator is random, so the expectation of the ratio is not the ratio of expectations. A per-draw denominator also prevents averaging over draws before combining with content (Section 9).

**State side: positive and normalized (ELU+1) is fine.** The state path contains no draws, so no expectation over draws is involved.

## 7. What one layer computes in expectation

With the identity feature map, the expected attention weight between rows $i$ and $j$ is

$$
\mathbb{E}\left[ q[k,i,:]^\top \kappa[k,j,:] \right] = \sum_{m} A_{i,m} \, B_{j,m} \, \mathbb{E}\left[ \psi_m(\xi_i) \, \psi_m(\xi_j) \right] .
$$

The expectation on the right is a cross-moment matrix of the prior. The expected attention pattern is therefore the prior's kernel, weighted channel by channel by the content of the two rows.

**Gaussian prior.** For a single basis function $\phi$, Stein's identity gives

$$
\mathbb{E}\left[ \xi_i \, \phi(\xi_j) \right] = \left(\Sigma_P\right)_{ij} \, \mathbb{E}\left[ \phi'(\xi_j) \right],
$$

so every odd basis function contributes $\Sigma_P$ up to a rescaling of each row. With $\psi_m = \mathrm{id}$ on both sides, the expected weight is exactly $A_{i,m} B_{j,m} \left(\Sigma_P\right)_{ij}$: the covariance applied to vectors the state chooses.

## 8. The positional basis

The basis is fixed, not learned. Learned linear projections may follow it, as with sinusoidal positional encodings.

**Advantages of a fixed basis**

- no learned network runs per draw, so no hidden activations need storing;
- the moments each feature contributes are known;
- symmetry can be designed in explicitly.

**Parity rule.** For a prior symmetric under $\xi \to -\xi$, a term has zero expectation if its total degree in the draw values is odd. The query and key features must therefore have matching parity:

| Query feature | Key feature | Expected contribution |
|---|---|---|
| odd | odd | covariance-type moments, such as $\Sigma_P$ |
| even | even | magnitude co-movement, such as $\mathbb{E}\left[\xi_i^2 \xi_j^2\right]$; useful for sparse or scale-mixture priors |
| odd | even (or the reverse) | zero; adds only noise |

**Candidate features**

| Feature | Parity | Use |
|---|---|---|
| $\xi$ | odd | the covariance itself |
| $\tanh(\xi / s)$, $\sin(\omega \xi)$ | odd | bounded, lower-variance versions; sensitive to different scales for non-Gaussian priors |
| $1$ | even | pairs only with even features |
| $\xi^2$, $\lvert \xi \rvert$, $\cos(\omega \xi)$ | even | magnitude structure; enables terms such as $\Sigma_P \, \mathrm{diag}(u) \, \Sigma_P$ in deeper paths |

**Scale.** Frequencies and thresholds are meaningful only relative to the draw scale. Divide each row's draws by a running estimate of its scale before applying the basis. That estimate is an average over past draws, so it is a deterministic function of the carried state and keeps the update unbiased.

## 9. Cheap variant

**Conditions**

1. one draw-side layer, whose output is averaged over draws immediately;
2. signed, unnormalized kernel with the identity feature map;
3. evaluation in the separable order below.

**Separable order.** Write $r(m)$ for the basis function used by channel $m$.

Step 1, one summary per draw, contracting over rows:

$$
S[k,m,e] = \frac{1}{N} \sum_{j} \phi_{r(m)}\left(\xi[k,j]\right) \, B_{j,m} \, V_{j,e} .
$$

Step 2, mean over draws before applying query content:

$$
T[i,m,e] = \frac{1}{K} \sum_{k} \phi_{r(m)}\left(\xi[k,i]\right) \, S[k,m,e] .
$$

Step 3, query content, once per step:

$$
\mathrm{out}[i,e] = \sum_{m} A_{i,m} \, T[i,m,e] .
$$

A learned projection that mixes basis functions into channels folds into step 1 by contracting over $r$ there.

**Memory.** The backward pass needs only $\phi$, shape $K \times N \times R$, and $S$, shape $K \times M \times E$, which has no row axis. No array with both a draw axis and a channel axis per row is ever formed, so no rematerialization is needed. Write the three steps as separate contractions, because a single multi-operand einsum may choose a different order.

**Relation to the current basis path.** The current path is this variant with the content weights $A$ and $B$ removed, and with values that depend on the pair of basis functions. In that sense the cheap variant is its direct generalization.

## 10. Generic variant and rematerialization

These choices leave the separable order and require per-draw tokens of shape $K \times N \times M$:

| Choice | Reason |
|---|---|
| more than one draw-side layer | later layers act on per-draw tokens |
| normalized attention | each draw and row has its own denominator |
| nonlinear feature maps applied after modulation | the nonlinearity acts on the modulated tensors |
| learned draw embedding networks | their hidden activations are per draw and per row |

Memory can then be bounded at a compute cost:

- **checkpoint each rollout step**, so activations do not accumulate across steps;
- **accumulate the draw term over chunks of draws** with a scan, checkpointing each chunk; the estimate is unchanged because it is still a mean of per-draw terms.

## 11. Relation to existing implementations

| Existing component | Place in this design |
|---|---|
| State path with ELU+1 linear attention | Unchanged; supplies the content tokens $c_j$ and the draw-free term |
| Basis draw path | Special case of the cheap variant |
| Row kernels | Several query-side encoding channels |
| Staged draw path with $L$ stages | An alternative depth mechanism that updates one per-draw summary vector instead of per-draw row tokens |

## 12. Implementation plan

1. **Attention block.** One linear-attention block class with the kernel as a parameter (signed and unnormalized, or ELU+1 and normalized) and an optional positional hook that modulates queries and keys.
2. **Draw path module.** Content projections from $c$, positional modulation from fixed basis features, one or more blocks with draws as the batch axis, a head, and the mean over draws.
3. **Cheap option.** A fast path selected when the conditions in Section 9 hold, implemented in the separable order.
4. **Tests**
   - the estimate equals the mean of single-draw estimates, and is unchanged by reordering draws;
   - the cheap path matches the generic path exactly when its conditions hold;
   - equivariance to permuting rows;
   - compiled memory of the cheap path matches the current basis path.
5. **Validation.** One run of the cheap variant against the current baseline at matched settings, then one run of the generic variant with two draw-side layers.

## 13. Open questions

- **Conditioning.** Content weighting in the attention may let the iteration learn some preconditioning. The decisive test remains feeding an explicit sketched top subspace and checking whether the conditioning gap closes.
- **Even features.** Even-even channels are needed for sparse and scale-mixture priors, and for second-power terms such as $\Sigma_P \, \mathrm{diag}(u) \, \Sigma_P$. They are expected to matter little for the Gaussian prior.
- **Depth.** Whether a second draw-side layer helps enough to justify the generic variant's cost.
