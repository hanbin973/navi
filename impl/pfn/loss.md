# Prior-data loss and predictive heads

Write $U=(C,x_*)$. A draw $(U,Y)$ from the simulator supplies an unbiased
sample of the single-query negative log-likelihood objective.

$$
\mathcal L(\theta)
= \mathbb E_{U,Y}[-\log q_\theta(Y\mid U)]
= \mathbb E_U\!\left[H(p(\cdot\mid U))+
D_{\mathrm{KL}}(p(\cdot\mid U)\,\Vert\,q_\theta(\cdot\mid U))\right].
$$

The second equality follows by adding and subtracting $\log p(Y\mid U)$
inside the expectation. Here $H$ is entropy for classes and differential
entropy for densities, when the quantities exist. The entropy term is
independent of $\theta$. A sufficiently expressive predictor at the
population optimum matches the simulator's posterior predictive on its
support. See [the original PFN paper](https://arxiv.org/abs/2112.10510).

For an episode with $m$ queries, use the mean of the query losses
$-\frac{1}{m}\sum_{j=1}^m\log q_\theta(y_j^*\mid x_j^*,C)$.
When batching variable-length episodes, exclude padded queries. Averaging
per episode weights tasks equally, while averaging over all valid queries
weights episodes in proportion to their query count.

## Classification

For $K$ classes, let $\ell\in\mathbb R^K$ be the query logits. The loss for
class $k$ is $\operatorname{logsumexp}(\ell)-\ell_k$, and softmax gives the
class probabilities. If the head has a fixed maximum class count, mask
unused classes before normalization.

## Continuous regression

A bounded bar distribution uses edges $b_0<\cdots<b_K$ and predicts masses
$\pi_k=\operatorname{softmax}(\ell)_k$ for bins
$I_k=[b_{k-1},b_k)$, including the right endpoint in the final bin. Within a
bin, the predictive density is uniform.

$$
q_\theta(y\mid U)=\frac{\pi_k}{b_k-b_{k-1}},\qquad
-\log q_\theta(y\mid U)=-\log\pi_k+\log(b_k-b_{k-1}),
\qquad y\in I_k.
$$

Bin width converts probability mass into density. Edges can be estimated
from synthetic target quantiles to give more resolution in high-density
regions. When targets can fall beyond the bounded support, define
normalized tail densities for the outer bins. Means and quantiles should
use the same within-bin and tail distributions as the loss.
