# Posterior predictive theory

Let $z$ describe a task, such as a latent function and its noise parameters.
The prior $p(z)$ and the simulator together induce a joint distribution over
a labeled context $C=\{(x_i,y_i)\}_{i=1}^n$ and a query pair $(x_*,y_*)$.
The Bayesian prediction integrates over tasks compatible with the evidence.

$$
p(y_* \mid x_*,C)
= \int p(y_* \mid x_*,C,z)\,p(z\mid x_*,C)\,dz.
$$

If $z$ makes the query target conditionally independent of the context given
$x_*$, the first term reduces to $p(y_*\mid x_*,z)$. If the query covariate
carries no additional information about $z$ given $C$, the second term reduces
to $p(z\mid C)$. The general form also covers tabular generators in which
feature distributions depend on $z$.

A PFN amortizes this integral by learning a mapping
$(C,x_*)\mapsto q_\theta(\cdot\mid x_*,C)$ across simulated episodes. The
learned weights $\theta$ encode a prediction rule shared across tasks.
At inference time, the context supplies the evidence used by that rule.

For exchangeable context rows, the prediction should be invariant to their
order. For multiple queries, the usual loss and readout supply one marginal per
query. A joint predictive distribution would also describe dependence between
query targets, such as uncertainty shared across points on a latent curve.

The training prior determines which prediction rules and uncertainty patterns
the network is encouraged to learn. Real datasets outside the simulator's
coverage can yield poorly calibrated predictions even with low synthetic loss.
See [Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510).
