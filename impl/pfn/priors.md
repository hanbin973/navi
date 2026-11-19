# Task priors and training episodes

## Generic PFN

1. Sample one task $z\sim p(z)$, including any shared simulator parameters.
2. Generate rows from that task and partition them into a labeled context $C$
   and queries $Q=\{(x_j^*,y_j^*)\}_{j=1}^m$.
3. Feed the context labels and query features to the Transformer. Keep
   $y_j^*$ in the supervision data used by the loss.

One shared draw of $z$ allows the context to inform query predictions.
For example, a Gaussian process episode can draw a function from a kernel
prior and evaluate it at context and query inputs with observation noise.
Drawing a different function for each row would remove the shared latent
function. Varying context size during episode generation teaches the model
to condition on different amounts of evidence.

## TabPFN

The original classification prior mixes neural-network generators and
structural causal models. In an SCM, a random graph specifies dependencies
among variables. Each node is generated from its parents and exogenous noise.

$$
v_k=f_k(v_{\operatorname{pa}(k)},\epsilon_k).
$$

Observed nodes supply columns and a target. A continuous target can be
discretized to form class labels. A preference for simpler mechanisms shapes
the prior over classification tasks. TabPFN v2 broadens the synthetic task
distribution and supports continuous targets for regression. See
[original TabPFN](https://arxiv.org/abs/2207.01848) and
[TabPFN v2](https://doi.org/10.1038/s41586-024-08328-6).

For a tabular episode generator, make row count, column count, target type,
and class count part of the task specification. Apply any class relabeling
consistently to the context and queries. Treat missingness, categorical
features, and preprocessing as explicit parts of the data contract so the
training distribution and inference inputs can be compared.
