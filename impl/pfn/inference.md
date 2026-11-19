# Inference and verification

## Prediction workflow

Treat the labeled training table as context and the prediction table as
queries. Align feature columns and apply the same per-episode encodings.
For a context-only prediction contract, fit data-dependent preprocessing
statistics using context rows so a query gives the same result alone or in
a larger query batch. Keep query targets absent from the forward inputs.

Run the pretrained model with fixed parameters and decode the scores as
class probabilities or a regression distribution. Cache context
representations only when they depend solely on that context and its
preprocessing configuration. If ensembling across class permutations,
undo each permutation before averaging probabilities. For regression
ensembles with different bin edges, combine CDFs on a shared target scale.

## Checks for a future implementation

| Contract | Observable check |
| --- | --- |
| Bayesian prediction | Compare a PFN trained on a Gaussian process prior with the analytic conditional mean and variance on held-out synthetic tasks. |
| Context order | Shuffle context rows with their labels and compare query predictions within numerical tolerance. |
| Query independence | Compare a prediction made alone with the same prediction in a reordered or expanded query batch. |
| Label isolation | Change held-out labels used by the loss and confirm that the forward logits stay fixed. |
| Padding | Compare valid predictions and mean loss with and without padded rows. |
| Classification | Check that probabilities sum to one over valid classes. |
| Regression | Integrate density across bins to recover predicted masses and evaluate the CDF at the support boundaries. |

For transfer to real tabular tasks, evaluate log loss and calibration for
classification, or predictive log density and interval coverage for
regression. Keep evaluation tasks separate from any decisions used to tune
the prior or preprocessing.
