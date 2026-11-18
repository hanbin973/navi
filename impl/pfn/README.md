# PFN and TabPFN design notes

A Prior-Data Fitted Network (PFN) learns to approximate posterior prediction
from examples of supervised learning tasks. TabPFN specializes this approach
to tabular data. These notes focus on the original PFN and TabPFN papers, with
TabPFN v2 as a reference for feature-wise attention and regression.

| Component | Design note |
| --- | --- |
| Bayesian target and amortized inference | [Theory](theory.md) |
| Task generators and context/query episodes | [Priors and episodes](priors.md) |
| Training objective and predictive heads | [Loss](loss.md) |
| Tokenization, attention masks, and readout | [Transformer](transformer.md) |
| Prediction workflow and verification | [Inference](inference.md) |

Use a Markdown preview that enables MathJax with single-dollar inline math
and double-dollar display math.
