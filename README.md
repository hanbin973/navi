# NAVI (Neural Accelerated Variance-component Inference)

NAVI (Neural Accelerated Variance-component Inference, 나비) is a neural estimator for inferring variance components of complex traits in large scale biobanks.

NAVI trains a graph neural network using simulated data from succinct tree sequences.

Once trained, it can instantaneously infer variance components of thousands of traits within seconds on a single GPU.

## Dependencies

NAVI's neural network uses a `jax` backend and is implemented in `flax`.
```bash
pip install jax flax
```

Training data is simulated from tree sequences
```bash
pip install tskit
```
