# NAVI (Neural Accelerated Variance-component Inference)

NAVI (__N__eural __A__ccelerated __V__ariance-component __I__nference, _나비_) is a neural estimator for inferring variance components of complex traits in large scale biobanks.

NAVI trains a graph neural network using simulated data from succinct tree sequences.

Once trained, it can instantaneously infer variance components of thousands of traits within seconds on a single GPU.

![](./navi.svg)

## Dependencies

NAVI's neural network uses a `jax` backend and is implemented in `flax`.
```bash
pip install jax flax
```

Training data is simulated from succinct tree sequences implemented in `tskit`.
```bash
pip install tskit
```
