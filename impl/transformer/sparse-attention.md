# Sliding-window and block-sparse attention

Sliding-window attention uses softmax over nearby keys. With a causal
window of width $w$ including the current position, query $i$ uses keys
$j$ satisfying $0\le i-j<w$. A kernel that skips out-of-window keys
evaluates at most $Nw$ scores. A dense pairwise masking implementation
can still allocate and compute a quadratic score tensor.

Block-sparse attention groups tokens into blocks and schedules only
allowed block pairs. A model can combine local windows with global tokens
to link distant regions. This changes the receptive field and requires a
kernel whose block layout matches the intended sparsity. See
[Longformer](https://arxiv.org/abs/2004.05150).

JAX exposes a local window argument on the same primitive as dense
attention. Check its backend-specific semantics and confirm the kernel
actually skips work before assigning a speedup. For unordered PFN rows,
a window based on raw row position makes predictions depend on the input
permutation. Feature-aware sparsity would need an explicit, permutation-
compatible selection rule and a policy for ties. See
[JAX dot-product attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html).
