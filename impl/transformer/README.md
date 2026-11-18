# Efficient Transformer variants

| Variant | What changes | Attention work for length $N$ |
| --- | --- | --- |
| [Fused dense softmax](dense-attention.md) | Kernel tiling and intermediate storage | $O(N^2)$ |
| [Grouped-query and multi-query attention](dense-attention.md#grouped-query-and-multi-query-attention) | Number of distinct key/value heads | $O(N^2)$ with query head count fixed |
| [Sliding-window and block-sparse softmax](sparse-attention.md) | Allowed query–key pairs and their execution | $O(Nw)$ for window width $w$ with a window-aware kernel |
| [Feature-map linear attention](linear-attention.md#feature-map-attention) | Similarity function and evaluation order | $O(N)$ for fixed feature and head dimensions |
| [Gated delta-rule attention](linear-attention.md#gated-delta-rule-attention) | Recurrent memory update and readout | $O(N)$ for fixed state dimensions in recurrent form |

Complexities describe sequence mixing, with projection and feed-forward
costs omitted. The [JAX implementation survey](jax-implementations.md)
records upstream code, maintenance evidence, and deployment scope. For the
context/query mask motivating these comparisons, see the
[PFN Transformer notes](../pfn/transformer.md). Math uses MathJax delimiters.
