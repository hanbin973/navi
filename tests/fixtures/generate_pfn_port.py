"""Generate the FP32 parity fixture from the pinned PFNs checkout.

From the repository root, with CPU Torch, NumPy and typing-extensions
available in an isolated uv environment, run
`python tests/fixtures/generate_pfn_port.py`.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

COMMIT = "f04d3cad1d4d464f1039ec04d55b8f004b60494a"
UPSTREAM = Path(__file__).resolve().parents[2] / "impl/pfn_pytorch/PFNs"


def generate():
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != COMMIT or dirty:
        raise RuntimeError("The PFNs reference must be the clean pinned checkout")

    sys.path.insert(0, str(UPSTREAM))
    from pfns.model.layer import PerFeatureLayer

    torch.set_num_threads(1)
    rng = np.random.default_rng(42)

    def sample(shape):
        return torch.from_numpy(rng.uniform(-0.8, 0.8, shape).astype(np.float32))

    layer = PerFeatureLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=7,
        attention_between_features=False,
        dtype=torch.float32,
        zero_init=False,
    )
    attn = layer.self_attn_between_items
    weights = [
        attn.w_qkv,
        attn.w_out,
        layer.mlp.linear1.weight,
        layer.mlp.linear2.weight,
    ]
    names = ["w_qkv", "w_out", "linear1", "linear2"]
    data = {"commit": COMMIT, "torch_version": str(torch.__version__)}
    with torch.no_grad():
        for name, weight in zip(names, weights, strict=True):
            weight.copy_(sample(weight.shape))
            data[name] = weight.detach().numpy()

    source = sample((1, 3, 4)).requires_grad_()
    query = sample((1, 1, 4)).requires_grad_()
    q, _, _, kv, _ = attn.compute_qkv(
        query,
        source,
        None,
        None,
        None,
        cache_kv=False,
        use_cached_kv=False,
        reuse_first_head_kv=False,
    )
    k, v = kv.unbind(dim=-3)
    data.update(q=q.detach().numpy(), k=k.detach().numpy(), v=v.detach().numpy())
    cases = (
        ("attention", attn(query, source), [query, source], weights[:2]),
        ("context", layer(source[:, :, None, :])[:, :, 0, :], [source], weights),
        (
            "query",
            layer(query[:, :, None, :], att_src=source[:, :, None, :])[:, :, 0, :],
            [query, source],
            weights,
        ),
    )
    data.update(
        input_query=query.detach().numpy(), input_source=source.detach().numpy()
    )
    for name, output, inputs, parameters in cases:
        cotangent = sample(output.shape)
        grads = torch.autograd.grad((output * cotangent).sum(), inputs + parameters)
        data[f"{name}_output"] = output.detach().numpy()
        data[f"{name}_cotangent"] = cotangent.numpy()
        for index, grad in enumerate(grads):
            data[f"{name}_grad_{index}"] = grad.numpy()
    np.savez_compressed(Path(__file__).with_name("pfn_port.npz"), **data)


if __name__ == "__main__":
    generate()
