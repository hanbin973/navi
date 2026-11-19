"""Compare Navi's noncausal attention layouts on a CUDA compute node."""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np


def evaluate(implementation, inputs, cotangent, *, backward, duplicate_query=False):
    def attention(q, k, v):
        if duplicate_query:
            q = jnp.concatenate((q, q), axis=1)
        output = jax.nn.dot_product_attention(
            q, k, v, is_causal=False, implementation=implementation
        )
        return output[:, :1] if duplicate_query else output

    if not backward:
        return (jax.device_get(jax.jit(attention)(*inputs)),)

    def contract(q, k, v):
        output = attention(q, k, v)
        return jnp.sum(output.astype(jnp.float32) * cotangent), output

    run = jax.jit(jax.value_and_grad(contract, has_aux=True, argnums=(0, 1, 2)))
    (_, output), grads = run(*inputs)
    return jax.device_get((output, *grads))


def compare(fused, *references):
    for name, candidate, *expected in zip(
        ("output", "dq", "dk", "dv")[: len(fused)], fused, *references, strict=True
    ):
        candidate = np.asarray(candidate, dtype=np.float32)
        for baseline in expected:
            np.testing.assert_allclose(
                candidate,
                np.asarray(baseline, dtype=np.float32),
                atol=0.04,
                rtol=0.04,
                err_msg=name,
            )


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected CUDA GPU execution, got {devices}")

    for package in ("jax", "jaxlib", "jax-cuda13-plugin", "nvidia-cudnn-cu13"):
        print(f"{package} {version(package)}", flush=True)
    print(f"Devices {devices}", flush=True)

    failures = []
    cases = ((16, 16, 32), (1, 16, 32), (1, 15, 32), (3, 3, 2), (1, 3, 2))
    for queries, keys, depth in cases:
        q, k, v, cotangent = (
            jax.random.normal(key, (2, length, 2, depth), dtype=jnp.float32)
            for key, length in zip(
                jax.random.split(jax.random.key(42), 4),
                (queries, keys, keys, queries),
                strict=True,
            )
        )
        inputs = (q, k, v)
        rounded = tuple(array.astype(jnp.bfloat16) for array in inputs)
        shape = f"q={queries} kv={keys} heads=2 depth={depth}"
        for backward in (False, True):
            phase = "training" if backward else "forward"
            fp32 = evaluate("xla", inputs, cotangent, backward=backward)
            bf16 = evaluate("xla", rounded, cotangent, backward=backward)
            print(f"PASS XLA FP32 and BF16 {shape} {phase}", flush=True)
            for duplicate in (
                (False, True) if (queries, keys, depth) == (1, 16, 32) else (False,)
            ):
                label = f"cuDNN BF16 {shape} {phase}"
                if duplicate:
                    label += " duplicated query"
                try:
                    fused = evaluate(
                        "cudnn",
                        rounded,
                        cotangent,
                        backward=backward,
                        duplicate_query=duplicate,
                    )
                    compare(fused, bf16, fp32)
                except NotImplementedError as error:
                    print(f"UNSUPPORTED {label}: {error}", flush=True)
                    if (queries, keys, depth) == (16, 16, 32) or duplicate:
                        failures.append(label)
                except (AssertionError, ValueError, RuntimeError) as error:
                    failures.append(label)
                    print(f"FAIL {label}: {error}", flush=True)
                else:
                    print(f"PASS {label}", flush=True)

    if failures:
        raise SystemExit(f"cuDNN verification failed for {len(failures)} cases")


if __name__ == "__main__":
    main()
