"""Minimal JAX/Equinox backend smoke check for Slurm jobs."""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()

    # Selecting the platform before execution makes missing GPU support fail.
    jax.config.update("jax_platform_name", args.device)
    devices = jax.devices()
    if not devices or any(device.platform != args.device for device in devices):
        raise RuntimeError(f"Expected {args.device} devices; found {devices}")

    model = eqx.nn.Linear(2, 1, key=jax.random.key(0))
    result = eqx.filter_jit(model)(jnp.ones(2)).block_until_ready()
    if result.shape != (1,) or not bool(jnp.all(jnp.isfinite(result))):
        raise RuntimeError(f"Unexpected smoke result: {result}")
    print(f"JAX {jax.__version__}; Equinox {eqx.__version__}; devices: {devices}")
    print(f"Smoke result: {result}")


if __name__ == "__main__":
    main()
