"""Generate and read versioned, memory-mapped spectral episode datasets."""

import json
import logging
from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np

from .distribution import FullSupportBarDistribution
from .helper import TrainingEpisode, validate_selections
from .simulation import (
    kernel_features,
    power_law_eigenvalues,
    sample_updater_draws,
    selected_factor,
    simulate_trait,
)

SCHEMA_VERSION = 1
_SPLITS = ("train", "validation", "test")
_PROBE_STEPS = 32
_FEATURE_CONVENTION = "population-kernel-entries"
_DTYPES = {"floating": "float32", "indices": "int32"}
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationConfig:
    """Sizes and generation parameters for a single shared spectrum."""

    n: int = 512
    m: int = 128
    alpha: float = 1.0
    c: int = 504
    q: int = 8
    k: int = 32
    train_horizon: int = 32
    eval_horizon: int = 512
    train_episodes: int = 256
    validation_episodes: int = 32
    test_episodes: int = 128
    shard_size: int = 16
    seed: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            minimum = 0 if name == "seed" else 1
            if name != "alpha" and (type(value) is not int or value < minimum):
                raise ValueError(f"invalid integer value for {name}")
        if type(self.alpha) not in (float, int):
            raise ValueError("alpha must be numeric")
        power_law_eigenvalues(self.n, self.m, self.alpha)
        if self.c + self.q > self.n or self.n >= 2**31:
            raise ValueError("require c + q <= n < 2**31")
        if self.train_episodes * self.q < 3:
            raise ValueError("require at least three training targets for bin edges")
        if self.eval_horizon < self.train_horizon:
            raise ValueError("eval_horizon must cover train_horizon")
        counts = (self.train_episodes, self.validation_episodes, self.test_episodes)
        if (
            max(self.seed, *counts, max(self.eval_horizon, _PROBE_STEPS) * self.k)
            >= 2**32
        ):
            raise ValueError("seed, episode counts and draw counts must fit in uint32")

    def count(self, split):
        return getattr(self, f"{split}_episodes")

    def horizon(self, split):
        return self.train_horizon if split == "train" else self.eval_horizon


def _array_specs(config, split):
    n, m, c, q, k = config.n, config.m, config.c, config.q, config.k
    specs = {
        "U": (np.float32, (n, m)),
        "eigenvalues": (np.float32, (m,)),
        "context": (np.int32, (c,)),
        "queries": (np.int32, (q,)),
        "observations": (np.float32, (c,)),
        "targets": (np.float32, (q,)),
        "variances": (np.float32, (2,)),
        "draws": (np.float32, (config.horizon(split), k, c)),
    }
    if split != "train":
        specs["probe_draws"] = (np.float32, (_PROBE_STEPS, k, c))
    return specs


def _shard_ranges(config, split):
    count = config.count(split)
    return [
        [first, min(first + config.shard_size, count)]
        for first in range(0, count, config.shard_size)
    ]


def _payload_bytes(config):
    return {
        split: sum(
            int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
            for dtype, shape in _array_specs(config, split).values()
        )
        * config.count(split)
        for split in _SPLITS
    }


def _shard_path(root, split, first):
    return root / split / f"{first:08d}"


def _write_draws(destination, context_factor, key, chunk_steps):
    horizon, k, c = destination.shape
    for first in range(0, horizon, chunk_steps):
        length = min(chunk_steps, horizon - first)
        indices = jnp.arange(length * k, dtype=jnp.uint32) + np.uint32(first * k)
        values = np.asarray(sample_updater_draws(context_factor, key, indices))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite updater draws")
        destination[first : first + length] = values.reshape(length, k, c)


def _write_episode(shard, offset, episode_key, spectrum, config, chunk_steps):
    trait_key, updater_key, probe_key = jax.random.split(episode_key, 3)
    trait = simulate_trait(trait_key, spectrum, n=config.n, c=config.c, q=config.q)
    arrays = jax.device_get(trait)
    basis = arrays.factors.U
    if (
        not np.isfinite(arrays.qr_diagonal).all()
        or np.any(arrays.qr_diagonal == 0)
        or not np.allclose(
            basis.T @ basis, np.eye(config.m, dtype=np.float32), rtol=2e-5, atol=2e-5
        )
    ):
        raise ValueError("reduced QR failed to produce an orthonormal basis")
    values = {
        "U": basis,
        "eigenvalues": arrays.factors.eigenvalues,
        "context": arrays.context,
        "queries": arrays.queries,
        "observations": arrays.observations,
        "targets": arrays.targets,
        "variances": arrays.variances,
    }
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite episode array {name}")
        shard[name][offset] = value
    factor = selected_factor(trait.factors, trait.context)
    _write_draws(shard["draws"][offset], factor, updater_key, chunk_steps)
    if "probe_draws" in shard:
        _write_draws(shard["probe_draws"][offset], factor, probe_key, chunk_steps)
    return arrays.targets


def generate_dataset(
    output: Path, config: SimulationConfig, *, draw_chunk_steps: int = 16
) -> str:
    """Write all splits beside the destination, then publish by directory rename."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if type(draw_chunk_steps) is not int or draw_chunk_steps < 1:
        raise ValueError("draw_chunk_steps must be a positive integer")
    payload = _payload_bytes(config)
    _LOG.info("Estimated dataset payload in bytes by split %s", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid4().hex
    spectrum = jnp.asarray(power_law_eigenvalues(config.n, config.m, config.alpha))
    root_key = jax.random.key(config.seed)
    ranges = {split: _shard_ranges(config, split) for split in _SPLITS}
    calibration = np.empty(min(8192, config.train_episodes * config.q), np.float32)
    filled = 0
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        root = Path(staging) / "dataset"
        for split_number, split in enumerate(_SPLITS):
            split_key = jax.random.fold_in(root_key, split_number)
            for first, stop in ranges[split]:
                path = _shard_path(root, split, first)
                path.mkdir(parents=True)
                shard = {
                    name: np.lib.format.open_memmap(
                        path / f"{name}.npy",
                        mode="w+",
                        dtype=dtype,
                        shape=(stop - first, *shape),
                    )
                    for name, (dtype, shape) in _array_specs(config, split).items()
                }
                for offset, episode_id in enumerate(range(first, stop)):
                    targets = _write_episode(
                        shard,
                        offset,
                        jax.random.fold_in(split_key, episode_id),
                        spectrum,
                        config,
                        draw_chunk_steps,
                    )
                    if split == "train" and filled < calibration.size:
                        size = min(targets.size, calibration.size - filled)
                        calibration[filled : filled + size] = targets[:size]
                        filled += size
                for array in shard.values():
                    array.flush()
                del array, shard
        quantiles = np.linspace(0, 1, min(33, calibration.size))
        edges = np.unique(np.quantile(calibration, quantiles).astype(np.float32))
        borders = FullSupportBarDistribution(edges).edges
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": 1,
            "dataset_id": dataset_id,
            "config": asdict(config),
            "shards": ranges,
            "dtypes": _DTYPES,
            "probe_steps": _PROBE_STEPS,
            "basis": "reduced-qr-of-gaussian-with-positive-r-diagonal",
            "feature_convention": _FEATURE_CONVENTION,
            "variance_recipe": "independent-base-2-log-uniform[-2,2)",
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "estimated_payload_bytes": payload,
            "bin_edges": list(borders),
            "calibration_target_count": filled,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        root.rename(output)
    return dataset_id


class StoredBatch(NamedTuple):
    episodes: TrainingEpisode
    draws: np.ndarray


class SimulationDataset:
    """Validate artifact metadata and open one split at a time."""

    def __init__(self, root: Path):
        self.root = Path(root)
        with (self.root / "manifest.json").open() as file:
            manifest = json.load(file)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("unsupported simulation dataset schema")
        try:
            settings = manifest["config"]
            if not isinstance(settings, dict) or settings.keys() != {
                field.name for field in fields(SimulationConfig)
            }:
                raise ValueError("incomplete or incompatible dataset configuration")
            self.config = SimulationConfig(**settings)
            self.dataset_id = manifest["dataset_id"]
            if not isinstance(self.dataset_id, str) or not self.dataset_id:
                raise ValueError("dataset ID must be a nonempty string")
            if manifest["feature_convention"] != _FEATURE_CONVENTION:
                raise ValueError("incompatible predictor feature convention")
            if manifest["dtypes"] != _DTYPES:
                raise ValueError("incompatible dataset dtypes")
            if manifest["probe_steps"] != _PROBE_STEPS:
                raise ValueError("incompatible probe horizon")
            self.bin_edges = FullSupportBarDistribution(manifest["bin_edges"]).edges
            self.ranges = manifest["shards"]
            if self.ranges != {
                split: _shard_ranges(self.config, split) for split in _SPLITS
            }:
                raise ValueError("invalid shard coverage")
        except (KeyError, TypeError) as error:
            raise ValueError("malformed simulation dataset manifest") from error

    def split(self, name: str, *, horizon: int):
        if name not in _SPLITS:
            raise ValueError(f"unknown dataset split {name!r}")
        if type(horizon) is not int or not 1 <= horizon <= self.config.horizon(name):
            raise ValueError(f"requested horizon exceeds stored coverage for {name!r}")
        return SimulationSplit(self, name, horizon)


class SimulationSplit:
    """Read validated episode arrays with a one-shard memory-map cache."""

    def __init__(self, dataset: SimulationDataset, name: str, horizon: int):
        self.dataset = dataset
        self.name = name
        self.horizon = horizon
        self.ranges = dataset.ranges[name]
        self.starts = tuple(first for first, _ in self.ranges)
        self.specs = _array_specs(dataset.config, name)
        self.spectrum = power_law_eigenvalues(
            dataset.config.n, dataset.config.m, dataset.config.alpha
        )
        self._cached = None
        for first, stop in self.ranges:
            self._open(first, stop)

    def __len__(self):
        return self.dataset.config.count(self.name)

    def _open(self, first, stop):
        path = _shard_path(self.dataset.root, self.name, first)
        shard = {}
        for name, (dtype, shape) in self.specs.items():
            array = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != (stop - first, *shape) or array.dtype != dtype:
                raise ValueError(f"invalid shape or dtype for {path / name}.npy")
            shard[name] = array
        return shard

    def _locate(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(f"episode index {index} outside {self.name} split")
        first, stop = self.ranges[bisect_right(self.starts, index) - 1]
        if self._cached is None or self._cached[0] != first:
            self._cached = (first, self._open(first, stop))
        return self._cached[1], index - first

    def episode(self, index: int) -> TrainingEpisode:
        shard, offset = self._locate(index)
        context = shard["context"][offset]
        queries = shard["queries"][offset]
        validate_selections(context, queries, self.dataset.config.n)
        selected = np.concatenate((context, queries))
        if np.unique(selected).size != selected.size:
            raise ValueError("context and query indices must be unique and disjoint")
        eigenvalues = shard["eigenvalues"][offset]
        if not np.allclose(eigenvalues, self.spectrum, rtol=3e-7, atol=0):
            raise ValueError("stored spectrum differs from manifest")
        factor = shard["U"][offset, selected] * np.sqrt(eigenvalues)
        fc, fq = factor[: context.size], factor[context.size :]
        observations = np.array(shard["observations"][offset, :, None])
        targets = np.array(shard["targets"][offset])
        variances = shard["variances"][offset]
        episode = TrainingEpisode(observations, *kernel_features(fc, fq), targets)
        if (
            not all(np.isfinite(value).all() for value in episode)
            or not np.isfinite(variances).all()
            or np.any(variances <= 0)
        ):
            raise ValueError("invalid stored data or derived features")
        return episode

    def draws(
        self,
        index: int,
        first: int = 0,
        stop: int | None = None,
        *,
        probe: bool = False,
    ):
        limit = _PROBE_STEPS if probe and self.name != "train" else self.horizon
        if stop is None:
            stop = limit
        if (probe and self.name == "train") or not 0 <= first < stop <= limit:
            raise ValueError("draw slice outside requested coverage")
        shard, offset = self._locate(index)
        name = "probe_draws" if probe else "draws"
        values = np.array(shard[name][offset, first:stop])
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite stored {name}")
        return values

    def batch(self, indices) -> StoredBatch:
        rows = [(self.episode(index), self.draws(index)) for index in indices]
        if not rows:
            raise ValueError("an episode batch must be nonempty")
        episodes, draws = zip(*rows, strict=True)
        return StoredBatch(
            TrainingEpisode(
                *(np.stack(field) for field in zip(*episodes, strict=True))
            ),
            np.stack(draws),
        )
