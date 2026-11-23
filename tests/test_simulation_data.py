import json
from dataclasses import replace

import numpy as np
import pytest

from navi.simulation_data import SimulationConfig, SimulationDataset, generate_dataset


@pytest.fixture
def stored_dataset(tmp_path):
    root = tmp_path / "dataset"
    config = SimulationConfig(
        n=6,
        m=2,
        alpha=1.5,
        c=3,
        q=2,
        k=2,
        train_horizon=2,
        eval_horizon=4,
        train_episodes=3,
        validation_episodes=1,
        test_episodes=2,
        shard_size=2,
    )
    generated_id = generate_dataset(root, config, draw_chunk_steps=7)
    dataset = SimulationDataset(root)
    assert dataset.dataset_id == generated_id
    return dataset


def test_round_trip_features_and_splits(stored_dataset, tmp_path):
    dataset = stored_dataset
    root = dataset.root
    train = dataset.split("train", horizon=2)
    test = dataset.split("test", horizon=4)
    index = 2
    shard = root / "train" / "00000002"
    arrays = {
        path.stem: np.load(path, allow_pickle=False) for path in shard.glob("*.npy")
    }
    assert set(arrays) == {
        "U",
        "eigenvalues",
        "context",
        "queries",
        "observations",
        "targets",
        "variances",
        "draws",
    }
    basis, spectrum, c, q = (
        arrays[name][0] for name in ("U", "eigenvalues", "context", "queries")
    )
    kernel = (basis * spectrum) @ basis.T
    batch = train.batch([index, 0])
    np.testing.assert_array_equal(
        batch.episodes.observations[0, :, 0], arrays["observations"][0]
    )
    np.testing.assert_array_equal(batch.episodes.targets[0], arrays["targets"][0])
    np.testing.assert_array_equal(batch.draws[0], arrays["draws"][0])
    np.testing.assert_allclose(
        batch.episodes.query_features[0, :, 0], kernel[q, q], atol=1e-6
    )
    np.testing.assert_allclose(
        batch.episodes.pair_features[0, ..., 0], kernel[np.ix_(q, c)], atol=1e-6
    )
    assert not np.array_equal(train.episode(0).targets, test.episode(0).targets)
    assert not np.array_equal(test.draws(0, 0, 1), test.draws(0, 0, 1, probe=True))
    calibration = np.r_[
        np.load(root / "train" / "00000000" / "targets.npy").ravel(),
        arrays["targets"].ravel(),
    ]
    np.testing.assert_array_equal(dataset.bin_edges, np.sort(calibration))

    other = tmp_path / "regenerated"
    generate_dataset(other, replace(dataset.config, shard_size=1), draw_chunk_steps=2)
    replay = SimulationDataset(other).split("test", horizon=4)
    np.testing.assert_array_equal(replay.episode(1).targets, test.episode(1).targets)
    np.testing.assert_allclose(replay.draws(1, 0, 4), test.draws(1, 0, 4), atol=1e-6)
    with pytest.raises(FileExistsError):
        generate_dataset(root, dataset.config)


def test_rejects_incompatible_or_incomplete_artifacts(stored_dataset):
    dataset = stored_dataset
    with pytest.raises(ValueError, match="horizon"):
        dataset.split("train", horizon=3)
    train = dataset.split("train", horizon=2)
    with pytest.raises(IndexError):
        train.episode(3)
    with pytest.raises(ValueError, match="coverage"):
        train.draws(0, 0, 3)
    with pytest.raises(ValueError, match="coverage"):
        train.draws(0, 0, 2, probe=True)
    path = dataset.root / "train" / "00000002" / "context.npy"
    np.save(path, np.full((1, 3), 6, dtype=np.int32))
    with pytest.raises(ValueError, match="population"):
        train.episode(2)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="context.npy"):
        dataset.split("train", horizon=2)
    manifest_path = dataset.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        SimulationDataset(dataset.root)
