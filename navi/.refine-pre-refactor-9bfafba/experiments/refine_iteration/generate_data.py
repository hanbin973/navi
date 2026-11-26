"""Generate reusable spectral-kernel data for the refine-iteration experiment."""

import argparse
import logging
from pathlib import Path

from navi.simulation_data import SimulationConfig, generate_dataset

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = SimulationConfig()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=defaults.n, help="population size")
    parser.add_argument("--m", type=int, default=defaults.m, help="spectral rank")
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--context-count", dest="c", type=int, help="defaults to n - q")
    parser.add_argument("--queries", dest="q", type=int, default=defaults.q)
    for name in (
        "k",
        "train_horizon",
        "eval_horizon",
        "train_episodes",
        "validation_episodes",
        "test_episodes",
        "shard_size",
        "seed",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    parser.add_argument("--draw-chunk-steps", type=int, default=16)
    args = parser.parse_args(argv)
    settings = vars(args).copy()
    output = settings.pop("output_dir")
    chunk_steps = settings.pop("draw_chunk_steps")
    if args.c is None:
        settings["c"] = args.n - args.q
    try:
        config = SimulationConfig(**settings)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset_id = generate_dataset(output, config, draw_chunk_steps=chunk_steps)
    logger.info("Wrote dataset %s to %s", dataset_id, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
