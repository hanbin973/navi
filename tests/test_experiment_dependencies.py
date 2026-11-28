"""Experiments must remain usable when sibling experiments are archived."""

import ast
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.mark.parametrize("experiment", ["attention_proposal", "refine_iteration"])
def test_experiment_imports_stay_within_own_package(experiment):
    for path in (EXPERIMENTS / experiment).glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module == "experiments" or module.startswith("experiments."):
                    assert module.startswith(f"experiments.{experiment}."), (
                        f"{path.name}:{node.lineno} depends on sibling {module}"
                    )
