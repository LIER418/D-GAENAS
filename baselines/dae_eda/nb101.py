"""A shared NAS-Bench-101 oracle and query ledger for new experiments.

All reported budgets count *unique* 108-epoch architecture evaluations,
including initialization.  Architecture performance is the mean of the three
benchmark repeats, matching D-GAENAS-release/scripts/saea_gae_nb101.py.
"""

from __future__ import annotations

import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFRECORD = ROOT / "data" / "nasbench_only108.tfrecord"
OPS = ("input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "output")
EDGE_POSITIONS = tuple((i, j) for i in range(7) for j in range(i + 1, 7))


def load_api(path: str | Path = DEFAULT_TFRECORD):
    # The released nasbench protobuf stubs need this compatibility setting with
    # recent protobuf versions.  It must precede importing nasbench.api.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    for name in ("nasbench.lib.evaluate", "nasbench.lib.model_builder", "nasbench.lib.training_time"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import tensorflow as tf

    if not hasattr(tf, "python_io"):
        tf.python_io = tf.compat.v1.python_io
    from nasbench import api

    return api.NASBench(str(Path(path).resolve())), api


@dataclass(frozen=True)
class Architecture:
    operations: tuple[int, ...]
    edges: tuple[int, ...]

    def __post_init__(self):
        if len(self.operations) != 7 or len(self.edges) != 21:
            raise ValueError("NAS-Bench-101 architecture requires 7 operations and 21 upper-triangle edges")
        if self.operations[0] != 0 or self.operations[-1] != 4:
            raise ValueError("Input and output operations must be fixed")
        if any(x not in (1, 2, 3) for x in self.operations[1:-1]):
            raise ValueError("Intermediate operations must be one of the three primitives")

    def matrix(self) -> np.ndarray:
        result = np.zeros((7, 7), dtype=np.int8)
        for (i, j), present in zip(EDGE_POSITIONS, self.edges):
            result[i, j] = present
        return result

    def vector(self) -> np.ndarray:
        one_hot = np.eye(3, dtype=np.float32)[np.asarray(self.operations[1:-1]) - 1].reshape(-1)
        return np.concatenate([one_hot, np.asarray(self.edges, dtype=np.float32)])

    @staticmethod
    def from_vector(vector: np.ndarray) -> "Architecture":
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if len(vector) != 36:
            raise ValueError("Expected 15 operation bits and 21 edge bits")
        ops = (0, *tuple(np.argmax(vector[:15].reshape(5, 3), axis=1).astype(int) + 1), 4)
        edges = tuple(int(x >= 0.5) for x in vector[15:])
        return Architecture(ops, edges)


class BudgetExhausted(Exception):
    pass


class Oracle:
    def __init__(self, budget: int, path: str | Path = DEFAULT_TFRECORD):
        self.nb, self.api = load_api(path)
        self.budget = int(budget)
        self.cache: dict[str, tuple[float, float]] = {}
        self.events: list[dict] = []
        self.invalid = 0
        self.repeats = 0
        self.best_architecture: Architecture | None = None
        self.best_val = -float("inf")
        self.best_test = float("nan")

    def spec(self, arch: Architecture):
        return self.api.ModelSpec(matrix=arch.matrix(), ops=[OPS[i] for i in arch.operations])

    def canonical_key(self, arch: Architecture) -> str:
        return self.spec(arch).hash_spec(self.nb.config["available_ops"])

    def valid(self, arch: Architecture) -> bool:
        try:
            return bool(self.nb.is_valid(self.spec(arch)))
        except (ValueError, KeyError):
            return False

    def evaluate(self, arch: Architecture) -> tuple[float, float] | None:
        if not self.valid(arch):
            self.invalid += 1
            return None
        key = self.canonical_key(arch)
        if key in self.cache:
            self.repeats += 1
            return self.cache[key]
        if len(self.cache) >= self.budget:
            raise BudgetExhausted
        try:
            _, metrics = self.nb.get_metrics_from_spec(self.spec(arch))
            runs = metrics[108]
            val = float(np.mean([r["final_validation_accuracy"] for r in runs]))
            test = float(np.mean([r["final_test_accuracy"] for r in runs]))
        except (KeyError, ValueError):
            self.invalid += 1
            return None
        self.cache[key] = (val, test)
        if val > self.best_val:
            self.best_architecture, self.best_val, self.best_test = arch, val, test
        self.events.append({
            "query": len(self.cache),
            "validation": val,
            "test": test,
            "best_validation": self.best_val,
            "best_test_of_best_validation": self.best_test,
        })
        return val, test

    def random_architecture(self, rng: random.Random) -> Architecture:
        # Generate valid cells without accessing performance labels.
        for _ in range(10000):
            operations = (0, *(rng.randint(1, 3) for _ in range(5)), 4)
            edge_set = set(rng.sample(range(21), rng.randint(6, 9)))
            arch = Architecture(tuple(operations), tuple(int(i in edge_set) for i in range(21)))
            if self.valid(arch):
                return arch
        raise RuntimeError("Unable to sample a valid NAS-Bench-101 cell")
