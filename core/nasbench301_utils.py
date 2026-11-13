
import hashlib
import random
import warnings
from collections import namedtuple

import numpy as np
import torch

warnings.filterwarnings('ignore')


PRIMITIVES_301 = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
N_OPS_301   = len(PRIMITIVES_301)
N_EDGES_301 = 8
N_CELLS_301 = 2

MAX_SRC_301 = [1, 1, 2, 2, 3, 3, 4, 4]

CONCAT_301 = [2, 3, 4, 5]

EDGES_301 = [
    (0, 2), (1, 2), 
    (0, 3), (1, 3), (2, 3), 
    (0, 4), (1, 4), (2, 4), (3, 4), 
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)
]

Genotype301 = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def _random_cell():
    edges = []
    for node_idx in range(4):
        max_s = node_idx + 1
        srcs  = random.sample(range(max_s + 1), 2)
        for src in srcs:
            op = random.choice(PRIMITIVES_301)
            edges.append((op, src))
    return edges


def _fix_cell_sources(cell: list) -> list:
    cell = list(cell)
    for node_idx in range(4):
        max_s  = node_idx + 1
        e1, e2 = 2 * node_idx, 2 * node_idx + 1
        op1, s1 = cell[e1]
        op2, s2 = cell[e2]
        if s1 == s2:
            opts = [s for s in range(max_s + 1) if s != s1]
            cell[e2] = (op2, random.choice(opts))
    return cell


def random_arch_301() -> Genotype301:
    return Genotype301(
        normal        = _random_cell(),
        normal_concat = CONCAT_301,
        reduce        = _random_cell(),
        reduce_concat = CONCAT_301,
    )


def genotype_hash(g: Genotype301) -> str:
    # The two incoming edges of a DARTS intermediate node are unordered.
    # Canonicalising each pair prevents spending another benchmark query on
    # an equivalent genotype whose two edges are merely swapped.
    def canonical_cell(cell):
        return tuple(
            tuple(sorted(cell[i:i + 2], key=lambda edge: (edge[1], edge[0])))
            for i in range(0, N_EDGES_301, 2)
        )

    key = repr((canonical_cell(g.normal), canonical_cell(g.reduce)))
    return hashlib.md5(key.encode()).hexdigest()


def query_nasbench301(perf_model, genotype: Genotype301,
                      with_noise: bool = False) -> float | None:
    try:
        acc = perf_model.predict(
            config         = genotype,
            representation = 'genotype',
            with_noise     = with_noise,
        )
        values = np.asarray(acc, dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            return None
        return float(values[0])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None


def load_nasbench301(model_dir: str = 'nb_models_1.0/xgb_v1.0'):
    try:
        import nasbench301 as nb
    except ImportError as exc:
        raise ImportError(
            "NAS-Bench-301 is not installed. Install the official package "
            "with `python -m pip install nasbench301`."
        ) from exc

    # NAS-Bench-301 0.3 removes and re-adds four hyperparameters by mutating
    # ConfigSpace 0.4.x internals. ConfigSpace >= 1.0 deliberately disallows
    # that mutation. Rewrite the corresponding JSON declarations before the
    # space is constructed; this produces the same final definitions while
    # retaining all conditions and forbiddens that refer to them by name.
    try:
        import ConfigSpace
        if int(ConfigSpace.__version__.split(".", 1)[0]) >= 1:
            import json
            from nasbench301.surrogate_models import utils as nb301_utils

            replacements = {
                "NetworkSelectorDatasetInfo:darts:layers": {
                    "type": "uniform_int", "lower": 1, "upper": 10000,
                    "default": 6, "log": False,
                },
                "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": {
                    "type": "uniform_int", "lower": 1, "upper": 10000,
                    "default": 50, "log": False,
                },
                "NetworkSelectorDatasetInfo:darts:init_channels": {
                    "type": "uniform_int", "lower": 1, "upper": 10000,
                    "default": 36, "log": False,
                },
                "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": {
                    "type": "uniform_float", "lower": 0.0, "upper": 10000.0,
                    "default": 0.0, "log": False,
                },
            }

            def _load_config_space_compat(path):
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                for hp in payload.get("hyperparameters", []):
                    name = hp.get("name")
                    if name in replacements:
                        hp.clear()
                        hp.update(name=name, **replacements[name])
                    elif hp.get("type") == "normal_float":
                        # Old normal-float declarations cannot be deserialised
                        # by current ConfigSpace/numpy combinations. This field
                        # is not an architecture variable used by NB301 queries;
                        # keep its feature slot through an ordinary uniform HP.
                        hp["type"] = "uniform_float"
                        hp["lower"] = 1e-6
                        hp["upper"] = 1.0
                        hp["log"] = False
                        hp.pop("mu", None)
                        hp.pop("sigma", None)
                return nb301_utils.config_space_json_r_w.read(json.dumps(payload))

            nb301_utils.ConfigLoader.load_config_space = staticmethod(
                _load_config_space_compat
            )
    except (ImportError, AttributeError, ValueError):
        # ConfigSpace 0.4.x follows the package's original code path.
        pass

    return nb.load_ensemble(model_dir)


def mutate_cell_301(cell: list, mut_prob: float = 0.15) -> list:
    cell = list(cell)
    for i, (op_name, src) in enumerate(cell):
        if random.random() < mut_prob:
            other_ops = [o for o in PRIMITIVES_301 if o != op_name]
            op_name   = random.choice(other_ops)
        if random.random() < mut_prob:
            max_s = MAX_SRC_301[i]
            if max_s > 0:
                other_srcs = [s for s in range(max_s + 1) if s != src]
                if other_srcs:
                    src = random.choice(other_srcs)
        cell[i] = (op_name, src)
    return _fix_cell_sources(cell)


def mutate_arch_301(genotype: Genotype301, mut_prob: float = 0.15) -> Genotype301:
    return Genotype301(
        normal        = mutate_cell_301(genotype.normal, mut_prob),
        normal_concat = CONCAT_301,
        reduce        = mutate_cell_301(genotype.reduce, mut_prob),
        reduce_concat = CONCAT_301,
    )


def mutate_one_arch_301(genotype: Genotype301) -> Genotype301:
    """Apply one ordinary DARTS mutation to one cell edge."""
    cells = [list(genotype.normal), list(genotype.reduce)]
    cell_idx = random.randrange(2)
    edge_idx = random.randrange(len(cells[cell_idx]))
    op_name, src = cells[cell_idx][edge_idx]
    if random.random() < 0.5:
        op_name = random.choice([op for op in PRIMITIVES_301 if op != op_name])
    else:
        max_src = MAX_SRC_301[edge_idx]
        paired_idx = edge_idx + 1 if edge_idx % 2 == 0 else edge_idx - 1
        paired_src = cells[cell_idx][paired_idx][1]
        choices = [
            candidate for candidate in range(max_src + 1)
            if candidate != src and candidate != paired_src
        ]
        # The first DARTS node has exactly two possible inputs, both already
        # occupied. In that case an input mutation is impossible, so mutate
        # the operation instead.
        if choices:
            src = random.choice(choices)
        else:
            op_name = random.choice([op for op in PRIMITIVES_301 if op != op_name])
    cells[cell_idx][edge_idx] = (op_name, src)
    return Genotype301(
        normal=cells[0], normal_concat=CONCAT_301,
        reduce=cells[1], reduce_concat=CONCAT_301,
    )


N_EDGES_DAG_301 = 14
FEAT_DIM_301    = 1 + N_OPS_301


def genotype_to_tensor(g: Genotype301) -> torch.Tensor:
    result = torch.zeros(N_CELLS_301, N_EDGES_DAG_301, FEAT_DIM_301, dtype=torch.float32)
    for c_idx, cell in enumerate((g.normal, g.reduce)):
        for i, (op_name, src) in enumerate(cell):
            dst = 2 + (i // 2)
            edge_idx = EDGES_301.index((src, dst))
            result[c_idx, edge_idx, 0] = 1.0
            op_idx = PRIMITIVES_301.index(op_name)
            result[c_idx, edge_idx, 1 + op_idx] = 1.0
            
        for edge_idx in range(N_EDGES_DAG_301):
            if result[c_idx, edge_idx, 0] == 0.0:
                result[c_idx, edge_idx, 1] = 1.0
                
    return result


def tensor_to_genotype(t: torch.Tensor) -> Genotype301:
    cells = []
    for cell_idx in range(N_CELLS_301):
        edges = []
        for edge_idx, (u, v) in enumerate(EDGES_301):
            if t[cell_idx, edge_idx, 0] > 0.5:
                op_idx = int(t[cell_idx, edge_idx, 1:].argmax())
                edges.append((PRIMITIVES_301[op_idx], u))
        cells.append(edges)
        
    return Genotype301(
        normal        = cells[0],
        normal_concat = CONCAT_301,
        reduce        = cells[1],
        reduce_concat = CONCAT_301,
    )


def set_seed_301(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
