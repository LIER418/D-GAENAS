
import hashlib
import random
import warnings
from collections import namedtuple

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
    key = str(g.normal) + str(g.reduce)
    return hashlib.md5(key.encode()).hexdigest()


def query_nasbench301(perf_model, genotype: Genotype301,
                      with_noise: bool = False) -> float | None:
    try:
        acc = perf_model.predict(
            config         = genotype,
            representation = 'genotype',
            with_noise     = with_noise,
        )
        return float(acc)
    except Exception:
        return None


def load_nasbench301(model_dir: str = 'nb_models_1.0/xgb_v1.0'):
    import nasbench301 as nb
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
