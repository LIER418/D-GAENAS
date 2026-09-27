
import random
import numpy as np
import torch
import torch.nn.functional as F


N_NODES = 4
N_EDGES = 6
N_OPS   = 5

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

_adj = torch.zeros(N_NODES, N_NODES)
for _i, _j in EDGES:
    _adj[_i, _j] = 1.0
FIXED_ADJ: torch.Tensor = _adj


def ops_list_to_tensor(ops_indices: list) -> torch.Tensor:
    return F.one_hot(
        torch.tensor(ops_indices, dtype=torch.long), num_classes=N_OPS
    ).float()


def tensor_to_ops_list(ops_tensor: torch.Tensor) -> list:
    if ops_tensor.dim() == 2:
        return ops_tensor.argmax(dim=-1).tolist()
    return [int(x) for x in ops_tensor.tolist()]


def ops_to_arch_str(ops_indices: list) -> str:
    n = [OPS[i] for i in ops_indices]
    return (
        f'|{n[0]}~0|'
        f'+|{n[1]}~0|{n[2]}~1|'
        f'+|{n[3]}~0|{n[4]}~1|{n[5]}~2|'
    )


def arch_str_to_ops(arch_str: str) -> list:
    parts    = [p for p in arch_str.replace('+', '').split('|') if p.strip()]
    op_names = [p.split('~')[0] for p in parts]
    return [OPS.index(name) for name in op_names]


def individual_hash(ops_tensor: torch.Tensor) -> tuple:
    return tuple(tensor_to_ops_list(ops_tensor))


def random_arch() -> torch.Tensor:
    ops = [random.randint(0, N_OPS - 1) for _ in range(N_EDGES)]
    return ops_list_to_tensor(ops)


def load_nasbench201(path: str):
    from nats_bench import create
    print(f"Loading NATS-Bench TSS: {path} …")
    api = create(path, 'tss', fast_mode=False, verbose=False)
    print(f"Loaded — {len(api)} architectures.\n")
    return api


def query_nasbench201(
    api,
    ops_tensor: torch.Tensor,
    dataset:    str = 'cifar10-valid',
    hp:         int = 200,
) -> tuple:
    try:
        ops_indices = tensor_to_ops_list(ops_tensor)
        arch_str    = ops_to_arch_str(ops_indices)
        idx         = api.query_index_by_arch(arch_str)
        if idx < 0:
            return None, None

        if dataset == 'cifar10-valid':
            val_info  = api.get_more_info(idx, 'cifar10-valid', hp=hp, is_random=False)
            test_info = api.get_more_info(idx, 'cifar10',       hp=hp, is_random=False)
            val_acc   = val_info['valid-accuracy']  / 100.0
            test_acc  = test_info['test-accuracy']  / 100.0
        elif dataset == 'cifar100':
            info     = api.get_more_info(idx, 'cifar100', hp=hp, is_random=False)
            val_acc  = info['valid-accuracy'] / 100.0
            test_acc = info['test-accuracy']  / 100.0
        elif dataset == 'ImageNet16-120':
            info     = api.get_more_info(idx, 'ImageNet16-120', hp=hp, is_random=False)
            val_acc  = info['valid-accuracy'] / 100.0
            test_acc = info['test-accuracy']  / 100.0
        else:
            return None, None

        return float(val_acc), float(test_acc)

    except Exception:
        return None, None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
