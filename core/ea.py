
import math
import hashlib
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


N_NODES   = 7
N_OPS     = 5
OP_INPUT  = 0
OP_OUTPUT = 4
INTER_OPS = [1, 2, 3]
MAX_EDGES = 9

Individual = Tuple[torch.Tensor, torch.Tensor]


def individual_hash(x: torch.Tensor, adj: torch.Tensor) -> str:
    raw = x.cpu().numpy().tobytes() + adj.cpu().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()


def _bfs(adj: torch.Tensor, src: int) -> set:
    N = adj.size(0)
    visited, queue = {src}, [src]
    while queue:
        n = queue.pop(0)
        for j in range(N):
            if adj[n, j] > 0.5 and j not in visited:
                visited.add(j)
                queue.append(j)
    return visited


def _bfs_rev(adj: torch.Tensor, dst: int) -> set:
    N = adj.size(0)
    visited, queue = {dst}, [dst]
    while queue:
        n = queue.pop(0)
        for i in range(N):
            if adj[i, n] > 0.5 and i not in visited:
                visited.add(i)
                queue.append(i)
    return visited


def is_valid(x: torch.Tensor, adj: torch.Tensor) -> bool:
    N   = x.size(0)
    cls = x.argmax(dim=-1).tolist()

    if cls[0] != OP_INPUT or cls[-1] != OP_OUTPUT:
        return False
    for c in cls[1:-1]:
        if c not in INTER_OPS:
            return False
    if int(adj.triu(diagonal=1).sum().item()) > MAX_EDGES:
        return False
    if (N - 1) not in _bfs(adj, 0):
        return False
    return True


def repair(x: torch.Tensor, adj: torch.Tensor) -> Individual:
    N, C   = x.shape
    device = x.device
    x_rep   = x.clone()
    adj_rep = adj.clone()

    x_rep[0]     = F.one_hot(torch.tensor(OP_INPUT),  num_classes=C).float().to(device)
    x_rep[N - 1] = F.one_hot(torch.tensor(OP_OUTPUT), num_classes=C).float().to(device)

    for i in range(1, N - 1):
        c = int(x_rep[i].argmax().item())
        if c not in INTER_OPS:
            nc = INTER_OPS[torch.randint(0, len(INTER_OPS), (1,)).item()]
            x_rep[i] = F.one_hot(torch.tensor(nc), num_classes=C).float().to(device)

    n_edges = int(adj_rep.triu(diagonal=1).sum().item())
    if n_edges > MAX_EDGES:
        existing = [(i, j) for i in range(N) for j in range(i + 1, N)
                    if adj_rep[i, j].item() > 0.5]
        perm = torch.randperm(len(existing)).tolist()
        for k in perm[: n_edges - MAX_EDGES]:
            ei, ej = existing[k]
            adj_rep[ei, ej] = 0.0

    for _ in range(N):
        from_input = _bfs(adj_rep, 0)
        if (N - 1) in from_input:
            break

        to_output = _bfs_rev(adj_rep, N - 1)

        candidates = []
        for s in sorted(from_input):
            for d in range(s + 1, N):
                if d not in from_input and adj_rep[s, d].item() < 0.5:
                    priority = 1 if d in to_output else 0
                    candidates.append((priority, s, d))

        if not candidates:
            break

        candidates.sort(key=lambda t: -t[0])

        if int(adj_rep.triu(diagonal=1).sum().item()) >= MAX_EDGES:
            non_critical = [
                (i, j) for i in range(N) for j in range(i + 1, N)
                if adj_rep[i, j] > 0.5
                and not (i in from_input and j in to_output)
            ]
            if not non_critical:
                break
            ri, rj = non_critical[torch.randint(0, len(non_critical), (1,)).item()]
            adj_rep[ri, rj] = 0.0

        _, s, d = candidates[0]
        adj_rep[s, d] = 1.0

    return x_rep, adj_rep


def random_individual(
    n_nodes: int          = N_NODES,
    n_ops:   int          = N_OPS,
    device:  torch.device = torch.device("cpu"),
) -> Individual:
    cls = torch.zeros(n_nodes, dtype=torch.long)
    cls[0]  = OP_INPUT
    cls[-1] = OP_OUTPUT
    for i in range(1, n_nodes - 1):
        cls[i] = INTER_OPS[torch.randint(0, len(INTER_OPS), (1,)).item()]
    x = F.one_hot(cls, num_classes=n_ops).float().to(device)

    adj = torch.zeros(n_nodes, n_nodes, device=device)
    edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    n_edges = torch.randint(1, MAX_EDGES + 1, (1,)).item()
    perm = torch.randperm(len(edges)).tolist()
    for k in perm[:n_edges]:
        i, j = edges[k]
        adj[i, j] = 1.0

    return repair(x, adj)


def mutate(
    x:             torch.Tensor,
    adj:           torch.Tensor,
    node_mut_prob: float = 0.1,
    edge_add_prob: float = 0.1,
    edge_del_prob: float = 0.1,
) -> Individual:
    N, C  = x.shape
    x_mut   = x.clone()
    adj_mut = adj.clone()

    for i in range(1, N - 1):
        if torch.rand(1).item() < node_mut_prob:
            curr  = int(x_mut[i].argmax().item())
            other = [op for op in INTER_OPS if op != curr]
            nc    = other[torch.randint(0, len(other), (1,)).item()]
            x_mut[i] = F.one_hot(torch.tensor(nc), num_classes=C).float().to(x.device)

    if torch.rand(1).item() < edge_add_prob:
        absent = [(i, j) for i in range(N) for j in range(i + 1, N)
                  if adj_mut[i, j].item() < 0.5]
        if absent:
            i, j = absent[torch.randint(0, len(absent), (1,)).item()]
            adj_mut[i, j] = 1.0

    if torch.rand(1).item() < edge_del_prob:
        present = [(i, j) for i in range(N) for j in range(i + 1, N)
                   if adj_mut[i, j].item() > 0.5]
        if present:
            i, j = present[torch.randint(0, len(present), (1,)).item()]
            adj_mut[i, j] = 0.0

    return x_mut, adj_mut


class EvalCache:

    def __init__(self, max_size: int = 10_000):
        self._store: OrderedDict[str, float] = OrderedDict()
        self.max_size = max_size
        self.hits     = 0
        self.misses   = 0

    def get(self, key: str) -> Optional[float]:
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: float) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class NASEvolution:

    def __init__(
        self,
        model:         Any,
        eval_fn:       Callable[[torch.Tensor, torch.Tensor], float],
        pop_size:      int   = 100,
        elite_ratio:   float = 0.1,
        node_mut_prob: float = 0.1,
        edge_add_prob: float = 0.1,
        edge_del_prob: float = 0.1,
        train_epochs:  int   = 5,
        lr:            float = 1e-3,
        node_loss_w:   float = 1.0,
        edge_loss_w:   float = 1.0,
        cache_size:    int   = 10_000,
        device:        str   = "cpu",
    ):
        self.model         = model.to(device)
        self.eval_fn       = eval_fn
        self.pop_size      = pop_size
        self.elite_ratio   = elite_ratio
        self.node_mut_prob = node_mut_prob
        self.edge_add_prob = edge_add_prob
        self.edge_del_prob = edge_del_prob
        self.train_epochs  = train_epochs
        self.node_loss_w   = node_loss_w
        self.edge_loss_w   = edge_loss_w
        self.device        = torch.device(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.cache     = EvalCache(max_size=cache_size)

        self.population: List[List] = []
        self.generation: int = 0


    def initialise(
        self,
        initial_pop: Optional[List[Individual]] = None,
    ) -> None:
        if initial_pop is not None:
            pool = [(x.to(self.device), adj.to(self.device))
                    for x, adj in initial_pop]
        else:
            pool = [random_individual(device=self.device)
                    for _ in range(self.pop_size)]

        self.population = []
        for x, adj in pool:
            fitness = self._evaluate(x, adj)
            self.population.append([x, adj, fitness])

        self._sort_population()


    def step(self) -> dict:
        self._sort_population()
        k = max(1, math.ceil(self.elite_ratio * len(self.population)))

        elites = [(ind[0], ind[1]) for ind in self.population[:k]]

        node_loss_avg, edge_loss_avg = self._train_model(elites)

        reconstructed = self._reconstruct_all()

        candidates, n_repaired = self._mutate_and_repair(reconstructed)

        evaluated = [(x, adj, self._evaluate(x, adj)) for x, adj in candidates]

        evaluated.sort(key=lambda t: -t[2])
        new_individuals = evaluated[:k]

        self.population = self.population[: len(self.population) - k]
        for x, adj, fit in new_individuals:
            self.population.append([x, adj, fit])
        self._sort_population()

        self.generation += 1
        return {
            "generation":      self.generation,
            "best_fitness":    self.population[0][2],
            "mean_fitness":    sum(ind[2] for ind in self.population) / len(self.population),
            "n_repaired":      n_repaired,
            "cache_hit_rate":  self.cache.hit_rate,
            "model_node_loss": node_loss_avg,
            "model_edge_loss": edge_loss_avg,
        }


    def run(self, n_generations: int, verbose: bool = True) -> List[dict]:
        if not self.population:
            raise RuntimeError("Call initialise() before run().")

        history = []
        for _ in range(n_generations):
            stats = self.step()
            history.append(stats)
            if verbose:
                print(
                    f"Gen {stats['generation']:4d} | "
                    f"best={stats['best_fitness']:.4f}  "
                    f"mean={stats['mean_fitness']:.4f}  "
                    f"repaired={stats['n_repaired']:3d}  "
                    f"cache_hit={stats['cache_hit_rate']:.1%}  "
                    f"node_loss={stats['model_node_loss']:.4f}  "
                    f"edge_loss={stats['model_edge_loss']:.4f}"
                )
        return history


    @property
    def best(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if not self.population:
            raise RuntimeError("Population is empty — call initialise() first.")
        ind = self.population[0]
        return ind[0], ind[1], ind[2]


    def _evaluate(self, x: torch.Tensor, adj: torch.Tensor) -> float:
        key    = individual_hash(x, adj)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        score = self.eval_fn(x, adj)
        self.cache.put(key, score)
        return score

    def _sort_population(self) -> None:
        self.population.sort(key=lambda ind: -ind[2])

    def _train_model(
        self, elites: List[Individual]
    ) -> Tuple[float, float]:
        self.model.train()

        x_batch   = torch.stack([e[0] for e in elites]).to(self.device)
        adj_batch = torch.stack([e[1] for e in elites]).to(self.device)

        total_node, total_edge = 0.0, 0.0
        for _ in range(self.train_epochs):
            self.optimizer.zero_grad()
            loss, info = self.model.compute_loss(
                x_batch, adj_batch,
                node_weight=self.node_loss_w,
                edge_weight=self.edge_loss_w,
            )
            loss.backward()
            self.optimizer.step()
            total_node += info["node_loss"]
            total_edge += info["edge_loss"]

        return total_node / self.train_epochs, total_edge / self.train_epochs

    def _reconstruct_all(self) -> List[Individual]:
        x_batch   = torch.stack([ind[0] for ind in self.population]).to(self.device)
        adj_batch = torch.stack([ind[1] for ind in self.population]).to(self.device)

        node_preds, adj_preds = self.model.reconstruct(x_batch, adj_batch)

        B, N, C = x_batch.shape
        reconstructed = []
        for b in range(B):
            x_rec   = F.one_hot(node_preds[b], num_classes=C).float()
            adj_rec = adj_preds[b]
            reconstructed.append((x_rec, adj_rec))
        return reconstructed

    def _mutate_and_repair(
        self,
        individuals: List[Individual],
    ) -> Tuple[List[Individual], int]:
        candidates = []
        n_repaired = 0
        for x, adj in individuals:
            x_mut, adj_mut = mutate(
                x, adj,
                node_mut_prob=self.node_mut_prob,
                edge_add_prob=self.edge_add_prob,
                edge_del_prob=self.edge_del_prob,
            )
            if not is_valid(x_mut, adj_mut):
                x_mut, adj_mut = repair(x_mut, adj_mut)
                n_repaired += 1
            candidates.append((x_mut, adj_mut))
        return candidates, n_repaired
