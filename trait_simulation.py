import tskit
import numpy as np
import scipy.sparse as sparse

import numba
from numba import i4, f8
from numba.experimental import jitclass

spec = [
        ('sample_weights', f8[:]),
        ('parent', i4[:]),
        ('num_samples', i4[:]),
        ('edges_left', f8[:]),
        ('edges_right', f8[:]),
        ('edges_parent', i4[:]),
        ('edges_child', i4[:]),
        ('edge_insertion_order', i4[:]),
        ('edge_removal_order', i4[:]),
        ('sequence_length', f8),
        ('nodes_time', f8[:]),
        ('samples', i4[:]),
        ('position', f8),
        ('virtual_root', i4),
        ('x', f8[:]),
        ('w', f8[:]),
        ('stack', f8[:]),
        ('NULL', i4)
       ] 

@jitclass(spec)
class TraitVector:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length
    ):
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        self.x = np.zeros(N, dtype=np.float64)
        self.stack = np.zeros(N, dtype=np.float64)
        self.NULL = -1 # to avoid tskit.NULL in numba

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.parent[v] = u
            self.num_samples[v] = 1

    def remove_edge(self, p, c):
        self.stack[c] += self.get_z(c)
        self.x[c] = self.position
        self.parent[c] = -1
        self.adjust_path_up(c, p, -1)

    def insert_edge(self, p, c):
        self.adjust_path_up(c, p, +1)
        self.x[c] = self.position
        self.parent[c] = p

    def adjust_path_up(self, c, p, sign):
        # sign = -1 for removing edges, +1 for adding
        while p != self.NULL:
            self.stack[p] += self.get_z(p)
            self.x[p] = self.position
            # check for floating point error
            prev_stack = self.stack[c]
            self.stack[c] -= sign * self.stack[p]
            p = self.parent[p]

    def get_z(self, u):
        p = self.parent[u]
        if p == self.NULL or u >= self.virtual_root:
            return 0.0
        time = self.nodes_time[p] - self.nodes_time[u]
        span = self.position - self.x[u]
        return np.sqrt(time * span) * np.random.normal()

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        left = 0
        self.position = left

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.remove_edge(p, c)
                k += 1
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                self.insert_edge(p, c)
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            left = right
            self.position = left

        # clear remaining things down to virtual samples
        for j, u in enumerate(self.samples):
            v = self.virtual_root + 1 + j
            self.remove_edge(u, v)

        out = np.zeros(len(self.samples))
        for out_i in range(len(self.samples)):
            i = out_i + self.virtual_root + 1
            out[out_i] = self.stack[i]
        return out

def genetic_value(ts, **kwargs):
    rv = TraitVector(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs,
    )
    return rv.run()

class DataLoader():
    def __init__(self,
                 ts: tskit.TreeSequence,
                 norm_factor: float,
                 receivers: np.ndarray,
                 senders: np.ndarray,
                 tau_range: list = [0.01, 1],
                 sigma_range: list = [0.1, 1],
                 batch_size: int = 200,
                 dynamic_size: bool = False,
                 ) -> (np.ndarray, np.ndarray):

        self.ts = ts
        self.norm_factor = norm_factor
        self.receivers = receivers
        self.senders = senders
        self.tau_range = tau_range
        self.sigma_range = sigma_range
        self.batch_size = batch_size
        self.dynamic_size = dynamic_size
        self.num_nodes = ts.num_individuals
        self.num_edges = receivers.size

    def __iter__(self):
        return self

    def __next__(self):
        traits = np.empty((self.batch_size, self.ts.num_individuals, 1))
        params = np.empty((self.batch_size, 2))
        factors = np.empty(self.batch_size)
        nodes_padding = np.ones((self.batch_size, self.num_nodes)) 
        edges_padding = np.ones((self.batch_size, self.num_edges))
        for i in range(self.batch_size):
            # sample params
            tau = np.random.uniform(self.tau_range[0], self.tau_range[1])
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # sample trait
            g = genetic_value(self.ts) / np.sqrt(self.norm_factor) * tau
            e = np.random.normal(size=self.ts.num_samples) * sigma
            y = g + e
            factor = 1.5 * y.std()
            y /= factor

            # construct graph
            traits[i] = y[:,None]
            params[i] = np.asarray([tau, sigma]) / factor
            factors[i] = factor

            # paddings
            if self.dynamic_size:
                # probability to keep nodes
                p_keep = np.random.uniform(low=0.5, high=1)
                # sample nodes to keep
                nodes_keep = np.random.binomial(1, p_keep, size=self.num_nodes)
                nodes_padding[i] = nodes_keep
                # pick edges
                nodes_keep_idx = np.arange(self.num_nodes)[nodes_keep.astype(bool)]
                receivers_keep = np.isin(self.receivers, nodes_keep_idx).astype(int)
                senders_keep = np.isin(self.senders, nodes_keep_idx).astype(int)
                edges_padding[i] = receivers_keep * senders_keep

        return traits, params, factors, nodes_padding, edges_padding



