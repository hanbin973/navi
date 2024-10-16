import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

class IndicatorWeights(nn.Module):
    distance_max: float
    n_bins: int
    def __call__(self,
                 distance: jnp.ndarray) -> jnp.ndarray:
        distance = distance.reshape(-1,1)
        bins = jnp.linspace(0, self.distance_max, self.n_bins + 1)
        indicator = (bins[None,:-1] - distance) * (distance - bins[None,1:])
        return jnp.heaviside(indicator, 0)

class MLPWeights(nn.Module):
    dim_hidden: int
    dim_out: int
    @nn.compact
    def __call__(self,
                 distance: jnp.ndarray) -> jnp.ndarray:
        distance = distance.reshape(-1,1)
        hidden = nn.Dense(self.dim_hidden)(distance)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(self.dim_out)(hidden)
        out = nn.relu(hidden)
        return out

class PowerDifference(nn.Module):
    a: float
    b: float
    a_init: Callable = nn.initializers.constant
    b_init: Callable = nn.initializers.constant
    @nn.compact
    def __call__(self,
                 x1: jnp.ndarray,
                 x2: jnp.ndarray) -> jnp.ndarray:
        a, b = self.param('a', self.a_init(self.a), shape=1), self.param('b', self.b_init(self.b), shape=1)
        a, b = jnp.clip(a, min=0, max=1), jnp.abs(b)        
        return jnp.pow(jnp.abs(a * x1 - (1 - a) * x2), b)

def normalize_edges(edges: jnp.ndarray,
                    segment_ids: jnp.ndarray,
                    num_segments:jnp.ndarray):
    edges_segment_sum = jax.ops.segment_sum(edges,
                                            segment_ids=segment_ids,
                                            num_segments=num_segments,
                                            indices_are_sorted=True)
    return edges / (edges_segment_sum[segment_ids] + 1e-5)

class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution
    """
    distance_max: float
    num_indicator_weight: int
    dim_mlp_hidden: int
    num_mlp_weight: int
    edges_padding: jnp.ndarray

    def setup(self):
        self.indicator_weight = IndicatorWeights(self.distance_max, self.num_indicator_weight)
        self.mlp_weight = MLPWeights(self.dim_mlp_hidden, self.num_mlp_weight)
        self.powerdiff = PowerDifference(.5, 2)
        self.gamma_self = nn.Dense(self.num_indicator_weight + self.num_mlp_weight, use_bias=False)
        self.gamma_gathered = nn.Dense(self.num_indicator_weight + self.num_mlp_weight)
        
    def __call__(self,
                 nodes: jnp.ndarray,
                 distance: jnp.ndarray,
                 receivers: jnp.ndarray,
                 senders: jnp.ndarray) -> jnp.ndarray:
        
        # message-passing
        indicator_weights = normalize_edges(self.indicator_weight(distance),
                                            segment_ids=receivers,
                                            num_segments=nodes.shape[0])
        mlp_weights = normalize_edges(self.mlp_weight(distance),
                                      segment_ids=receivers,
                                      num_segments=nodes.shape[0])
        weights = jnp.hstack((indicator_weights, mlp_weights)) * self.edges_padding.reshape(-1,1)
        edge_pds = self.powerdiff(nodes[receivers], nodes[senders])
        nodes_gathered = jax.ops.segment_sum(weights * edge_pds,
                                             segment_ids=receivers,
                                             num_segments=nodes.shape[0],
                                             indices_are_sorted=True)

        # convolution
        nodes = self.gamma_self(nodes) + self.gamma_gathered(nodes_gathered)
        nodes = nn.relu(nodes)
        
        return nodes            

class Readout(nn.Module):
    """
    DeepSets + ResNet Readout
    https://ieeexplore.ieee.org/document/8852103
    """
    nodes_padding: jnp.ndarray
    @nn.compact
    def __call__(self, nodes: jnp.ndarray) -> jnp.ndarray:
        nodes = nodes * self.nodes_padding.reshape(-1,1)
        nodes = jnp.hstack((nodes, nodes**2))
        sumstats = jnp.sum(nodes, axis=0) / jnp.sum(self.nodes_padding)
        sumstats_resid = nn.relu(nn.Dense(128)(nodes))
        sumstats_resid = nn.relu(nn.Dense(sumstats.size)(sumstats_resid))
        sumstats_resid = jnp.sum(sumstats_resid, axis=0) / jnp.sum(self.nodes_padding)
        return sumstats + sumstats_resid # jnp.concatenate

class Mapping(nn.Module):
    """
    Just MLP
    """
    dim_hiddens: Sequence[int]
    num_params: int
    @nn.compact
    def __call__(self, sumstats: jnp.ndarray) -> jnp.ndarray:
        theta = sumstats
        for dim in self.dim_hiddens:
            theta = nn.Dense(dim)(theta)
            theta = nn.relu(theta)
        theta = nn.Dense(self.num_params)(theta)
        theta = nn.relu(theta)
        return theta

class NAVI(nn.Module):
    """
    SpatialGraphConv -> Readout -> Mapping
    """
    num_params: int
    num_spatial_conv: int
    distance_max: float
    num_indicator_weight: int
    dim_mlp_hidden: int
    num_mlp_weight: int
    dim_mapping_hiddens: Sequence[int]
    @nn.compact
    def __call__(self, 
                traits: jnp.ndarray, 
                distance: jnp.ndarray, 
                receivers: jnp.ndarray, 
                senders: jnp.ndarray,
                nodes_padding: jnp.ndarray,
                edges_padding: jnp.ndarray) -> jnp.ndarray:
        nodes = traits
        for _ in range(self.num_spatial_conv):
            nodes = SpatialGraphConv(self.distance_max,
                                    self.num_indicator_weight,
                                    self.dim_mlp_hidden,
                                    self.num_mlp_weight,
                                    edges_padding
                                    )(nodes, distance, receivers, senders)
        readout = Readout(nodes_padding)(nodes)
        params = Mapping(self.dim_mapping_hiddens, self.num_params)(readout)
        return params 
