import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

from simulation import DataLoader

@jax.jit
def loss_fn(state: train_state.TrainState,
            params: jnp.ndarray,
            traits: jnp.ndarray,
            distance: jnp.ndarray,
            receivers: jnp.ndarray,
            senders: jnp.ndarray,
            vcs: jnp.ndarray,
            nodes_padding: jnp.ndarray,
            edges_padding: jnp.ndarray):
    
    vcs_hat = state.apply_fn(params,
                             traits,
                             distance,
                             receivers,
                             senders,
                             nodes_padding,
                             edges_padding)
    error_sum_squared = jnp.mean(jnp.power(vcs - vcs_hat, 2), axis=0).mean()
    return error_sum_squared

@jax.jit
def train_step(state: train_state.TrainState,
               traits: jnp.ndarray,
               distance: jnp.ndarray,
               receivers: jnp.ndarray,
               senders: jnp.ndarray,
               vcs: jnp.ndarray,
               nodes_padding: jnp.ndarray,
               edges_padding: jnp.ndarray):

    grad_fn = jax.value_and_grad(loss_fn, argnums=1)
    loss, grads = grad_fn(state,
                          state.params,
                          traits,
                          distance,
                          receivers,
                          senders,
                          vcs,
                          nodes_padding,
                          edges_padding)
    state = state.apply_gradients(grads=grads)
    return state, loss

class TrainOnTheFly:
    def __init__(self,
                 state: train_state.TrainState,
                 distance: jnp.ndarray,
                 receivers: jnp.ndarray,
                 senders: jnp.ndarray):
        self.state = state
        self.distance = distance
        self.receivers = receivers
        self.senders = senders
        self.optimization_trajectory = []

    def train(self,
              data_loader: DataLoader,
              num_refresh: int = 100,
              num_epochs: int = 2000,
              verbose: bool = False,
              callback: bool = True):

        traits, vcs, _, nodes_padding, edges_padding = next(data_loader)
        for epoch in range(num_epochs):
            state, loss_val = train_step(self.state,
                                         traits,
                                         self.distance,
                                         self.receivers,
                                         self.senders,
                                         vcs,
                                         nodes_padding,
                                         edges_padding)
            if epoch % num_refresh == 0:
                traits, vcs, _, nodes_padding, edges_padding = next(data_loader)
            if epoch % num_refresh == 1:
                if verbose:
                    print('Epoch: %d, Loss: %.8f' % (epoch, loss_val)) 

            self.optimization_trajectory.append(loss_val)
            self.state = state
