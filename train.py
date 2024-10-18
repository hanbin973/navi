import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

import optax

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
