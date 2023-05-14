"""Simple linear layer."""

from typing import Dict

import haiku as hk
import jax.numpy as jnp
from jax import vmap


class Linear(hk.Module):
    """Model defining linear relation between input nodes and targets."""

    def __init__(self, dim_out):
        super().__init__()
        self.mlp = hk.nets.MLP([dim_out], activate_final=False, name="MLP")

    def __call__(self, features: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # transform
        x = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features[0]
        ]
        # call
        return vmap(self.mlp)(jnp.concatenate(x, axis=-1))
