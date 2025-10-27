from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray


def maybe_split_rng(key: PRNGKeyArray | None, num: int = 2):
    """Splits a random key into multiple random keys while handling None/num==1."""
    if key is None:
        return [None] * num
    if num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    return jrandom.split(key, num)

