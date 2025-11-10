import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import jax.tree_util as jtu


def maybe_split_key(key: PRNGKeyArray | None, num: int = 2) -> tuple[PRNGKeyArray | None, ...]:
    """Splits a random key into multiple random keys while handling None/num==1."""
    if key is None:
        return (None, ) * num
    if num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    return jax.random.split(key, num)



def is_array_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")

def slice_out(tree, i ,size):
    def take(leaf):
        if is_array_like(leaf) and leaf.shape[0] == size:
            return leaf[i]
        else:
            return leaf

    return jtu.tree_map(take, tree)



def is_array_like_with_leading_size(x, size):
    return hasattr(x, "shape") and hasattr(x, "dtype") and x.shape[0] == size
