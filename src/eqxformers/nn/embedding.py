import jax
from jaxtyping import Array, Int, PRNGKeyArray
import equinox as eqx
from ..modeling_utils import Module


sentinel = object()
Initializer = jax.nn.initializers.Initializer

normal_init = jax.nn.initializers.normal()


class Embedding(Module):
    weight: Array

    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    initializer: Initializer = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        initializer: Initializer = normal_init,
        key: PRNGKeyArray
    ):
        self.weight = initializer(key, (num_embeddings, embedding_dim))

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.initializer = initializer 

    def __call__(
        self,
        x: Int[Array, "..."],
    ) -> Array:
        x = self.maybe_prepare_input(x)
        output = self.weight.take(x, axis = 0)
        return self.maybe_prepare_output(output)
