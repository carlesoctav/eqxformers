import jax
import jax.numpy as jnp
from ..modeling_utils import Module
from jaxtyping import PRNGKeyArray, Array, Real
import equinox as eqx



def dropout(
        x: Real[Array, "..."] ,
        dropout_rate: float,
        *,
        key: PRNGKeyArray
):
    if dropout_rate < 0.0 or dropout_rate > 1.0:
        raise ValueError("Dropout rate must be in the range [0.0, 1.0].")
    if dropout_rate == 0.0:
        return jnp.zeros_like(x)
    if dropout_rate == 1.0:
        return x

    keep_prob = 1.0 - dropout_rate
    keep_mask = jax.random.bernoulli(key, keep_prob, shape = x.shape)
    output = jnp.where(keep_mask, x / keep_prob, 0)
    return output



class Dropout(Module):
    p: float = eqx.field(static = True)
    inference: bool 

    def __init__(
        self,
        p: int,
        inference = False,
    ):
        self.p = p
        self.inference = inference

        if self.p >1 or self.p < 0:
            raise ValueError("Dropout probability must be between 0 and 1.")



    def __call__(
        self,
        x: Real[Array, "..."],
        *, 
        key: PRNGKeyArray | None = None
    ):
        if key is None and not self.inference:
            raise ValueError("Dropout layer requires a PRNGKey during non-inference mode.") 

        if self.inference:
            return self.maybe_prepare_output(x)

        if self.p == 0:
            return self.maybe_prepare_output(x)

        x = self.maybe_prepare_input(x)

        keep_prob = 1.0 - self.p
        keep_mask = jax.random.bernoulli(key, keep_prob, shape = x.shape)
        output = jnp.where(keep_mask, x / keep_prob, 0)

        return self.maybe_prepare_output(output)

