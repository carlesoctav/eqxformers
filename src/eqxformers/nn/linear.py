import jax
from jaxtyping import Array, Float 
import equinox as eqx
from ..modeling_utils import Module
from jaxtyping import PRNGKeyArray


Initializer = jax.nn.initializers.Initializer

normal_init = jax.nn.initializers.normal()

class Linear(Module):
    weight: Array
    bias: Array

    in_features: int = eqx.field(static = True)
    out_features: int = eqx.field(static = True)
    use_bias: int = eqx.field(static = True)
    initializer: Initializer  = eqx.field(static = True, default = normal_init)


    def __init__(
        self
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer = normal_init,
        key: PRNGKeyArray
    ):
        wkey, bkey = jax.random.split(key, 2)

        self.weight = initializer(wkey, (out_features, in_features))
        if use_bias:
            self.bias = zero_init(bkey, (out_features, ))

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.initializer = initializer


    def __call__(self, x: Float[Array, "*B in_features"]):
        x = self.maybe_prepare_input(x)
        x = x @ self.weight.T
        if self.use_bias:
            x = x + self.bias

        return self.maybe_prepare_output(x)

