import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
from ..modeling_utils import Module

Initializer = jax.nn.initializers.Initializer

one_init = jax.nn.initializers.ones
zero_init = jax.nn.initializers.zeros

class LayerNorm(Module):
    weight: Array | None
    bias: Array | None

    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)
    initializer: Initializer = eqx.field(static=True)
    use_fast_variance: bool = eqx.field(static=True)

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
        use_fast_variance: bool = True,
        *,
        initializer: Initializer = one_init,  
        key: PRNGKeyArray, 
    ):
        normalized_shape = (
                (normalized_shape,)
                if isinstance(normalized_shape, int)
                else normalized_shape
        )

        if elementwise_affine:
            wkey, bkey = jax.random.split(key)
            self.weight = initializer(wkey, normalized_shape)
            if use_bias:
                self.bias = zero_init(bkey, normalized_shape, )
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

        self.normalized_shape = normalized_shape
        self.eps = eps 
        self.elementwise_affine = elementwise_affine 
        self.initializer = initializer
        self.use_fast_variance = use_fast_variance



    def __call__(
        self,
        x: Float[Array, " *normalized_shape"],
    ) -> Array:


        x = self.maybe_prepare_input(x)

        nd = x.ndim
        k = len(self.normalized_shape)

        if k == 0 or nd < k:
            raise ValueError(
                f"Input rank {nd} too small for normalized_shape {self.normalized_shape}"
            )

        if tuple(x.shape[-k:]) != tuple(self.normalized_shape):
            raise ValueError(
                f"Trailing shape {x.shape[-k:]} does not match normalized_shape {self.normalized_shape}"
            )

        axes = tuple(range(nd - k, nd))
        orig_dtype = x.dtype
        acc_dtype = jnp.promote_types(orig_dtype, jnp.float32)
        x_acc = jnp.asarray(x, acc_dtype)
        mean = jnp.mean(x_acc, axis=axes, keepdims=True)
        diff = x_acc - mean
        if self.use_fast_variance:
            mean_sq = jnp.mean(jnp.square(x_acc), axis=axes, keepdims=True)
            var = jnp.maximum(0.0, mean_sq - jnp.square(mean))
        else:
            var = jnp.mean(jnp.square(diff), axis=axes, keepdims=True)
        inv = jax.lax.rsqrt(var + self.eps)
        y = diff * inv

        if self.weight is not None:
            weight = jnp.asarray(self.weight, acc_dtype)
            y = weight * y

        if self.bias is not None:
            bias = jnp.asarray(self.bias, acc_dtype)
            y = y + bias

        y = jnp.asarray(y, orig_dtype)

        return self.maybe_prepare_output(y)
