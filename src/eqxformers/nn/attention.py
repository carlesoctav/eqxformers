import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ..modeling_utils import Module
from ..utils import GeneralInterface
from .dropout import dropout


class AttentionFn(tp.Protocol):
    def __call__(
        self,
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Float[Array, "B T N S"] | None = None, 
        attention_mask: Bool[Array, "B T N S"] | Bool[Array, "B T S"] | None = None,
        *,
        inference: bool = False,
        dropout_rate: float = 0.0,
        dropout_key: PRNGKeyArray | None = None,
        **kwargs: tp.Any,
    ) -> Float[Array, "B T N H"]:
        ...

def jax_nn_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Float[Array, "B T N S"] | None = None, 
    attention_mask: Bool[Array, "B T N S"] | Bool[Array, "B T S"] | None = None, 
    *,
    inference: bool = False,
    dropout_rate: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    **kwargs: tp.Any,
) -> Float[Array, "B T N H"]:

    # query: ArrayLike,
    # key: ArrayLike,
    # value: ArrayLike,
    # bias: ArrayLike | None = None,
    # mask: ArrayLike | None = None,
    # *,
    # scale: float | None = None,
    # is_causal: bool = False,
    # query_seq_lengths: ArrayLike | None = None,
    # key_value_seq_lengths: ArrayLike | None = None,
    # local_window_size: int | tuple[int, int] | None = None,
    # implementation: Literal['xla', 'cudnn'] | None = None) -> Array:
    # deterministic = inference or dropout_rate == 0.0
    if dropout_rate > 0.0 and not inference: 
        raise NotImplementedError(
            "jax.nn.dot_product_attention does not support dropout."
            "please set dropout_rate to 0.0 or inference to True." 
        )

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias.swap_axes(1, 2) if bias is not None else None,
        mask=attention_mask.swap_axes(1, 2) if attention_mask is not None else None,
        **kwargs,
    )


def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Float[Array, "B T N S"] | None = None, 
    attention_mask: Bool[Array, "B T N S"] | Bool[Array, "B T S"] | None = None, 
    *,
    inference: bool = False,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    **kwargs,
) -> Float[Array, "B T N H"]:
    query = query / jnp.sqrt(query.shape[-1])

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape

    if Hk != H or Hv != H:
        raise ValueError("Query, key, and value must share the same head dimension")

    if Kv != K:
        raise ValueError("Value tensor must share the key head axis for attention")

    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = N // K
        key = jnp.repeat(key, repeat_factor, axis=-2)
        value = jnp.repeat(value, repeat_factor, axis=-2)
        K = N

    scores = jnp.einsum("b t n h, b s n h -> b t n s", query, key)

    if attention_mask is not None:
        #mask: (B, T, N, S) or (B, T, S), score: (B, T, N, S)
        neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
        attention_mask = attention_mask[:, :, None, :] if attention_mask.ndim == 3 else attention_mask
        scores = jnp.where(attention_mask, scores, neg_inf)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(scores.dtype, jnp.float32)
    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if not inference: 
        weights = dropout(weights, dropout_rate, inference, key = dropout_key)

    attn = jnp.einsum("btns, bsnh -> btnh", weights, value)
    return attn


class AttentionFnInterface(GeneralInterface[str, AttentionFn]):
    _global_mapping: dict[str, AttentionFn] = {
        "sdpa": jax_nn_dot_product_attention,
        "eager": eager_dot_product_attention,
    }


ALL_ATTENTION_FUNCTIONS = AttentionFnInterface()


