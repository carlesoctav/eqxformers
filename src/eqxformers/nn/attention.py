from __future__ import annotations

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
        key: Float[Array, "B S N H"],
        value: Float[Array, "B S N H"],
        bias: Float[Array, "B N T S"] | None = None,
        attention_mask: Bool[Array, "B N T S"] | None = None,
        *,
        inference: bool = False,
        dropout_rate: float = 0.0,
        dropout_key: PRNGKeyArray | None = None,
        **kwargs: tp.Any,
    ) -> Float[Array, "B T N H"]:
        ...


def _validate_heads(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S N H"],
    value: Float[Array, "B S N H"],
) -> None:
    if query.shape[-2] != key.shape[-2] or query.shape[-2] != value.shape[-2]:
        raise ValueError("Query, key, and value must share the same number of heads")

    if key.shape[-1] != query.shape[-1] or value.shape[-1] != query.shape[-1]:
        raise ValueError("Query, key, and value must share the same head dimension")


def jax_nn_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S N H"],
    value: Float[Array, "B S N H"],
    bias: Float[Array, "B N T S"] | None = None,
    attention_mask: Bool[Array, "B N T S"] | None = None,
    *,
    inference: bool = False,
    dropout_rate: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    **kwargs: tp.Any,
) -> Float[Array, "B T N H"]:
    _validate_heads(query, key, value)

    deterministic = inference or dropout_rate == 0.0
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=attention_mask,
        dropout_rng=dropout_key,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        **kwargs,
    )


def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S N H"],
    value: Float[Array, "B S N H"],
    bias: Float[Array, "B N T S"] | None = None,
    attention_mask: Bool[Array, "B N T S"] | None = None,
    *,
    inference: bool = False,
    dropout_rate: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    **kwargs: tp.Any,
) -> Float[Array, "B T N H"]:
    _validate_heads(query, key, value)

    scale = jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))
    query = query / scale

    attn_scores = jnp.einsum("btnh,bsnh->bnts", query, key)

    if bias is not None:
        attn_scores = attn_scores + bias

    if attention_mask is not None:
        mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        large_neg = jnp.finfo(attn_scores.dtype).min
        attn_scores = jnp.where(mask, attn_scores, large_neg)

    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    if not inference and dropout_rate > 0.0:
        if dropout_key is None:
            raise ValueError("dropout_rate > 0.0 requires providing a dropout_key")
        attn_weights = dropout(attn_weights, dropout_rate, inference=False, key=dropout_key)

    attn_output = jnp.einsum("bnt s, bsnh -> btnh", attn_weights, value)
    return attn_output


class AttentionFnInterface(GeneralInterface[str, AttentionFn]):
    _global_mapping: dict[str, AttentionFn] = {
        "sdpa": jax_nn_dot_product_attention,
        "eager": eager_dot_product_attention,
    }


ALL_ATTENTION_FUNCTIONS = AttentionFnInterface()


class Attention(Module):
    attn_impl: str = eqx.field(static=True)
    attention_fn: AttentionFn

    def __init__(self, attn_impl: str):
        self.attn_impl = attn_impl
        self.attention_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    def __call__(
        self,
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S N H"],
        value: Float[Array, "B S N H"],
        bias: Float[Array, "B N T S"] | None = None,
        attention_mask: Bool[Array, "B N T S"] | None = None,
        *,
        inference: bool = False,
        dropout_rate: float = 0.0,
        dropout_key: PRNGKeyArray | None = None,
        **kwargs: tp.Any,
    ) -> Float[Array, "B T N H"]:
        prepared = self.maybe_prepare_input(query, key, value, bias, attention_mask)
        if isinstance(prepared, tuple):
            query, key, value, bias, attention_mask = prepared
        else:
            raise TypeError("prepare_input hook must return a tuple for attention inputs")

        output = self.attention_fn(
            query,
            key,
            value,
            bias=bias,
            attention_mask=attention_mask,
            inference=inference,
            dropout_rate=dropout_rate,
            dropout_key=dropout_key,
            **kwargs,
        )

        return self.maybe_prepare_output(output)
