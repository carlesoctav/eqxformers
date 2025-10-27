import math
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array
from transformers import PretrainedConfig
from ..utils import first_from


RotaryInitFn = Callable[[Any, jnp.dtype], Tuple[Array, Array]]


def _base_angles(dim: int, base: float, dtype: jnp.dtype) -> Array:
    half_dim = dim // 2
    exponent = jnp.arange(half_dim, dtype=jnp.float32) * (2.0 / float(dim))
    inv_freq = (base ** (-exponent)).astype(jnp.float32)
    return inv_freq


def r_from_inv(inv_freq: Array, seq_len: int, dtype: jnp.dtype) -> Array:
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return jnp.exp(1j * angles)


def default_rope(
    config: PretrainedConfig,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[Array, Array]:
    base = float(config.rope_theta)
    assert base > 1.0, f"rope_theta/base must be > 1, got {base}"

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    assert head_dim > 0, f"head_dim must be > 0, got {head_dim}"
    assert head_dim % 2 == 0, f"RoPE requires even head_dim, got {head_dim}"

    rot_dim = head_dim

    seq_len = first_from(
        getattr(config, "max_position_embeddings"),
        error_msg="max_position_embeddings must be set for RoPE initialization.",
    )
    assert seq_len >= 1, f"seq_len must be >= 1, got {seq_len}"

    half = rot_dim // 2
    exponent = jnp.arange(half, dtype=jnp.float32) * (2.0 / float(rot_dim))
    inv_freq = (base ** (-exponent)).astype(jnp.float32)

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    r_theta = jnp.exp(1j * angles)

    attention_scaling = jnp.asarray(1.0, dtype=jnp.float32)

    return r_theta, attention_scaling


ROPE_INIT_FUNCTIONS: Dict[str, RotaryInitFn] = {
    "default": default_rope,
}


def make_rope_init_fn(rope_type: str) -> RotaryInitFn:
    try:
        return ROPE_INIT_FUNCTIONS[rope_type]
    except KeyError as err:
        raise KeyError(
            f"Unsupported RoPE type '{rope_type}'. Available: {sorted(ROPE_INIT_FUNCTIONS)}"
        ) from err

