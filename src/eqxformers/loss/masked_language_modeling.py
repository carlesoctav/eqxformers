from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from eqxformers.training_utils import LossFn, LossFunctionConfig


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    where: Array | None = None,
) -> Array:
    if where is not None:
        safe_labels = jnp.where(where, labels, 0)
    else:
        safe_labels = labels

    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - jax.lax.stop_gradient(logits_max)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))
    true_logit = jnp.take_along_axis(shifted, safe_labels[..., None], axis=-1)[..., 0]
    loss = log_sum_exp - true_logit
    if where is not None:
        loss = jnp.where(where, loss, 0.0)
    return loss


def masked_language_modeling_loss(
    model: eqx.Module,
    batch: Any,
    *,
    ignore_index: int = - 100,
    key: Array
):
    _, dropout_key = jr.split(key)

    logits = model(
        input_ids=batch.input_ids,
        position_ids= batch.position_ids,
        token_type_ids=batch.token_type_ids,
        attention_mask=batch.attention_mask,
        segment_ids=getattr(batch, "segment_ids", None),
        key=dropout_key,
    )
    logits = logits.astype(jnp.float32)

    labels = batch.labels
    valid_mask = labels != ignore_index

    loss_per_token = softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=jnp.where(valid_mask, labels, 0),
        where=valid_mask,
    )

    total_loss = jnp.sum(loss_per_token)
    num_valid_tokens = jnp.sum(valid_mask)
    accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == labels) & valid_mask)

    aux = {
        "loss": (total_loss, num_valid_tokens),
        "acc": (accuracy, num_valid_tokens),
        "total_token": num_valid_tokens,
    }
    return total_loss, aux



@LossFunctionConfig.register_subclass("mlm")
@dataclass
class MaskedLanguageModelingLossConfig(LossFunctionConfig):
    ignore_index: int = -100
    def make(self) -> LossFn:
        return functools.partial(
            masked_language_modeling_loss,
            ignore_index = self.ignore_index
        )

__all__ = [
    "masked_language_modeling_loss",
    "MaskedLanguageModelingLossConfig",
    "softmax_cross_entropy_with_integer_labels",
]
