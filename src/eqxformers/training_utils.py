from __future__ import annotations

import abc
from typing import Any, Protocol

import draccus
import equinox as eqx
import optax
from jax import Array


class LossFn(Protocol):
    def __call__(self, model: eqx.Module, batch: Any, *, key: Array) -> tuple[Array, dict[str, Any]]:
        ...


class TrainStepFn(Protocol):
    def __call__(self, state: "State", batch: Any, *, key: Array) -> tuple["State", dict[str, Any]]:
        ...


class LossFunctionConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    @abc.abstractmethod
    def make(self, data_config: Any) -> LossFn:
        raise NotImplementedError


class State(eqx.Module):
    """Container holding the trainable module and optimizer state."""

    model: eqx.Module
    opt_state: Any
    grad_tx: optax.GradientTransformation = eqx.field(static=True)
    wrt: Any = eqx.field(static=True)

    def __init__(
        self,
        model: eqx.Module,
        grad_tx: optax.GradientTransformation,
        *,
        wrt: Any = eqx.is_inexact_array,
        opt_state: Any | None = None,
    ):
        params = eqx.filter(model, wrt)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "grad_tx", grad_tx)
        object.__setattr__(self, "wrt", wrt)
        if opt_state is None:
            opt_state = grad_tx.init(params)
        object.__setattr__(self, "opt_state", opt_state)

    def apply_gradients(self, grads: Any) -> "State":
        params = eqx.filter(self.model, self.wrt)
        updates, next_opt_state = self.grad_tx.update(grads, self.opt_state, params)
        next_model = eqx.apply_updates(self.model, updates)
        return eqx.tree_at(
            lambda s: (s.model, s.opt_state),
            self,
            (next_model, next_opt_state),
        )


def make_train_step(loss_function: LossFn, *, gradient_accumulation_steps: int) -> TrainStepFn:
    if gradient_accumulation_steps != 1:
        raise NotImplementedError("Gradient accumulation > 1 is not yet supported in the benchmark loop")

    grad_fn = eqx.filter_value_and_grad(loss_function, has_aux=True)

    def train_step(state: State, batch: Any, *, key: Array) -> tuple[State, dict[str, Any]]:
        (loss, aux), grads = grad_fn(state.model, batch, key=key)
        del loss
        new_state = state.apply_gradients(grads)
        return new_state, aux

    return eqx.filter_jit(train_step)


__all__ = ["LossFn", "LossFunctionConfig", "State", "TrainStepFn", "make_train_step"]
