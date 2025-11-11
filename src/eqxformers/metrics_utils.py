import logging
import typing as tp

import jax.tree_util as jtu


LOGGER = logging.getLogger(__name__)


def _is_tuple_leaf(x: tp.Any) -> bool:
    return isinstance(x, tuple)


class SufficientMetric:
    """Utility to accumulate streaming metrics and emit step / per-N aggregates."""

    def __init__(self, name: str, log_every_n_step: int | None = None) -> None:
        self.name = name
        self.log_every_n_step = log_every_n_step or 0
        self._buffer_tree: tp.Any = None
        self._last_added: tp.Any = None
        self.per_N_metrics_buffer: dict[int, dict[str, float]] = {}
        self._count = 0
        self._warned_log_every = False

    def __iadd__(self, other: tp.Any) -> "SufficientMetric":
        return self.add(other)

    def reduce_fn(self, tree: tp.Any, count: int) -> tp.Any:
        def _reduce_leaf(x: tp.Any) -> float:
            if isinstance(x, tuple) and len(x) == 2:
                value, normaliser = x
                normaliser = normaliser or count
                return float(value / normaliser) if normaliser else float(value)
            return float(x)

        return jtu.tree_map(_reduce_leaf, tree, is_leaf=_is_tuple_leaf)

    def add(self, other: tp.Any) -> "SufficientMetric":
        if other is None:
            return self
        self._last_added = other
        self._count += 1

        if self._buffer_tree is None:
            self._buffer_tree = other
        else:
            self._buffer_tree = jtu.tree_map(lambda a, b: a + b, self._buffer_tree, other)
        return self

    def step_metrics(self) -> dict[str, float]:
        if self._last_added is None:
            return {}
        reduced = self.reduce_fn(self._last_added, count=1)
        return {f"{self.name}/{k}": v for k, v in reduced.items()}

    def per_N_metrics(self, step: int, *, skip_check: bool = False) -> dict[str, float]:
        cached = self.per_N_metrics_buffer.get(step, None)
        if cached is not None:
            return cached

        if self._buffer_tree is None:
            return {}

        if not skip_check:
            if self.log_every_n_step <= 0:
                if not self._warned_log_every:
                    self._warned_log_every = True
                    LOGGER.warning(
                        "Metric %s has log_every_n_step=%s; skipping per-N logging.",
                        self.name,
                        self.log_every_n_step,
                    )
                return {}
            if self._count == 0 or step % self.log_every_n_step != 0:
                return {}

        reduced = self.reduce_fn(self._buffer_tree, count=self._count)
        self.per_N_metrics_buffer[step] = {**reduced, "count": self._count}
        self._buffer_tree = None
        self._count = 0
        return {f"{self.name}_per_N/{k}": v for k, v in reduced.items()}

    def summary(self) -> dict[str, tp.Any]:
        return {
            "name": self.name,
            "count": self._count,
            "per_N_cache": dict(self.per_N_metrics_buffer),
        }


__all__ = ["SufficientMetric"]
