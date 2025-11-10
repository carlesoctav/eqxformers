from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import draccus
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jax import Array
from jax.sharding import Mesh, set_mesh
from tqdm.auto import tqdm

from eqxformers.config import ModelConfig
from eqxformers.data import DatasetConfig, HFMLMDatasetConfig
from eqxformers.loss import MaskedLanguageModelingLossConfig
from eqxformers.models.bert import BertConfig
from eqxformers.optim.adam import AdamConfig
from eqxformers.optimizer_utils import OptimizerConfig
from eqxformers.tracker_utils import Tracker, TrackerConfig, TrackioTrackerConfig
from eqxformers.training_utils import (
    LossFn,
    LossFunctionConfig,
    State,
    TrainStepFn,
    make_train_step,
)


LOGGER = logging.getLogger(__name__)


class TrainStepConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    @abc.abstractmethod
    def make(self, loss_function: LossFn, *, gradient_accumulation_steps: int) -> TrainStepFn:
        raise NotImplementedError


@dataclass
class BenchmarkConfig:
    data: DatasetConfig = field(default_factory=HFMLMDatasetConfig)
    num_train_steps: int = 400
    gradient_accumulation_steps: int = 1
    seed: int = 42
    mesh_shape: tuple[int, ...] = (4,)
    mesh_axis_names: tuple[str, ...] = ("dp",)
    model: ModelConfig = field(default_factory=lambda: BertConfig(task="mlm"))
    optimizer: OptimizerConfig = field(
        default_factory=lambda: AdamConfig(
            lr=5e-5,
            lr_schedule="cosine",
            warmup=0.1,
            lr_decay=0.9,
            cycle_length=1.0,
            weight_decay=0.01,
        )
    )
    tracker: TrackerConfig | None = field(default_factory=TrackioTrackerConfig)
    loss_function: LossFunctionConfig = field(default_factory=MaskedLanguageModelingLossConfig)
    train_step: TrainStepConfig | None = None
    hlo_path: Path | None = field(default_factory=lambda: Path("benchmark/train_hlo.txt"))


def benchmark_loop(
    state: State,
    train_step_fn: TrainStepFn,
    train_loader: Iterable[Any],
    *,
    num_steps: int,
    key: Array,
) -> tuple[State, dict[str, Any]]:
    step_idx = -1
    train_times: list[float] = []
    next_batch_times: list[float] = []
    compile_time: float | None = None

    iterator = iter(train_loader)
    progress = tqdm(total=num_steps + 1, desc="Benchmarking", disable=jax.process_index() != 0)
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                LOGGER.info("Data loader exhausted during benchmark loop")
                break
            batch_end = time.perf_counter()

            key, step_key = jr.split(key)

            step_idx += 1
            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
                state, aux = train_step_fn(state, batch, key=step_key)
            _ = aux
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
            step_end = time.monotonic()

            if step_idx == 0:
                compile_time = step_end - step_start
                LOGGER.info("Compilation (step 0) took %.4fs", compile_time)
            else:
                train_times.append(step_end - step_start)
                next_batch_times.append(batch_end - batch_start)

            progress.update(1)
    finally:
        progress.close()

    train_times_arr = np.asarray(train_times) if train_times else np.asarray([0.0])
    batch_times_arr = np.asarray(next_batch_times) if next_batch_times else np.asarray([0.0])
    stats = {
        "train_step_time_mean": float(train_times_arr.mean()),
        "train_step_time_std": float(train_times_arr.std()),
        "next_batch_time_mean": float(batch_times_arr.mean()),
        "compile_time": float(compile_time) if compile_time is not None else None,
    }
    LOGGER.info("Benchmark stats: %s", stats)
    return state, stats


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class BenchmarkLoop:
    config: BenchmarkConfig
    state: State
    train_step_fn: TrainStepFn = field(repr=False)
    data_loader: Iterable[Any] = field(repr=False)
    num_train_step: int = field(default=0)
    tracker: Tracker | None = field(default=None, repr=False)
    key: Array = field(default_factory=lambda: jr.PRNGKey(0), repr=False)
    mesh: Mesh = field(repr=False, default=None)
    _tracker_finished: bool = field(default=False, init=False, repr=False)

    @classmethod
    def make(cls, cfg: BenchmarkConfig) -> BenchmarkLoop:
        LOGGER.info("Initializing benchmark loop")
        key = jr.PRNGKey(cfg.seed)
        key, model_key = jr.split(key)

        mesh = jax.make_mesh(tuple(cfg.mesh_shape), tuple(cfg.mesh_axis_names))
        tracker = cfg.tracker.make() if cfg.tracker is not None else None

        loss_function = cfg.loss_function.make(cfg.data)
        if cfg.train_step is not None:
            raise NotImplementedError("Custom train steps are not supported yet; configure loss_function instead.")

        train_step_fn = make_train_step(
            loss_function=loss_function,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )

        train_loader = cfg.data.make(mesh=mesh, seed=cfg.seed)

        with set_mesh(mesh):
            model = cfg.model.make(key=model_key)
            grad_tx = cfg.optimizer.make(cfg.num_train_steps)
            state = State(model, grad_tx)

            start_time = time.monotonic()
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
            diff = time.monotonic() - start_time
            LOGGER.info("Model and optimizer initialization took %.4fs", diff)

            iterator = iter(train_loader)
            first_batch = next(iterator)
            key, lower_key = jr.split(key)
            start = time.monotonic()
            lowered = train_step_fn.lower(state, first_batch, key=lower_key)
            compiled = lowered.compile()
            compile_time = time.monotonic() - start
            LOGGER.info("Train step compilation took %.4fs", compile_time)

        if cfg.hlo_path is not None:
            hlo_path = Path(cfg.hlo_path)
            hlo_path.parent.mkdir(parents=True, exist_ok=True)
            hlo_path.write_text(lowered.as_text())
            LOGGER.info("Compilation artifact written to %s", hlo_path)

        return cls(
            config=cfg,
            state=state,
            train_step_fn=compiled,
            data_loader=train_loader,
            num_train_step=cfg.num_train_steps,
            tracker=tracker,
            key=key,
            mesh=mesh,
        )

    def run(self) -> dict[str, Any]:
        LOGGER.info("Starting benchmark for %d steps", self.num_train_step)
        with set_mesh(self.mesh):
            try:
                self.state, stats = benchmark_loop(
                    self.state,
                    self.train_step_fn,
                    self.data_loader,
                    num_steps=self.num_train_step,
                    key=self.key,
                )
                if self.tracker is not None:
                    self.tracker.log("benchmark_stats", stats)
                return stats
            finally:
                if self.tracker is not None and not self._tracker_finished:
                    self.tracker.finish()
                    self._tracker_finished = True


__all__ = [
    "BenchmarkConfig",
    "BenchmarkLoop",
    "DatasetConfig",
    "HFMLMDatasetConfig",
    "LossFunctionConfig",
    "MaskedLanguageModelingLossConfig",
    "TrainStepConfig",
    "benchmark_loop",
    "make_train_step",
    "setup_logging",
]
