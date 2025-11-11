import abc
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import draccus
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jax import Array
import equinox as eqx
from jax.sharding import Mesh, set_mesh
from tqdm.auto import tqdm

from eqxformers.config import ModelConfig
from eqxformers.data import DatasetConfig, HFMLMDatasetConfig
from eqxformers.loss import MaskedLanguageModelingLossConfig
from eqxformers.models.bert import BertConfig
from eqxformers.optim.adam import AdamConfig
from eqxformers.optimizer_utils import OptimizerConfig
from eqxformers.logging_utils import setup_logger
from eqxformers.tracker_utils import Tracker, TrackerConfig, TrackioTrackerConfig
from eqxformers.training_utils import (
    LossFn,
    LossFunctionConfig,
    State,
    TrainStepFn,
    make_train_step,
)
from eqxformers.metrics_utils import SufficientMetric


logger = logging.getLogger("distributed_logger")


def _device_scalar_to_float(value: Any) -> float:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        return float(value.reshape(()))
    return float(value)


def _extract_loss(aux: Any) -> float | None:
    if not isinstance(aux, dict):
        return None
    loss_entry = aux.get("loss")
    if loss_entry is None:
        return None

    try:
        if isinstance(loss_entry, (tuple, list)) and len(loss_entry) == 2:
            total, denom = loss_entry
            total_f = _device_scalar_to_float(total)
            denom_f = _device_scalar_to_float(denom)
            if denom_f != 0:
                return total_f / denom_f
            return total_f
        return _device_scalar_to_float(loss_entry)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Failed to extract loss from %s", loss_entry, exc_info=True)
        return None


def _format_loss_history(history: list[tuple[int, float]], max_points: int = 20) -> str:
    if not history:
        return ""
    if len(history) <= max_points:
        return ", ".join(f"{step}:{loss:.4f}" for step, loss in history)
    head_count = max_points // 2
    tail_count = max_points - head_count
    head = ", ".join(f"{step}:{loss:.4f}" for step, loss in history[:head_count])
    tail = ", ".join(f"{step}:{loss:.4f}" for step, loss in history[-tail_count:])
    return f"{head}, ..., {tail}"


def _log_loss_history(history: list[tuple[int, float]]) -> None:
    if not history:
        return
    losses = [loss for _, loss in history]
    mean_loss = float(np.mean(losses))
    min_loss = float(np.min(losses))
    max_loss = float(np.max(losses))
    last_step, last_loss = history[-1]
    logger.info(
        "Loss stats (per-token average over %d steps): min=%.6f | max=%.6f | mean=%.6f | last(step=%d)=%.6f",
        len(history),
        min_loss,
        max_loss,
        mean_loss,
        last_step,
        last_loss,
    )
    logger.info("Loss trajectory sample: %s", _format_loss_history(history))


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
    log_file: Path = field(default_factory=lambda: Path("distributed.log"))

    def make(self) -> "BenchmarkLoop":
        logger.info("Initializing benchmark loop")
        key = jr.PRNGKey(self.seed)
        key, model_key = jr.split(key)

        mesh = jax.make_mesh(tuple(self.mesh_shape), tuple(self.mesh_axis_names))
        run_config = asdict(self)
        tracker = self.tracker.make(run_config) if self.tracker is not None else None

        loss_function = self.loss_function.make()
        if self.train_step is not None:
            raise NotImplementedError("Custom train steps are not supported yet; configure loss_function instead.")

        train_step_fn = make_train_step(
            loss_function=loss_function,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        train_loader = self.data.make(mesh=mesh, seed=self.seed)

        def unbox_params(tree):
            def f(leaf):
                if eqx.is_array(leaf):
                    return jax.lax.with_sharding_constraint(leaf, jax.P())
                return leaf
            return jtu.tree_map(f, tree)

        with set_mesh(mesh):
            model = self.model.make(key=model_key)
            model = unbox_params(model)
            grad_tx = self.optimizer.make(self.num_train_steps)
            state = State(model, grad_tx)

            start_time = time.monotonic()
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
            diff = time.monotonic() - start_time
            logger.info("Model and optimizer initialization took %.4fs", diff)

            iterator = iter(train_loader)
            first_batch = next(iterator)
            key, lower_key = jr.split(key)
            start = time.monotonic()
            lowered = train_step_fn.lower(state, first_batch, key=lower_key)
            compiled = lowered.compile()
            compile_time = time.monotonic() - start
            logger.info("Train step compilation took %.4fs", compile_time)

        if self.hlo_path is not None:
            hlo_path = Path(self.hlo_path)
            hlo_path.parent.mkdir(parents=True, exist_ok=True)
            hlo_path.write_text(lowered.as_text())
            logger.info("Compilation artifact written to %s", hlo_path)

        return BenchmarkLoop(
            config=self,
            state=state,
            train_step_fn=compiled,
            data_loader=train_loader,
            num_train_step=self.num_train_steps,
            tracker=tracker,
            key=key,
            mesh=mesh,
            log_file=self.log_file,
        )


def benchmark_loop(
    state: State,
    train_step_fn: TrainStepFn,
    train_loader: Iterable[Any],
    tracker: Tracker | None = None,
    *,
    num_steps: int,
    key: Array,
) -> tuple[State, dict[str, Any]]:
    step_idx = 0
    train_times: list[float] = []
    next_batch_times: list[float] = []
    compile_time: float | None = None
    loss_history: list[tuple[int, float]] = []

    iterator = iter(train_loader)
    progress = tqdm(total=num_steps, desc="Benchmarking", disable=jax.process_index() != 0)
    metrics = SufficientMetric(name = "train")
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                logger.info("Data loader exhausted during benchmark loop")
                break
            batch_end = time.perf_counter()

            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
                state, aux = train_step_fn(state, batch, key=key)
            metrics += aux
            logs= {**metrics.step_metrics(), **metrics.per_N_metrics(step_idx)}
            tracker.log(logs, step = step_idx)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
            step_end = time.monotonic()

            if step_idx == 0:
                compile_time = step_end - step_start
                logger.info("Compilation (step 0) took %.4fs", compile_time)
            else:
                train_times.append(step_end - step_start)
                next_batch_times.append(batch_end - batch_start)

            progress.update(1)
            step_idx += 1
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
    logger.info("Benchmark stats: %s", stats)
    return state, stats


def setup_logging(log_file: Path | str = "distributed.log") -> logging.LoggerAdapter:
    return setup_logger(log_file)


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
    log_file: Path = field(default_factory=lambda: Path("distributed.log"))

    def run(self) -> dict[str, Any]:
        logger.info("Starting benchmark for %d steps", self.num_train_step)
        with set_mesh(self.mesh):
            try:
                self.state, stats = benchmark_loop(
                    self.state,
                    self.train_step_fn,
                    self.data_loader,
                    self.tracker,
                    num_steps=self.num_train_step,
                    key=self.key,
                )
                if self.tracker is not None:
                    self.tracker.log_hparams(stats)
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
