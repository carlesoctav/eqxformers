import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import draccus
import jax
import numpy as np
import trackio


def _convert_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, jax.Array):
        if value.ndim == 0:
            return value.item()
        return np.array(value).tolist()
    return value


def _is_primary_process() -> bool:
    try:
        return jax.process_index() == 0
    except Exception:  # pragma: no cover - jax not initialized
        return True


class Tracker(abc.ABC):
    @abc.abstractmethod
    def log(self, tag: str, data: Mapping[str, Any], *, step: int | None = None) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        """Optional cleanup hook."""


@dataclass
class TrackerConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    @abc.abstractmethod
    def make(self) -> Tracker:
        raise NotImplementedError


@dataclass
class TrackioTracker(Tracker):
    project: str
    name: str | None = None
    space_id: str | None = None
    space_storage: Any = None
    dataset_id: str | None = None
    config: Mapping[str, Any] | None = None
    resume: str = "never"
    settings: Any = None
    private: bool | None = None
    embed: bool = True

    _run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if _is_primary_process():
            self._run = trackio.init(
                project=self.project,
                name=self.name,
                space_id=self.space_id,
                space_storage=self.space_storage,
                dataset_id=self.dataset_id,
                config=self.config,
                resume=self.resume,
                settings=self.settings,
                private=self.private,
                embed=self.embed,
            )

    def log(self, tag: str, data: Mapping[str, Any], *, step: int | None = None) -> None:
        if not _is_primary_process() or self._run is None:
            return
        payload = {tag: _convert_value(data)}
        kwargs: dict[str, Any] = {}
        if step is not None:
            kwargs["step"] = step
        self._run.log(payload, **kwargs)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


@TrackerConfig.register_subclass("trackio")
@dataclass
class TrackioTrackerConfig(TrackerConfig):
    project: str = "benchmark"
    name: str | None = None
    space_id: str | None = None
    space_storage: Any = None
    dataset_id: str | None = None
    config: Mapping[str, Any] | None = None
    resume: str = "never"
    settings: Any = None
    private: bool | None = None
    embed: bool = True

    def make(self) -> TrackioTracker:
        return TrackioTracker(
            project=self.project,
            name=self.name,
            space_id=self.space_id,
            space_storage=self.space_storage,
            dataset_id=self.dataset_id,
            config=self.config,
            resume=self.resume,
            settings=self.settings,
            private=self.private,
            embed=self.embed,
        )


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - import side effect
        raise RuntimeError("wandb is not installed; please pip install wandb to use the wandb tracker") from exc
    return wandb


@dataclass
class WandbTracker(Tracker):
    project: str = "benchmark"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    config: Mapping[str, Any] | None = None
    resume: str | bool | None = "allow"
    mode: str | None = None
    id: str | None = None
    settings: Any = None

    _run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _is_primary_process():
            return
        wandb = _import_wandb()
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            group=self.group,
            tags=self.tags,
            config=_convert_value(self.config) if self.config is not None else None,
            resume=self.resume,
            mode=self.mode,
            id=self.id,
            settings=self.settings,
        )

    def log(self, tag: str, data: Mapping[str, Any], *, step: int | None = None) -> None:
        if self._run is None or not _is_primary_process():
            return
        flattened = {f"{tag}/{k}": _convert_value(v) for k, v in data.items()}
        self._run.log(flattened, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


@TrackerConfig.register_subclass("wandb")
@dataclass
class WandbTrackerConfig(TrackerConfig):
    project: str = "benchmark"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    config: Mapping[str, Any] | None = None
    resume: str | bool | None = "allow"
    mode: str | None = None
    id: str | None = None
    settings: Any = None

    def make(self) -> WandbTracker:
        return WandbTracker(
            project=self.project,
            entity=self.entity,
            name=self.name,
            group=self.group,
            tags=self.tags,
            config=self.config,
            resume=self.resume,
            mode=self.mode,
            id=self.id,
            settings=self.settings,
        )


__all__ = [
    "Tracker",
    "TrackerConfig",
    "TrackioTracker",
    "TrackioTrackerConfig",
    "WandbTracker",
    "WandbTrackerConfig",
]
