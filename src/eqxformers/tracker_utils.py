import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import draccus
import jax
import numpy as np
import trackio
from .utils import rank_zero


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

class Tracker(abc.ABC):
    @abc.abstractmethod
    def log(self, logs: dict[str, Any], *, step: int | None = None) -> None:
        raise NotImplementedError


    @abc.abstractmethod
    def log_hparams():
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self) -> None:
        raise NotImplementedError


@dataclass
class TrackerConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    @abc.abstractmethod
    def make(self, config) -> Tracker:
        raise NotImplementedError


@dataclass
class TrackioTracker(Tracker):
    run: "TrackioRun" 

    @rank_zero
    def log(self, logs: dict[str, Any], *, step: int | None = None) -> None:
        self.run.log(logs, step)

    @rank_zero
    def log_hparams(self, hparams) -> None:
        self.run.config.update(hparams) 

    @rank_zero
    def finish(self) -> None:
        self.run.finish()


@TrackerConfig.register_subclass("trackio")
@dataclass
class TrackioTrackerConfig(TrackerConfig):
    project: str = "benchmark"
    name: str | None = None
    space_id: str | None = None
    space_storage: Any = None
    dataset_id: str | None = None
    resume: str = "never"
    settings: Any = None
    private: bool | None = None
    embed: bool = True

    def make(self, config) -> TrackioTracker:
        run = None
        if jax.process_index() == 0:
            run = trackio.init(
                project=self.project,
                name=self.name,
                space_id=self.space_id,
                space_storage=self.space_storage,
                dataset_id=self.dataset_id,
                config=config,
                resume=self.resume,
                settings=self.settings,
                private=self.private,
                embed=self.embed,
            )

        return TrackioTracker(run = run)

__all__ = [
    "Tracker",
    "TrackerConfig",
    "TrackioTracker",
    "TrackioTrackerConfig",
]
