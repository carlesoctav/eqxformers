import abc
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import draccus
from jax.sharding import Mesh, PartitionSpec


Batch = Any


def _default_partition_spec_axes() -> tuple[str | tuple[str, ...] | None, ...]:
    return ("dp",)


@dataclass
class HFSource:
    """Configuration describing a Hugging Face dataset source."""

    path: str = "carlesoctav/skripsi_UI_membership_30K"
    name: str | None = None
    data_files: str | Sequence[str] | Mapping[str, Any] | None = None
    data_dir: str | None = None
    split: str = "train"
    streaming: bool = False


@dataclass
class DataLoaderConfig:
    """Configuration for Grain dataloader construction."""

    batch_size: int = 64
    partition_spec: tuple[str | tuple[str, ...] | None, ...] = field(default_factory=_default_partition_spec_axes)
    num_epochs: int | None = None
    dataset_weights: Sequence[float] | None = None
    dataloading_host_index: int | None = None
    dataloading_host_count: int | None = None
    is_not_sharded: bool = True
    read_num_threads: int = 0
    read_prefetch_buffer_size: int = 0
    shuffle: bool = True
    seed: int = 0
    worker_count: int = 0
    worker_buffer_size: int = 0
    drop_remainder: bool = True

    def partition_spec_obj(self) -> PartitionSpec:
        return PartitionSpec(*self.partition_spec)


class DataConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
):
    dataloader: DataLoaderConfig

    @abc.abstractmethod
    def make(self, *, mesh: Mesh, seed: int) -> Iterable[Batch]:
        """Build the dataset iterator tied to the provided mesh."""
        raise NotImplementedError


__all__ = ["DataConfig", "DataLoaderConfig", "HFSource"]
