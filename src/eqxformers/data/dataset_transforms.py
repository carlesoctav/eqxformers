import dataclasses as dc
from collections.abc import Callable, Sequence
import typing as tp

import jax.tree_util as jtu
import numpy as np

import grain


class BaseDatasetTransform:
    """Marker base class for dataset-level transforms."""

    def __call__(
        self, dataset: grain.MapDataset | grain.IterDataset
    ) -> grain.MapDataset | grain.IterDataset:  # pragma: no cover - interface
        raise NotImplementedError


class _HuggingFaceIterableDatasetIterator(grain.DatasetIterator):
    """Iterator wrapper around a HuggingFace streaming dataset."""

    def __init__(self, dataset: tp.Any):
        super().__init__()
        self._iterator = iter(dataset)

    def __next__(self):
        return next(self._iterator)

    def get_state(self):
        raise RuntimeError("Checkpointing HuggingFace streaming datasets is not supported.")

    def set_state(self, state):  # pragma: no cover - streaming datasets are not checkpointable
        raise RuntimeError("Checkpointing HuggingFace streaming datasets is not supported.")


class HuggingFaceIterableDataset(grain.IterDataset):
    """Grain ``IterDataset`` backed by a HuggingFace ``IterableDataset``."""

    def __init__(self, dataset: tp.Any):
        super().__init__()
        self._dataset = dataset

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceIterableDatasetIterator(self._dataset)

    def __str__(self) -> str:
        return "HuggingFaceIterableDataset"

    def shard(self, *, num_shards: int, index: int) -> "HuggingFaceIterableDataset":
        if not hasattr(self._dataset, "shard"):
            raise TypeError("Underlying dataset does not support sharding.")
        return HuggingFaceIterableDataset(
            self._dataset.shard(num_shards=num_shards, index=index)
        )

    def shuffle(
        self,
        *,
        seed: int | None = None,
        buffer_size: int | None = None,
    ) -> "HuggingFaceIterableDataset":
        kwargs = {}
        if buffer_size is not None and buffer_size > 0:
            kwargs["buffer_size"] = buffer_size
        return HuggingFaceIterableDataset(self._dataset.shuffle(seed=seed, **kwargs))

    def repeat(self, num_epochs: int | None) -> "HuggingFaceIterableDataset":
        return HuggingFaceIterableDataset(self._dataset.repeat(num_epochs))


@dc.dataclass
class EnsureMapDataset(BaseDatasetTransform):
    """Wrap raw datasets into Grain map datasets."""

    dataset_type: str

    def __call__(self, dataset: tp.Any) -> grain.MapDataset:
        if isinstance(dataset, grain.MapDataset):
            return dataset
        if self.dataset_type == "huggingface":
            try:
                import datasets as hf_datasets  # pylint: disable=import-error
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "datasets package is required for huggingface datasets"
                ) from exc
            if isinstance(dataset, hf_datasets.Dataset):
                return grain.MapDataset.source(dataset)
            if isinstance(dataset, hf_datasets.IterableDataset):
                return HuggingFaceIterableDataset(dataset)
            raise TypeError(
                "Expected a HuggingFace Dataset or IterableDataset instance for huggingface data"
            )
        if self.dataset_type == "arrayrecord":
            from grain import sources

            if isinstance(dataset, sources.ArrayRecordDataSource):
                return grain.MapDataset.source(dataset)
            if isinstance(dataset, str):
                data_source = sources.ArrayRecordDataSource([dataset])
                return grain.MapDataset.source(data_source)
            if isinstance(dataset, Sequence):
                data_source = sources.ArrayRecordDataSource(list(dataset))
                return grain.MapDataset.source(data_source)
            raise TypeError("Unsupported input for arrayrecord dataset type")
        raise NotImplementedError(
            f"Dataset type {self.dataset_type!r} is not supported"
        )


@dc.dataclass
class ToIterDataset(BaseDatasetTransform):
    """Convert map dataset to iter dataset if required."""

    def __call__(
        self, dataset: grain.MapDataset | grain.IterDataset
    ) -> grain.IterDataset:
        if isinstance(dataset, grain.IterDataset):
            return dataset
        return dataset.to_iter_dataset()


@dc.dataclass
class ApplyFirstFitPacking(BaseDatasetTransform):
    """Apply Grain first-fit packing transformation."""

    length_struct: dict[str, int]
    num_packing_bins: int | None = None
    shuffle_bins: bool = True

    def __call__(
        self, dataset: grain.IterDataset | grain.MapDataset
    ) -> grain.IterDataset:
        bins = self.num_packing_bins or max(self.length_struct.values())
        packed = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct=self.length_struct,
            num_packing_bins=bins,
            shuffle_bins=self.shuffle_bins,
        )
        return packed

@dc.dataclass
class BatchDataset(BaseDatasetTransform):
    """Batch dataset with a fixed batch size."""

    batch_size: int
    drop_remainder: bool = True

    def __call__(
        self, dataset: grain.IterDataset
    ) -> grain.IterDataset:
        return dataset.batch(
            batch_size=self.batch_size, drop_remainder=self.drop_remainder
        )
