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
            if not isinstance(dataset, hf_datasets.Dataset):
                raise TypeError(
                    "Expected a `datasets.Dataset` instance for huggingface data"
                )
            return grain.MapDataset.source(dataset)
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
        if not isinstance(
            dataset, grain.IterDataset
        ):  # pragma: no cover - sanity guard
            dataset = dataset.to_iter_dataset()
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
        self, dataset: grain.MapDataset | grain.IterDataset
    ) -> grain.IterDataset:
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset()
        return dataset.batch(
            batch_size=self.batch_size, drop_remainder=self.drop_remainder
        )


@dc.dataclass
class BatchRampUpDataset(BaseDatasetTransform):
    """Adapt the batch size dynamically following schedule factors."""

    batch_size: int
    rampup_factors: dict[str, float]
    drop_remainder: bool = True

    def __call__(
        self, dataset: grain.MapDataset | grain.IterDataset
    ) -> grain.IterDataset:
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset()
        schedule = _build_batch_schedule(self.batch_size, self.rampup_factors)
        return _BatchRampUpIterDataset(
            dataset, schedule, drop_remainder=self.drop_remainder
        )


def _build_batch_schedule(
    batch_size: int, factors: dict[str, float]
) -> Callable[[int], int]:
    parsed = sorted((int(step), scale) for step, scale in factors.items())

    def schedule(step: int) -> int:
        scaled = batch_size
        for boundary, scale in parsed:
            if step >= boundary:
                scaled = max(1, int(round(batch_size * scale)))
            else:
                break
        return scaled

    return schedule


class _BatchRampUpIterDataset(grain.IterDataset):
    """IterDataset that adapts the batch size using a user-defined schedule."""

    def __init__(
        self,
        parent: grain.IterDataset,
        schedule: Callable[[int], int],
        *,
        drop_remainder: bool,
    ) -> None:
        super().__init__(parent)
        self._schedule = schedule
        self._drop_remainder = drop_remainder

    def __iter__(self):  # pragma: no cover - thin wrapper around parent iterator
        parent_iter = iter(self._parent)
        step = 0
        while True:
            batch_size = self._schedule(step)
            if batch_size <= 0:
                raise ValueError("Batch size schedule produced a non-positive value")
            elems = []
            try:
                for _ in range(batch_size):
                    elems.append(next(parent_iter))
            except StopIteration:
                if not elems or self._drop_remainder:
                    raise
            if not elems:
                break
            step += 1
            yield jtu.tree_map(lambda *xs: np.stack(xs), *elems)
