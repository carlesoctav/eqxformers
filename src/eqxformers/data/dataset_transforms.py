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
