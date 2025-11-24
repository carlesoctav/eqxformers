import grain
import warnings
from datasets import IterableDataset


class AlwaysFirst:
    pass


class _HuggingFaceSourceIterator(grain.DatasetIterator, AlwaysFirst):

    def __init__(
        self,
        dataset: IterableDataset
    ):
        super().__init__()
        self._dataset = dataset
        self._iterator = iter(self._dataset)

    def __next__(self):
        return next(self._iterator)

    def get_state(self):
        return self._dataset.state_dict()

    def set_state(self, state):
        self._dataset.load_state_dict(state)
        self._iterator = iter(self._dataset)


class HuggingFaceSourceIterableDataset(grain.IterDataset, AlwaysFirst):

    def __init__(
        self,
        source: IterableDataset,
    ):
        super().__init__()
        self._source = source

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceSourceIterator(self._source)

    def __str__(self) -> str:
        return "HuggingFaceIterableDataset"

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
    ):
        return self._source.shard(num_shards, index, contiguous)

    def set_slice(
        self,
        sl: slice,
        sequential_slice: bool = True
    ) -> None:

        if sl.step is None or sl.step <= 0:
            raise ValueError("slice.step (num_workers) must be a positive integer.")
        worker_index = 0 if sl.start is None else sl.start
        contiguous = bool(sequential_slice)

        if self._source.num_shards < sl.step:
            warnings.warn("The number of shards in the HuggingFace dataset is smaller than the number of workers. Some workers will not receive any data.")

        self._source = self._source.shard(
            num_shards=sl.step,
            index=worker_index,
            contiguous=contiguous,
        )

    def shuffle(
        self,
        seed: int | None = None,
        buffer_size: int | None = 1000,
    ) -> "HuggingFaceSourceIterableDataset":
        return HuggingFaceSourceIterableDataset(self._source.shuffle(seed=seed, buffer_size = buffer_size))
