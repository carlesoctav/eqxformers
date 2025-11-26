"""
Fast opportunistic multiprocessing Iterator.
"""
from __future__ import annotations

import multiprocessing as mp
import typing
from multiprocessing import queues
from queue import Empty

import cloudpickle
import grain


@typing.runtime_checkable
class SupportsInPlaceSlicing(typing.Protocol):
  """Datasets that support mutation by setting the processed data slice."""

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    ...


def _set_slice_iter_dataset(ds: grain.IterDataset, sl: slice, sequential_slice: bool) -> bool:
    if isinstance(ds, SupportsInPlaceSlicing):
        ds.set_slice(sl, sequential_slice=sequential_slice)
        return 

    if not ds.parents:
        raise ValueError(f"Cannot slice `IterDataset` source. {type(ds)}") 

    for parent in ds.parents:
        if  isinstance(parent, grain.MapDataset):
            _set_slice_map_dataset(parent, sl, sequential_slice) 
        else:
            _set_slice_iter_dataset(parent, sl, sequential_slice) 
    return 

def _set_slice_map_dataset(ds: grain.MapDataset, sl: slice, sequential_slice: bool) -> bool:
    if isinstance(ds, SupportsInPlaceSlicing):
        ds.set_slice(sl, sequential_slice=sequential_slice)
        return 

    if not ds.parents:
        raise ValueError(f"Cannot slice `MapDataset` source. {type(ds)}")


    for parent in ds.parents:
        if isinstance(parent, grain.MapDataset): 
            _set_slice_map_dataset(parent, sl, sequential_slice) 
        else:
            _set_slice_iter_dataset(parent, sl, sequential_slice) 

    return 



def _ensure_picklable(ds: grain.IterDataset) -> None:
    try:
        cloudpickle.dumps(ds)
    except Exception as exc:
        raise RuntimeError(
            "Dataset is not picklable; mp_prefetch would also fail. "
            "Check transforms/state and ensure callables are cloudpickle-able."
        ) from exc


def _worker_loop(
    worker_id: int,
    num_workers: int,
    sequential_slice: bool,
    dataset_bytes: bytes | None,
    out_q: queues.Queue,
    stop_event: mp.Event,
):
    try:
        ds = cloudpickle.loads(dataset_bytes) if dataset_bytes is not None else None
        if ds is None:
            raise RuntimeError("Worker did not receive a dataset.")

        sl = slice(worker_id, None, num_workers)
        _set_slice_iter_dataset(ds, sl, sequential_slice)

        for row in ds:
            if stop_event.is_set():
                break
            try:
                out_q.put((worker_id, row), block=True)
            except Exception as exc:  
                out_q.put((worker_id, {"error": str(exc)}), block=True)
                break
    except Exception as exc: 
        out_q.put((worker_id, {"error": str(exc)}), block=True)
    finally:
        out_q.put((worker_id, None), block=True)


class FastMultiprocessingIterDataset(grain.IterDataset):
    """IterDataset that rebuilds the dataset per worker and reads opportunistically."""

    def __init__(
        self,
        dataset: grain.IterDataset, 
        num_workers: int = 1,
        *,
        sequential_slice: bool = True,
        worker_buffer_size: int = 1000,
    ):
        if dataset is None:
            raise ValueError("Provide a picklable `dataset`.")
        super().__init__()
        self._dataset = dataset
        self._num_workers = num_workers 
        self._worker_buffer_size = worker_buffer_size
        self._sequential_slice = bool(sequential_slice)

    def __iter__(self) -> grain.DatasetIterator:
        if self._num_workers == 0:
            return self._dataset.__iter__()
        return _FastMPIterator(
            dataset=self._dataset,
            num_workers=self._num_workers,
            worker_buffer_size=self._worker_buffer_size,
            sequential_slice=self._sequential_slice,
        )

    def __str__(self) -> str:
        return f"FastMultiprocessingIterDataset(num_workers={self._num_workers})"


class _FastMPIterator(grain.DatasetIterator):
    """Iterator that pulls from a shared queue (opportunistic, no round-robin)."""

    def __init__(
        self,
        dataset: grain.IterDataset, 
        num_workers: int,
        worker_buffer_size: int,
        sequential_slice: bool,
    ):
        super().__init__()
        self._dataset = dataset
        self._num_workers = num_workers
        self._queue_mul = worker_buffer_size
        self._sequential_slice = sequential_slice
        self._ctx = mp.get_context("spawn")
        self._queue = None
        self._stop_event = None
        self._processes = None
        self._finished = 0
        self._dataset_bytes = None

    def __iter__(self):
        if self._queue is None:
            self._start_workers()
        if self._queue is None:
            raise RuntimeError("Failed to start workers: queue not initialized.")
        return self

    def __next__(self):
        if self._queue is None:
            self._start_workers()
        if self._queue is None:
            raise RuntimeError("Failed to start workers: queue not initialized.")

        while self._finished < self._num_workers:
            try:
                worker_id, payload = self._queue.get(timeout=1.0)
            except Empty:
                if self._processes and all(not p.is_alive() for p in self._processes):
                    raise RuntimeError("All workers exited without producing data.")
                continue
            if payload is None:
                self._finished += 1
                continue
            if isinstance(payload, dict) and "error" in payload:
                self._stop_event.set()
                raise RuntimeError(f"Worker {worker_id} error: {payload['error']}")
            return payload
        raise StopIteration

    def close(self):
        if self._stop_event:
            self._stop_event.set()
        if self._processes:
            for p in self._processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        self._queue = None
        self._stop_event = None
        self._processes = None

    def __del__(self):
        self.close()

    def get_state(self):
        raise NotImplementedError("FastMultiprocessingIterDataset is not checkpointable.")

    def set_state(self, state):
        raise NotImplementedError("FastMultiprocessingIterDataset is not checkpointable.")

    def _start_workers(self):
        self._queue = self._ctx.Queue(maxsize=self._num_workers * self._queue_mul)
        self._stop_event = self._ctx.Event()
        try:
            _ensure_picklable(self._dataset)
            self._dataset_bytes = cloudpickle.dumps(self._dataset)
        except Exception as exc: 
            raise RuntimeError(
                "Failed to pickle dataset for multiprocessing. Ensure map ops are cloudpickle-able "
                "and avoid embedding non-picklable state (e.g., lazily load tokenizers)."
            ) from exc
        self._processes = [
            self._ctx.Process(
                target=_worker_loop,
                args=(
                    i,
                    self._num_workers,
                    self._sequential_slice,
                    self._dataset_bytes,
                    self._queue,
                    self._stop_event,
                ),
                daemon=True,
            )
            for i in range(self._num_workers)
        ]
        for p in self._processes:
            p.start()

def make_mp(
    dataset: grain.IterDataset,
    num_workers: int,
    sequential_slice: bool = True,
    worker_buffer_size: int = 1000,
) -> grain.IterDataset:
    return FastMultiprocessingIterDataset(
        dataset=dataset,
        num_workers=num_workers,
        sequential_slice=sequential_slice,
        worker_buffer_size=worker_buffer_size,
    )
