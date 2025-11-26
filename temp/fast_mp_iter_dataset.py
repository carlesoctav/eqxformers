"""
Fast, opportunistic multiprocessing iterator for HuggingFace streaming datasets.

This intentionally bypasses Grain's round-robin reader and checkpoints to
maximize throughput for simple map-style work (e.g., counting tokens).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import queues
from queue import Empty
from typing import Callable, Any, Iterable

import cloudpickle

import grain


def _shard_dataset(ds: Any, num_workers: int, worker_id: int):
    """Shard a HuggingFace iterable/map dataset for this worker."""
    if hasattr(ds, "shard"):
        try:
            return ds.shard(num_shards=num_workers, index=worker_id, contiguous=True)
        except Exception:
            pass
    # Fallback for map datasets that support len/select.
    try:
        total = len(ds)
        start = worker_id * total // num_workers
        end = (worker_id + 1) * total // num_workers
        return ds.select(range(start, end))
    except Exception:
        # As a last resort, return the dataset unchanged.
        return ds


def _worker_loop(worker_id: int, num_workers: int, sequential_slice: bool,
                 dataset_bytes: bytes | None,
                 ds_factory: Callable[[], Any] | None,
                 out_q: queues.Queue, stop_event: mp.Event):
    try:
        if ds_factory is not None:
            ds = ds_factory()
        else:
            ds = cloudpickle.loads(dataset_bytes) if dataset_bytes is not None else None
        if ds is None:
            raise RuntimeError("Worker did not receive a dataset or factory.")
        # Try in-place slice for Grain/HF streaming datasets.
        if hasattr(ds, "set_slice"):
            try:
                ds = ds  # shallow
                ds.set_slice(slice(worker_id, None, num_workers), sequential_slice=sequential_slice)
            except Exception:
                ds = _shard_dataset(ds, num_workers=num_workers, worker_id=worker_id)
        else:
            ds = _shard_dataset(ds, num_workers=num_workers, worker_id=worker_id)

        for row in ds:
            if stop_event.is_set():
                break
            try:
                out_q.put((worker_id, row), block=True)
            except Exception as exc:  # noqa: BLE001
                out_q.put((worker_id, {"error": str(exc)}), block=True)
                break
    except Exception as exc:  # noqa: BLE001
        out_q.put((worker_id, {"error": str(exc)}), block=True)
    finally:
        # Signal completion for this worker.
        out_q.put((worker_id, None), block=True)


class FastMultiprocessingIterDataset(grain.IterDataset):
    """Minimal IterDataset that uses an opportunistic shared queue for speed."""

    def __init__(
        self,
        dataset: Any | None = None,
        num_workers: int,
        *,
        sequential_slice: bool = True,
        queue_mul: int = 1000,
        ds_factory: Callable[[], Any] | None = None,
    ):
        """
        Args:
            dataset: HuggingFace Iterable/Map dataset or a Grain IterDataset. Must be picklable. Optional if ds_factory is provided.
            num_workers: number of worker processes.
            sequential_slice: if True use contiguous slices (like mp_prefetch sequential_slice),
                else strided slices.
            queue_mul: queue size multiplier (queue size = num_workers * queue_mul).
            ds_factory: optional zero-arg callable returning a fresh dataset instance. Use when
                the dataset object is not picklable.
        """
        super().__init__()
        if dataset is None and ds_factory is None:
            raise ValueError("Provide either `dataset` or `ds_factory`.")
        self._dataset = dataset
        self._ds_factory = ds_factory
        self._num_workers = max(1, num_workers)
        self._queue_mul = max(1, queue_mul)
        self._sequential_slice = bool(sequential_slice)

    def __iter__(self) -> grain.DatasetIterator:
        return _FastMPIterator(
            dataset=self._dataset,
            ds_factory=self._ds_factory,
            num_workers=self._num_workers,
            queue_mul=self._queue_mul,
            sequential_slice=self._sequential_slice,
        )

    def __str__(self) -> str:
        return f"FastMultiprocessingIterDataset(num_workers={self._num_workers})"


class _FastMPIterator(grain.DatasetIterator):
    """Iterator that pulls from a shared queue (opportunistic, no round-robin)."""

    def __init__(
        self,
        dataset: Any | None,
        num_workers: int,
        queue_mul: int,
        sequential_slice: bool,
        ds_factory: Callable[[], Any] | None,
    ):
        super().__init__()
        self._dataset = dataset
        self._ds_factory = ds_factory
        self._num_workers = num_workers
        self._queue_mul = queue_mul
        self._sequential_slice = sequential_slice
        self._ctx = mp.get_context("spawn")
        self._queue = None
        self._stop_event = None
        self._processes = None
        self._finished = 0

    def __iter__(self):
        if self._queue is None:
            self._start_workers()
        return self

    def __next__(self):
        while self._finished < self._num_workers:
            try:
                worker_id, payload = self._queue.get(timeout=1.0)
            except Empty:
                continue
            if payload is None:
                self._finished += 1
                continue
            if isinstance(payload, dict) and "error" in payload:
                # Stop on first error.
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
        dataset_bytes = None
        if self._dataset is not None and self._ds_factory is None:
            dataset_bytes = cloudpickle.dumps(self._dataset)
        self._processes = [
            self._ctx.Process(
                target=_worker_loop,
                args=(
                    i,
                    self._num_workers,
                    self._sequential_slice,
                    dataset_bytes,
                    self._ds_factory,
                    self._queue,
                    self._stop_event,
                ),
                daemon=True,
            )
            for i in range(self._num_workers)
        ]
        for p in self._processes:
            p.start()
