"""
Fast opportunistic multiprocessing Iterator (Pool + async startup).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import typing
from multiprocessing import queues
from queue import Empty
import cloudpickle
import grain
import traceback

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
        if isinstance(parent, grain.MapDataset):
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

_WORKER_OUT_Q: typing.Optional[queues.Queue] = None
_WORKER_STOP_EVENT: typing.Optional[mp.Event] = None


def _pool_worker_initializer(out_q, stop_event):
    global _WORKER_OUT_Q, _WORKER_STOP_EVENT
    _WORKER_OUT_Q = out_q
    _WORKER_STOP_EVENT = stop_event


def _pool_worker_main(worker_id: int, num_workers: int, sequential_slice: bool, dataset_bytes: bytes):
    """
    This function runs in the worker process inside the Pool.
    It deserializes the dataset, sets its slice, iterates and pushes rows into the shared queue.
    It must be robust and always send final sentinel (worker_id, None).
    """
    global _WORKER_OUT_Q, _WORKER_STOP_EVENT
    try:
        if _WORKER_OUT_Q is None:
            raise RuntimeError("Worker output queue not initialized in worker process.")

        ds = cloudpickle.loads(dataset_bytes) if dataset_bytes is not None else None
        if ds is None:
            raise RuntimeError("Worker did not receive a dataset.")

        sl = slice(worker_id, None, num_workers)
        _set_slice_iter_dataset(ds, sl, sequential_slice)

        for row in ds:
            if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
                break
            try:
                _WORKER_OUT_Q.put((worker_id, row), block=True)
            except Exception as exc:
                try:
                    _WORKER_OUT_Q.put((worker_id, {"error": f"Queue put failed: {exc}"}), block=True)
                except Exception:
                    pass
                break
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            _WORKER_OUT_Q.put((worker_id, {"error": f"{exc}\n{tb}"}), block=True)
        except Exception:
            pass
    finally:
        try:
            _WORKER_OUT_Q.put((worker_id, None), block=True)
        except Exception:
            pass
    return None


class FastMultiprocessingIterDataset(grain.IterDataset):
    """IterDataset that rebuilds the dataset per worker and reads opportunistically (Pool + async startup)."""

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
        self._num_workers = int(num_workers)
        self._worker_buffer_size = int(worker_buffer_size)
        self._sequential_slice = bool(sequential_slice)

    def __iter__(self) -> grain.DatasetIterator:
        if self._num_workers <= 0:
            return self._dataset.__iter__()
        return _FastMPPoolIterator(
            dataset=self._dataset,
            num_workers=self._num_workers,
            worker_buffer_size=self._worker_buffer_size,
            sequential_slice=self._sequential_slice,
        )

    def __str__(self) -> str:
        return f"FastMultiprocessingIterDataset(num_workers={self._num_workers})"


class _FastMPPoolIterator(grain.DatasetIterator):
    """Iterator that uses a background thread to start an mp.Pool and reads opportunistically from a shared queue."""

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

        self._mp_queue: typing.Optional[queues.Queue] = None
        self._mp_stop_event: typing.Optional[mp.Event] = None

        self._pool: typing.Optional[mp.pool.Pool] = None
        self._pool_thread: typing.Optional[threading.Thread] = None
        self._pool_ready = threading.Event()

        self._async_results: list = []

        self._finished = 0
        self._dataset_bytes: typing.Optional[bytes] = None

        self._lock = threading.Lock()

        self._closed = False

    def __iter__(self):
        with self._lock:
            if self._pool_thread is None:
                self._start_pool_in_background()
        return self

    def __next__(self):
        with self._lock:
            if self._pool_thread is None:
                self._start_pool_in_background()

        if self._mp_queue is None:
            raise RuntimeError("Failed to initialize multiprocessing queue.")

        while self._finished < self._num_workers:
            try:
                worker_id, payload = self._mp_queue.get(timeout=0.05)
            except Empty:
                if self._pool_ready.is_set() and self._pool is not None:
                    if all(r.ready() for r in self._async_results) and self._mp_queue.empty():
                        if self._finished >= self._num_workers:
                            break
                continue

            if payload is None:
                self._finished += 1
                continue
            if isinstance(payload, dict) and "error" in payload:
                if self._mp_stop_event is not None:
                    self._mp_stop_event.set()
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(f"Worker {worker_id} error: {payload['error']}")
            return payload
        raise StopIteration

    def close(self):
        """
        Request shutdown. This sets the stop event for workers, terminates the pool if needed,
        and waits briefly for processes to exit.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._mp_stop_event is not None:
                try:
                    self._mp_stop_event.set()
                except Exception:
                    pass

            if self._pool is not None:
                try:
                    # terminate is harsh but ensures workers stop quickly on close()
                    self._pool.terminate()
                except Exception:
                    pass
                try:
                    self._pool.join()
                except Exception:
                    pass
                self._pool = None

            if self._pool_thread is not None:
                try:
                    self._pool_thread.join(timeout=2.0)
                except Exception:
                    pass
                self._pool_thread = None

            self._mp_queue = None
            self._mp_stop_event = None
            self._async_results = []
            self._pool_ready.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_state(self):
        raise NotImplementedError("FastMultiprocessingIterDataset is not checkpointable.")

    def set_state(self, state):
        raise NotImplementedError("FastMultiprocessingIterDataset is not checkpointable.")

    def _start_pool_in_background(self):
        """Start a background thread that will create the mp.Pool and submit worker tasks."""
        self._mp_queue = self._ctx.Queue(maxsize=self._num_workers * self._queue_mul)
        self._mp_stop_event = self._ctx.Event()

        try:
            _ensure_picklable(self._dataset)
            self._dataset_bytes = cloudpickle.dumps(self._dataset)
        except Exception as exc:
            self._mp_queue = None
            self._mp_stop_event = None
            raise RuntimeError(
                "Failed to pickle dataset for multiprocessing. Ensure map ops are cloudpickle-able "
                "and avoid embedding non-picklable state (e.g., lazily load tokenizers)."
            ) from exc

        self._pool_thread = threading.Thread(target=self._pool_thread_main, daemon=True)
        self._pool_thread.start()

    def _pool_thread_main(self):
        try:
            pool = self._ctx.Pool(
                processes=self._num_workers,
                initializer=_pool_worker_initializer,
                initargs=(self._mp_queue, self._mp_stop_event),
            )

            async_results = []
            for i in range(self._num_workers):
                ar = pool.apply_async(
                    _pool_worker_main,
                    args=(i, self._num_workers, self._sequential_slice, self._dataset_bytes),
                )
                async_results.append(ar)

            with self._lock:
                self._pool = pool
                self._async_results = async_results
                self._pool_ready.set()

            pool.close()
            pool.join()

        except Exception as exc:
            try:
                if self._mp_queue is not None:
                    tb = traceback.format_exc()
                    self._mp_queue.put((-1, {"error": f"{exc}\n{tb}"}), block=False)
            except Exception:
                pass
        finally:
            self._pool_ready.set()

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
