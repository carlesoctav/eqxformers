import grain
from datasets import load_dataset

from grain.experimental import DatasetOptions, WithOptionsIterDataset, ThreadPrefetchIterDataset
from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterableDataset

def close_thread_it(it):
    if it is None:
        return
    if hasattr(it, "close"):
        try:
            it.close()
        except Exception:
            pass

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
    ds = WithOptionsIterDataset(
        HuggingFaceSourceIterableDataset(dataset),
        DatasetOptions(min_shm_size=1 << 60),
    )

    tp_ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=4)
    tp_it = iter(tp_ds)
    first = next(tp_it)
    second = next(tp_it)
    print(f"DEBUG: thread-prefetch first={first}")
    print(f"DEBUG: thread-prefetch second={second}")
    close_thread_it(tp_it)
