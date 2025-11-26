import grain
from datasets import load_dataset
import os
import multiprocessing as mp

from grain.experimental import DatasetOptions, WithOptionsIterDataset
from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterDataset


def close_mp_it(it):
    if it is None:
        return
    raw = getattr(it, "_raw_iterator", None)
    gen = getattr(it, "_iterator", None)
    # Close the generator wrapper to trigger __exit__ on raw.
    if gen is not None:
        try:
            gen.close()
        except Exception:
            pass
    # Public close on newer grains.
    if hasattr(it, "close"):
        try:
            it.close()
        except Exception:
            pass
    # Fallback: stop raw iterator/pool.
    if raw is not None:
        try:
            raw.stop_prefetch()
        except Exception:
            pass


def mp_alive(it) -> bool:
    """Best-effort check if mp_prefetch control thread is still alive."""
    raw = getattr(it, "_raw_iterator", None)
    if raw is None:
        return False
    thread = getattr(raw, "_reader_thread", None)
    return thread is not None and thread.is_alive()


if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train")
    ds = WithOptionsIterDataset(
        HuggingFaceSourceIterDataset(dataset),
        DatasetOptions(min_shm_size=1 << 60),
    )

    mp_ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=2))

    # First iterator: advance three items and capture state after two.
    mp_it1 = iter(mp_ds)
    first = next(mp_it1)
    second = next(mp_it1)
    state = mp_it1.get_state()
    third_expected = next(mp_it1)
    print(f"DEBUG: first mp-prefetch sample={first}")
    print(f"DEBUG: second mp-prefetch sample={second}")
    print(f"DEBUGPRINT[11]: temp_test_mp.py:17: state={state}")
    close_mp_it(mp_it1)

    # Resume from captured state.
    mp_it2 = iter(mp_ds)
    mp_it2.set_state(state)
    resumed = next(mp_it2)
    assert resumed == third_expected, "State restore did not resume at the correct element"
    print(f"DEBUG: resumed mp-prefetch sample={resumed}")
    print(f"DEBUG: mp iterator alive after resume? {mp_alive(mp_it2)}")
    close_mp_it(mp_it2)
    print(f"DEBUG: mp iterator alive after close? {mp_alive(mp_it2)}")
    # Ensure no stray children remain before shutdown.
    for child in mp.active_children():
        child.terminate()
        child.join()
    # Avoid interpreter finalization racing with background worker threads.
    os._exit(0)
