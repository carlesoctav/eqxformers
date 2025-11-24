import grain
from datasets import load_dataset
from grain.experimental import ThreadPrefetchIterDataset

from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterableDataset




if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", streaming = True, split = "train")
    ds = HuggingFaceSourceIterableDataset(dataset)

    mp_ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=2))
    mp_it = iter(mp_ds)
    resumed_it = None
    try:
        first = next(mp_it)
        second = next(mp_it)
        state = mp_it.get_state()
        third_expected = next(mp_it)
        mp_it._raw_iterator.stop_prefetch()
        print(f"DEBUG: first mp-prefetch sample={first}")
        print(f"DEBUG: second mp-prefetch sample={second}")
        print(f"DEBUGPRINT[11]: temp_test_mp.py:17: state={state}")
        resumed_it = iter(mp_ds)
        resumed_it.set_state(state)
        resumed = next(resumed_it)
        assert resumed == third_expected, "State restore did not resume at the correct element"
        print(f"DEBUG: resumed mp-prefetch sample={resumed}")
        print(resumed == third_expected)
    finally:
        for it in (mp_it, resumed_it):
            if it is None:
                continue
            # Prefer public close; fall back to stopping the raw iterator if present.
            if hasattr(it, "close"):
                it.close()
            elif hasattr(it, "_raw_iterator") and it._raw_iterator is not None:
                it._raw_iterator.stop_prefetch()

    # ThreadPrefetchIterDataset path removed for this smoke test.
