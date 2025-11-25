from datasets import load_dataset
import argparse
import os
import sys
import pyarrow
import pyarrow.dataset
import time
import multiprocessing as mp
import grain
from grain.experimental import DatasetOptions, WithOptionsIterDataset

# Add parent directory to path to import eqxformers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterableDataset


def close_mp_it(it):
    """Helper to cleanly close mp_prefetch iterator."""
    if it is None:
        return
    raw = getattr(it, "_raw_iterator", None)
    gen = getattr(it, "_iterator", None)
    if gen is not None:
        try:
            gen.close()
        except Exception:
            pass
    if hasattr(it, "close"):
        try:
            it.close()
        except Exception:
            pass
    if raw is not None:
        try:
            raw.stop_prefetch()
        except Exception:
            pass


def main(data_dir: str, report_every: int, num_workers: int, buffer_size: int) -> None:
    print(f"Starting multiprocessing data loading with {num_workers} workers")
    
    resolved_dir = os.path.expanduser(data_dir)
    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=1,
            range_size_limit=128 << 20
        ),
    )

    ds = load_dataset(
        "parquet",
        data_dir=resolved_dir,
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )["train"]
    
    # Wrap in HuggingFaceSourceIterableDataset
    hf_dataset = HuggingFaceSourceIterableDataset(ds)
    # Force large min_shm_size so small arrays are copied, not put in shared memory.
    ds_with_options = WithOptionsIterDataset(
        hf_dataset,
        DatasetOptions(min_shm_size=1 << 30),
    )

    mp_options = grain.MultiprocessingOptions(
        num_workers=num_workers,
        per_worker_buffer_size=buffer_size,
    )
    mp_ds = ds_with_options.mp_prefetch(
        mp_options,
        sequential_slice=True,  # use contiguous sharding; strided slices are much slower on HF streaming
    )
    
    start = time.time()
    last_report = start
    total_rows = 0
    
    mp_it = None
    try:
        mp_it = iter(mp_ds)
        for row in mp_it:
            total_rows += 1

            if report_every and total_rows % report_every == 0:
                now = time.time()
                elapsed = now - start
                window_rate = report_every / (now - last_report) if now > last_report else 0.0
                avg_rate = total_rows / elapsed if elapsed else 0.0
                print(
                    f"Progress: {total_rows:,} rows | avg {avg_rate:,.2f} rows/s | "
                    f"recent {window_rate:,.2f} rows/s | elapsed {elapsed:,.1f}s"
                )
                last_report = now
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        close_mp_it(mp_it)
        elapsed = time.time() - start
        if total_rows:
            avg_rate = total_rows / elapsed if elapsed else 0.0
            print(f"Finished iterating {total_rows:,} rows in {elapsed:,.1f}s ({avg_rate:,.2f} rows/s avg)")
        else:
            print("No rows were processed.")
        
        # Clean up any remaining child processes
        for child in mp.active_children():
            child.terminate()
            child.join()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Iterate a streaming dataset with multiprocessing and measure throughput.")
    parser.add_argument("--data-dir", default="~/data/100BT", help="Path to the dataset directory (supports ~ expansion).")
    parser.add_argument(
        "--report-every",
        type=int,
        default=10_000,
        help="Print a progress update every N rows (set to 0 to disable periodic logs).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Buffer size per worker.",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        report_every=args.report_every,
        num_workers=args.num_workers,
        buffer_size=args.buffer_size,
    )
    
    # Avoid interpreter finalization racing with background worker threads
    os._exit(0)
