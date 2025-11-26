import argparse
import os
import time

import pyarrow
import pyarrow.dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import grain
from grain import transforms as grain_transforms
from grain._src.python.dataset.transformations.prefetch import ThreadPrefetchIterDataset

# Add parent directory to path to import eqxformers
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eqxformers.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from eqxformers.data.masked_language_modeling import MLMBatch  # noqa: F401 - used for type clarity
from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterDataset
from eqxformers.data.dataset_transforms import BaseDatasetTransform


def apply_ops(ds, ops, seed: int | None = None):
    """Apply the MLM transform pipeline to a HuggingFace streaming dataset."""
    for op in ops:
        if isinstance(op, BaseDatasetTransform):
            ds = op(ds)
        elif isinstance(op, grain_transforms.RandomMap):
            ds = ds.random_map(op, seed=seed)
        elif isinstance(op, grain_transforms.Map):
            ds = ds.map(op)
        else:
            raise TypeError(f"Unsupported op type: {type(op)}")
    return ds


def main(args):
    resolved_dir = os.path.expanduser(args.data_dir)
    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=1,
            range_size_limit=128 << 20,
        ),
    )

    ds = load_dataset(
        "parquet",
        data_dir=resolved_dir,
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )["train"]
    ds = HuggingFaceSourceIterDataset(ds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=True, local_files_only=args.local_files_only
    )

    ops, _batch_cls = masked_language_modeling_transforms(
        dataset_type="huggingface",
        column=args.column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        mask_replace_prob=args.mask_replace_prob,
        random_replace_prob=args.random_replace_prob,
        pad_to_multiple_of=None,
        packing=False,
        packing_bins=None,
    )

    ds = apply_ops(ds, ops, seed=args.seed)
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size = 4256 * 4)

    start = time.time()
    last_report = start
    total_rows = 0
    try:
        for row in ds:
            total_rows += 1
            if args.report_every and total_rows % args.report_every == 0:
                now = time.time()
                elapsed = now - start
                window_rate = args.report_every / (now - last_report) if now > last_report else 0.0
                avg_rate = total_rows / elapsed if elapsed else 0.0
                print(
                    f"Progress: {total_rows:,} rows | avg {avg_rate:,.2f} rows/s | "
                    f"recent {window_rate:,.2f} rows/s | elapsed {elapsed:,.1f}s"
                )
                last_report = now
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        elapsed = time.time() - start
        if total_rows:
            avg_rate = total_rows / elapsed if elapsed else 0.0
            print(
                f"Finished iterating {total_rows:,} rows in {elapsed:,.1f}s "
                f"({avg_rate:,.2f} rows/s avg)"
            )
        else:
            print("No rows were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming MLM preprocessing test (single process).")
    parser.add_argument("--data-dir", default="~/data/100BT", help="Path to the dataset directory.")
    parser.add_argument("--column", default="text", help="Text column to tokenize.")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased", help="Tokenizer to use.")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="Masking probability.")
    parser.add_argument("--mask-replace-prob", type=float, default=0.8, help="Fraction of masks replaced with [MASK].")
    parser.add_argument("--random-replace-prob", type=float, default=0.1, help="Fraction of masks replaced with random token.")
    parser.add_argument("--report-every", type=int, default=1_000, help="Progress log interval.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download tokenizer weights.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for random_map transforms.")
    args = parser.parse_args()
    main(args)
