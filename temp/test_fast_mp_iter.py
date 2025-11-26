import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from eqxformers.data.fast_mp import FastMultiprocessingIterDataset
from eqxformers.data.huggingface_datasets import HuggingFaceSourceIterDataset


@dataclass
class Tokenize:
    """Picklable map op that keeps a tokenizer instance (like MLM transforms)."""

    tokenizer_path: str
    column: str = "messages"

    def __post_init__(self):
        self.tokenizer = None

    def map(self, data):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return len(self.tokenizer.apply_chat_template(data[self.column]))

    def __call__(self, data):
        # HF/Grain map expects a callable; delegate to map().
        return self.map(data)


def main(args):
    dataset = load_dataset(
        "allenai/Dolci-Think-SFT",
        streaming=True,
    )["train"]

    dataset = HuggingFaceSourceIterDataset(dataset)
    tokenize_fn = Tokenize(tokenizer_path=args.tokenizer, column=args.column)
    dataset = dataset.map(tokenize_fn)
    dataset = dataset.filter(lambda x: x < 2000) 

    fast_ds = FastMultiprocessingIterDataset(
        dataset=dataset,
        num_workers=args.num_workers,
        sequential_slice=not args.strided,
        worker_buffer_size=args.queue_mul,
    )

    start = time.time()
    last = start
    total = 0
    counter = []
    try:
        for count in tqdm(fast_ds):
            counter.append(count)
            total += 1
            if args.report_every and total % args.report_every == 0:
                now = time.time()
                elapsed = now - start
                window = args.report_every / (now - last) if now > last else 0.0
                avg = total / elapsed if elapsed else 0.0
                print(
                    f"Progress: {total:,} rows | avg {avg:,.2f} rows/s | "
                    f"recent {window:,.2f} rows/s | elapsed {elapsed:,.1f}s"
                )
                last = now

        series = pd.Series(counter)
        desc = series.describe()
        print(f"DEBUGPRINT[20]: test_fast_mp_iter.py:73: desc={desc}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        elapsed = time.time() - start
        if total:
            print(f"Finished {total:,} rows in {elapsed:,.1f}s ({total/elapsed:,.2f} rows/s avg)")
        else:
            print("No rows were processed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Test FastMultiprocessingIterDataset throughput.")
    parser.add_argument("--cache-dir", default="temp/hf_cache", help="Writable HuggingFace cache dir.")
    parser.add_argument("--tokenizer", default="google/gemma-3-1b-it", help="HF tokenizer id to use.")
    parser.add_argument("--column", default="messages", help="Column containing chat messages.")
    parser.add_argument("--num-workers", type=int, default=96, help="Number of worker processes.")
    parser.add_argument("--queue-mul", type=int, default=1000, help="Queue size multiplier.")
    parser.add_argument("--report-every", type=int, default=10000, help="Progress interval.")
    parser.add_argument("--strided", action="store_true", help="Use strided slicing instead of contiguous.")
    args = parser.parse_args()
    main(args)
