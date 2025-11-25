from datasets import load_dataset, load_from_disk
import argparse
import os
import pyarrow
import pyarrow.dataset
import time


def main(data_dir: str, report_every: int) -> None:
    resolved_dir = os.path.expanduser(data_dir)

    ds = load_from_disk("~/data/fineweb")
    print(f"Loaded dataset iterator from {data_dir}")

    start = time.time()
    last_report = start
    total_rows = 0

    try:
        for row in ds:
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
        elapsed = time.time() - start
        if total_rows:
            avg_rate = total_rows / elapsed if elapsed else 0.0
            print(f"Finished iterating {total_rows:,} rows in {elapsed:,.1f}s ({avg_rate:,.2f} rows/s avg)")
        else:
            print("No rows were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate a streaming dataset and measure throughput.")
    parser.add_argument("--data-dir", default="~/read/100BT", help="Path to the dataset directory (supports ~ expansion).")
    parser.add_argument(
        "--report-every",
        type=int,
        default=10_000,
        help="Print a progress update every N rows (set to 0 to disable periodic logs).",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        report_every=args.report_every,
    )
