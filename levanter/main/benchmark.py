from __future__ import annotations

import logging

import draccus

from eqxformers.benchmarking_utils import BenchmarkConfig, BenchmarkLoop, setup_logging


@draccus.wrap()
def main(cfg: BenchmarkConfig) -> None:
    setup_logging()
    logging.getLogger(__name__).info("Launching benchmark CLI")
    loop = BenchmarkLoop.make(cfg)
    stats = loop.run()
    logging.getLogger(__name__).info("Benchmark finished: %s", stats)


if __name__ == "__main__":  # pragma: no cover
    main()
