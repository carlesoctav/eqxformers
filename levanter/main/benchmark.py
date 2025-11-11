import draccus

from eqxformers.benchmarking_utils import BenchmarkConfig, setup_logging


@draccus.wrap()
def main(cfg: BenchmarkConfig) -> None:
    logger = setup_logging(cfg.log_file)
    logger.info("Launching benchmark CLI")
    loop = cfg.make()
    stats = loop.run()
    logger.info("Benchmark finished: %s", stats)


if __name__ == "__main__":  # pragma: no cover
    main()
