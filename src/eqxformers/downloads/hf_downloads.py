import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus
import fsspec
from datasets import load_dataset
from huggingface_hub import HfFileSystem


logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for downloading datasets from the Hugging Face Hub."""

    path: str = "owner/dataset"
    revision: str | None = None
    hf_urls_glob: list[str] = field(default_factory=list)

    output_path: str = "./data"
    hf_repo_type_prefix: str = "datasets"
    num_proc: int = 8
    chunk_size_bytes: int = 16 * 1024 * 1024
    max_retries: int = 10
    action: Literal["stream", "save_to_disk"] = "stream"

    name: str | None = None
    split: str | None = None
    streaming: bool = False
    max_shard_size: str = "500MB"

    hf_token: str | None = None
    dump_config: bool = False
    dump_config_path: str | None = None


def ensure_fsspec_path_writable(output_path: str) -> None:
    """Check if the fsspec path is writable by trying to create and delete a temporary file."""
    fs, _ = fsspec.core.url_to_fs(output_path)
    try:
        test_path = os.path.join(output_path, "test_write_access")
        with fs.open(test_path, "w") as f:
            f.write("test")
        fs.rm(test_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No write access to fsspec path: {output_path} ({exc})") from exc


def stream_file_to_fsspec(cfg: DownloadConfig, file_path: str, fsspec_file_path: str) -> None:
    """Stream a file from HfFileSystem to another fsspec path with retry logic.

    We re-create per-thread filesystem clients to avoid sharing stateful objects that
    may not be thread-safe across a ThreadPool.
    """
    hf_fs = HfFileSystem(token=cfg.hf_token or os.environ.get("HF_TOKEN", False))
    target_fs, _ = fsspec.core.url_to_fs(cfg.output_path)

    for attempt in range(cfg.max_retries):
        try:
            with hf_fs.open(file_path, "rb") as src_file:
                target_fs.mkdirs(os.path.dirname(fsspec_file_path), exist_ok=True)
                with target_fs.open(fsspec_file_path, "wb") as dest_file:
                    while chunk := src_file.read(cfg.chunk_size_bytes):
                        dest_file.write(chunk)
            logger.info("Streamed %s successfully to %s", file_path, fsspec_file_path)
            return
        except Exception as exc:  # noqa: BLE001
            wait_time = (2**attempt) + random.uniform(0, 5)
            logger.warning("Attempt %s failed for %s: %s; retrying in %.1fs", attempt + 1, file_path, exc, wait_time)
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to download {file_path} after {cfg.max_retries} attempts")


def write_provenance_json(output_path: str, cfg: DownloadConfig, files: list[str]) -> None:
    """Persist a small provenance file alongside the downloaded shards."""
    fs, _ = fsspec.core.url_to_fs(output_path)
    fs.mkdirs(output_path, exist_ok=True)
    provenance_path = os.path.join(output_path, "provenance.json")
    payload = {"dataset": cfg.path, "version": cfg.revision, "links": sorted(files)}
    with fs.open(provenance_path, "w") as fp:
        json.dump(payload, fp)
    logger.info("Wrote provenance to %s", provenance_path)


def list_repo_files(cfg: DownloadConfig, hf_fs: HfFileSystem) -> list[str]:
    """List files in the HF repo, applying any glob filters."""
    hf_repo_prefix = cfg.hf_repo_type_prefix.strip("/")
    repo_path = f"{hf_repo_prefix}/{cfg.path}" if hf_repo_prefix else cfg.path

    if not cfg.hf_urls_glob:
        files = hf_fs.find(repo_path, revision=cfg.revision)
    else:
        files = []
        for hf_url_glob in cfg.hf_urls_glob:
            pattern = os.path.join(repo_path, hf_url_glob)
            files.extend(hf_fs.glob(pattern, revision=cfg.revision))

    return files


def download_hf(cfg: DownloadConfig) -> None:
    """Download raw files from a Hugging Face dataset repo to an fsspec path."""
    logging.basicConfig(level=logging.INFO)
    ensure_fsspec_path_writable(cfg.output_path)

    hf_fs = HfFileSystem(token=cfg.hf_token or os.environ.get("HF_TOKEN", False))
    files = list_repo_files(cfg, hf_fs)
    if not files:
        raise ValueError(f"No files found for dataset `{cfg.path}`. Used glob patterns: {cfg.hf_urls_glob}")

    tasks = []
    for file in files:
        fsspec_file_path = os.path.join(cfg.output_path, file.split("/", 3)[-1])
        tasks.append((cfg, file, fsspec_file_path))

    logger.info("Total number of files to process: %s", len(tasks))

    with ThreadPoolExecutor(max_workers=cfg.num_proc) as executor:
        futures = [executor.submit(stream_file_to_fsspec, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()

    write_provenance_json(cfg.output_path, cfg, files)
    logger.info("Streamed all files and wrote provenance JSON; check %s.", cfg.output_path)


def save_to_disk(cfg: DownloadConfig) -> str:
    """Materialize a dataset locally using `datasets.save_to_disk` (Arrow format)."""
    target_path = cfg.output_path

    if target_path is None:
        raise ValueError("Provide `save_to_disk_path` or `gcs_output_path` for save_to_disk.")
    if cfg.streaming:
        raise ValueError("save_to_disk requires `streaming=False` to materialize the dataset.")

    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
        revision=cfg.revision,
        num_proc=cfg.num_proc,
        streaming=False,
    )
    dataset.save_to_disk(target_path, max_shard_size=cfg.max_shard_size, num_proc=cfg.num_proc)
    logger.info("Saved dataset `%s` (revision `%s`) to %s", cfg.path, cfg.revision, target_path)
    return target_path


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    if cfg.dump_config:
        template = DownloadConfig()
        if cfg.dump_config_path:
            with open(cfg.dump_config_path, "w", encoding="utf-8") as f:
                draccus.dump(template, f)
            print(f"Wrote template config to {cfg.dump_config_path}")
        else:
            print(draccus.dump(template))
        return

    if cfg.action == "save_to_disk":
        save_to_disk(cfg)
        return
    download_hf(cfg)


if __name__ == "__main__":
    main()
