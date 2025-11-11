from dataclasses import dataclass, field
from typing import Iterable

from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer

from eqxformers.data_utils import DataConfig, DataLoaderConfig, HFSource

from .masked_language_modeling import MLMProcessingConfig, masked_language_modeling_transforms
from .training import make_dataloader


@DataConfig.register_subclass("hf_mlm")
@dataclass
class HFMLMDatasetConfig(DataConfig):
    source: HFSource = field(default_factory=HFSource)
    processing: MLMProcessingConfig = field(default_factory=MLMProcessingConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    def make(self, *, mesh: Mesh, seed: int | None = None) -> Iterable[object]:
        tokenizer = AutoTokenizer.from_pretrained(self.processing.tokenizer_name)

        dataset = load_dataset(
            path=self.source.path,
            name=self.source.name,
            data_files=self.source.data_files,
            data_dir=self.source.data_dir,
            split=self.source.split,
            streaming=self.source.streaming,
        )

        operations, batch_class = masked_language_modeling_transforms(
            dataset_type="huggingface",
            column=self.processing.column_name,
            tokenizer=tokenizer,
            max_length=self.processing.max_length,
            mlm_probability=self.processing.mlm_probability,
            mask_replace_prob=self.processing.mask_replace_prob,
            random_replace_prob=self.processing.random_replace_prob,
            pad_to_multiple_of=self.processing.pad_to_multiple_of,
            packing=self.processing.packing,
            packing_bins=self.processing.packing_bins,
        )

        data_seed = self.dataloader.seed if seed is None else seed

        return make_dataloader(
            datasets=[dataset],
            operations=operations,
            global_batch_size=self.dataloader.batch_size,
            pspec=self.dataloader.partition_spec_obj(),
            mesh=mesh,
            num_epochs=self.dataloader.num_epochs,
            dataset_weights=self.dataloader.dataset_weights,
            dataloading_host_index=self.dataloader.dataloading_host_index,
            dataloading_host_count=self.dataloader.dataloading_host_count,
            is_not_sharded=self.dataloader.is_not_sharded,
            read_num_threads=self.dataloader.read_num_threads,
            read_prefetch_buffer_size=self.dataloader.read_prefetch_buffer_size,
            shuffle=self.dataloader.shuffle,
            seed=data_seed,
            worker_count=self.dataloader.worker_count,
            worker_buffer_size=self.dataloader.worker_buffer_size,
            drop_remainder=self.dataloader.drop_remainder,
            batch_class=batch_class,
        )


__all__ = ["HFMLMDatasetConfig"]
