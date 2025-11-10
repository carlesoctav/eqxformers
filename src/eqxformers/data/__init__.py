from ..data_utils import DataConfig, DataLoaderConfig, HFSource
from .dataset_config import HFMLMDatasetConfig
from .masked_language_modeling import MLMProcessingConfig, masked_language_modeling_transforms
from .next_token_prediction import next_token_prediction_transforms
from .training import IterDatasetWithInputSpec, make_dataloader

# Backwards compatibility alias.
DatasetConfig = DataConfig


__all__ = [
    "DataConfig",
    "DatasetConfig",
    "DataLoaderConfig",
    "HFSource",
    "HFMLMDatasetConfig",
    "MLMProcessingConfig",
    "masked_language_modeling_transforms",
    "next_token_prediction_transforms",
    "make_dataloader",
    "IterDatasetWithInputSpec",
]
