import dataclasses as dc
import typing as tp

import numpy as np

import grain
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase
import jax.tree_util as jtu

from .dataset_transforms import (
    ApplyFirstFitPacking,
    BaseDatasetTransform,
    EnsureMapDataset,
    ToIterDataset,
)


@jtu.register_dataclass
@dc.dataclass
class LLMBatch:
    """Batch container for next-token prediction setups."""

    inputs: Array
    targets: Array
    inputs_position: Array
    targets_position: Array
    inputs_segmentation: Array
    targets_segmentation: Array
    attention_mask: Array | None = None


def _shift_right(
    array: np.ndarray,
    *,
    axis: int = -1,
    padding_value: int,
) -> np.ndarray:
    """Shift elements one position to the right along an axis."""

    axis = axis % array.ndim
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (1, 0)
    padded = np.pad(array, pad_width, mode="constant", constant_values=padding_value)

    slicer = [slice(None)] * array.ndim
    slicer[axis] = slice(0, -1)
    return padded[tuple(slicer)]


@dc.dataclass
class TokenizeText(grain_transforms.Map):
    """Tokenize raw text coming from a column."""

    column: str
    tokenizer: PreTrainedTokenizerBase
    packing: bool
    max_length: int | None = None

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        text = features[self.column]
        encoded = self.tokenizer(
            text,
            truncation=self.max_length is not None,
            padding="max_length" if not self.packing else None,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokens = np.asarray(encoded["input_ids"], dtype=np.int32)
        if tokens.ndim > 1:
            tokens = tokens.squeeze(axis=0)
        return {"input_ids": tokens}


@dc.dataclass
class EnsureTokenIds(grain_transforms.Map):
    """Ensure numeric token ids are provided in the expected field."""

    column: str

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        token_ids = np.asarray(features[self.column], dtype=np.int32)
        return {"input_ids": token_ids}


@dc.dataclass
class MakeInputsTargets(grain_transforms.Map):
    """Duplicate input ids into inputs/targets dictionary."""

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        tokens = np.asarray(features["input_ids"], dtype=np.int32)
        out = {"inputs": tokens, "targets": tokens}
        return out


@dc.dataclass
class ShiftTokensForNTD(grain_transforms.Map):
    """Shift inputs to create teacher-forced targets."""

    bos_token_id: int
    pad_token_id: int
    axis: int = -1

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        targets = np.asarray(features["targets"], dtype=np.int32)
        inputs = np.asarray(features["inputs"], dtype=np.int32)

        if inputs.shape != targets.shape:
            raise ValueError(
                "inputs and targets must have identical shapes before shifting"
            )

        features["inputs"] = _shift_right(
            inputs, axis=self.axis, padding_value=self.bos_token_id
        )
        features["targets"] = targets

        mask = features.get("attention_mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=np.int32)
        else:
            mask = (targets != self.pad_token_id).astype(np.int32)

        return features


@dc.dataclass
class ReformatPackedKeys(grain_transforms.Map):
    """Rename packed dataset keys to the canonical naming."""

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        renames = {
            "inputs_segment_ids": "inputs_segmentation",
            "targets_segment_ids": "targets_segmentation",
            "inputs_positions": "inputs_position",
            "targets_positions": "targets_position",
        }
        for old, new in renames.items():
            if old in features:
                features[new] = features.pop(old)
        return features


def next_token_prediction_transforms(
    *,
    dataset_type: str,
    column: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    is_tokenized: bool,
    packing: bool = False,
    packing_bins: int | None = None,
) -> tuple[
    list[grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform],
    type[LLMBatch],
]:
    """Build the list of transforms required for next-token prediction."""

    operations = [EnsureMapDataset(dataset_type=dataset_type)]

    if is_tokenized:
        operations.append(EnsureTokenIds(column=column))
    else:
        operations.append(
            TokenizeText(
                column=column,
                tokenizer=tokenizer,
                max_length=max_length,
                packing=packing,
            )
        )

    operations.append(MakeInputsTargets())

    operations.append(ToIterDataset())

    if packing:
        length_struct = {"inputs": max_length, "targets": max_length}
        operations.append(
            ApplyFirstFitPacking(
                length_struct=length_struct, num_packing_bins=packing_bins
            )
        )
        operations.append(ReformatPackedKeys())

    return operations, LLMBatch
