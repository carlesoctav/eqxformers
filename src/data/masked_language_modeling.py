import dataclasses as dc
from dataclasses import dataclass

import grain
import jax.tree_util as jtu
import numpy as np
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from .dataset_transforms import (
    ApplyFirstFitPacking,
    BaseDatasetTransform,
    EnsureMapDataset,
    ToIterDataset,
)


@dataclass
class DataTransformsForMaskedLMGivenText(grain.transforms.RandomMap):
    tokenizer: PreTrainedTokenizerBase
    columns: str
    max_length: int | None = None
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int | None = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            )

    def random_map(
        self,
        texts: dict[str, str],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if self.columns not in texts:
            raise ValueError(f"Column {self.columns} not found in the input data.")

        tokenized = self.tokenizer(
            texts[self.columns],
            return_tensors="np",
            max_length=self.max_length,
            padding="max_length" if self.max_length is not None else None,
            truncation=self.max_length is not None,
        )

        # Tokenizers return shape (1, seq_len) for a single string; squeeze it.
        if tokenized["input_ids"].ndim == 2 and tokenized["input_ids"].shape[0] == 1:
            for k in list(tokenized.keys()):
                tokenized[k] = np.squeeze(tokenized[k], axis=0)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            tokenized["input_ids"].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = np.asarray(special_tokens_mask, dtype=bool)

        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int32)

        input_ids, labels = self.mask_tokens(
            input_ids,
            len(self.tokenizer),
            self.tokenizer.mask_token_id,
            special_tokens_mask=special_tokens_mask,
            rng=rng,
        )

        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int32)
        else:
            attention_mask = np.asarray(attention_mask, dtype=np.int32)

        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int32)
        else:
            token_type_ids = np.asarray(token_type_ids, dtype=np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": np.asarray(labels, dtype=np.int32),
        }

    def mask_tokens(
        self,
        input_ids: np.ndarray,
        vocab_size: int,
        mask_token_id: int,
        special_tokens_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.mlm_probability or self.mlm_probability <= 0:
            labels = np.full_like(input_ids, fill_value=-100)
            return input_ids, labels

        masked_draw = rng.binomial(1, self.mlm_probability, size=input_ids.shape)
        masked_indices = masked_draw & ~special_tokens_mask

        mask_replace_draw = rng.binomial(
            1, self.mask_replace_prob, size=input_ids.shape
        )
        replace_with_mask_token_indices = masked_indices & mask_replace_draw

        denom = max(1e-12, 1.0 - self.mask_replace_prob)
        random_replace_prob = self.random_replace_prob / denom

        random_replace_prob = float(min(1.0, max(0.0, random_replace_prob)))
        random_replace_draw = rng.binomial(1, random_replace_prob, size=input_ids.shape)

        replace_with_random_indices = (
            masked_indices & (~replace_with_mask_token_indices) & random_replace_draw
        )

        random_token = rng.integers(
            low=0,
            high=vocab_size,
            size=input_ids.shape,
            dtype=input_ids.dtype,
        )

        output_ids = np.where(replace_with_mask_token_indices, mask_token_id, input_ids)
        output_ids = np.where(replace_with_random_indices, random_token, output_ids)

        labels = np.where(
            masked_indices,
            input_ids,
            np.asarray(-100, dtype=input_ids.dtype),
        )

        return output_ids, labels


@jtu.register_dataclass
@dc.dataclass
class MLMBatch:
    """Batch container for masked language modeling."""

    input_ids: Array
    attention_mask: Array
    token_type_ids: Array
    labels: Array
    segment_ids: Array | None = None
    position_ids: Array | None = None


@dc.dataclass
class ReformatPackedForMLM(grain_transforms.Map):
    """Rename packed dataset keys to the expected MLM naming."""

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        def _pop_first(*names: str) -> Array | None:
            for name in names:
                if name in features:
                    return features.pop(name)
            return None

        if "segment_ids" not in features:
            seg = _pop_first("input_ids_segmentation", "input_ids_segment_ids")
            if seg is not None:
                features["segment_ids"] = seg

        if "position_ids" not in features:
            pos = _pop_first("input_ids_position", "input_ids_positions")
            if pos is not None:
                features["position_ids"] = pos

        return features


def masked_language_modeling_transforms(
    dataset_type: str,
    column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    *,
    mlm_probability: float = 0.15,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
    pad_to_multiple_of: int | None = None,
    packing: bool = False,
    packing_bins: int | None = None,
) -> tuple[
    list[grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform],
    type[MLMBatch],
]:
    """Build transforms for creating MLM datasets."""

    random_transform = DataTransformsForMaskedLMGivenText(
        tokenizer=tokenizer,
        columns=column,
        max_length=max_length,
        mlm_probability=mlm_probability,
        mask_replace_prob=mask_replace_prob,
        random_replace_prob=random_replace_prob,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    operations = [
        EnsureMapDataset(dataset_type=dataset_type),
        ToIterDataset(),
        random_transform,
    ]

    if packing:
        length_struct = {
            "input_ids": max_length,
            "labels": max_length,
            "attention_mask": max_length,
            "token_type_ids": max_length,
        }
        operations.append(
            ApplyFirstFitPacking(
                length_struct=length_struct,
                num_packing_bins=packing_bins,
            )
        )
        operations.append(ReformatPackedForMLM())

    return operations, MLMBatch
