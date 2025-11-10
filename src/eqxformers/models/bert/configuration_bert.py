from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from jaxtyping import PRNGKeyArray

from ...config import ModelConfig
from .modeling_bert import BertModel
from .modeling_bert import BertForMaskedLM



@ModelConfig.register_subclass("bert")
@dataclass
class BertConfig(ModelConfig):
    """Dataclass describing BERT hyper-parameters buildable via Draccus."""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    pad_token_id: int | None = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_scan: bool = True
    _attn_implementation: str = "eager"

    task: str = "model"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

    def make(self, task: str | None = None, *, key: PRNGKeyArray, **kwargs: Any) -> Any:
        task = task if task else self.task 
        match task:
            case "model":
                return BertModel(self, key = key, **kwargs)
            case "mlm":
                return BertForMaskedLM(self, key = key, **kwargs)
            case _:
                raise NotImplementedError(f"Unknown task '{task}' for BertConfig")
