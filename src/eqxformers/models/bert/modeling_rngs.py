from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import transformers
from equinox import field
from jax.nn.initializers import normal, ones as ones_init, zeros as zeros_init
from jaxtyping import Array, Float, Int
from transformers.models.bert.configuration_bert import BertConfig

from ... import nn
from ...huggingface import HuggingFaceCompatibleModule
from ...masking_utils import make_full_mask
from ...modeling_utils import PrepareableModule, Rngs
from ...nn import (
    AttentionModule,
    make_attention_module,
)
eqx.nn.Sequential


Pytree = Any


class BertEmbeddings(PrepareableModule):
    word_embeddings: nn.Embedding
    position_embeddings: nn.Embedding
    token_type_embeddings: nn.Embedding
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            rngs=rngs,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            rngs=rngs,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            rngs=rngs,
        )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids: Int[Array, " ..."],
        position_ids: Int[Array, " ..."],
        token_type_ids: Int[Array, " ..."],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, " ... H"]:
        input_ids, position_ids, token_type_ids = self.maybe_prepare_input(
            (input_ids, position_ids, token_type_ids)
        )

        inputs_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings, rngs=rngs)

        return self.maybe_prepare_output(embeddings)


class BertSelfAttention(PrepareableModule):
    query: nn.Linear
    value: nn.Linear
    key: nn.Linear
    dropout_rate: float = field(static=True)
    inference: bool = field(static=True)
    sdpa: AttentionModule
    num_attention_heads: int = field(static=True)
    attention_head_size: int = field(static=True)
    all_head_size: int = field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size})"
                "is not a multiple of the number of attention"
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout_rate = config.attention_probs_dropout_prob
        self.inference = False

        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            rngs=rngs,
        )
        self.key = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            rngs=rngs,
        )
        self.value = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            rngs=rngs,
        )

        self.sdpa = make_attention_module(
            config=config,
        )

    def __call__(
        self,
        hidden_states: Float[Array, " B T H"],
        attention_mask: Int[Array, " B T"] | None = None,
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, " ... T H"]:
        hidden_states, attention_mask = self.maybe_prepare_input(
            (hidden_states, attention_mask)
        )

        if hidden_states.ndim == 1:
            raise ValueError(
                "hidden_states must have (..., seq_len, hidden_size) shape"
            )

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        def _to_heads(x, num_heads):
            return x.reshape(*x.shape[:-1], num_heads, self.attention_head_size)

        q_heads = _to_heads(q, self.num_attention_heads)
        k_heads = _to_heads(k, self.num_attention_heads)
        v_heads = _to_heads(v, self.num_attention_heads)

        attn_heads = self.sdpa(
            query=q_heads,
            key=k_heads,
            value=v_heads,
            mask=attention_mask,
            dropout_rate=self.dropout_rate,
            rngs=rngs,
        )

        attn = attn_heads.reshape(*attn_heads.shape[:-2], self.all_head_size)
        return self.maybe_prepare_output(attn)


class BertSelfOutput(PrepareableModule):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        input_tensor: Float[Array, "T H"],
        *,
        rngs: Rngs | None = None,
    ):
        hidden_states, input_tensor = self.maybe_prepare_input(
            (hidden_states, input_tensor)
        )

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, rngs=rngs)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return self.maybe_prepare_output(hidden_states)


class BertAttention(PrepareableModule):
    self: BertSelfAttention
    output: BertSelfOutput

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.self = BertSelfAttention(
            config,
            rngs=rngs,
        )

        self.output = BertSelfOutput(config, rngs=rngs)

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        attention_mask: Int[Array, " T"] | None = None,
        /,
        *,
        rngs: Rngs | None = None,
    ):
        hidden_states, attention_mask = self.maybe_prepare_input(
            (hidden_states, attention_mask)
        )
        self_output = self.self(
            hidden_states,
            attention_mask,
            rngs=rngs,
        )
        attention_output = self.output(self_output, hidden_states, rngs=rngs)
        return self.maybe_prepare_output(attention_output)


class BertIntermediate(PrepareableModule):
    dense: nn.Linear
    # intermediate_act_fn: Callable  todo: think about this later

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T intermediate_size"]:
        (hidden_states,) = self.maybe_prepare_input((hidden_states,))

        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)

        return self.maybe_prepare_output(hidden_states)


class BertOutput(PrepareableModule):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "T intermediate_size"],
        input_tensor: Float[Array, "T H"],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T H"]:
        hidden_states, input_tensor = self.maybe_prepare_input(
            (hidden_states, input_tensor)
        )
        hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states, rngs=rngs)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return self.maybe_prepare_output(hidden_states)


# thinking about layer
class BertLayer(PrepareableModule):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.attention = BertAttention(
            config,
            rngs=rngs,
        )
        self.intermediate = BertIntermediate(
            config,
            rngs=rngs,
        )
        self.output = BertOutput(
            config,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        attention_mask: Int[Array, " T"] | None = None,
        *,
        rngs: Rngs | None = None,
    ):
        hidden_states, attention_mask = self.maybe_prepare_input(
            (hidden_states, attention_mask)
        )

        attention_output = self.attention(
            hidden_states,
            attention_mask,
            rngs=rngs,
        )
        intermediate_output = self.intermediate(attention_output, rngs=rngs)
        layer_output = self.output(intermediate_output, attention_output, rngs=rngs)
        return self.maybe_prepare_output(layer_output)


class BertEncoder(PrepareableModule):
    layer: tuple[BertLayer, ...]

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.layer = tuple(
            BertLayer(
                config,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        )

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        attention_mask: Int[Array, " T"] | None = None,
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T H"]:
        # layer_dyn_list, layer_static_list = eqx.partition(self.layer, eqx.is_array)
        #
        # static_template = layer_static_list[0]
        #
        # dynamic_stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *layer_dyn_list)
        # num_layers = next(iter(jax.tree_util.tree_leaves(dynamic_stacked))).shape[0]
        #
        # def f(carry, dyn_t):  
        #     hidden_states, attention_mask = carry
        #     module_t = eqx.combine(dyn_t, static_template)  
        #     hidden_states = module_t(hidden_states, attention_mask, rngs=rngs)
        #     return (hidden_states, attention_mask), None
        #
        # (hidden_states, _), _ = jax.lax.scan(
        #     f,
        #     (hidden_states, attention_mask),
        #     dynamic_stacked, 
        # )

        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask, rngs = rngs)

        return self.maybe_prepare_output(hidden_states)


class BertModelWeightPlanMixin:
    """
    - Linear: weight ~ N(0, initializer_range), bias zeros
    - Embedding: weight ~ N(0, initializer_range); if pad_token_id is set, zero that row
    - LayerNorm: weight ones, bias zeros
    """

    def init_weights_plan(self, module, key):
        cfg = getattr(self, "config", None)
        std = getattr(cfg, "initializer_range", 0.02)
        pad_idx = getattr(cfg, "pad_token_id", None)

        if isinstance(module, nn.Linear):
            wkey, bkey = jax.random.split(key, 2)
            w_shape = (module.out_features, module.in_features)
            new_w = normal(std)(wkey, w_shape, dtype=jnp.float32)
            new_bias = None
            if module.use_bias and module.bias is not None:
                b_shape = (module.out_features,)
                new_bias = zeros_init(bkey, b_shape, dtype=jnp.float32)
            new_mod = module
            new_mod = eqx.tree_at(lambda m: m.weight, new_mod, new_w)
            if module.use_bias and module.bias is not None:
                new_mod = eqx.tree_at(lambda m: m.bias, new_mod, new_bias)
            return new_mod

        if isinstance(module, nn.Embedding):
            w_shape = (module.num_embeddings, module.embedding_dim)
            new_w = normal(std)(key, w_shape, dtype=jnp.float32)
            if pad_idx is not None and 0 <= int(pad_idx) < module.num_embeddings:
                new_w = new_w.at[int(pad_idx)].set(
                    jnp.zeros((module.embedding_dim,), dtype=jnp.float32)
                )
            return eqx.tree_at(lambda m: m.weight, module, new_w)

        if isinstance(module, nn.LayerNorm):
            split_keys = jax.random.split(key, 2)
            w_key, b_key = split_keys[0], split_keys[1]
            w_shape = module.normalized_shape
            new_w = ones_init(w_key, w_shape, dtype=jnp.float32)
            new_mod = eqx.tree_at(lambda m: m.weight, module, new_w)
            if module.bias is not None:
                new_b = zeros_init(b_key, w_shape, dtype=jnp.float32)
                new_mod = eqx.tree_at(lambda m: m.bias, new_mod, new_b)
            return new_mod

        return module


class BertModel(
    BertModelWeightPlanMixin,
    PrepareableModule,
    HuggingFaceCompatibleModule[transformers.BertModel],
):
    embeddings: BertEmbeddings
    encoder: BertEncoder
    config: BertConfig | None = field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        store_config=True,
        rngs: Rngs,
    ):
        self.embeddings = BertEmbeddings(config, rngs=rngs)
        self.encoder = BertEncoder(config, rngs=rngs)

        if store_config:
            self.config = config
        else:
            self.config = None

    def __call__(
        self,
        input_ids: Int[Array, " *B T"],
        position_ids: Int[Array, " *B T"],
        token_type_ids: Int[Array, "*B T"],
        attention_mask: Int[Array, "*B T"] | None = None,
        segment_ids: Int[Array, "*B T"] | None = None,
        *,
        rngs: Rngs | None = None,
    ):
        args = self.maybe_prepare_input(
            (input_ids, position_ids, token_type_ids, attention_mask, segment_ids)
        )
        input_ids, position_ids, token_type_ids, attention_mask, segment_ids = args
        hidden_states = self.embeddings(
            input_ids, position_ids, token_type_ids, rngs=rngs
        )

        attention_mask = make_full_mask(
            self.config._attn_implementation,
            hidden_states,
            attention_mask,
            segment_ids,
        )

        hidden_states = self.encoder(
            hidden_states,
            attention_mask,
            rngs=rngs,
        )

        return self.maybe_prepare_output(hidden_states)


class BertPredictionHeadTransform(PrepareableModule):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T H"]:
        (hidden_states,) = self.maybe_prepare_input((hidden_states,))

        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return self.maybe_prepare_output(hidden_states)


class BertLMPredictionHead(PrepareableModule):
    transform: BertPredictionHeadTransform
    bias: Array

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.transform = BertPredictionHeadTransform(config, rngs=rngs)
        self.bias = jnp.zeros((config.vocab_size,), dtype=jnp.float32)

    def __call__(
        self,
        hidden_states: Float[Array, "T H"],
        embedding_weight: Float[Array, "vocab_size H"],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T vocab_size"]:
        hidden_states, embedding_weight = self.maybe_prepare_input(
            (hidden_states, embedding_weight)
        )
        hs = self.transform(hidden_states, rngs=rngs)
        logits = jnp.einsum("...d, vd->...v", hs, embedding_weight)
        logits = logits + self.bias
        return self.maybe_prepare_output(logits)


class BertOnlyMLMHead(PrepareableModule):
    predictions: BertLMPredictionHead

    def __init__(
        self,
        config: BertConfig,
        *,
        rngs: Rngs,
    ):
        self.predictions = BertLMPredictionHead(config, rngs=rngs)

    def __call__(
        self,
        sequence_output: Float[Array, "T H"],
        embedding_weight: Float[Array, "vocab_size H"],
        *,
        rngs: Rngs | None = None,
    ) -> Float[Array, "T vocab_size"]:
        sequence_output, embedding_weight = self.maybe_prepare_input(
            (sequence_output, embedding_weight)
        )
        logits = self.predictions(sequence_output, embedding_weight, rngs=rngs)
        return self.maybe_prepare_output(logits)


class BertForMaskedLM(
    BertModelWeightPlanMixin,
    PrepareableModule,
    HuggingFaceCompatibleModule[transformers.BertForMaskedLM],
):
    bert: BertModel
    cls: BertOnlyMLMHead
    config: BertConfig | None = eqx.field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        store_config: bool = True,
        rngs: Rngs,
    ):
        self.bert = BertModel(
            config,
            store_config=True,
            rngs=rngs,
        )
        self.cls = BertOnlyMLMHead(config, rngs=rngs)

        if store_config:
            self.config = config
        else:
            self.config = None

    def __call__(
        self,
        input_ids: Int[Array, " ... T"],
        position_ids: Int[Array, "... T"],
        token_type_ids: Int[Array, "... T"],
        *,
        attention_mask: Int[Array, "... T"] | None = None,
        segment_ids: Int[Array, "... T"] | None = None,
        rngs: Rngs | None = None,
    ):
        input_ids, position_ids, token_type_ids, attention_mask, segment_ids = (
            self.maybe_prepare_input(
                (input_ids, position_ids, token_type_ids, attention_mask, segment_ids)
            )
        )

        sequence_output = self.bert(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            segment_ids,
            rngs=rngs,
        )

        w = self.bert.embeddings.word_embeddings.weight
        if hasattr(w, "value"):
            w = w.value
        logits = self.cls(sequence_output, w, rngs=rngs)
        return self.maybe_prepare_output(logits)

    @classmethod
    def normalize_hf_key_for_eqx(cls, key: str) -> str | None:
        if key.startswith("cls.predictions.decoder.weight"):
            return None
        return key
