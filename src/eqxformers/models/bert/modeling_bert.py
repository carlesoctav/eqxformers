from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from transformers.models.bert.configuration_bert import BertConfig

from ...masking_utils import make_full_mask
from ...modeling_utils import Module
from ...nn.attention import ALL_ATTENTION_FUNCTIONS
from ...nn.dropout import Dropout
from ...nn.embedding import Embedding
from ...nn.linear import Linear
from ...nn.normalisation import LayerNorm




def _split_key(key: PRNGKeyArray | None, num: int) -> tuple[PRNGKeyArray | None, ...]:
    if key is None:
        return (None,) * num
    return tuple(jax.random.split(key, num))


class BertEmbeddings(Module):
    word_embeddings: Embedding
    position_embeddings: Embedding
    token_type_embeddings: Embedding
    layer_norm: LayerNorm
    dropout: Dropout

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        w_key, p_key, tt_key, ln_key = jax.random.split(key, 4)
        self.word_embeddings = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            key=w_key,
        )
        self.position_embeddings = Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            key=p_key,
        )
        self.token_type_embeddings = Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            key=tt_key,
        )

        self.layer_norm = LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            key=ln_key,
        )
        self.dropout = Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        position_ids: Int[Array, "B T"],
        token_type_ids: Int[Array, "B T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(input_ids, position_ids, token_type_ids)
        input_ids, position_ids, token_type_ids = prepared

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, key=key)

        return self.maybe_prepare_output(embeddings)


class BertSelfAttention(Module):
    query: Linear
    key: Linear
    value: Linear
    attention_impl: str = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    inference: bool

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        q_key, k_key, v_key = jax.random.split(key, 3)

        self.query = Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            key=q_key,
        )
        self.key = Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            key=k_key,
        )
        self.value = Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            key=v_key,
        )

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout_rate = config.attention_probs_dropout_prob
        self.inference = False
        self.attention_impl = getattr(config, "_attn_implementation", "sdpa")

    def _shape_to_heads(
        self, x: Float[Array, "B T H"], batch_size: int
    ) -> Float[Array, "B T N D"]:
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, attention_mask)
        hidden_states, attention_mask = prepared

        batch_size, _seq_len, _hidden = hidden_states.shape
        query_layer = self._shape_to_heads(self.query(hidden_states), batch_size)
        key_layer = self._shape_to_heads(self.key(hidden_states), batch_size)
        value_layer = self._shape_to_heads(self.value(hidden_states), batch_size)

        if attention_mask is not None:
            if attention_mask.ndim != 3:
                raise ValueError("attention_mask must have shape (B, T, S)")
            attention_mask = attention_mask[:, None, :, :]

        attention_fn = ALL_ATTENTION_FUNCTIONS[self.attention_impl]
        attn_output = attention_fn(
            query_layer,
            key_layer,
            value_layer,
            attention_mask=attention_mask,
            dropout_rate=self.dropout_rate,
            dropout_key=key,
            inference=self.inference,
        )

        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)
        return self.maybe_prepare_output(attn_output)


class BertSelfOutput(Module):
    dense: Linear
    layer_norm: LayerNorm
    dropout: Dropout

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        dense_key, ln_key = jax.random.split(key, 2)
        self.dense = Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            key=dense_key,
        )
        self.layer_norm = LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            key=ln_key,
        )
        self.dropout = Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        input_tensor: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, input_tensor)
        hidden_states, input_tensor = prepared

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, key=key)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return self.maybe_prepare_output(hidden_states)


class BertAttention(Module):
    self_attention: BertSelfAttention
    output: BertSelfOutput

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        attn_key, output_key = jax.random.split(key, 2)
        self.self_attention = BertSelfAttention(config, key=attn_key)
        self.output = BertSelfOutput(config, key=output_key)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, attention_mask)
        hidden_states, attention_mask = prepared

        attn_key, output_key = _split_key(key, 2)
        attention_output = self.self_attention(
            hidden_states,
            attention_mask,
            key=attn_key,
        )
        attention_output = self.output(
            attention_output,
            hidden_states,
            key=output_key,
        )
        return self.maybe_prepare_output(attention_output)


class BertIntermediate(Module):
    dense: Linear
    activation: tp.Callable[[Array], Array]

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        self.dense = Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            key=key,
        )
        self.activation = jax.nn.gelu

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T I"]:
        hidden_states = self.maybe_prepare_input(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.maybe_prepare_output(hidden_states)


class BertOutput(Module):
    dense: Linear
    layer_norm: LayerNorm
    dropout: Dropout

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        dense_key, ln_key = jax.random.split(key, 2)
        self.dense = Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            key=dense_key,
        )
        self.layer_norm = LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            key=ln_key,
        )
        self.dropout = Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "B T I"],
        input_tensor: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, input_tensor)
        hidden_states, input_tensor = prepared

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, key=key)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return self.maybe_prepare_output(hidden_states)


class BertLayer(Module):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        attn_key, inter_key, out_key = jax.random.split(key, 3)
        self.attention = BertAttention(config, key=attn_key)
        self.intermediate = BertIntermediate(config, key=inter_key)
        self.output = BertOutput(config, key=out_key)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, attention_mask)
        hidden_states, attention_mask = prepared

        attn_key, ffn_key = _split_key(key, 2)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            key=attn_key,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(
            intermediate_output,
            attention_output,
            key=ffn_key,
        )
        return self.maybe_prepare_output(layer_output)


class BertEncoder(Module):
    layers: tuple[BertLayer, ...]

    def __init__(self, config: BertConfig, *, key: PRNGKeyArray):
        layer_keys = jax.random.split(key, config.num_hidden_layers)
        self.layers = tuple(BertLayer(config, key=layer_key) for layer_key in layer_keys)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(hidden_states, attention_mask)
        hidden_states, attention_mask = prepared

        if key is not None:
            layer_keys = jax.random.split(key, len(self.layers))
        else:
            layer_keys = (None,) * len(self.layers)

        for bert_layer, layer_key in zip(self.layers, layer_keys):
            hidden_states = bert_layer(
                hidden_states,
                attention_mask,
                key=layer_key,
            )

        return self.maybe_prepare_output(hidden_states)


class BertModel(Module):
    embeddings: BertEmbeddings
    encoder: BertEncoder
    config: BertConfig | None = eqx.field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        store_config: bool = True,
    ):
        embed_key, encoder_key = jax.random.split(key, 2)
        self.embeddings = BertEmbeddings(config, key=embed_key)
        self.encoder = BertEncoder(config, key=encoder_key)
        self.config = config if store_config else None

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        position_ids: Int[Array, "B T"],
        token_type_ids: Int[Array, "B T"],
        attention_mask: Array | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        prepared = self.maybe_prepare_input(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            segment_ids,
        )
        input_ids, position_ids, token_type_ids, attention_mask, segment_ids = prepared

        embed_key, encoder_key = _split_key(key, 2)
        hidden_states = self.embeddings(
            input_ids,
            position_ids,
            token_type_ids,
            key=embed_key,
        )

        mask_impl = getattr(self.config, "_attn_implementation", "sdpa") if self.config else "sdpa"
        attention_mask = make_full_mask(
            mask_impl,
            hidden_states,
            attention_mask,
            segment_ids,
        )

        hidden_states = self.encoder(
            hidden_states,
            attention_mask,
            key=encoder_key,
        )

        return self.maybe_prepare_output(hidden_states)


class BertMLMHead(Module):
    dense: Linear
    layer_norm: LayerNorm
    activation: tp.Callable[[Array], Array]
    lm_head: Linear | eqx.nn.Shared
    lm_head_linear: Linear
    bias: Array
    tie_word_embeddings: bool = eqx.field(static=True)

    def __init__(
        self,
        config: BertConfig,
        embedding: Embedding,
        *,
        key: PRNGKeyArray,
    ):
        dense_key, ln_key, head_key = jax.random.split(key, 3)
        self.dense = Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            key=dense_key,
        )
        self.layer_norm = LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            key=ln_key,
        )
        self.activation = jax.nn.gelu

        lm_linear = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            use_bias=False,
            key=head_key,
        )

        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if self.tie_word_embeddings:
            self.lm_head = eqx.nn.Shared(
                (embedding, lm_linear),
                where=lambda modules: modules[1].weight,
                get=lambda modules: modules[0].weight,
            )
        else:
            self.lm_head = lm_linear

        self.lm_head_linear = lm_linear
        self.bias = jnp.zeros((config.vocab_size,), dtype=jnp.float32)

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T V"]:
        hidden_states = self.maybe_prepare_input(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if isinstance(self.lm_head, eqx.nn.Shared):
            _, lm_linear = self.lm_head()
        else:
            lm_linear = self.lm_head

        logits = lm_linear(hidden_states)
        logits = logits + self.bias
        return self.maybe_prepare_output(logits)


class BertForMaskedLM(Module):
    bert: BertModel
    mlm_head: BertMLMHead
    config: BertConfig | None = eqx.field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        key: PRNGKeyArray,
        store_config: bool = True,
    ):
        bert_key, head_key = jax.random.split(key, 2)
        self.bert = BertModel(config, key=bert_key, store_config=True)
        self.mlm_head = BertMLMHead(
            config,
            self.bert.embeddings.word_embeddings,
            key=head_key,
        )
        self.config = config if store_config else None

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        position_ids: Int[Array, "B T"],
        token_type_ids: Int[Array, "B T"],
        *,
        attention_mask: Array | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T V"]:
        prepared = self.maybe_prepare_input(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            segment_ids,
        )
        input_ids, position_ids, token_type_ids, attention_mask, segment_ids = prepared

        bert_key, cls_key = _split_key(key, 2)
        sequence_output = self.bert(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
            segment_ids,
            key=bert_key,
        )

        logits = self.mlm_head(sequence_output, key=cls_key)
        return self.maybe_prepare_output(logits)
