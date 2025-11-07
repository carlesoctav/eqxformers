import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from datasets import load_dataset
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertConfig

from src.eqxformers.data import make_dataloader
from src.eqxformers.data.masked_language_modeling import masked_language_modeling_transforms
from src.eqxformers.models.bert import BertForMaskedLM



LOGGER = logging.getLogger(__name__)


DATASET_NAME = "carlesoctav/skripsi_UI_membership_30K"
DATASET_SPLIT = "train"
DATASET_SUBSET = None
COLUMN_NAME = "id_title"
MAX_LENGTH = 512
BATCH_SIZE = 64
NUM_STEPS = 10000
LEARNING_RATE = 5e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
SEED = 42
NUM_WORKERS = 4
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
LOG_EVERY_N_STEPS = 10
EVAL_INTERVAL = 500
SAVE_INTERVAL_STEPS = 100
MESH_SHAPE = (4,)
MESH_AXIS_NAMES = ("dp",)


class Optimizer(eqx.Module):
    opt_state: Any
    grad_tx: optax.GradientTransformation = eqx.field(static=True)
    wrt: Any = eqx.field(static=True)

    def __init__(self, module: eqx.Module, grad_tx: optax.GradientTransformation, *, wrt: Any = eqx.is_inexact_array):
        self.grad_tx = grad_tx
        self.wrt = wrt
        params = eqx.filter(module, self.wrt)
        self.opt_state = grad_tx.init(params)

    def __call__(self, grads: Any, module: eqx.Module) -> tuple[eqx.Module, "Optimizer"]:
        params = eqx.filter(module, self.wrt)
        updates, opt_state = self.grad_tx.update(grads, self.opt_state, params)
        new_module = eqx.apply_updates(module, updates)
        new_self = eqx.tree_at(lambda s: s.opt_state, self, opt_state)
        return new_module, new_self


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    where: Array | None = None,
) -> Array:
    if where is not None:
        safe_labels = jnp.where(where, labels, 0)
    else:
        safe_labels = labels

    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - jax.lax.stop_gradient(logits_max)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))
    true_logit = jnp.take_along_axis(shifted, safe_labels[..., None], axis=-1)[..., 0]
    loss = log_sum_exp - true_logit
    if where is not None:
        loss = jnp.where(where, loss, 0.0)
    return loss


def make_train_step(
    loss_function: Callable[[eqx.Module, Optimizer, Any], tuple[jax.Array, dict[str, Any]]],
    *,
    gradient_accumulation_steps: int = 1,
) -> Callable[[eqx.Module, Optimizer, Any], tuple[eqx.Module, Optimizer, dict[str, Any]]]:
    grad_fn = eqx.filter_value_and_grad(loss_function, has_aux=True)

    def train_step(
        module: eqx.Module,
        optimizer: Optimizer,
        batch: Any,
        *,
        key: Array,
    ) -> tuple[eqx.Module, Optimizer, dict[str, Any]]:
        if gradient_accumulation_steps != 1:
            raise NotImplementedError("Gradient accumulation > 1 is not yet supported in this benchmark script")

        (loss, aux), grads = grad_fn(module, optimizer, batch, key=key)
        del loss
        new_module, new_optimizer = optimizer(grads, module)
        return new_module, new_optimizer, aux

    return eqx.filter_jit(train_step)


def benchmark_loop(
    module: eqx.Module,
    optimizer: Optimizer,
    train_step_fn: Callable[[eqx.Module, Optimizer, Any], tuple[eqx.Module, Optimizer, dict[str, Any]]],
    train_loader: Iterable[Any],
    *,
    key: Array,
    num_steps: int = 100,
) -> tuple[eqx.Module, Optimizer, dict[str, Any]]:
    step_idx = -1
    train_times: list[float] = []
    next_batch_times: list[float] = []
    compile_time: float | None = None

    iterator = iter(train_loader)
    progress = tqdm(total=num_steps + 1, desc="Benchmarking", disable=jax.process_index() != 0)
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                LOGGER.info("Data loader exhausted during benchmark loop")
                break
            batch_end = time.perf_counter()

            key, step_key = jr.split(key)

            step_idx += 1
            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
                module, optimizer, aux = train_step_fn(module, optimizer, batch, key=step_key)
            _ = aux
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, module)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, optimizer)
            step_end = time.monotonic()

            if step_idx == 0:
                compile_time = step_end - step_start
                LOGGER.info("Compilation (step 0) took %.4fs", compile_time)
            else:
                train_times.append(step_end - step_start)
                next_batch_times.append(batch_end - batch_start)

            progress.update(1)
    finally:
        progress.close()

    train_times_arr = np.asarray(train_times) if train_times else np.asarray([0.0])
    batch_times_arr = np.asarray(next_batch_times) if next_batch_times else np.asarray([0.0])
    stats = {
        "train_step_time_mean": float(train_times_arr.mean()),
        "train_step_time_std": float(train_times_arr.std()),
        "next_batch_time_mean": float(batch_times_arr.mean()),
        "compile_time": float(compile_time) if compile_time is not None else None,
    }
    LOGGER.info("Benchmark stats: %s", stats)
    return module, optimizer, stats



def _cast_floating(tree: Any, dtype: jnp.dtype):
    def _cast(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jtu.tree_map(_cast, tree, is_leaf=lambda x: isinstance(x, jax.Array))


def _get_position_ids(batch: Any, seq_length: int):
    if getattr(batch, "position_ids", None) is not None:
        return batch.position_ids
    batch_size = batch.input_ids.shape[0]
    return jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))


def loss_function(model: BertForMaskedLM, optimizer: Optimizer, batch: Any, *, key: Array):
    del optimizer
    _, dropout_key = jr.split(key)

    logits = model(
        input_ids=batch.input_ids,
        position_ids=_get_position_ids(batch, MAX_LENGTH),
        token_type_ids=batch.token_type_ids,
        attention_mask=batch.attention_mask,
        segment_ids=getattr(batch, "segment_ids", None),
        key=dropout_key,
    )
    logits = logits.astype(jnp.float32)

    labels = batch.labels
    valid_mask = labels != -100

    loss_per_token = softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=jnp.where(valid_mask, labels, 0),
        where=valid_mask,
    )

    total_loss = jnp.sum(loss_per_token)
    num_valid_tokens = jnp.sum(valid_mask)
    accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == labels) & valid_mask)

    aux = {
        "loss": (total_loss, num_valid_tokens),
        "acc": (accuracy, num_valid_tokens),
        "total_token": num_valid_tokens,
    }
    return total_loss, aux


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _build_mesh():
    devices = jax.devices()
    expected = int(np.prod(MESH_SHAPE))
    if expected <= 0:
        raise ValueError("MESH_SHAPE must describe at least one device")
    if len(devices) < expected:
        LOGGER.warning(
            "Requested mesh with %d devices but only %d available; using available devices",
            expected,
            len(devices),
        )
        mesh_devices = np.array(devices[: len(devices)])
        actual_shape = (len(mesh_devices),)
    else:
        mesh_devices = np.array(devices[:expected])
        actual_shape = MESH_SHAPE
    mesh_devices = mesh_devices.reshape(actual_shape)
    return Mesh(mesh_devices, MESH_AXIS_NAMES)


def main():
    setup_logging()
    LOGGER.info("Starting benchmark training run")

    key = jr.PRNGKey(SEED)
    key, model_key = jr.split(key)

    mesh = _build_mesh()

    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        _attn_implementation="eager",
        use_scan = False,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def unbox(model):
        def f(leaf):
            if isinstance(leaf, jax.Array):
                return jax.lax.with_sharding_constraint(leaf, jax.P())
            return leaf

        return jtu.tree_map(f, model)


    with mesh:
        model = BertForMaskedLM(config, key=model_key)
        print(f"DEBUGPRINT[82]: train.py:276: model={model}")
        # model = _cast_floating(model, jnp.bfloat16)
        model = unbox(model) 

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            decay_steps=NUM_STEPS,
            end_value=0.0,
        )

        grad_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
        )

        optimizer = Optimizer(model, grad_tx)

    start_time = time.monotonic()
    jax.block_until_ready(model)
    jax.block_until_ready(optimizer)
    diff = time.monotonic() - start_time
    LOGGER.info("Model and optimizer initialization took %.4fs", diff)

    train_step_fn = make_train_step(
        loss_function=loss_function,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=False,
    )

    operations, batch_class = masked_language_modeling_transforms(
        dataset_type="huggingface",
        column=COLUMN_NAME,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        mlm_probability=MLM_PROBABILITY,
        packing=False,
    )

    train_loader = make_dataloader(
        datasets=[dataset],
        operations=operations,
        global_batch_size=BATCH_SIZE,
        pspec=PartitionSpec("dp"),
        mesh=mesh,
        num_epochs=None,
        shuffle=True,
        seed=SEED,
        worker_count=NUM_WORKERS,
        worker_buffer_size=WORKER_BUFFER_SIZE,
        drop_remainder=True,
        batch_class=batch_class,
    )

    first_batch = next(iter(train_loader))
    key, lower_key = jr.split(key)
    start = time.monotonic() 
    lowered = train_step_fn.lower(model, optimizer, first_batch, key=lower_key)
    compiled = lowered.compile()
    diff = time.monotonic()  - start
    print(f"DEBUGPRINT[75]: train.py:341: diff={diff}")
    hlo_text = lowered.as_text()
    Path("benchmark").mkdir(parents=True, exist_ok=True)
    (Path("benchmark") / "train_hlo.txt").write_text(hlo_text)

    LOGGER.info("Compilation artifact written to benchmark/train_hlo.txt")

    model, optimizer, stats = benchmark_loop(model, optimizer, compiled, train_loader, key=key)
    print(f"DEBUGPRINT[81]: train.py:349: stats={stats}")


if __name__ == "__main__":
    main()
