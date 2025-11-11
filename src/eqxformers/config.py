import abc
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, ClassVar, Mapping, Sequence

import draccus
from jaxtyping import PRNGKeyArray


ModelBuilder = Callable[..., Any]
ConfigSource = str | Path | Mapping[str, Any] | IO[str]


class ModelConfig(
    draccus.ChoiceRegistry,
    abc.ABC,
): 
    @abc.abstractmethod
    def make(self, *, key: PRNGKeyArray, **kwargs: Any) -> Any:
        """Materialize the configured model."""
        raise NotImplementedError


def parse_model_config(config: ConfigSource, *, args: Sequence[str] | None = None) -> ModelConfig:
    """Parse a model config (or path) into a :class:`ModelConfigRegister`."""

    parsed_args: Sequence[str] | None = [] if args is None else args
    return draccus.parse(ModelConfig, config, args=parsed_args)


def init_model_from_yaml(
    config: ConfigSource,
    *,
    key: PRNGKeyArray,
    args: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Parse ``config`` with Draccus and initialize the requested model."""

    model_config = parse_model_config(config, args=args)
    return model_config.make(key=key, **kwargs)


def _ensure_builtin_configs_loaded() -> None:
    for module in (
        "eqxformers.models.bert.configuration_bert",
    ):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            continue


_ensure_builtin_configs_loaded()


__all__ = [
    "ModelConfig",
    "parse_model_config",
    "init_model_from_yaml",
]
