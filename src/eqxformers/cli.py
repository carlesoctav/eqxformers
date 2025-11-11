import argparse
import json
from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.random as jr

from .config import init_model_from_yaml


def _count_parameters(module: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    return sum(int(getattr(leaf, "size", 0)) for leaf in leaves)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Initialize an eqxformers model from a YAML config")
    parser.add_argument("config", type=Path, help="Path to the YAML file describing the model")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the initialization RNG")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a small JSON summary including parameter count",
    )
    args = parser.parse_args(argv)

    key = jr.PRNGKey(args.seed)
    model = init_model_from_yaml(args.config, key=key)

    print(f"Initialized {model.__class__.__name__} from {args.config} with seed {args.seed}")
    if args.summary:
        summary = {
            "class": model.__class__.__name__,
            "parameters": _count_parameters(model),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
