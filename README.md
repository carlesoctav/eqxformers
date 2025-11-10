# eqxformers

## YAML-driven model configs

You can now describe eqxformers models with [Draccus](https://github.com/dlwh/draccus) YAML files and initialize them without touching Python code.

```bash
python -m eqxformers.cli config/bert_mlm.yaml --summary
```

The example in `config/bert_mlm.yaml` registers a BERT config (`type: bert`) and chooses the masked language modeling head via `task: mlm`. The CLI will parse the file, instantiate the requested module, and optionally print a JSON summary.

Programmatic use is available through `eqxformers.config.init_model_from_yaml` or by parsing configs via `eqxformers.ModelConfigRegister` and calling `.build(...)` yourself.
