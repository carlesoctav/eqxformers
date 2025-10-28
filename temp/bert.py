import equinox as eqx
import jax
import numpy as np
from transformers import BertConfig, BertTokenizer

from src.eqxformers.models.bert import BertModel


tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
config = BertConfig(use_scan = True)
print(f"DEBUGPRINT[65]: bert.py:10: config={config}")
model= BertModel(config, key = jax.random.key(10))


token = tokenizer("hallo dunia", return_tensors = "np")
print(f"DEBUGPRINT[58]: bert.py:11: token={token}")
model = eqx.nn.inference_mode(model)
output = model(
    **{
        "position_ids":np.arange(token["input_ids"].shape[1]),
        **token
    }
)
print(f"DEBUGPRINT[59]: bert.py:13: output={output}")


