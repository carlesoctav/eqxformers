import torchax
from transformers import BertModel, BertConfig, BertTokenizer

torchax.enable_globally()

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("hallo", return_tensors = "pt")

print(f"DEBUGPRINT[69]: torchax_transformers.py:12: inputs={inputs}")
inputs = {
    "input_ids": inputs["input_ids"].to("jax"),
#     # "positions_ids": torch.arange(inputs['token_ids'].shape[1]).to("jax"),
    "attention_mask": inputs["attention_mask"].to("jax"),
}
print(f"DEBUGPRINT[68]: torchax_transformers.py:12: inputs={inputs}")


model = BertModel(BertConfig()).to("jax")
outputs = model(**inputs)



