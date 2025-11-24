from datasets import load_dataset

ds  = load_dataset("HuggingFaceFW/fineweb", streaming = True)
print(f"DEBUGPRINT[10]: hf_ds.py:1: ds={ds}")
