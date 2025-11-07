from datasets import load_dataset, load_from_disk


ds = load_from_disk("gs://carles-git-good/ww")
print(f"DEBUGPRINT[98]: loaddata.py:6: ds={ds}")
