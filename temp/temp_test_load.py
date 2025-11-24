from datasets import load_dataset

from eqxformers.data.huggingface_datasets import HuggingFaceIterableDataset


dataset = load_dataset("carlesoctav/skripsi_UI_membership_30K", streaming = True, split = "train")
ds = HuggingFaceIterableDataset(dataset)

it = iter(ds)

data = next(it)
print(f"DEBUGPRINT[1]: temp_test_load.py:10: data={data}")
count = 0
while count < 5:
    data = next(it)
    count+=1

state = it.get_state()
print(f"DEBUGPRINT[2]: temp_test_load.py:18: state={state}")

final_data = next(it)

dataset = load_dataset("carlesoctav/skripsi_UI_membership_30K", streaming = True, split = "train")
ds = HuggingFaceIterableDataset(dataset)
it =  iter(ds)
data = next(it)
print(f"DEBUGPRINT[1]: temp_test_load.py:10: data={data}")

it.set_state(state)

new_final_data = next(it)
print(f"DEBUGPRINT[6]: temp_test_load.py:31: new_final_data={new_final_data}")
print(f"DEBUGPRINT[7]: temp_test_load.py:33: final_data={final_data}")
print(f"DEBUGPRINT[8]: temp_test_load.py:31: new_final_data={type(new_final_data)}")
print( final_data == new_final_data)




