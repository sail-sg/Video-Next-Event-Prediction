import datasets
from datasets import load_dataset, load_from_disk

DATASET_PATH='/path/to/futurebench.json'

ds = load_dataset("json", data_files={'test': DATASET_PATH})
ds.save_to_disk('./futurebench')

dataset = load_from_disk('./futurebench')

print(dataset)
