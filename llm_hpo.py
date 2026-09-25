#%%
from datasets import load_dataset

dataset = load_dataset("legacy-datasets/banking77")
# %%
print(dataset)
# %%

splits = dataset["train"]