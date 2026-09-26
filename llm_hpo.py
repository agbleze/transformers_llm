#%%
from datasets import load_dataset
import ray
import os
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import transformers
import torch

dataset = load_dataset("legacy-datasets/banking77")
# %%
print(dataset)
# %%

splits = dataset["train"]
# %%
splits.to_list()
# %%
type(splits)
# %%
def is_valid(exp):
    return isinstance(exp.get("text"), str)
# %%
splits.filter(is_valid)
# %%
train_data = ray.data.from_items(dataset["train"].to_list())
# %%
test_data = ray.data.from_items(dataset["test"].to_list())
# %%
model_name = "microsoft/deberta-v3-small"
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# %%
tokenizer
# %%
tokenizer

dataset.data["train"].schema
# %%
dataset.column_names
# %%
os.confstr_names
# %%
dataset.data.keys()
# %%
train_data.random_sample(0.1)
# %%
exp_text = dataset.data["train"].to_pylist()[0]
# %%
tokenizer(exp_text["text"])
# %%
for i in train_data.iter_rows():
    print(i)
    break
# %%
ray_train_from_hf_data = ray.data.from_huggingface(dataset["train"])
# %%
ray_train_from_hf_data.schema
# %%
train_data.get_dataset_id()
# %%
for i in train_data.iter_torch_batches(batch_size=32):
    print(i)
    break
# %%
for i in ray_train_from_hf_data.iter_torch_batches(batch_size=32):
    print(i)
    break
# %%
for b in ray_train_from_hf_data.iter_batches(batch_size=4):
    print(b)
    text = b["text"]
    print(text)
    print(type(text))
    token = tokenizer(text.tolist(), truncation=True, padding="longest", return_tensors="pt")#, return_tensors="pt")
    print(token)
    break
# %%
def tokenize_batch(batch, tokenizer=tokenizer, device="cuda"):
    text = batch["text"]
    label = batch["label"]
    text = text.tolist() if isinstance(text, np.ndarray) else text
    token = tokenizer(text, max_length=128, truncation=True, 
                      padding="longest", 
                      return_tensors="pt"
                      )
    token["label"] = torch.tensor(label, dtype=torch.long)
    token = {k: v.to(device) for k,v in token.items()}
    return token


#%%

for b in train_data.iter_batches(batch_size=4):
    token = tokenize_batch(b, tokenizer)
    print(token)
    break


#%%

for b in ray_train_from_hf_data.iter_torch_batches(batch_size=4, collate_fn=tokenize_batch):
    print(b)
    break
