
#%%
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
import evaluate
from huggingface_hub import notebook_login
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, Checkpoint
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers.async_hyperband import ASHAScheduler
import ray

# %%
use_gpu = True
num_workers = 1
# %%
wandb.login(key="")

#%%
task = "cb"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

#%%
hf_dataset = load_dataset("super_glue", task)

def is_valid(example):
    return isinstance(example.get("premise"), str) and isinstance(example.get("hypothesis"), str)

#%%
hf_dataset["train"] = hf_dataset["train"].filter(is_valid)
hf_dataset["validation"] = hf_dataset["validation"].filter(is_valid)
hf_dataset["test"] = hf_dataset["test"].filter(is_valid)


#%% preprocessing the data with ray data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#%%
ray_datasets = {
    "train": ray.data.from_items(hf_dataset["train"].to_list()),
    "validation": ray.data.from_items(hf_dataset["validation"].to_list()),
    "test": ray.data.from_items(hf_dataset["test"].to_list())
}

ray_datasets

# %% preprocess the samples

def tokenize_fn(batch):
    premises = batch["premise"]
    hypothesis = batch["hypothesis"]
    labela = batch["label"]
    
    
