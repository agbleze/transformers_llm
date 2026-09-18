
#%%
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import evaluate
from huggingface_hub import notebook_login
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, Checkpoint
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers.async_hyperband import ASHAScheduler

# %%
use_gpu = True
num_workers = 1
# %%
wandb.login(key="")

#%%
task = "cb"
