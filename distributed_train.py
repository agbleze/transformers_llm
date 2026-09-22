
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
from glob import glob
import logging
import uuid

#%%

from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, CheckpointConfig
#%%

logging.getLogger("ray").setLevel(logging.ERROR)
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
os.environ["RAY_CLUSTER_NAME"] = "local"
#os.environ["RAY_ADDRESS"] = "localhost:6379"
os.environ["RAY_TMPDIR"] = "/tmp/ray_native"
os.environ["RAY_STORAGE"] = "/mnt/d/ray_spill/tune_results"
os.environ["RAY_DATA_CACHE_DIR"] = "/mnt/d/ray_spill/data_cache"

# 2. Add these to disable the broken WSL internal state health checks
os.environ["RAY_ENABLE_STATE_API"] = "0"
os.environ["RAY_TRAIN_ENABLE_MONITORING"] = "0"

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

ray.init(
    #address="local",
    
    _temp_dir="/tmp/ray_native",
    
    
    # 2. OFFLOAD THE HEAVY WEIGHTS: Large arrays spill to your D: drive
    _system_config={
        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": ["/mnt/d/ray_spill"]}}'
    },
    
    include_dashboard=False,
    # 3. CODE REPOSITORY: Points to your external storage directory
    #runtime_env={"working_dir": "/mnt/d/data_store/distributed_work"}
)



#%%

working_dir = "/mnt/d/data_store/distributed_work"

#os.listdir(data_store)

glob(f"{working_dir}/*")


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
    hypotheses = batch["hypothesis"]
    labels = batch["label"]
    
    premises = premises.tolist() if isinstance(premises, np.ndarray) else premises
    hypotheses = hypotheses.tolist() if isinstance(hypotheses, np.ndarray) else hypotheses
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
    
    tokenized = tokenizer(premises,
                          hypotheses,
                          truncation=True,
                          padding="longest",
                          return_tensors="pt"
                          )
    tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
    tokenized = {k: v.to("cuda") for k, v in tokenized.items()}
    return tokenized
    
#%% finetuning the model
num_labels = 3
metric_name = ("accuracy")
model_name = model_checkpoint.split("/")[-1]
validation_key = ("validation")
name = f"{model_name}-finetuned-{task}"

max_steps_per_epoch = ray_datasets["train"].count() // (batch_size * num_workers)

def train_func(config):
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    metric = evaluate.load("super_glue", config_name=task)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                               num_labels=num_labels
                                                            )  
    train = ray.train.get_dataset_shard("train")
    eval = ray.train.get_dataset_shard("eval")
    
    train_iterable = train.iter_torch_batches(batch_size=batch_size,
                                              collate_fn=tokenize_fn
                                              )  
    eval_iterable = eval.iter_torch_batches(batch_size=batch_size,
                                            collate_fn=tokenize_fn
                                            )
    args = TrainingArguments(name, eval_strategy="epoch",
                             save_strategy="no",
                             logging_strategy="epoch",
                             per_device_train_batch_size=config.get("batch_size", 64),
                             per_device_eval_batch_size=config.get("batch_size", 64),
                             learning_rate=config.get("learning_rate", 2e-5),
                             num_train_epochs=config.get("epochs", 6),
                             weight_decay=config.get("weight_decay", 0.001),
                             max_steps=max_steps_per_epoch * config.get("epochs", 6),
                             disable_tqdm=False,
                             no_cuda=not torch.cuda.is_available(),
                             report_to="wandb",
                             run_name="superglue_cb"
                             )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_iterable,
                      eval_dataset=eval_iterable,
                      compute_metrics=compute_metrics
                      )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    
    print("Beginning training...")
    trainer.train()
    
    eval_metrics = trainer.evaluate()
    print(f"Final evaluation:", eval_metrics)
    
    unique_id = str(uuid.uuid4())[:8]
    output_dir = f"/mnt/d/ray_spill/scratch_checkpoints/{unique_id}/hf_model"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    ray.train.report({"eval_loss": eval_metrics.get("eval_loss"),
                      "eval_accuracy": eval_metrics.get("eval_accuracy")
                      },
                     checkpoint=Checkpoint.from_directory(output_dir)
                     )
    ray.tune.report(metrics=eval_metrics, checkpoint=Checkpoint.from_directory(output_dir))
    

# %% setup torchtrainer
ray_trainer = TorchTrainer(train_loop_per_worker=train_func,
                       scaling_config=ScalingConfig(num_workers=num_workers,
                                                    use_gpu=use_gpu,
                                                    ),
                       datasets = {"train": ray_datasets["train"],
                                   "eval": ray_datasets["validation"]
                                   },
                    #    run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1,
                    #                                                            checkpoint_score_attribute="eval_loss",
                    #                                                            checkpoint_score_order="min",
                    #                                                            )
                    #                         )
                       )


#%%

#ray_trainer.fit()

trainable_with_datasets = tune.with_parameters(
    train_func,
    # This matches the dataset shard extraction lines inside your train_func!
    # They will be injected transparently into the workers
    datasets={
        "train": ray_datasets["train"],
        "eval": ray_datasets["validation"]
    }
)

trainable_with_resources = tune.with_resources(
    trainable_with_datasets,
    resources={"cpu": num_workers, "gpu": 1 if use_gpu else 0}
)


#%% tune hyperparameters
import warnings
from ray.train import CheckpointConfig, RunConfig, SyncConfig
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")


tuner = Tuner(trainable_with_resources,
              param_space={
                    "learning_rate": tune.grid_search([2e-5, 2e-4, 2e-3, 2e-2]),
                    "epochs": tune.choice([2, 4, 6, 8]),
                    "batch_size": tune.choice([16, 32, 64, 128]),
                    "weight_decay": tune.grid_search([0.0, 0.01, 0.1, 0.001])
            
            },
            tune_config=tune.TuneConfig(
                metric="eval_loss",
                mode="min",
                num_samples=1,
                scheduler=ASHAScheduler(max_t=8, #max([2, 4, 6, 8]),
                                        grace_period=1,
                                        reduction_factor=2
                                        ),
            ),
            run_config=RunConfig(
                                name="tune_transformers",
                                storage_path="/mnt/d/ray_spill/tune_results", # Massively protects local WSL storage
                                checkpoint_config=CheckpointConfig(
                                    num_to_keep=1, 
                                    checkpoint_score_attribute="eval_loss",
                                    checkpoint_score_order="min"
                                ),
                            ),

            
            )

#tuner_run_config.sync_config = custom_sync_config

"""
CheckpointConfig(
                    num_to_keep=1, 
                    checkpoint_score_attribute="eval_loss",
                    checkpoint_score_order="min",
                    checkpoint_at_end=False
                )
                
                
RunConfig(
                name="tune_transformers",
                #verbose=1,
                checkpoint_config=custom_checkpoint_config,
                storage_path="/mnt/d/ray_spill/tune_results",
            ),
"""

#%%

tune_results = tuner.fit()

#%%

tune_results.get_dataframe().sort_values("eval_loss")

#%% best hyperparameters

best_hyperparameters = tune_results.get_best_result()
best_trial = tune_results.get_best_result(metric="eval_loss", mode="min")
train_loop_config = best_trial.config("train_loop_config")

#%% train the model with best hyperparameters
trainer = TorchTrainer(train_loop_per_worker=train_func,
                       train_loop_config=train_loop_config,
                       scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
                       datasets={"train": ray_datasets["train"],
                                 "eval": ray_datasets["validatiion"],
                                 },
                       run_config=RunConfig(checkpoint_config=CheckpointConfig(
                           num_to_keep=1, checkpoint_score_attribute="eval_loss",
                           checkpoint_score_order="min",
                       )
                                            )
                       )


result = trainer.fit()

#%%
wandb.finish()

#%%
result.checkpoint

#%%
with result.checkpoint.as_directory() as checkpoint_dir:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    print(f"Model loaded from {checkpoint_dir}")


