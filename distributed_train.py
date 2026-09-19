
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
                             save_strategy="epoch",
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
    
    output_dir = os.path.join(ray.train.get_context().get_trial_dir(), "hf_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    ray.train.report({"eval_loss": eval_metrics.get("eval_loss"),
                      "eval_accuracy": eval_metrics.get("eval_accuracy")
                      },
                     checkpoint=Checkpoint.from_directory(output_dir)
                     )
    

# %% setup torchtrainer
trainer = TorchTrainer(train_func,
                       scaling_config=ScalingConfig(num_workers=num_workers,
                                                    use_gpu=use_gpu,
                                                    ),
                       datasets = {"train": ray_datasets["train"],
                                   "eval": ray_datasets["validation"]
                                   },
                       run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1,
                                                                               checkpoint_score_attribute="eval_boss",
                                                                               checkpoint_score_order="min",
                                                                               )
                                            )
                       )

#%% tune hyperparameters
tuner = Tuner(trainer,
              param_space={"train_loop_config": {
                  "learning_rate": tune.grid_search([2e-5, 2e-4, 2e-3, 2e-2]),
                  "epochs": tune.choice([2, 4, 6, 8]),
                  "batch_size": tune.choice([16, 32, 64, 128]),
                  "weight_decay": tune.grid_search([0.0, 0.01, 0.1, 0.001])
              }
            },
            tune_config=tune.TuneConfig(
                metric="eval_loss",
                mode="min",
                num_samples=1,
                scheduler=ASHAScheduler(max_t=max([2, 4, 6, 8]),
                                        grace_period=1,
                                        reduction_factor=2
                                        ),
            ),
            run_config=RunConfig(
                name="tune_transformers",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1, 
                    checkpoint_score_attribute="eval_loss",
                    checkpoint_score_order="min",
                )
            ),
            )


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


