#%%
from datasets import load_dataset
import ray
import os
from ray import train, tune
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import RunConfig
import evaluate
import numpy as np
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, TrainingArguments, 
                          Trainer, pipeline,
                          PrinterCallback
                          )
import transformers
import torch
from collections import namedtuple


#%%
dataset = load_dataset("legacy-datasets/banking77")
device = "cuda"
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

#%%

evaluate.list_evaluation_modules()
# %%
metric = evaluate.load("accuracy")

#%%

metric.data
# %%
def compute_metric(eval_pred, metric_fn=metric):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric_fn.compute(predictions=preds, references=labels)
# %%

exp_eval = {"pred": [[1,2,3,4,5]], "label": [[1,2,3,3,4]]}


exp_eval.values
# %%
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           use_safetensors=True,
                                                           num_labels=77
                                                           )
# %%
dummy_inputs = model.dummy_inputs
dummy_inputs = {k: v.to(device) for k,v in dummy_inputs.items()}
# %%
model.num_labels
# %%
classifier = pipeline("text-classification",model=model, tokenizer=tokenizer)
# %%
exptest ="my card failed"

pred_trial = classifier(exptest)
# %%

input_token_ids = tokenizer(exptest,return_tensors="pt", padding="longest", max_length=128,
                            truncation=True
                            )

input_token_ids = {k: v.to(device) for k,v in input_token_ids.items()}
# %%
with torch.no_grad():
    outputs = model(**dummy_inputs)
# %%
outputs
# %%
torch.argmax(outputs.logits, dim=-1)
# %%
dummy_batch_size = dummy_inputs["input_ids"].shape[0]
# %%
mock_labels = torch.randint(low=0, high=3, size=(dummy_batch_size,))
# %%
np.argmax(mock_labels)
# %%
EvalPred = namedtuple("EvalPred", ["predictions", "label_ids"])
# %%
# %%
pred_payload = EvalPred(predictions=outputs.logits.cpu().numpy(),label_ids=mock_labels)
# %%
compute_metric(eval_pred=pred_payload, metric_fn=metric)

#%%

train_data_iter = train_data.iter_torch_batches(batch_size=32, collate_fn=tokenize_batch)

#%%

for b in train_data_iter:
    print(b)
    break
# %%
def train_model(config):
    from ray.train.huggingface.transformers import prepare_trainer
    import evaluate
    import numpy as np
    from transformers import (AutoModelForSequenceClassification, 
                            AutoTokenizer, TrainingArguments, 
                            Trainer, pipeline,
                            PrinterCallback
                            )
    
    model_name = config.get("model_name")
    dataset_name = config.get("dataset_name")
    device = config.get("device")
    batch_size = config.get("batch_size")
    metric_name = config.get("metric_name", "accuracy")
    epochs = config.get("epochs")
    dataset = load_dataset(dataset_name)
    
    name = f"trial-{dataset_name}-{model_name}-finetuned"
    
    metric = evaluate.load(metric_name)
    print("Getting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def compute_metric(eval_pred, metric=metric):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    def tokenize_batch(batch, tokenizer=tokenizer, device=device):
        text = batch["text"]
        label = batch["labels"]
        text = text.tolist() if isinstance(text, np.ndarray) else text
        token = tokenizer(text, max_length=128, truncation=True, 
                        padding="longest", 
                        return_tensors="pt"
                        )
        token["labels"] = torch.tensor(label, dtype=torch.long)
        token = {k: v.to(device) for k,v in token.items()}
        return token



    print("Getting Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                use_safetensors=True,
                                                                num_labels=77
                                                                )
    print("Creating ray data")
    train_data = ray.data.from_items(dataset["train"].to_list()).rename_columns({"label": "labels"})
    test_data = ray.data.from_items(dataset["test"].to_list()).rename_columns({"label": "labels"})
    
    max_steps_per_epoch = train_data.count() // batch_size
    max_step = max_steps_per_epoch * epochs
    print(f"creating iterable torch batches")
    train_data_iterable = train_data.iter_torch_batches(batch_size=batch_size, collate_fn=tokenize_batch)
    test_data_iterable = test_data.iter_torch_batches(batch_size=batch_size, collate_fn=tokenize_batch)
    
    print("Configuring TrainingArguments and Trainer")
    args = TrainingArguments(num_train_epochs=epochs,
                             per_device_train_batch_size=batch_size,
                             per_device_eval_batch_size=batch_size,
                             learning_rate=config.get("learning_rate"),
                             lr_scheduler_type=config.get("lr_scheduler_type"),
                             optim=config.get("optim"),
                             eval_strategy="epoch",
                             save_strategy="best",
                             metric_for_best_model="accuracy",
                             max_steps=max_step,
                             #enable_jit_checkpoint=True,
                             )
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_data_iterable,
                      eval_dataset=test_data_iterable,
                      compute_metrics=compute_metric
                      )
    print("add callbacks")
    #trainer.add_callback(RayTrainReportCallback())
    trainer.add_callback(PrinterCallback())
    print("prepare trainer")
    trainer = prepare_trainer(trainer)
    
    print(" training...")
    trainer.train()
    print("evaluating ...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")
    
    ray.tune.report(metrics=eval_metrics)
    

#%%
config = {"batch_size": tune.choice([4,8,16]),
          "learning_rate": tune.loguniform(1e-5, 1e-1),
          "lr_scheduler_type": tune.choice(categories=["linear", "cosine", "constant", "constant_with_warmup"]),
          "optim": tune.choice(categories=["adamw_torch", "sgd", "adafactor"]),
          "dataset_name": "legacy-datasets/banking77",
          "model_name": "microsoft/deberta-v3-small",
          "device": "cuda",
          "epochs":10,
          "cpu": 2,
          "gpu": 0.33
          }

#%%
def main(config):
    storage_path = "/mnt/d/distributed_work/cluster_storage/ray-results"
    os.makedirs(storage_path, exist_ok=True)
    print("Configuring scheduler")
    scheduler = ASHAScheduler(time_attr="training_iteration",
                              max_t=config.get("epochs"),
                              grace_period=2,
                              reduction_factor=2
                              )
    print("Configuring Tuner")
    tuner = tune.Tuner(trainable=tune.with_resources(trainable=tune.with_parameters(train_model),
                                           resources={"cpu": config.get("cpu"),
                                                      "gpu": config.get("gpu")
                                                      }
                                           ),
                       tune_config=tune.TuneConfig(metric="eval_loss",
                                                   mode="min",
                                                   scheduler=scheduler,
                                                   num_samples=1
                                                   ),
                       run_config=RunConfig(name=f"{config.get('dataset_name')}_tune_demo",
                                            storage_path=storage_path
                                            ),
                       param_space=config,
        
                    )
    print(f"fitting tuner")
    results = tuner.fit()
    print(f"successfully fitted tuner")
    best_result = results.get_best_result(metric="loss", mode="min")
    print(f"Best Validation loss: {best_result.metrics['loss']}")
    print(f"Best validation acczracy: {best_result.metrics['accuracy']}")
    
    return results, best_result


# %%
ray.init(
    ignore_reinit_error=True,
    _temp_dir="/tmp/ray_native",
    
    
    # 2. OFFLOAD THE HEAVY WEIGHTS: Large arrays spill to your D: drive
    _system_config={
        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": ["/mnt/d/ray_spill"]}}'
    },
    
)
res = main(config)
# %%
