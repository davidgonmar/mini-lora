from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import Trainer
from mini_lora import lorafy, count_params, count_trainable_params

dataset = load_dataset("yelp_review_full")


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5
)

# only lorafy query and value weights, so lamdba x: "query" in x.name or "value" in x.name
model = lorafy(model, r=1, should_lorafy=lambda mod, name: "query" in name or "value" in name, should_freeze=lambda x, name: "classifier" not in name)
params, trainable_params = count_params(model), count_trainable_params(model)
print(f"Total parameters: {params}, Trainable parameters: {trainable_params}")
print("Percentage of trainable parameters: ", trainable_params / params * 100, "%")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=3, per_device_train_batch_size=4, per_device_eval_batch_size=4
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
