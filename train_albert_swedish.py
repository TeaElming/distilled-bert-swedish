import evaluate
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

##########################################################
# 1. Load Each Dataset
##########################################################
absabank_dataset = load_from_disk("./data/processed_absabank_dataset")
trustpilot_dataset = load_from_disk("./data/processed_trustpilot_dataset")
twitter_dataset = load_from_disk("./data/processed_twitter_dataset")

##########################################################
# 2. Merge Splits
##########################################################
# Concatenate train sets, val sets, test sets
train_all: Dataset = absabank_dataset["train"] \
    .concatenate(trustpilot_dataset["train"]) \
    .concatenate(twitter_dataset["train"])

val_all: Dataset = absabank_dataset["validation"] \
    .concatenate(trustpilot_dataset["validation"]) \
    .concatenate(twitter_dataset["validation"])

test_all: Dataset = absabank_dataset["test"] \
    .concatenate(trustpilot_dataset["test"]) \
    .concatenate(twitter_dataset["test"])

# Build a single DatasetDict
dataset = DatasetDict({
    "train": train_all,
    "validation": val_all,
    "test": test_all
})

##########################################################
# 3. Load Tokenizer
##########################################################
ALBERT_MODEL_NAME = "KBLab/albert-base-swedish-cased-alpha"
tokenizer = AutoTokenizer.from_pretrained(ALBERT_MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Tokenize all splits
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Rename 'label' -> 'labels' if needed
for split in ["train", "validation", "test"]:
    if "label" in tokenized_dataset[split].column_names:
        tokenized_dataset[split] = tokenized_dataset[split].rename_column("label", "labels")

# Set PyTorch format
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

##########################################################
# 4. Load ALBERT Model
##########################################################
model = AutoModelForSequenceClassification.from_pretrained(
    ALBERT_MODEL_NAME,
    num_labels=3  # negative, neutral, positive
)

##########################################################
# 5. Training Setup
##########################################################
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(dim=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./albert-sentiment-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

##########################################################
# 6. Train & Evaluate
##########################################################
trainer.train()

test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test Accuracy:", test_results["eval_accuracy"])

##########################################################
# 7. Save the Final Model
##########################################################
trainer.save_model("./albert-sentiment-finetuned")
tokenizer.save_pretrained("./albert-sentiment-finetuned")

print("All done - trained on all datasets in one go!")
