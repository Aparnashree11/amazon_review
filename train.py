import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load CSV file using pandas
df = pd.read_csv("data/cleaned_data.csv")

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train/test (80/20 split)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split['train']
test_dataset = split['test']

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['content'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization to datasets
train_tokenized = train_dataset.map(tokenize, batched=True)
test_tokenized = test_dataset.map(tokenize, batched=True)

# Set PyTorch format columns
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/distilbert-local",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=2,
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save final model and tokenizer to a clean directory
final_model_path = "./models/distilbert-final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Evaluate
metrics = trainer.evaluate()
print(metrics)

