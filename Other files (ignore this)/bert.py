# sentiment_analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch

# %% [1. Data Preparation] ------------------------------------------------
# Load Amazon reviews_hugging
reviews = pd.read_csv("amazon_reviews.csv")  # Replace with your data
reviews_hugging = reviews[['text', 'rating']]  # Keep only needed columns

# Convert ratings to 3 classes
def rating_to_label(rating):
    if rating <= 2: return 'negative'
    elif rating == 3: return 'neutral'
    else: return 'positive'

reviews_hugging['label'] = reviews_hugging['rating'].apply(rating_to_label)

# Encode labels
le = LabelEncoder()
reviews_hugging['encoded_label'] = le.fit_transform(reviews_hugging['label'])

# Split data
train_df, test_df = train_test_split(reviews_hugging, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# %% [2. Dataset Conversion] ----------------------------------------------
# Convert to HuggingFace datasets
train_ds = Dataset.from_pandas(train_df[['text', 'encoded_label']])
val_ds = Dataset.from_pandas(val_df[['text', 'encoded_label']])
test_ds = Dataset.from_pandas(test_df[['text', 'encoded_label']])

# %% [3. Model & Tokenizer] -----------------------------------------------
model_name = "bert-base-uncased"  # For English reviews_hugging
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={i: label for i, label in enumerate(le.classes_)}
)

# %% [4. Tokenization] ----------------------------------------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "encoded_label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "encoded_label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "encoded_label"])

# %% [5. Training Setup] --------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Reduced for stability
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True,  # Enable mixed precision
    report_to="none"
)

# %% [6. Training] --------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# Start training with crash protection
try:
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted! Saving model...")
    trainer.save_model("interrupted_model")

# %% [7. Evaluation] ------------------------------------------------------
# Final test evaluation
test_results = trainer.evaluate(test_ds)
print("\nTest Set Performance:")
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Macro F1: {test_results['eval_f1_macro']:.4f}")

# Generate predictions
preds = trainer.predict(test_ds)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

# Detailed report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=le.classes_
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
           xticklabels=le.classes_,
           yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# %% [8. Saving] ----------------------------------------------------------
# Save model and predictions
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

pd.DataFrame({
    "text": test_df["text"],
    "true_label": test_df["label"],
    "pred_label": le.inverse_transform(y_pred)
}).to_csv("predictions.csv", index=False)