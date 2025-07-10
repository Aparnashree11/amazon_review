from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt

# Load test data
df = pd.read_csv("data/cleaned_data.csv").sample(500)
texts = df["content"].tolist()
labels = df["label"].tolist()

# Load model
model_path = "./models/distilbert-final/"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Predict
predictions = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    predictions.append(pred)

# Confusion matrix
cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(labels, predictions, target_names=["Negative", "Positive"]))