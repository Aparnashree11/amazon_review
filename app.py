import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./models/distilbert-final/"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Load cleaned test data
@st.cache_data
def load_test_data():
    test_df = pd.read_csv("./data/cleaned_data.csv").sample(500)
    return test_df

test_df = load_test_data()
texts = test_df["content"].tolist()
true_labels = test_df["label"].tolist()

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Predict on test set for accuracy
@st.cache_data
def get_predictions(texts):
    return [predict_sentiment(text) for text in texts]

predictions = get_predictions(texts)

# Compute metrics
acc = accuracy_score(true_labels, predictions)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

# Streamlit UI
st.title("Amazon Review Sentiment Analyzer")
st.write("A DistilBERT model fine-tuned on Amazon reviews")

# Show model metrics
st.subheader("Model Performance on Test Set (500 samples)")
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**Precision:** {prec:.4f}")
st.write(f"**Recall:** {rec:.4f}")
st.write(f"**F1-score:** {f1:.4f}")

# Display confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
fig, ax = plt.subplots(figsize=(4, 4))
disp.plot(ax=ax, cmap="Blues", values_format="d")
st.pyplot(fig)

# Misclassified examples
st.subheader("Sample Misclassified Reviews")
misclassified = [
    (text, true, pred)
    for text, true, pred in zip(texts, true_labels, predictions)
    if true != pred
]

for text, true, pred in misclassified[:3]:
    st.write(f"> *{text[:300]}...*")
    st.write(f"**Actual:** {'Positive' if true == 1 else 'Negative'}, **Predicted:** {'Positive' if pred == 1 else 'Negative'}")
    st.write("---")

# User input
st.subheader("Predict Sentiment for Your Review")
user_input = st.text_area("Enter your Amazon product review here:")

if st.button("Predict Sentiment"):
    pred = predict_sentiment(user_input)
    sentiment = "Positive" if pred == 1 else "Negative"
    st.success(f"Predicted Sentiment: **{sentiment}**")