import joblib

# Load the saved model
pipe = joblib.load("model.joblib")
examples = [
    "I love this new movie, it was amazing!",
    "The traffic today was horrible and exhausting.",
    "I'm not sure how I feel about this update."
]
preds = pipe.predict(examples)
print(preds)  # 0/1/2 → Negative/Neutral/Positive
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
for text, pred in zip(examples, preds):
    print(f"Text: {text}\nPredicted Sentiment: {labels[pred]}\n")
