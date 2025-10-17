import pandas as pd
import re
import emoji
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Robustly load & inspect the messy CSV
path = "sentimentdataset.csv"
raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
for i in range(10):
    print(f"Row {i}:", raw.iloc[i].tolist())

# Auto-detect header row if it contains 'text' and 'sentiment'
header_row = None
for i in range(10):
    row = [str(x).lower() for x in raw.iloc[i]]
    if any("text" in c for c in row) and any("sentiment" in c for c in row):
        header_row = i
        break

df = pd.read_csv(path, header=header_row) if header_row is not None else pd.read_csv(path)
print("Columns:", df.columns.tolist())

# 2. Extract only Text and Sentiment columns (normalize names)
df.columns = [c.strip() for c in df.columns.astype(str)]
text_col = next((c for c in df.columns if "text" in c.lower() or "post" in c.lower()), None)
sent_col = next((c for c in df.columns if "sent" in c.lower() or "label" in c.lower()), None)

df = df[[text_col, sent_col]].rename(columns={text_col: "text", sent_col: "sentiment"})
df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

# 3. Clean & preprocess text
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(t):
    t = str(t)
    t = emoji.demojize(t)
    t = re.sub(r'http\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#', '', t)
    t = re.sub(r'[^A-Za-z0-9\s:]', ' ', t)
    t = t.lower().strip()
    doc = nlp(t)
    tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.lemma_.strip()]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# 4. Quick EDA
print(df['sentiment'].value_counts())
df['len'] = df['clean_text'].str.split().apply(len)
print(df['len'].describe())

# 5. Label normalization & imbalance handling
df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
mapping = {'Positive':'Positive','Negative':'Negative','Neutral':'Neutral'}
df = df[df['sentiment'].isin(mapping.keys())]
label2id = {'Negative':0, 'Neutral':1, 'Positive':2}
df['label'] = df['sentiment'].map(label2id)

# 6. Baseline classical approach
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000),
                     LogisticRegression(max_iter=1000, class_weight='balanced'))
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Negative','Neutral','Positive']))

# 10. Save your model & preprocessing
joblib.dump(pipe, 'model.joblib')
print('Model saved as model.joblib')
