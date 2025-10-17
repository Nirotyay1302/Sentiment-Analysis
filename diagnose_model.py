import joblib
import pandas as pd
import re

print('Loading model...')
pipe = joblib.load('model.joblib')
print('Model type:', type(pipe))
cls = None
if hasattr(pipe, 'classes_'):
    cls = list(pipe.classes_)
elif hasattr(pipe, 'named_steps'):
    last = list(pipe.named_steps.values())[-1]
    cls = getattr(last, 'classes_', None)
print('Model classes:', cls)

# Load sample data
print('\nReading sample CSV (sentimentdataset.csv)...')
df = pd.read_csv('sentimentdataset.csv')
textcol = next((c for c in df.columns if 'text' in c.lower()), df.columns[0])
print('Detected text column:', textcol)

texts = df[textcol].astype(str).head(10).tolist()
cleaned = [re.sub(r'[^A-Za-z0-9\s.,!?]', '', t).strip() for t in texts]
print('\nSample texts (cleaned):')
for i, t in enumerate(cleaned, 1):
    print(i, '-', t[:200])

print('\nRunning model.predict on sample texts...')
try:
    preds = pipe.predict(cleaned)
    print('Raw predictions:', preds)
except Exception as e:
    print('Predict error:', e)

# If model supports predict_proba, show top probs
try:
    if hasattr(pipe, 'predict_proba'):
        probs = pipe.predict_proba(cleaned)
        print('\nPredict_proba sample (first row):', probs[0])
except Exception as e:
    print('Predict_proba error:', e)

print('\nDone.')
