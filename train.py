import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score


def clean(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


df = pd.read_csv("data/Resume.csv")
df = df[["Resume_str", "Category"]].rename(columns={"Resume_str": "Resume"})
df = df.dropna()
print("Rows before dedupe:", len(df))
df = df.drop_duplicates(subset="Resume")
print("Rows after dedupe:", len(df))

df["clean"] = df["Resume"].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["Category"],
    test_size=0.2, stratify=df["Category"], random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print(classification_report(y_test, preds))

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, preds, ax=ax, xticks_rotation=90, colorbar=False
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("Saved model.joblib, vectorizer.joblib, confusion_matrix.png")