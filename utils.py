import re


def clean(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()