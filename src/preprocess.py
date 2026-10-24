"""Tokenization and bag-of-words construction."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


def tokenize(doc: str):
    """Lowercase, split on word characters and hyphens, and drop English stop words."""
    tokens = re.findall(r"\b[\w\-]+\b", doc.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def docs_to_bow(docs, min_df=5, max_df=0.9):
    """Build a sparse document-term count matrix.

    Terms must appear in at least ``min_df`` documents and in at most a
    ``max_df`` fraction of documents. Returns the matrix and the fitted
    ``CountVectorizer``.
    """
    cv = CountVectorizer(tokenizer=tokenize,
                         min_df=min_df, max_df=max_df)
    X = cv.fit_transform(docs)
    print(f"Created sparse count matrix of shape {X.shape} "
          f"({X.nnz} non-zeros).")
    return X, cv
