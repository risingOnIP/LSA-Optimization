import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
def tokenize(doc: str):
    tokens = re.findall(r"\b[\w\-]+\b", doc.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

def docs_to_bow(docs, min_df=5, max_df=0.9):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(tokenizer=tokenize,
                         min_df=min_df, max_df=max_df)
    X = cv.fit_transform(docs)
    print(f"Created sparse count matrix of shape {X.shape} "
          f"({X.nnz} non-zeros).")
    return X, cv
