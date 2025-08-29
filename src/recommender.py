
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_text(df: pd.DataFrame, cols=("title","short_description","tags")) -> pd.Series:
    txt = []
    for _, row in df[list(cols)].fillna("").iterrows():
        txt.append(" ".join(str(v) for v in row.values))
    return pd.Series(txt)

def build_content_matrix(text_series: pd.Series, max_features=20000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1,2))
    mat = vec.fit_transform(text_series)
    return vec, mat

def content_similarity(mat):
    return cosine_similarity(mat)

def compute_hybrid_scores(content_scores: np.ndarray, popularity: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-9)
    pop_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-9)
    return alpha * content_norm + (1 - alpha) * pop_norm

def recommend(df: pd.DataFrame, sim_matrix: np.ndarray, popularity: np.ndarray, title: str, topn: int = 5, alpha: float = 0.6):
    idx_map = {t:i for i,t in enumerate(df["title"])}
    if title not in idx_map:
        return pd.DataFrame(columns=["title","hybrid_score","popularity"])
    i = idx_map[title]
    hybrid = compute_hybrid_scores(sim_matrix[i], popularity, alpha=alpha)
    order = np.argsort(-hybrid)
    out = []
    for j in order:
        if j == i: 
            continue
        out.append([df.loc[j,"title"], float(hybrid[j]), float(popularity[j])])
        if len(out) >= topn:
            break
    return pd.DataFrame(out, columns=["title","hybrid_score","popularity"])
