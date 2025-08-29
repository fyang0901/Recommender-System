
import os
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path="data/processed/items.csv"):
    return pd.read_csv(path)

def build_text(df):
    return (df["title"].fillna("") + " " + df.get("short_description","").fillna("") + " " + df.get("tags","").fillna("")).tolist()

def launch_app(path="data/processed/items.csv"):
    df = load_data(path)
    texts = build_text(df)
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2))
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat)
    pop = df.get("popularity", pd.Series(np.zeros(len(df)))).to_numpy()

    def _recommend(title, alpha, topn):
        if title not in set(df["title"]):
            return pd.DataFrame(columns=["title","score","popularity"])
        i = df.index[df["title"]==title][0]
        cs = sim[i]
        cs_n = (cs - cs.min())/(cs.max()-cs.min()+1e-9)
        pop_n = (pop - pop.min())/(pop.max()-pop.min()+1e-9)
        hybrid = alpha*cs_n + (1-alpha)*pop_n
        order = np.argsort(-hybrid)
        out = []
        for j in order:
            if j==i: continue
            out.append([df.loc[j,"title"], float(hybrid[j]), float(pop[j])])
            if len(out)>=topn: break
        return pd.DataFrame(out, columns=["title","hybrid_score","popularity"])

    with gr.Blocks(title="Hybrid Recommender") as demo:
        gr.Markdown("## Hybrid Recommender Demo")
        with gr.Row():
            title = gr.Dropdown(choices=list(df["title"]), value=df["title"].iloc[0], label="Title")
            alpha = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Alpha (content vs popularity)")
            topn = gr.Slider(3, 15, value=5, step=1, label="Top N")
        btn = gr.Button("Recommend")
        out = gr.Dataframe(headers=["title","hybrid_score","popularity"], label="Recommendations")
        btn.click(fn=_recommend, inputs=[title, alpha, topn], outputs=out)
    demo.launch()

if __name__ == "__main__":
    launch_app()
