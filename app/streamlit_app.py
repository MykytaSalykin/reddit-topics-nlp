# app/streamlit_app.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import load_npz
import altair as alt

ARTIFACTS = Path("artifacts")
CLUSTERS = ARTIFACTS / "clusters"
REPORTS = ARTIFACTS / "reports"
DATA_PROCESSED = Path("data/processed/comments_clean.csv.gz")


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


@st.cache_data(show_spinner=False)
def load_artifacts():
    missing = []

    tfidf_path = first_existing(ARTIFACTS / "tfidf.npz")
    if tfidf_path is None:
        missing.append("artifacts/tfidf.npz")

    vect_path = first_existing(
        ARTIFACTS / "tfidf_vectorizer.joblib",
        ARTIFACTS / "vectorizer.pkl",
    )
    if vect_path is None:
        missing.append(
            "artifacts/tfidf_vectorizer.joblib (or artifacts/vectorizer.pkl)"
        )

    umap_path = first_existing(ARTIFACTS / "umap_2d.npy")
    if umap_path is None:
        missing.append("artifacts/umap_2d.npy")

    labels_path = first_existing(CLUSTERS / "labels.csv.gz")
    if labels_path is None:
        missing.append("artifacts/clusters/labels.csv.gz")

    kmeans_path = first_existing(
        CLUSTERS / "kmeans.joblib",
        CLUSTERS / "model.pkl",
    )
    if kmeans_path is None:
        missing.append(
            "artifacts/clusters/kmeans.joblib (or artifacts/clusters/model.pkl)"
        )

    rows_index_path = first_existing(ARTIFACTS / "rows_index.csv.gz")
    if rows_index_path is None:
        missing.append("artifacts/rows_index.csv.gz")

    data_path = first_existing(DATA_PROCESSED)
    if data_path is None:
        missing.append(str(DATA_PROCESSED))

    if missing:
        raise FileNotFoundError("Missing artifacts:\n- " + "\n- ".join(missing))

    X_tfidf = load_npz(tfidf_path)
    vect: TfidfVectorizer = joblib.load(vect_path)
    umap_2d = np.load(umap_path)
    labels_df = pd.read_csv(
        labels_path
    )  # columns: cluster_id, label, top_terms (≈K rows)
    kmeans: KMeans = joblib.load(kmeans_path)
    rows_index = pd.read_csv(
        rows_index_path
    )  # includes row_id, subreddit, clean_text, ...
    df = pd.read_csv(data_path)  # data/processed/comments_clean.csv.gz

    return {
        "X_tfidf": X_tfidf,
        "vectorizer": vect,
        "umap_2d": umap_2d,
        "labels_df": labels_df,
        "kmeans": kmeans,
        "rows_index": rows_index,
        "df": df,
    }


def coalesce(df: pd.DataFrame, out: str, a: str, b: str):
    has_a, has_b = a in df.columns, b in df.columns
    if has_a and has_b:
        df[out] = df[a].where(df[a].notna(), df[b])
        df.drop(columns=[a, b], inplace=True)
    elif has_a:
        df.rename(columns={a: out}, inplace=True)
    elif has_b:
        df.rename(columns={b: out}, inplace=True)
    if out not in df.columns:
        df[out] = None


def main():
    st.title("Reddit Topic Explorer")
    st.caption("Interactive clustering demo: TF-IDF → UMAP → KMeans")

    try:
        art = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Failed to load artifacts:\n\n{e}")
        st.info(
            "Run these commands first:\n\n"
            "```bash\n"
            "python -m src.pipeline.clean_text\n"
            "python -m src.pipeline.features\n"
            "python -m src.pipeline.cluster\n"
            "python -m src.pipeline.report\n"
            "python -m src.pipeline.labeling\n"
            "```\n"
        )
        return

    df = art["df"]
    umap_2d = art["umap_2d"]
    labels_df = art["labels_df"]  # K rows: cluster_id, label, top_terms
    rows_index = art["rows_index"]  # N rows: row_id + metadata
    kmeans: KMeans = art["kmeans"]  # has .labels_ (N,)

    # --- per-document merge ---
    # Ensure df has row_id to match rows_index (which we built in features.py)
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})

    # Join per-row metadata
    merged = rows_index.merge(df, on="row_id", how="left")

    # Add per-document cluster assignments
    merged["cluster_id"] = kmeans.labels_.astype(int)

    # Attach readable labels per cluster
    merged = merged.merge(
        labels_df, on="cluster_id", how="left"
    )  # adds: label, top_terms
    merged.rename(columns={"label": "cluster_label"}, inplace=True)

    # UMAP coordinates for scatter
    merged["x"] = umap_2d[:, 0].astype(float)
    merged["y"] = umap_2d[:, 1].astype(float)

    # If earlier artifacts created *_x/_y duplicates, normalize them
    coalesce(merged, "subreddit", "subreddit_x", "subreddit_y")
    coalesce(merged, "submission_id", "submission_id_x", "submission_id_y")
    coalesce(merged, "score", "score_x", "score_y")
    coalesce(merged, "clean_text", "clean_text_x", "clean_text_y")

    # --- sidebar filters ---
    st.sidebar.header("Filters")

    subs = (
        sorted(merged["subreddit"].dropna().unique().tolist())
        if "subreddit" in merged.columns
        else []
    )
    sel_subs = st.sidebar.multiselect("Subreddits", subs, default=subs)

    clusters = sorted(merged["cluster_id"].dropna().unique().tolist())
    sel_clusters = st.sidebar.multiselect("Clusters (id)", clusters, default=clusters)

    q = st.sidebar.text_input("Search text (contains)")

    # Apply filters
    view = merged
    if sel_subs:
        view = view[view["subreddit"].isin(sel_subs)]
    if sel_clusters:
        view = view[view["cluster_id"].isin(sel_clusters)]
    if q:
        view = view[view["clean_text"].str.contains(q, case=False, na=False)]

    st.caption(f"Current rows: {len(view):,}")

    # --- scatter plot ---
    st.subheader("UMAP scatter (colored by cluster)")
    chart = (
        alt.Chart(view)
        .mark_circle(size=35)
        .encode(
            x=alt.X("x:Q", title="UMAP-1"),
            y=alt.Y("y:Q", title="UMAP-2"),
            color=alt.Color("cluster_id:N", title="Cluster"),
            tooltip=[
                alt.Tooltip(
                    "subreddit:N",
                    title="subreddit",
                ),
                alt.Tooltip("cluster_id:N", title="cluster"),
                alt.Tooltip("cluster_label:N", title="label"),
                alt.Tooltip("clean_text:N", title="text"),
            ],
        )
        .interactive()
        .properties(height=520)
    )
    st.altair_chart(chart, use_container_width=True)

    # --- sample table ---
    st.subheader("Sample rows")
    cols = [
        c
        for c in ["subreddit", "cluster_id", "cluster_label", "clean_text", "score"]
        if c in view.columns
    ]
    st.dataframe(view[cols].head(1000), use_container_width=True, height=420)


if __name__ == "__main__":
    main()
