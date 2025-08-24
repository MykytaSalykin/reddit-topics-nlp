import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import load_npz

ARTIFACTS = Path("artifacts")
CLUSTERS = ARTIFACTS / "clusters"
REPORTS = ARTIFACTS / "reports"
DATA_PROCESSED = Path("data/processed/comments_clean.csv.gz")


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
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

    # FIX: load sparse matrix
    from scipy.sparse import load_npz

    X_tfidf = load_npz(tfidf_path)

    vect: TfidfVectorizer = joblib.load(vect_path)
    umap_2d = np.load(umap_path)
    labels_df = pd.read_csv(labels_path)
    kmeans: KMeans = joblib.load(kmeans_path)
    rows_index = pd.read_csv(rows_index_path)
    df = pd.read_csv(data_path)

    return {
        "X_tfidf": X_tfidf,
        "vectorizer": vect,
        "umap_2d": umap_2d,
        "labels": labels_df,
        "kmeans": kmeans,
        "rows_index": rows_index,
        "df": df,
    }


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
            "```\n"
        )
        return

    df = art["df"]
    umap_2d = art["umap_2d"]
    labels_df = art["labels"]
    rows_index = art["rows_index"]

    merged = rows_index.merge(df, left_on="row_id", right_index=True, how="left").merge(
        labels_df, on="row_id", how="left"
    )
    merged["x"] = umap_2d[:, 0]
    merged["y"] = umap_2d[:, 1]

    st.sidebar.header("Filters")
    subs = sorted(merged["subreddit"].dropna().unique().tolist())
    sel_subs = st.sidebar.multiselect("Subreddits", subs, default=subs)

    clusters = sorted(merged["cluster"].dropna().unique().tolist())
    sel_clusters = st.sidebar.multiselect("Clusters", clusters, default=clusters)

    q = st.sidebar.text_input("Search text (contains)")

    view = merged.copy()
    if sel_subs:
        view = view[view["subreddit"].isin(sel_subs)]
    if sel_clusters:
        view = view[view["cluster"].isin(sel_clusters)]
    if q:
        view = view[view["clean_text"].str.contains(q, case=False, na=False)]

    st.subheader("UMAP scatter (colored by cluster)")
    st.write(
        "Tip: Use sidebar filters to subset points. Hover the table below to inspect rows."
    )

    import altair as alt

    chart = (
        alt.Chart(view)
        .mark_circle(size=45)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("cluster:N", legend=None),
            tooltip=["subreddit:N", "cluster:N", "body:Q", "clean_text:N"],
        )
        .interactive()
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Sample rows")
    st.dataframe(
        view[["subreddit", "cluster", "clean_text"]].head(1000),
        use_container_width=True,
        height=400,
    )


if __name__ == "__main__":
    main()
