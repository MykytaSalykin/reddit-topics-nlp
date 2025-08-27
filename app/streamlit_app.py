# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import load_npz
import altair as alt
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image

# --- Paths ---
ARTIFACTS = Path("artifacts")
CLUSTERS = ARTIFACTS / "clusters"
REPORTS = ARTIFACTS / "reports"
DATA_PROCESSED = Path("data/processed/comments_clean.csv.gz")

st.set_page_config(page_title="Reddit Topic Explorer", layout="wide")


# --- Utils ---
def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None


def sample_points(
    df: pd.DataFrame, n: int = 5000, group_cols=("subreddit", "cluster")
) -> pd.DataFrame:
    """Sample up to n rows keeping diversity across groups."""
    if len(df) <= n:
        return df
    out = []
    groups = df.groupby(list(group_cols), dropna=False)
    per_group = max(1, n // max(1, len(groups)))
    for _, g in groups:
        k = min(len(g), per_group)
        out.append(g.sample(n=k, random_state=42))
    big = pd.concat(out, ignore_index=True)
    if len(big) > n:
        big = big.sample(n=n, random_state=42)
    return big


def coalesce(df: pd.DataFrame, out: str, a: str, b: str) -> None:
    """Make a single column out of a/b (left-pref), drop suffixes."""
    has_a, has_b = a in df.columns, b in df.columns
    if has_a and has_b:
        df[out] = df[a].where(df[a].notna(), df[b])
        df.drop(columns=[a, b], inplace=True)
    elif has_a:
        df.rename(columns={a: out}, inplace=True)
    elif has_b:
        df.rename(columns={b: out}, inplace=True)
    if out not in df.columns:
        df[out] = pd.NA


def make_wordcloud(texts: list[str]) -> Image.Image:
    """Make a word cloud image from texts."""
    text = " ".join(t for t in texts if isinstance(t, str))
    if not text.strip():
        return Image.new("RGB", (800, 400), color=(255, 255, 255))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


# --- Load artifacts ---
@st.cache_data(show_spinner=False)
def load_artifacts():
    tfidf_path = first_existing(ARTIFACTS / "tfidf.npz")
    vect_path = first_existing(
        ARTIFACTS / "tfidf_vectorizer.joblib", ARTIFACTS / "vectorizer.pkl"
    )
    umap_path = first_existing(ARTIFACTS / "umap_2d.npy")
    kmeans_path = first_existing(CLUSTERS / "kmeans.joblib", CLUSTERS / "model.pkl")
    rows_index_path = first_existing(ARTIFACTS / "rows_index.csv.gz")
    data_path = first_existing(DATA_PROCESSED)
    labels_map_path = first_existing(CLUSTERS / "labels.csv.gz")
    svd_model_path = first_existing(ARTIFACTS / "svd.joblib")
    svd_npy_path = first_existing(ARTIFACTS / "svd_300.npy")
    toxicity_path = first_existing(REPORTS / "toxicity.csv.gz")

    missing = []
    for p, desc in [
        (tfidf_path, "artifacts/tfidf.npz"),
        (vect_path, "vectorizer"),
        (umap_path, "artifacts/umap_2d.npy"),
        (kmeans_path, "artifacts/clusters/kmeans.joblib"),
        (rows_index_path, "artifacts/rows_index.csv.gz"),
        (data_path, str(DATA_PROCESSED)),
        (labels_map_path, "artifacts/clusters/labels.csv.gz"),
    ]:
        if p is None:
            missing.append(desc)
    if missing:
        raise FileNotFoundError("Missing artifacts:\n- " + "\n- ".join(missing))

    X_tfidf = load_npz(tfidf_path)
    vect: TfidfVectorizer = joblib.load(vect_path)
    umap_2d = np.load(umap_path)
    kmeans: KMeans = joblib.load(kmeans_path)
    rows_index = pd.read_csv(rows_index_path)
    df_raw = pd.read_csv(data_path)
    labels_map = pd.read_csv(labels_map_path)
    tox_df = pd.read_csv(toxicity_path) if toxicity_path else None
    svd_model = joblib.load(svd_model_path) if svd_model_path else None
    X_svd = np.load(svd_npy_path) if svd_npy_path else None

    return {
        "X_tfidf": X_tfidf,
        "vectorizer": vect,
        "umap_2d": umap_2d,
        "kmeans": kmeans,
        "rows_index": rows_index,
        "df_raw": df_raw,
        "labels_map": labels_map,
        "tox": tox_df,
        "svd_model": svd_model,
        "X_svd": X_svd,
    }


def pick_features_for_kmeans(X_tfidf, X_svd, svd_model, kmeans: KMeans):
    """Pick a feature matrix whose dim matches KMeans centers."""
    need_dim = kmeans.cluster_centers_.shape[1]
    if X_svd is not None and X_svd.shape[1] == need_dim:
        return X_svd
    if svd_model is not None and getattr(svd_model, "n_components", None) == need_dim:
        return svd_model.transform(X_tfidf)
    if X_tfidf.shape[1] == need_dim:
        return X_tfidf
    if svd_model is not None:
        X_try = svd_model.transform(X_tfidf)
        if X_try.shape[1] == need_dim:
            return X_try
    raise ValueError(
        f"Cannot match KMeans feature dim={need_dim} with available matrices."
    )


# --- Main ---
def main():
    st.title("Reddit Topic Explorer")
    st.caption("TF-IDF → UMAP → KMeans + Toxicity and Sentiment")

    try:
        art = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Failed to load artifacts:\n\n{e}")
        return

    df_raw = art["df_raw"].reset_index().rename(columns={"index": "row_id"})
    rows_index = art["rows_index"]
    umap_2d = art["umap_2d"]
    kmeans: KMeans = art["kmeans"]
    labels_map = art["labels_map"]
    tox_df = art["tox"]

    X_feat = pick_features_for_kmeans(
        art["X_tfidf"], art["X_svd"], art["svd_model"], kmeans
    )
    clusters = kmeans.predict(X_feat)
    cluster_df = pd.DataFrame({"row_id": np.arange(len(clusters)), "cluster": clusters})

    # Keep only columns we need from df_raw
    text_cols = ["row_id"] + [
        c for c in ["clean_text", "score", "subreddit"] if c in df_raw.columns
    ]
    df_text = df_raw[text_cols].copy()

    # Merge rows_index + df_text (may produce *_x/_y)
    merged = rows_index.merge(df_text, on="row_id", how="left")

    # Normalize overlapping columns back to single names
    coalesce(merged, "subreddit", "subreddit_x", "subreddit_y")
    coalesce(merged, "score", "score_x", "score_y")
    coalesce(merged, "clean_text", "clean_text_x", "clean_text_y")

    # Attach clusters and labels
    merged = merged.merge(cluster_df, on="row_id", how="left")
    if "cluster_id" in labels_map.columns:
        labels_map = labels_map.rename(columns={"cluster_id": "cluster"})
    merged = merged.merge(labels_map, on="cluster", how="left")

    # Fallbacks
    if "subreddit" not in merged.columns:
        merged["subreddit"] = pd.NA
    if "clean_text" not in merged.columns and "body" in df_raw.columns:
        merged["clean_text"] = df_raw["body"]
    if "label" not in merged.columns:
        merged["label"] = pd.NA
    if "score" not in merged.columns:
        merged["score"] = 0

    # UMAP coords
    merged["x"] = umap_2d[:, 0]
    merged["y"] = umap_2d[:, 1]

    tab_overview, tab_explorer, tab_toxicity = st.tabs(
        ["📊 Clusters Overview", "🔎 Explorer", "🧪 Toxicity and Sentiment"]
    )

    # --- Cluster Overview ---
    with tab_overview:
        st.subheader("Cluster sizes")
        sizes = merged.groupby("cluster", dropna=True).size().reset_index(name="count")
        st.bar_chart(sizes.set_index("cluster")["count"])

        st.subheader("Top clusters word clouds")
        top_clusters = (
            sizes.sort_values("count", ascending=False)["cluster"].head(6).tolist()
        )
        cols = st.columns(3)
        for i, cl in enumerate(top_clusters):
            col = cols[i % 3]
            sample_texts = (
                merged.loc[merged["cluster"] == cl, "clean_text"]
                .dropna()
                .sample(min(500, (merged["cluster"] == cl).sum()), random_state=42)
                .tolist()
            )
            if sample_texts:
                img = make_wordcloud(sample_texts)
                with col:
                    st.image(
                        img,
                        caption=f"Cluster {int(cl)} — word cloud",
                        use_container_width=True,
                    )

    # --- Explorer ---
    with tab_explorer:
        st.subheader("Interactive UMAP scatter")
        subs = sorted(
            [
                s
                for s in merged["subreddit"].dropna().astype(str).unique().tolist()
                if s and s.strip().lower() != "nan"
            ]
        )
        sel_subs = st.multiselect("Subreddits", subs, default=subs)
        clusters_list = sorted(merged["cluster"].dropna().unique().tolist())
        sel_clusters = st.multiselect("Clusters", clusters_list, default=clusters_list)
        q = st.text_input("Search text (contains)", "")

        view = merged.copy()
        if sel_subs:
            view = view[view["subreddit"].isin(sel_subs)]
        if sel_clusters:
            view = view[view["cluster"].isin(sel_clusters)]
        if q:
            view = view[
                view["clean_text"].astype(str).str.contains(q, case=False, na=False)
            ]

        view_sampled = sample_points(view, n=5000)
        st.caption(f"Showing {len(view_sampled):,} points out of {len(view):,}")

        scatter = (
            alt.Chart(view_sampled)
            .mark_circle(size=45)
            .encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("cluster:N", legend=None),
                tooltip=[
                    "subreddit:N",
                    "cluster:N",
                    "label:N",
                    "score:Q",
                    "clean_text:N",
                ],
            )
            .interactive()
            .properties(height=550)
        )
        st.altair_chart(scatter, use_container_width=True)

        st.subheader("Sample rows")
        cols_exist = [
            c
            for c in ["subreddit", "cluster", "label", "score", "clean_text"]
            if c in view.columns
        ]
        st.dataframe(view[cols_exist].head(1000), use_container_width=True, height=400)

    # --- Toxicity (inside the app only) ---
    with tab_toxicity:
        st.subheader("Toxicity and Sentiment")
        if tox_df is None:
            st.info("No toxicity report found. Run: `python -m src.pipeline.toxicity`.")
        else:
            # Build a safe meta and ensure the column is named exactly 'subreddit'
            if "subreddit" in merged.columns:
                meta = merged[["row_id", "subreddit"]].drop_duplicates()
            else:
                meta = merged[["row_id"]].copy()
                meta["subreddit"] = pd.NA

            tox_join = tox_df.merge(meta, on="row_id", how="left")

            # If for any reason merge created suffixed names, normalize them
            if "subreddit" not in tox_join.columns:
                for cand in ("subreddit_x", "subreddit_y", "subreddit_meta"):
                    if cand in tox_join.columns:
                        tox_join.rename(columns={cand: "subreddit"}, inplace=True)
                        break
            if "subreddit" not in tox_join.columns:
                tox_join["subreddit"] = pd.NA  # final fallback

            subs_t = sorted(
                [
                    s
                    for s in tox_join["subreddit"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                    if s and s.strip().lower() != "nan"
                ]
            )
            sel_sub = st.selectbox("Subreddit", ["(all)"] + subs_t, index=0)
            v = tox_join.copy()
            if sel_sub != "(all)":
                v = v[v["subreddit"] == sel_sub]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows", f"{len(v):,}")
            with c2:
                st.metric("Mean toxicity", f"{v['toxicity'].mean():.3f}")
            with c3:
                st.metric("Median toxicity", f"{v['toxicity'].median():.3f}")
            with c4:
                pos_share = (v["sentiment"] == "positive").mean() if len(v) else 0.0
                st.metric("Positive share", f"{pos_share:.0%}")

            for p in [
                REPORTS / "toxicity_distribution.png",
                REPORTS / "sentiment_distribution.png",
                REPORTS / "toxicity_by_subreddit.png",
            ]:
                if p.exists():
                    st.image(str(p), use_container_width=True)

            st.subheader("Sample rows")
            show_cols = [
                c
                for c in ["row_id", "subreddit", "toxicity", "sentiment"]
                if c in v.columns
            ]
            st.dataframe(v[show_cols].head(1000), use_container_width=True, height=400)


if __name__ == "__main__":
    main()
