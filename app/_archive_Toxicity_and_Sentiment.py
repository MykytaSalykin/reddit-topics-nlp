# app/pages/2_Toxicity_and_Sentiment.py
from __future__ import annotations
from pathlib import Path
import io

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Toxicity & Sentiment", layout="wide")

REPORTS = Path("artifacts/reports")
CSV_PATH = REPORTS / "toxicity.csv.gz"

st.title("🧪 Toxicity & Sentiment")

# --- Load data ---
if not CSV_PATH.exists():
    st.warning(
        "Dataset with toxicity/sentiment not found. "
        "Run:\n\n"
        "```bash\npython -m src.pipeline.toxicity && python -m src.pipeline.plots\n```",
        icon="⚠️",
    )
    st.stop()

df = pd.read_csv(CSV_PATH, compression="gzip")

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    # Optional filters if columns exist
    if "subreddit" in df.columns:
        subs = ["(all)"] + sorted(df["subreddit"].dropna().unique().tolist())[:200]
        sub_choice = st.selectbox("Subreddit", subs, index=0)
        if sub_choice != "(all)":
            df = df[df["subreddit"] == sub_choice]

    if "cluster_id" in df.columns:
        clusters = ["(all)"] + sorted(df["cluster_id"].dropna().unique().tolist())
        cl_choice = st.selectbox("Cluster", clusters, index=0)
        if cl_choice != "(all)":
            df = df[df["cluster_id"] == cl_choice]

    st.caption(f"Current rows: **{len(df):,}**")

# --- KPI header ---
c1, c2, c3, c4 = st.columns(4)
if len(df):
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Mean toxicity", f"{df['toxicity'].mean():.3f}")
    c3.metric("Median toxicity", f"{df['toxicity'].median():.3f}")
    if "sentiment" in df.columns:
        share_pos = (df["sentiment"] == "positive").mean()
        c4.metric("Positive share", f"{share_pos:.0%}")
else:
    st.info("No rows after filters.")
    st.stop()


# --- Helper to show existing PNG or render inline ---
def show_png_or_plot(png_name: str, plot_fn):
    png_path = REPORTS / png_name
    if png_path.exists():
        st.image(str(png_path), caption=png_name, use_container_width=True)
    else:
        fig = plot_fn()
        st.pyplot(fig)


# --- Row 1: distributions ---
st.subheader("Distributions")
colA, colB = st.columns(2)

with colA:

    def plot_tox_hist():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["toxicity"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Toxicity Scores")
        ax.set_xlabel("toxicity")
        ax.set_ylabel("count")
        plt.tight_layout()
        return fig

    show_png_or_plot("toxicity_distribution.png", plot_tox_hist)

with colB:
    if "sentiment" in df.columns:

        def plot_sent_counts():
            fig, ax = plt.subplots(figsize=(7, 4))
            order = ["negative", "neutral", "positive"]
            present = [o for o in order if o in df["sentiment"].unique()]
            sns.countplot(x="sentiment", data=df, order=present, ax=ax)
            ax.set_title("Sentiment Distribution")
            ax.set_xlabel("sentiment")
            ax.set_ylabel("count")
            plt.tight_layout()
            return fig

        show_png_or_plot("sentiment_distribution.png", plot_sent_counts)

# --- Row 2: by subgroup ---
st.subheader("Breakdowns")

if "subreddit" in df.columns:

    def plot_by_subreddit():
        # limit categories for readability
        top = df["subreddit"].value_counts().head(15).index
        d = df[df["subreddit"].isin(top)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x="subreddit", y="toxicity", data=d, ax=ax)
        ax.set_title("Toxicity by Subreddit (top 15 by count)")
        ax.set_xlabel("subreddit")
        ax.set_ylabel("toxicity")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig

    show_png_or_plot("toxicity_by_subreddit.png", plot_by_subreddit)

if "cluster_id" in df.columns:

    def plot_by_cluster():
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.violinplot(
            x="cluster_id",
            y="toxicity",
            data=df,
            scale="width",
            inner="quartile",
            ax=ax,
        )
        ax.set_title("Toxicity by Cluster")
        ax.set_xlabel("cluster_id")
        ax.set_ylabel("toxicity")
        plt.tight_layout()
        return fig

    show_png_or_plot("toxicity_by_cluster.png", plot_by_cluster)

# --- Table preview + download ---
st.subheader("Sample rows")
preview_cols = [
    c
    for c in ["clean_text", "toxicity", "sentiment", "cluster_id", "subreddit"]
    if c in df.columns
]
st.dataframe(df[preview_cols].head(100), use_container_width=True)

# Download filtered slice
buf = io.BytesIO()
df.to_parquet(buf, index=False)
st.download_button(
    "⬇️ Download filtered (Parquet)",
    data=buf.getvalue(),
    file_name="toxicity_filtered.parquet",
    mime="application/octet-stream",
)
