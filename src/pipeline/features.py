# src/pipeline/features.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

DATA_IN = Path("data/processed/comments_clean.csv.gz")
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)


def build_tfidf(
    df: pd.DataFrame,
    min_df: int = 5,
    max_df: float = 0.6,
    max_features: int = 50000,
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    # Use 1–2 grams with sublinear tf to better separate short comments
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        norm="l2",
        sublinear_tf=True,
    )
    X = vec.fit_transform(df["clean_text"].astype(str).values)
    return X, vec


def reduce_dimensionality(
    X: sparse.csr_matrix,
    n_components_svd: int = 300,
    n_components_umap: int = 2,
    random_state: int = 42,
) -> dict:
    out: dict[str, np.ndarray] = {}

    # TruncatedSVD for dense semantic features (kept for downstream tasks)
    if n_components_svd and n_components_svd < X.shape[1]:
        svd = TruncatedSVD(n_components=n_components_svd, random_state=random_state)
        X_svd = svd.fit_transform(X)
        joblib.dump(svd, ART / "svd.joblib")
        out["svd"] = X_svd

    # UMAP for 2D visualization (on SVD output if available, else TF-IDF)
    base = out.get("svd", X)
    if isinstance(base, sparse.csr_matrix):
        base = base.toarray()
    um = umap.UMAP(
        n_components=n_components_umap,
        random_state=random_state,
        n_neighbors=30,
        min_dist=0.1,
    )
    X_umap = um.fit_transform(base)
    joblib.dump(um, ART / "umap.joblib")
    out["umap"] = X_umap

    return out


def main():
    # Ensure a clean 0..N-1 index so row_id aligns with TF-IDF rows
    df = pd.read_csv(DATA_IN, compression="gzip")
    df = df.reset_index(drop=True)

    # Build TF-IDF and persist
    X, vec = build_tfidf(df)
    sparse.save_npz(ART / "tfidf.npz", X)
    joblib.dump(vec, ART / "tfidf_vectorizer.joblib")
    with open(ART / "tfidf_meta.json", "w") as f:
        json.dump({"n_docs": X.shape[0], "n_features": X.shape[1]}, f, indent=2)

    # Dimensionality reduction artifacts
    embeds = reduce_dimensionality(X)
    np.save(ART / "umap_2d.npy", embeds["umap"])
    if "svd" in embeds:
        np.save(ART / "svd_300.npy", embeds["svd"])

    # Row mapping used by the Streamlit app; include explicit row_id
    rows_index = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=int),
            "subreddit": df.get("subreddit"),
            "submission_id": df.get("submission_id"),
            "score": df.get("score"),
            "clean_text": df.get("clean_text"),
        }
    )
    rows_index.to_csv(ART / "rows_index.csv.gz", index=False, compression="gzip")

    print(f"TF-IDF: {X.shape}. Saved artifacts to {ART}")


if __name__ == "__main__":
    main()
