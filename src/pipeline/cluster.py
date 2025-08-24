# src/pipeline/cluster.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ART = Path("artifacts")
OUT = ART / "clusters"
OUT.mkdir(parents=True, exist_ok=True)


def choose_k(X, k_grid=(5, 8, 10, 12, 15, 20), sample=20000, random_state=42):
    # silhouette on a sample for speed
    if isinstance(X, sparse.csr_matrix):
        X_eval = X
        if X_eval.shape[0] > sample:
            idx = np.random.RandomState(random_state).choice(
                X_eval.shape[0], size=sample, replace=False
            )
            X_eval = X_eval[idx]
    else:
        X_eval = X
        if X_eval.shape[0] > sample:
            idx = np.random.RandomState(random_state).choice(
                X.shape[0], size=sample, replace=False
            )
            X_eval = X_eval[idx]
    scores = {}
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X_eval)
        try:
            s = silhouette_score(X_eval, labels)
        except Exception:
            s = float("nan")
        scores[k] = s
        print(f"k={k:>3} | silhouette={s:.4f}")
    best_k = max(
        (k for k in scores if not np.isnan(scores[k])), key=lambda t: scores[t]
    )
    return best_k, scores


def main():
    X = sparse.load_npz(ART / "tfidf.npz")
    rows = pd.read_csv(ART / "rows_index.csv.gz", compression="gzip")

    best_k, scores = choose_k(X)
    with open(OUT / "k_selection.json", "w") as f:
        json.dump(scores, f, indent=2)

    km = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    rows["cluster"] = labels
    rows.to_csv(OUT / "labels.csv.gz", index=False, compression="gzip")
    joblib.dump(km, OUT / "kmeans.joblib")
    print(f"Saved KMeans(k={best_k}) and labels to {OUT}")


if __name__ == "__main__":
    main()
