# src/pipeline/labeling.py
from pathlib import Path
import gzip
import csv
import numpy as np
import joblib

ARTIFACTS = Path("artifacts")
CLUSTERS_DIR = ARTIFACTS / "clusters"
CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)


def top_terms_for_clusters(vectorizer, kmeans, topn=6):
    centroids = kmeans.cluster_centers_
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = []
    for cid, c in enumerate(centroids):
        top_idx = np.argsort(-c)[:topn]
        terms = vocab[top_idx]
        label = ", ".join(terms[:3])
        labels.append(
            {"cluster_id": cid, "label": label, "top_terms": ", ".join(terms)}
        )
    return labels


def main():
    vec = joblib.load(ARTIFACTS / "tfidf_vectorizer.joblib")
    kmeans = joblib.load(CLUSTERS_DIR / "kmeans.joblib")
    labels = top_terms_for_clusters(vec, kmeans, topn=8)
    out = CLUSTERS_DIR / "labels.csv.gz"
    with gzip.open(out, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "label", "top_terms"])
        w.writeheader()
        for row in labels:
            w.writerow(row)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
