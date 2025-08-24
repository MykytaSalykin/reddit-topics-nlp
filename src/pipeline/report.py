# src/pipeline/report.py
from pathlib import Path
import textwrap
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

ART = Path("artifacts")
CL = ART / "clusters"
REPORTS = ART / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def top_terms_per_cluster(X, labels, vectorizer, topn=15):
    # X: csr matrix, labels: array, vectorizer: fitted TfidfVectorizer
    vocab = np.array(vectorizer.get_feature_names_out())
    tops = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        Xc = X[idx]
        # mean tfidf within cluster
        mean_w = np.asarray(Xc.mean(axis=0)).ravel()
        top_idx = mean_w.argsort()[::-1][:topn]
        tops[int(c)] = vocab[top_idx].tolist()
    return tops


def plot_cluster_sizes(labels, path_png: Path):
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Cluster sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Docs")
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def sample_rows(rows: pd.DataFrame, labels_col="cluster", per_cluster=5, seed=42):
    out = []
    rng = np.random.RandomState(seed)
    for c in sorted(rows[labels_col].unique()):
        sub = rows[rows[labels_col] == c]
        take = min(per_cluster, len(sub))
        out.append(sub.sample(take, random_state=rng))
    return pd.concat(out, ignore_index=True)


def main():
    rows = pd.read_csv(CL / "labels.csv.gz", compression="gzip")
    X = sparse.load_npz(ART / "tfidf.npz")
    vec = joblib.load(ART / "tfidf_vectorizer.joblib")
    km = joblib.load(CL / "kmeans.joblib")
    labels = rows["cluster"].to_numpy()

    # top terms
    tops = top_terms_per_cluster(X, labels, vec, topn=15)

    # plot sizes
    plot_path = REPORTS / "cluster_sizes.png"
    plot_cluster_sizes(labels, plot_path)

    # sample texts
    samples = sample_rows(rows, per_cluster=6)
    # build markdown
    md_lines = ["# Topic Clusters Report\n"]
    md_lines.append(
        f"- Documents: **{len(rows):,}**  \n- Clusters: **{len(np.unique(labels))}**\n"
    )
    md_lines.append(f"![cluster sizes]({plot_path.as_posix()})\n")
    md_lines.append("## Top Terms by Cluster\n")
    for c in sorted(tops.keys()):
        md_lines.append(f"### Cluster {c}\n")
        md_lines.append("`" + "`, `".join(tops[c]) + "`\n")

    md_lines.append("\n## Sample Comments per Cluster\n")
    for c in sorted(rows["cluster"].unique()):
        md_lines.append(f"### Cluster {c}\n")
        sub = samples[samples["cluster"] == c]
        for _, r in sub.iterrows():
            txt = str(r["clean_text"])
            txt = textwrap.shorten(txt, width=220, placeholder="…")
            md_lines.append(f"- {txt}")
        md_lines.append("")

    report_md = REPORTS / "topics_report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {report_md}")
    print(f"Also saved {plot_path}")


if __name__ == "__main__":
    main()
