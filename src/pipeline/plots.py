# src/pipeline/plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ART = Path("artifacts/reports")
IN_PATH = ART / "toxicity.csv.gz"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH, compression="gzip")

    ART.mkdir(parents=True, exist_ok=True)

    # --- 1. Distribution of toxicity scores ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df["toxicity"], bins=50, kde=True, color="red")
    plt.title("Distribution of Toxicity Scores")
    plt.xlabel("Toxicity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ART / "toxicity_distribution.png")
    plt.close()

    # --- 2. Sentiment distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(
        x="sentiment",
        data=df,
        order=["negative", "neutral", "positive"],
        palette="coolwarm",
    )
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ART / "sentiment_distribution.png")
    plt.close()

    # --- 3. Toxicity by subreddit ---
    if "subreddit" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="subreddit", y="toxicity", data=df)
        plt.title("Toxicity by Subreddit")
        plt.xlabel("Subreddit")
        plt.ylabel("Toxicity")
        plt.tight_layout()
        plt.savefig(ART / "toxicity_by_subreddit.png")
        plt.close()

    # --- 4. Toxicity by cluster ---
    if "cluster_id" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x="cluster_id", y="toxicity", data=df, scale="width", inner="quartile"
        )
        plt.title("Toxicity by Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Toxicity")
        plt.tight_layout()
        plt.savefig(ART / "toxicity_by_cluster.png")
        plt.close()

    print("Saved plots to artifacts/reports/")


if __name__ == "__main__":
    main()
