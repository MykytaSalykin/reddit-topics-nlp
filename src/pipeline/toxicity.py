# src/pipeline/toxicity.py
import pandas as pd
from detoxify import Detoxify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
from tqdm import tqdm

ART = Path("artifacts")
IN_PATH = ART / "rows_index.csv.gz"
OUT_PATH = ART / "reports" / "toxicity.csv.gz"

BATCH_SIZE = 512


def main():
    df = pd.read_csv(IN_PATH, compression="gzip")
    if "clean_text" not in df.columns:
        raise ValueError(f"'clean_text' column not found in {IN_PATH}")

    print("Loading Detoxify model (1–2 minutes)...")
    model = Detoxify("original")

    analyzer = SentimentIntensityAnalyzer()
    toxicity_scores, sentiments = [], []

    print("Scoring in batches...")
    texts = df["clean_text"].astype(str).tolist()
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        tox = model.predict(batch)
        toxicity_scores.extend(tox["toxicity"])
        scores = [analyzer.polarity_scores(t)["compound"] for t in batch]

        def to_label(v: float) -> str:
            return "positive" if v > 0.05 else "negative" if v < -0.05 else "neutral"

        sentiments.extend([to_label(v) for v in scores])

    df["toxicity"] = toxicity_scores
    df["sentiment"] = sentiments

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"Saved toxicity+sentiment → {OUT_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
