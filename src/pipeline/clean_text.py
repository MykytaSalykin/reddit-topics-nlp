import re
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


def lemmatize(text: str) -> str:
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and len(t) > 2]
    return " ".join(tokens)


def process_comments(
    in_path="data/raw/comments.csv.gz", out_path="data/processed/comments_clean.csv.gz"
):
    df = pd.read_csv(in_path, compression="gzip")
    tqdm.pandas(desc="Cleaning")
    df["clean_text"] = df["body"].astype(str).progress_apply(basic_clean)
    df["clean_text"] = df["clean_text"].progress_apply(lemmatize)
    df = df[df["clean_text"].str.len() > 0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")
    print(f"Saved cleaned dataset → {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    process_comments()
