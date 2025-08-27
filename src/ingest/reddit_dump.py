# src/ingest/reddit_dump.py
import os
import time
from pathlib import Path
from typing import List
import pandas as pd
import praw
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env explicitly from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

DEFAULT_SUBS = "fitness,chess,technology,movies,politics"


def get_reddit():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "reddit-topics-nlp/0.1"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
    )


def fetch_subreddit_comments(
    reddit,
    subreddit: str,
    time_filter: str = "year",
    posts_limit: int = 200,
    max_comments_per_post: int = 50,
    sleep_between_posts: float = 0.2,
) -> pd.DataFrame:
    rows = []
    sub = reddit.subreddit(subreddit)
    for post in tqdm(
        sub.top(time_filter=time_filter, limit=posts_limit), desc=f"{subreddit}"
    ):
        try:
            post.comments.replace_more(limit=0)
            taken = 0
            for c in post.comments.list():
                body = getattr(c, "body", None)
                if not body:
                    continue
                rows.append(
                    {
                        "subreddit": subreddit,
                        "submission_id": post.id,
                        "submission_title": post.title,
                        "created_utc": int(getattr(c, "created_utc", 0)),
                        "author": str(getattr(c, "author", "")),
                        "score": int(getattr(c, "score", 0)),
                        "body": str(body),
                        "permalink": f"https://www.reddit.com{getattr(c, 'permalink', '')}",
                    }
                )
                taken += 1
                if taken >= max_comments_per_post:
                    break
            time.sleep(sleep_between_posts)
        except Exception:
            continue
    return pd.DataFrame(rows)


def main():
    out_path = Path("data/raw/comments.csv.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subs = os.getenv("SUBREDDITS", DEFAULT_SUBS).split(",")
    subs = [s.strip() for s in subs if s.strip()]
    posts = int(os.getenv("POSTS_PER_SUB", "200"))
    per_post = int(os.getenv("MAX_COMMENTS_PER_POST", "50"))
    time_filter = os.getenv("TIME_FILTER", "year")

    client_ok = bool(
        os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")
    )
    if not client_ok:
        # Fallback: if a local dump already exists, don't error — just inform.
        if out_path.exists():
            print(f"No API creds found — using existing dump: {out_path}")
            return
        print("No API creds found and no local dump exists — nothing to do.")
        return

    reddit = get_reddit()
    frames: List[pd.DataFrame] = []
    for s in subs:
        df = fetch_subreddit_comments(reddit, s, time_filter, posts, per_post)
        frames.append(df)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["subreddit", "submission_id", "body"])
        df_all.to_csv(out_path, index=False, compression="gzip")
        print(f"Saved: {out_path} ({len(df_all):,} rows)")
    else:
        print("No data fetched.")


if __name__ == "__main__":
    main()
