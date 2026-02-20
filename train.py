"""
train.py — Offline pipeline (run once, not part of the Streamlit app)
Downloads real Airbnb review data for Barcelona from Inside Airbnb,
scores each review across 8 lifestyle dimensions using TextBlob sentiment
analysis, aggregates scores per neighbourhood, and saves neighbourhood_scores.csv.

Usage:
    python train.py

Output:
    neighbourhood_scores.csv
"""

import pandas as pd
import requests
import io
from textblob import TextBlob

# ── 1. Download real Barcelona reviews from Inside Airbnb ──────────────────────
REVIEWS_URL  = "https://data.insideairbnb.com/spain/catalonia/barcelona/2025-09-14/data/reviews.csv.gz"
LISTINGS_URL = "https://data.insideairbnb.com/spain/catalonia/barcelona/2025-09-14/data/listings.csv.gz"


def download_csv_gz(url, label, **kwargs):
    print(f"Downloading {label}...")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {label}: HTTP {r.status_code}")
    return pd.read_csv(io.BytesIO(r.content), compression="gzip", **kwargs)

listings = download_csv_gz(LISTINGS_URL, "listings",
                           usecols=["id", "neighbourhood_cleansed"])
listings = listings.rename(columns={"id": "listing_id"})

reviews = download_csv_gz(REVIEWS_URL, "reviews",
                          usecols=["listing_id", "comments"])

# ── 2. Merge reviews with neighbourhood labels ─────────────────────────────────
df = reviews.merge(listings, on="listing_id", how="left")
df = df.dropna(subset=["comments", "neighbourhood_cleansed"])
df["comments"] = df["comments"].astype(str).str.lower()

# ── 3. Keyword dictionaries per lifestyle dimension ────────────────────────────
DIMENSION_KEYWORDS = {
    "Nightlife & Bars":    ["bar", "nightlife", "club", "pub", "party", "drinks", "cocktail", "tapas", "nightout"],
    "Peaceful & Quiet":    ["quiet", "peaceful", "calm", "relaxing", "tranquil", "silent", "noisy", "loud", "noise"],
    "Walkability":         ["walk", "walking distance", "stroll", "walkable", "on foot", "nearby", "close to everything"],
    "Nature & Parks":      ["park", "garden", "nature", "green", "trees", "outdoor", "fresh air", "beach"],
    "Food & Restaurants":  ["restaurant", "food", "eat", "cafe", "coffee", "market", "cuisine", "bakery", "brunch"],
    "Safety":              ["safe", "safety", "secure", "dangerous", "unsafe", "sketchy", "feel safe"],
    "Public Transport":    ["metro", "bus", "transport", "subway", "train", "tram", "transit", "connection"],
    "Family-Friendly":     ["family", "kids", "children", "stroller", "playground", "child-friendly", "families"],
}

def score_review_for_dimension(text, keywords):
    if not any(kw in text for kw in keywords):
        return None
    return TextBlob(text).sentiment.polarity

def score_reviews(df):
    results = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  Scoring review {i}/{total}...")
        text = row["comments"]
        nbhd = row["neighbourhood_cleansed"]
        for dim, keywords in DIMENSION_KEYWORDS.items():
            score = score_review_for_dimension(text, keywords)
            if score is not None:
                results.append({"neighbourhood": nbhd, "dimension": dim, "polarity": score})
    return pd.DataFrame(results)

print("Scoring reviews (this takes a few minutes)...")
scores_long = score_reviews(df)

# ── 4. Aggregate and scale to 0-100 ───────────────────────────────────────────
agg = (
    scores_long
    .groupby(["neighbourhood", "dimension"])["polarity"]
    .mean()
    .reset_index()
)
agg["score"] = ((agg["polarity"] + 1) / 2 * 100).round(1)

pivot = agg.pivot(index="neighbourhood", columns="dimension", values="score").reset_index()
pivot.columns.name = None

review_counts = df.groupby("neighbourhood_cleansed").size().reset_index(name="n_reviews")
review_counts = review_counts.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
pivot = pivot.merge(review_counts, on="neighbourhood")
pivot = pivot[pivot["n_reviews"] >= 100].drop(columns="n_reviews")

for col in DIMENSION_KEYWORDS.keys():
    if col in pivot.columns:
        pivot[col] = pivot[col].fillna(pivot[col].median())

print(f"\nDone. {len(pivot)} neighbourhoods scored.")
print(pivot[["neighbourhood"]].to_string())

# ── 5. Save ───────────────────────────────────────────────────────────────────
pivot.to_csv("neighbourhood_scores.csv", index=False)
print("\nSaved: neighbourhood_scores.csv")
