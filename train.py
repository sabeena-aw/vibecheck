"""
train.py — Offline pipeline (run once, not part of the Streamlit app)
=======================================================================
Downloads real Airbnb review data for Barcelona from Inside Airbnb,
scores each review across 8 lifestyle dimensions using keyword-based
sentiment analysis (TextBlob), aggregates scores per neighbourhood,
and saves neighbourhood_scores.csv for use by app.py.

Usage:
    pip install pandas textblob requests
    python train.py

Output:
    neighbourhood_scores.csv
"""

import pandas as pd
import requests
import io
from textblob import TextBlob

# ── 1. Download real Barcelona reviews from Inside Airbnb ─────────────────────
# Inside Airbnb publishes open data under CC license: http://insideairbnb.com
REVIEWS_URL = "http://data.insideairbnb.com/spain/catalonia/barcelona/2024-09-10/data/reviews.csv.gz"
LISTINGS_URL = "http://data.insideairbnb.com/spain/catalonia/barcelona/2024-09-10/data/listings.csv.gz"

print("Downloading listings...")
listings_raw = requests.get(LISTINGS_URL, timeout=60).content
listings = pd.read_csv(io.BytesIO(listings_raw), compression="gzip",
                       usecols=["id", "neighbourhood_cleansed"])
listings = listings.rename(columns={"id": "listing_id"})

print("Downloading reviews...")
reviews_raw = requests.get(REVIEWS_URL, timeout=120).content
reviews = pd.read_csv(io.BytesIO(reviews_raw), compression="gzip",
                      usecols=["listing_id", "comments"])

# ── 2. Merge reviews with neighbourhood labels ─────────────────────────────────
df = reviews.merge(listings, on="listing_id", how="left")
df = df.dropna(subset=["comments", "neighbourhood_cleansed"])
df["comments"] = df["comments"].astype(str).str.lower()

# ── 3. Keyword dictionaries per lifestyle dimension ───────────────────────────
# Each dimension has positive and negative signal words.
# Reviews mentioning these keywords are scored on that dimension via TextBlob.
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

def score_review_for_dimension(text: str, keywords: list) -> float | None:
    """
    If the review mentions any keyword for this dimension,
    return its TextBlob polarity (-1 to 1). Otherwise return None.
    'Peaceful & Quiet' is inverted for negative words like 'noisy'.
    """
    if not any(kw in text for kw in keywords):
        return None
    blob = TextBlob(text)
    return blob.sentiment.polarity


def score_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Score each review for each dimension. Returns long-form scores."""
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

# ── 4. Aggregate: mean polarity per neighbourhood per dimension ───────────────
# Convert polarity (-1 to 1) → 0-100 scale
agg = (
    scores_long
    .groupby(["neighbourhood", "dimension"])["polarity"]
    .mean()
    .reset_index()
)
agg["score"] = ((agg["polarity"] + 1) / 2 * 100).round(1)

# Pivot to wide format
pivot = agg.pivot(index="neighbourhood", columns="dimension", values="score").reset_index()
pivot.columns.name = None

# Keep only neighbourhoods with enough data (at least 100 reviews)
review_counts = df.groupby("neighbourhood_cleansed").size().reset_index(name="n_reviews")
review_counts = review_counts.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
pivot = pivot.merge(review_counts, on="neighbourhood")
pivot = pivot[pivot["n_reviews"] >= 100].drop(columns="n_reviews")

# Fill any missing dimension scores with the city-wide median
for col in DIMENSION_KEYWORDS.keys():
    if col in pivot.columns:
        pivot[col] = pivot[col].fillna(pivot[col].median())

print(f"\nDone. {len(pivot)} neighbourhoods scored.")
print(pivot.head())

# ── 5. Save ───────────────────────────────────────────────────────────────────
pivot.to_csv("neighbourhood_scores.csv", index=False)
print("\nSaved: neighbourhood_scores.csv")
print("You can now run:  streamlit run app.py")
