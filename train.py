"""
train.py — Offline pipeline (run once, not part of the Streamlit app)
=======================================================================
Downloads real Airbnb review data for Barcelona from Inside Airbnb,
scores each neighbourhood across 8 lifestyle dimensions using keyword
frequency analysis on real guest reviews, and saves neighbourhood_scores.csv.

Why frequency instead of sentiment?
-------------------------------------
TextBlob sentiment gives near-identical scores across all neighbourhoods
because Airbnb guests write positively regardless of area. Keyword frequency
— how often dimension-relevant phrases appear per review — captures genuine
differences in what guests actually talk about in each neighbourhood.

Why high-specificity phrases instead of single words?
------------------------------------------------------
Single words like "bar" match "harbour", "barbecue", "sidebar".
"tapas" appears in every Barcelona neighbourhood review.
Multi-word phrases like "nightlife", "cocktail bar", "felt safe" are
specific enough to only appear when guests are genuinely describing that
dimension of the neighbourhood.

Usage:
    python train.py

Output:
    neighbourhood_scores.csv
"""

import pandas as pd
import numpy as np
import requests
import io

# ── 1. Download real Barcelona data from Inside Airbnb ─────────────────────────
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

# ── 2. Merge and clean ─────────────────────────────────────────────────────────
df = reviews.merge(listings, on="listing_id", how="left")
df = df.dropna(subset=["comments", "neighbourhood_cleansed"])
df["comments"] = df["comments"].astype(str).str.lower()
df["word_count"] = df["comments"].apply(lambda x: max(len(x.split()), 1))

# ── 3. High-specificity keyword dictionaries ───────────────────────────────────
# Multi-word phrases chosen to avoid false matches.
# Positive = dimension is present, Negative = dimension is absent.
DIMENSION_KEYWORDS = {
    "Nightlife & Bars": {
        "positive": [
            "nightlife", "night life", "night out", "nights out",
            "cocktail bar", "rooftop bar", "wine bar", "craft beer",
            "clubbing", "live music", "late night", "open late",
            "party scene", "great bars", "lots of bars", "many bars",
            "born bars", "gothic bars", "raval bars",
        ],
        "negative": ["quiet at night", "nothing open at night", "dead at night"],
    },
    "Peaceful & Quiet": {
        "positive": [
            "quiet", "peaceful", "calm", "tranquil", "serene",
            "very quiet", "no noise", "residential feel",
            "away from the crowds", "relaxing area",
        ],
        "negative": [
            "noisy", "loud noise", "street noise", "too noisy",
            "disturbed by noise", "rowdy", "very loud",
        ],
    },
    "Walkability": {
        "positive": [
            "walkable", "walking distance", "walk everywhere",
            "everything on foot", "steps away", "minutes walk",
            "stroll to", "very central", "walk to the beach",
            "great location for walking", "walk to everything",
        ],
        "negative": ["need a taxi", "far from everything", "not walkable"],
    },
    "Nature & Parks": {
        "positive": [
            "park nearby", "next to the park", "walk to the park",
            "ciutadella", "montjuic", "tibidabo",
            "green area", "beachfront", "right on the beach",
            "sea view from", "steps from the beach",
            "surrounded by nature", "very green area",
            "parc de la ciutadella", "parc güell",
        ],
        "negative": [],
    },
    "Food & Restaurants": {
        "positive": [
            "great restaurants", "amazing restaurants", "restaurants nearby",
            "food scene", "local restaurants", "tapas nearby",
            "boqueria", "market nearby", "great cafes",
            "coffee shops", "bakery nearby", "foodie paradise",
            "lots of restaurants",
        ],
        "negative": [],
    },
    "Safety": {
        "positive": [
            "felt safe", "feel safe", "very safe", "safe area",
            "safe neighbourhood", "safe at night", "no issues",
            "felt comfortable", "no problems", "safe to walk",
            "safe place", "safe location", "always felt safe",
            "never felt unsafe", "perfectly safe",
        ],
        "negative": [
            "unsafe", "dangerous", "sketchy", "dodgy",
            "robbery", "pickpocket", "watch your belongings",
            "avoid at night", "not safe", "felt unsafe",
        ],
    },
    "Public Transport": {
        "positive": [
            "metro nearby", "close to metro", "metro station",
            "great transport", "well connected", "bus stop nearby",
            "public transport", "easy to get around",
            "tram stop", "train station nearby", "minutes from metro",
        ],
        "negative": ["far from metro", "no metro nearby", "poor transport"],
    },
    "Family-Friendly": {
        "positive": [
            "great for families", "family friendly", "good for kids",
            "child friendly", "playground nearby", "stroller friendly",
            "safe for kids", "families with children", "travelling with kids",
        ],
        "negative": ["not suitable for children", "not family friendly"],
    },
}

# ── 4. Frequency scoring ───────────────────────────────────────────────────────
# For each review, count how often dimension phrases appear per 100 words.
# Positive phrases add, negative phrases subtract.

print("Computing keyword frequencies per review...")

def compute_freq_score(text, word_count, pos_kws, neg_kws):
    pos_count = sum(text.count(kw) for kw in pos_kws)
    neg_count = sum(text.count(kw) for kw in neg_kws)
    return (pos_count - neg_count) / word_count * 100

results = []
total = len(df)
for i, (_, row) in enumerate(df.iterrows()):
    if i % 10000 == 0:
        print(f"  Processing review {i:,}/{total:,}...")
    text = row["comments"]
    nbhd = row["neighbourhood_cleansed"]
    wc   = row["word_count"]
    for dim, kws in DIMENSION_KEYWORDS.items():
        freq = compute_freq_score(text, wc, kws["positive"], kws["negative"])
        if freq != 0:
            results.append({"neighbourhood": nbhd, "dimension": dim, "freq": freq})

scores_long = pd.DataFrame(results)

# ── 5. Aggregate per neighbourhood ────────────────────────────────────────────
print("\nAggregating scores per neighbourhood...")

# Merge review counts before aggregating so we can weight by volume
review_counts_early = df.groupby("neighbourhood_cleansed").size().reset_index(name="n_reviews")
review_counts_early = review_counts_early.rename(columns={"neighbourhood_cleansed": "neighbourhood"})

agg = (
    scores_long
    .groupby(["neighbourhood", "dimension"])["freq"]
    .mean()
    .reset_index()
)

# Merge review counts and apply Bayesian smoothing:
# blend each neighbourhood's score toward the city-wide mean
# weighted by how many reviews it has. Neighbourhoods with few reviews
# get pulled toward the mean, preventing small-sample outliers from
# dominating the top/bottom of the ranking.
agg = agg.merge(review_counts_early, on="neighbourhood", how="left")
city_means = agg.groupby("dimension")["freq"].mean().rename("city_mean")
agg = agg.merge(city_means, on="dimension")
K = 500  # smoothing factor — higher = more pull toward mean for small neighbourhoods
agg["freq_smooth"] = (agg["freq"] * agg["n_reviews"] + agg["city_mean"] * K) / (agg["n_reviews"] + K)

# Min-max scale smoothed scores to 10-90 range
def minmax_scale(series):
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - mn) / (mx - mn) * 80 + 10).round(1)

agg["score"] = agg.groupby("dimension")["freq_smooth"].transform(minmax_scale)

pivot = agg.pivot(index="neighbourhood", columns="dimension", values="score").reset_index()
pivot.columns.name = None

# ── 6. Filter: keep only neighbourhoods relevant to Airbnb guests ─────────────
# We filter by both review count AND listing count.
# Working-class residential areas with few listings produce noisy scores
# because guests staying nearby mention those areas in passing — not because
# the neighbourhood itself has the features being scored.
listing_counts = listings.groupby("neighbourhood_cleansed").size().reset_index(name="n_listings")
listing_counts = listing_counts.rename(columns={"neighbourhood_cleansed": "neighbourhood"})

pivot = pivot.merge(review_counts_early, on="neighbourhood", how="left")
pivot = pivot.merge(listing_counts, on="neighbourhood", how="left")

# Keep only neighbourhoods with enough reviews AND enough listings
# to be a realistic option for Airbnb guests
pivot = pivot[(pivot["n_reviews"] >= 300) & (pivot["n_listings"] >= 50)]
pivot = pivot.drop(columns=["n_reviews", "n_listings"])

print(f"After filtering: {len(pivot)} neighbourhoods remain")

for col in DIMENSION_KEYWORDS.keys():
    if col in pivot.columns:
        pivot[col] = pivot[col].fillna(pivot[col].median())

# ── 7. Print summary ───────────────────────────────────────────────────────────
print(f"\nDone. {len(pivot)} neighbourhoods scored.")
print(f"\nNightlife range: {pivot['Nightlife & Bars'].min():.1f} – {pivot['Nightlife & Bars'].max():.1f}")
print(f"Nature range:    {pivot['Nature & Parks'].min():.1f} – {pivot['Nature & Parks'].max():.1f}")
print(f"Safety range:    {pivot['Safety'].min():.1f} – {pivot['Safety'].max():.1f}")

print("\nTop 5 by Nightlife:")
print(pivot[["neighbourhood","Nightlife & Bars"]].sort_values("Nightlife & Bars", ascending=False).head(5).to_string(index=False))
print("\nTop 5 by Nature & Parks:")
print(pivot[["neighbourhood","Nature & Parks"]].sort_values("Nature & Parks", ascending=False).head(5).to_string(index=False))
print("\nTop 5 by Safety:")
print(pivot[["neighbourhood","Safety"]].sort_values("Safety", ascending=False).head(5).to_string(index=False))

# ── 8. Save ────────────────────────────────────────────────────────────────────
pivot.to_csv("neighbourhood_scores.csv", index=False)
print("\nSaved: neighbourhood_scores.csv")