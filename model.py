"""
model.py — Prediction logic

Loaded once by app.py at startup. Contains:
  - load_scores()         : loads pre-computed neighbourhood scores
  - rank_neighbourhoods() : cosine similarity ranking
  - fit_label()           : human-readable score interpretation
  - get_top_matches()     : dimension-level match/mismatch analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DIMENSIONS = [
    "Nightlife & Bars",
    "Peaceful & Quiet",
    "Walkability",
    "Nature & Parks",
    "Food & Restaurants",
    "Safety",
    "Public Transport",
    "Family-Friendly",
]


def load_scores(path: str = "neighbourhood_scores.csv") -> pd.DataFrame:
    """Load pre-computed neighbourhood scores from the offline pipeline."""
    df = pd.read_csv(path)
    # Ensure all dimension columns are present
    for dim in DIMENSIONS:
        if dim not in df.columns:
            raise ValueError(f"Missing dimension column in CSV: {dim}")
    return df


def rank_neighbourhoods(user_prefs: dict, scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Core ranking model using cosine similarity.

    Why cosine similarity?
    ----------------------
    We care about the *direction* of the preference vector, not its magnitude.
    A user who rates everything 5/5 and one who rates everything 3/3 have the same
    relative priorities — cosine similarity captures this correctly, whereas
    Euclidean distance would treat them differently.

    Parameters
    ----------
    user_prefs : dict  {dimension: int (1–5)}
    scores_df  : DataFrame with neighbourhood scores (0–100 per dimension)

    Returns
    -------
    DataFrame sorted by fit_score descending, with added columns:
      similarity  : raw cosine similarity (0–1)
      fit_score   : similarity scaled to 0–100
    """
    user_vec = np.array([user_prefs[d] for d in DIMENSIONS]).reshape(1, -1)
    nbhd_matrix = scores_df[DIMENSIONS].values

    # L2-normalise both vectors before computing similarity
    user_norm = user_vec / (np.linalg.norm(user_vec) + 1e-9)
    nbhd_norms = np.linalg.norm(nbhd_matrix, axis=1, keepdims=True) + 1e-9
    nbhd_norm = nbhd_matrix / nbhd_norms

    similarities = cosine_similarity(user_norm, nbhd_norm)[0]

    result = scores_df.copy()
    result["similarity"] = similarities
    result["fit_score"] = (similarities * 100).round(1)
    result = result.sort_values("fit_score", ascending=False).reset_index(drop=True)
    return result


def fit_label(score: float) -> tuple[str, str]:
    """Map a numeric fit score to a human-readable label and hex colour."""
    if score >= 88: return "Excellent match", "#00A699"
    if score >= 78: return "Great match",     "#FF385C"
    if score >= 68: return "Good match",      "#FC642D"
    return "Fair match", "#B0B0B0"


def get_match_analysis(user_prefs: dict, nbhd_row: pd.Series) -> dict:
    """
    Identify the strongest matches and friction points between the user's
    priorities and the neighbourhood's scores.

    Returns
    -------
    dict with keys:
      strengths  : list of (dimension, score) where score >= 70 and priority >= 4
      frictions  : list of (dimension, score) where score < 55 and priority >= 3
      model_note : short string explaining confidence level
    """
    strengths = [
        (d, nbhd_row[d]) for d in DIMENSIONS
        if nbhd_row[d] >= 70 and user_prefs.get(d, 3) >= 4
    ]
    frictions = [
        (d, nbhd_row[d]) for d in DIMENSIONS
        if nbhd_row[d] < 55 and user_prefs.get(d, 3) >= 3
    ]
    # Honest model confidence note
    high_priority_dims = [d for d in DIMENSIONS if user_prefs.get(d, 3) >= 4]
    if len(high_priority_dims) >= 4:
        note = "High confidence — you have strong preferences across many dimensions."
    elif len(high_priority_dims) == 0:
        note = "Lower confidence — your preferences are evenly distributed. Try setting some dimensions higher to get a sharper recommendation."
    else:
        note = "Moderate confidence — the model is weighting your top priorities."

    return {"strengths": strengths, "frictions": frictions, "model_note": note}
