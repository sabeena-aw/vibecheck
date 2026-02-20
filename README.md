# 🏠 NeighbourhoodFit · Airbnb Prototype

> A Streamlit prototype that ranks Barcelona neighbourhoods by how well they match your personal travel style — using NLP sentiment analysis on real Airbnb reviews and cosine similarity ranking.

---

## How it works

**Offline pipeline (`train.py`):**
1. Downloads real Airbnb guest reviews for Barcelona from [Inside Airbnb](http://insideairbnb.com) (open data)
2. Filters reviews by lifestyle dimension keywords (nightlife, safety, food, etc.)
3. Scores each relevant review using **TextBlob sentiment analysis** (polarity −1 → +1)
4. Aggregates scores per neighbourhood → saves `neighbourhood_scores.csv`

**App runtime (`app.py`):**
1. Loads `neighbourhood_scores.csv`
2. User sets preference sliders (1–5) for 8 lifestyle dimensions
3. **Cosine similarity** between user preference vector and each neighbourhood's score vector
4. Neighbourhoods ranked by fit score — the one most aligned with your priorities comes first

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app (loads CSV, runs similarity model, shows results) |
| `train.py` | Offline pipeline (downloads data, runs NLP, saves CSV) — run once |
| `neighbourhood_scores.csv` | Pre-computed scores from the pipeline (committed so app works without re-training) |
| `requirements.txt` | Python dependencies |

---

## Step-by-step: Run locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/neighbourhoodfit.git
cd neighbourhoodfit
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Re-run the training pipeline
This downloads fresh data from Inside Airbnb and regenerates `neighbourhood_scores.csv`.
Skip this step if you just want to use the pre-computed CSV already in the repo.
```bash
python train.py
```
⚠️ This takes ~5–10 minutes (downloads ~150MB of review data and scores them).

### Step 5 — Run the Streamlit app
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`

---

## Step-by-step: Deploy to Streamlit Cloud (free, for the submission link)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/neighbourhoodfit.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository → set main file to `app.py` → click **Deploy**
5. In ~2 minutes you'll have a public URL like `https://yourapp.streamlit.app`

That URL is your **Deliverable 2** for the assignment.

---

## Data source

Inside Airbnb — open data published under Creative Commons license.  
Barcelona dataset: http://insideairbnb.com/barcelona

---

## Assignment deliverables checklist

- [x] GitHub repository with all code
- [x] Live Streamlit app link (after deploying)
- [x] 2-page process document (`report.docx`)
