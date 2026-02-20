# Vibe Check · Airbnb Prototype

A Streamlit prototype built as an Airbnb feature concept that scores how well a listing's neighbourhood matches a user's personal travel style. It uses NLP sentiment analysis on real guest reviews and cosine similarity to rank all Barcelona neighbourhoods against the user's preferences.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app — loads scores, runs similarity model, displays results |
| `model.py` | Prediction logic — cosine similarity ranking, separated from the UI |
| `train.py` | Offline pipeline — downloads Inside Airbnb data, runs TextBlob NLP, saves CSV |
| `neighbourhood_scores.csv` | Pre-computed neighbourhood scores (ready to use without re-training) |
| `requirements.txt` | Python dependencies |
| `images/` | Listing photos used in the app |
| `airbnb_logo.png` | Airbnb logo displayed in the nav bar |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate scores from real data (optional, takes ~10 min):
```bash
python train.py
```

## Data source

Inside Airbnb — Barcelona dataset, published under Creative Commons licence.  
http://insideairbnb.com/barcelona
