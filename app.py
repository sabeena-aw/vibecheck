"""
app.py — Vibe Check · Airbnb prototype
=======================================
A new Airbnb feature that helps users find the Barcelona neighbourhood
that best matches their travel style, powered by NLP-scored review data
and cosine similarity ranking.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from model import load_scores, rank_neighbourhoods, fit_label, get_match_analysis, DIMENSIONS

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vibe Check · Airbnb",
    page_icon="🏠",
    layout="wide",
)

# ── CSS — Airbnb palette, font, minimal overrides ──────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif !important;
    color: #222222;
  }

  /* Airbnb red primary button */
  .stButton > button {
    background-color: #FF385C !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
    transition: background 0.15s !important;
  }
  .stButton > button:hover {
    background-color: #E0314F !important;
    color: white !important;
  }

  /* Slider accent */
  [data-testid="stSlider"] > div > div > div > div {
    background: #FF385C !important;
  }

  /* Tab underline */
  div[data-baseweb="tab-highlight"] {
    background-color: #FF385C !important;
    height: 2px !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #222222 !important;
    font-weight: 700 !important;
  }
  button[data-baseweb="tab"] {
    font-family: 'Nunito Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #717171 !important;
  }

  /* Progress bar */
  [data-testid="stProgress"] > div > div {
    background: #FF385C !important;
  }

  /* Metric styling */
  [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #222222 !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #717171 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
  }

  /* Expander */
  [data-testid="stExpander"] summary {
    font-weight: 700 !important;
    font-size: 0.9rem !important;
  }

  /* Divider colour */
  hr { border-color: #EBEBEB !important; }

  /* Hide Streamlit default chrome */
  #MainMenu, footer { visibility: hidden; }

  /* Nav bar */
  .nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0 14px 0;
    border-bottom: 1px solid #EBEBEB;
    margin-bottom: 28px;
  }
  .nav-links {
    font-size: 0.88rem;
    font-weight: 600;
    color: #222222;
  }
  .nav-links span {
    margin-left: 24px;
    cursor: pointer;
  }
  .nav-links span:hover { text-decoration: underline; }

  /* Listing image placeholders */
  .img-main {
    background: linear-gradient(135deg, #f0e6ff 0%, #e8d5f5 100%);
    border-radius: 12px 0 0 12px;
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    color: #999;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .img-sub {
    background: linear-gradient(135deg, #f5f0ff 0%, #ede0ff 100%);
    height: 165px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    color: #999;
    font-weight: 600;
  }
  .img-sub:first-child { border-radius: 0 12px 0 0; margin-bottom: 10px; }
  .img-sub:last-child  { border-radius: 0 0 12px 0; }

  /* Ranking table row */
  .rank-row {
    display: flex;
    align-items: center;
    padding: 13px 16px;
    border: 1px solid #EBEBEB;
    border-radius: 10px;
    margin-bottom: 8px;
    gap: 14px;
  }
  .rank-row.top {
    border: 2px solid #FF385C;
    background: #FFFAFB;
  }
  .rank-num   { font-weight: 800; color: #BBBBBB; min-width: 28px; font-size: 0.95rem; }
  .rank-num.top { color: #FF385C; }
  .rank-name  { font-weight: 700; font-size: 0.95rem; flex: 1; }
  .rank-desc  { font-size: 0.8rem; color: #717171; margin-top: 2px; }
  .rank-score { font-weight: 800; font-size: 0.95rem; margin-left: auto; white-space: nowrap; }
  .score-good { color: #00A699; }
  .score-mid  { color: #FF385C; }
  .score-low  { color: #BBBBBB; }

  /* Result score display */
  .fit-score {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
  }

  /* Vibe Check badge on tab */
  .vc-badge {
    background: #FF385C;
    color: white;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 8px;
    margin-left: 5px;
    vertical-align: middle;
    letter-spacing: 0.3px;
  }
</style>
""", unsafe_allow_html=True)

# ── Static data ────────────────────────────────────────────────────────────────
NBHD_INFO = {
    "Gràcia": (
        "Bohemian village in the heart of the city. Leafy squares, independent cafés, and a strong local creative scene. Popular with young professionals and long-term expats.",
        ["Parc Güell (10 min walk)", "Weekend market at Plaça de la Llibertat", "Strong independent restaurant scene"],
        ["Can feel noisy on weekend nights", "Hilly streets heading north"]
    ),
    "Eixample": (
        "Barcelona's elegant Modernist grid. Wide boulevards, Gaudí landmarks, high-end restaurants, and the best metro connectivity in the city.",
        ["Sagrada Família & La Pedrera", "Passeig de Gràcia shopping", "9 metro lines within walking distance"],
        ["Very busy and touristy in summer", "Street noise from traffic"]
    ),
    "Barceloneta": (
        "The city's beach neighbourhood. Mediterranean energy, fresh seafood, and year-round coastal life. Best for those who want the beach within minutes.",
        ["Direct beach access", "Fresh seafood at Port Olímpic", "Barceloneta market"],
        ["Very crowded in summer", "Loud at night on weekends", "Limited supermarket options"]
    ),
    "El Raval": (
        "Gritty, multicultural, and genuinely creative. Home to MACBA, the city's best street food, and a nightlife scene that runs until dawn.",
        ["MACBA contemporary art museum", "La Boqueria market", "Diverse international food scene"],
        ["Some areas feel unsafe at night", "Can be chaotic and loud"]
    ),
    "Sant Pere, Santa Caterina i la Ribera": (
        "El Born — Barcelona's most fashionable neighbourhood. Medieval streets, world-class tapas bars, independent boutiques, and a relaxed daytime energy.",
        ["El Born Cultural Centre", "Picasso Museum", "Best tapas concentration in the city"],
        ["Very touristy around El Born market", "Limited parking"]
    ),
    "Les Corts": (
        "A quiet, residential neighbourhood adjacent to the business district. Calm streets, good transport links, and away from the tourist crowds.",
        ["Camp Nou stadium", "L'Illa Diagonal shopping centre", "Very safe and residential"],
        ["Limited nightlife", "Far from historic centre"]
    ),
    "Sarrià-Sant Gervasi": (
        "Upscale hillside area above the city. Peaceful, green, and strongly family-oriented. The quietest option with the best air quality.",
        ["Tibidabo views and amusement park", "Quiet neighbourhood parks", "Excellent international schools nearby"],
        ["Far from central Barcelona", "Limited late-night options", "Hilly terrain"]
    ),
    "Nou Barris": (
        "Authentic working-class Barcelona. No tourist infrastructure, genuine local life, and significantly lower prices than central neighbourhoods.",
        ["Local covered market", "Strong community identity", "Best value accommodation"],
        ["Far from major attractions", "Limited English spoken", "Less developed for visitors"]
    ),
    "Sant Andreu": (
        "An up-and-coming residential neighbourhood with a village feel inside the city. Strong community identity and good value.",
        ["Mercat de Sant Andreu", "Rambla del Poblenou nearby", "Good metro and rail connections"],
        ["35 minutes from city centre on foot", "Limited tourist-facing services"]
    ),
    "Sant Martí": (
        "Home to Poblenou, Barcelona's rapidly developing tech and design district. Close to the beach and well connected.",
        ["Rambla del Poblenou", "Beach access via Bogatell", "Growing restaurant and café scene"],
        ["Construction noise in parts", "Some areas still being developed"]
    ),
    "Sants-Montjuïc": (
        "Gateway to Montjuïc hill and home to Sants station. Good transport hub, strong local identity, and access to great outdoor spaces.",
        ["Montjuïc Castle and MNAC museum", "Sants station (AVE trains)", "Parc de l'Espanya Industrial"],
        ["Busy around the station", "Some parts feel transitional"]
    ),
    "Horta-Guinardó": (
        "Green, tranquil, and genuinely off the tourist trail. One of the best neighbourhoods for outdoor space and authentic local life.",
        ["Parc del Laberint d'Horta", "Bunkers del Carmel (city views)", "Very local and authentic"],
        ["40 minutes from Las Ramblas", "Limited direct tourist transport"]
    ),
}

NBHD_COORDS = {
    "Gràcia": (41.4036, 2.1571),
    "Eixample": (41.3929, 2.1610),
    "Barceloneta": (41.3795, 2.1894),
    "El Raval": (41.3797, 2.1686),
    "Sant Pere, Santa Caterina i la Ribera": (41.3852, 2.1810),
    "Les Corts": (41.3840, 2.1267),
    "Sarrià-Sant Gervasi": (41.4049, 2.1283),
    "Nou Barris": (41.4388, 2.1740),
    "Sant Andreu": (41.4296, 2.1895),
    "Sant Martí": (41.4048, 2.1981),
    "Sants-Montjuïc": (41.3698, 2.1497),
    "Horta-Guinardó": (41.4196, 2.1623),
}

DIM_LABELS = {
    "Nightlife & Bars":    "Nightlife",
    "Peaceful & Quiet":    "Peace & quiet",
    "Walkability":         "Walkability",
    "Nature & Parks":      "Nature & parks",
    "Food & Restaurants":  "Food scene",
    "Safety":              "Safety",
    "Public Transport":    "Public transport",
    "Family-Friendly":     "Family-friendly",
}

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_scores():
    return load_scores("neighbourhood_scores.csv")

scores_df = get_scores()

# ── Radar chart ────────────────────────────────────────────────────────────────
def make_radar(user_prefs, nbhd_row):
    labels  = list(DIM_LABELS.values())
    user_v  = [user_prefs[d] * 20 for d in DIMENSIONS]
    nbhd_v  = [float(nbhd_row[d]) for d in DIMENSIONS]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=nbhd_v + [nbhd_v[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(255,56,92,0.10)",
        line=dict(color="#FF385C", width=2),
        name="Neighbourhood profile",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_v + [user_v[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(0,166,153,0.08)",
        line=dict(color="#00A699", width=2, dash="dot"),
        name="Your priorities",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#EBEBEB", tickfont=dict(size=9, color="#AAAAAA"),
            ),
            angularaxis=dict(tickfont=dict(size=10, color="#444444")),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10, family="Nunito Sans")),
        height=360,
        margin=dict(t=20, b=55, l=50, r=50),
        paper_bgcolor="white",
        font=dict(family="Nunito Sans"),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_nav = st.columns([1, 3])
with col_logo:
    st.markdown("""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 168 50" height="38" style="margin-top:4px;display:block;">
      <!-- Bélo symbol -->
      <path fill="#FF385C" d="M31.8 33.6c-.4-.7-.9-1.5-1.3-2.2-.4-.8-.9-1.6-1.3-2.4
        2.6-3.1 4-6.8 4-10.4 0-5.4-3-8.6-8.1-8.6s-8.1 3.2-8.1 8.6c0 3.6 1.5 7.3 4 10.4
        -.4.8-.8 1.6-1.3 2.4-.4.7-.9 1.5-1.3 2.2C15 30.9 12 25.9 12 20.5 12 11.8 17.1 7
        25.1 7s13.1 4.8 13.1 13.5c0 5.4-3 10.4-6.4 13.1z
        m-6.7 2.4c-1.1-1.4-2.2-2.9-3.2-4.4 1 .2 2.1.3 3.2.3s2.2-.1 3.2-.3
        c-1 1.5-2.1 3-3.2 4.4z
        M17.3 36.8C15.5 34.6 14.5 32 14.5 29.2c0-2 .7-3.6 2.2-4.6 1.3-.9 2.9-1.2
        4.4-1 .5.8.9 1.7 1.4 2.6.5.8.9 1.7 1.5 2.6-2.6 2.4-5 5.4-6.7 8z
        m15.5 0c-1.7-2.6-4.2-5.6-6.7-8 .5-.9 1-1.7 1.5-2.6.5-.8.9-1.7 1.4-2.6
        1.5-.2 3.1.1 4.4 1 1.5 1 2.2 2.6 2.2 4.6-.1 2.8-1.1 5.4-2.8 7.6z"/>
      <!-- airbnb wordmark -->
      <path fill="#FF385C" d="M52 28.8V16.4h3.2v1.7c.9-1.3 2.2-2 3.9-2 3 0 5 2.1 5 5.5v7.2h-3.3v-6.7
        c0-1.9-.9-3-2.5-3s-2.7 1.1-2.7 3.1v6.6H52z
        M76.4 23.6c0 3.1-2.2 5.5-5.8 5.5s-5.8-2.4-5.8-5.5 2.2-5.5 5.8-5.5 5.8 2.4 5.8 5.5z
        m-3.3 0c0-1.8-1-3-2.5-3s-2.5 1.2-2.5 3 1 3 2.5 3 2.5-1.2 2.5-3z
        M84.6 29.1c-1.9 0-3.3-.7-4.1-2v1.7h-3.2V11.5h3.3v6.6c.8-1.2 2.2-2 4-2
        3.3 0 5.6 2.5 5.6 6.5s-2.3 6.5-5.6 6.5zm-.9-9.9c-1.7 0-2.9 1.3-2.9 3.4s1.2 3.4
        2.9 3.4 2.9-1.3 2.9-3.4-1.2-3.4-2.9-3.4z
        M91.8 11.5h3.3v17.3h-3.3z
        M107.4 25.5c-.6 2.2-2.6 3.6-5.1 3.6-3.4 0-5.7-2.4-5.7-5.6 0-3.3 2.2-5.5
        5.6-5.5 3.3 0 5.4 2.2 5.4 5.4 0 .4 0 .8-.1 1.1H99.9c.2 1.5 1.2 2.4 2.5
        2.4 1.1 0 1.9-.5 2.3-1.4h2.7zm-7.4-3.3h4.5c-.2-1.3-1-2.1-2.2-2.1s-2.1.8-2.3 2.1z
        M117.9 16.1c.4 0 .7 0 1 .1l-.3 3.2c-.4-.1-.7-.1-1.1-.1-1.7 0-2.8 1.1-2.8
        3V28.8h-3.3V16.4h3.1l.3 2.2c.6-1.5 1.8-2.5 3.1-2.5z
        M129.4 28.8l-.3-1.5c-1 1.2-2.5 1.8-4.3 1.8-2.8 0-4.5-1.6-4.5-4 0-2.6
        2-4.2 5.5-4.2h3v-.6c0-1.3-.8-2-2.5-2-1.3 0-2.5.5-3.8 1.3l-1.2-2.2
        c1.6-1 3.4-1.5 5.4-1.5 3.3 0 5.2 1.6 5.2 4.5v8.4h-2.5zm-.6-5.5h-2.4
        c-1.6 0-2.4.6-2.4 1.7 0 1 .7 1.7 2 1.7 1.6 0 2.8-1 2.8-2.6v-.8z"/>
    </svg>
    """, unsafe_allow_html=True)

with col_nav:
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;align-items:center;height:46px;gap:24px;font-size:0.88rem;font-weight:600;">
      <span style="cursor:pointer;">Airbnb your home</span>
      <span style="cursor:pointer;">Help</span>
      <span style="cursor:pointer;border:1px solid #DDDDDD;border-radius:22px;padding:6px 14px;">Sign in</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# LISTING HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h2 style="margin:0 0 8px;font-size:1.5rem;font-weight:800;color:#222;">
  Bright apartment with private terrace — El Born, Barcelona
</h2>
<div style="display:flex;gap:8px;align-items:center;font-size:0.88rem;color:#222;flex-wrap:wrap;">
  <span>&#9733; <strong>4.93</strong></span>
  <span style="color:#CCCCCC;">·</span>
  <span style="text-decoration:underline;cursor:pointer;">218 reviews</span>
  <span style="color:#CCCCCC;">·</span>
  <span style="font-weight:700;">Superhost</span>
  <span style="color:#CCCCCC;">·</span>
  <span style="text-decoration:underline;cursor:pointer;">El Born, Barcelona, Spain</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Photo grid — fixed height, cropped to fit, matching Airbnb listing layout
import base64

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

liv = img_to_b64("images/living.jpg")
bed = img_to_b64("images/bedroom.jpg")
kit = img_to_b64("images/kitchen.jpg")

st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:200px 200px;gap:8px;border-radius:12px;overflow:hidden;margin-bottom:24px;">
  <div style="grid-row:1/3;overflow:hidden;">
    <img src="data:image/jpeg;base64,{liv}" style="width:100%;height:100%;object-fit:cover;display:block;">
  </div>
  <div style="overflow:hidden;">
    <img src="data:image/jpeg;base64,{bed}" style="width:100%;height:100%;object-fit:cover;display:block;">
  </div>
  <div style="overflow:hidden;">
    <img src="data:image/jpeg;base64,{kit}" style="width:100%;height:100%;object-fit:cover;display:block;">
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_vibecheck = st.tabs([
    "Overview",
    "Vibe Check  \u2728",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    col_info, col_book = st.columns([1.55, 1], gap="large")

    with col_info:
        # Host line
        st.markdown("""
        <div style="padding-bottom:24px;border-bottom:1px solid #EBEBEB;margin-bottom:24px;">
          <div style="font-size:1.1rem;font-weight:700;margin-bottom:6px;">Entire apartment hosted by Marta</div>
          <div style="color:#717171;font-size:0.88rem;">4 guests &nbsp;·&nbsp; 2 bedrooms &nbsp;·&nbsp; 2 beds &nbsp;·&nbsp; 1 bathroom</div>
        </div>
        """, unsafe_allow_html=True)

        # Highlights
        highlights = [
            ("★", "Superhost", "Marta has a 4.93 rating across 218 stays."),
            ("✔", "Great location", "95% of recent guests rated the location 5 stars."),
            ("✔", "Free cancellation", "Full refund if cancelled before 48 hours of check-in."),
        ]
        for icon, title, desc in highlights:
            st.markdown(f"""
            <div style="display:flex;gap:16px;margin-bottom:18px;align-items:flex-start;">
              <div style="font-size:1.2rem;min-width:24px;margin-top:1px;">{icon}</div>
              <div>
                <div style="font-weight:700;font-size:0.92rem;">{title}</div>
                <div style="color:#717171;font-size:0.85rem;margin-top:2px;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Description
        st.markdown("""
        <div style="font-size:0.92rem;line-height:1.75;color:#444;">
          A sunlit apartment on the second floor of a classic Barcelona townhouse, right in the heart of El Born.
          The terrace overlooks a quiet internal courtyard and gets afternoon sun year-round. The neighbourhood
          is extremely walkable — the Picasso Museum, Santa Maria del Mar basilica, and some of the city's best
          tapas bars are all within a five-minute walk.
          <br><br>
          The apartment has been recently renovated while keeping the original tile floors and exposed brick details.
          The kitchen is fully equipped for cooking, and there is a dedicated workspace if you need to get things done.
        </div>
        <div style="margin-top:14px;font-size:0.9rem;font-weight:600;text-decoration:underline;cursor:pointer;color:#222;">
          Show more
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Amenities
        st.markdown("<div style='font-size:1rem;font-weight:700;margin-bottom:16px;'>What this place offers</div>", unsafe_allow_html=True)
        amenities = [
            ("Terrace", "Private courtyard-facing terrace"),
            ("Kitchen", "Fully equipped, espresso machine"),
            ("Fast WiFi", "Fibre connection, 600 Mbps"),
            ("Air conditioning", "Split system, hot & cold"),
            ("Washer", "In-unit, with dryer"),
            ("Smart TV", "Netflix and Prime included"),
            ("Workspace", "Dedicated desk with monitor"),
            ("Self check-in", "Lockbox, available 24h"),
        ]
        cols = st.columns(2)
        for i, (name, detail) in enumerate(amenities):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                  <div style="font-weight:600;font-size:0.88rem;">{name}</div>
                  <div style="color:#717171;font-size:0.82rem;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_book:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom:16px;">
              <span style="font-size:1.3rem;font-weight:800;">€112</span>
              <span style="font-size:0.88rem;color:#717171;"> / night</span>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.date_input("Check-in", key="checkin")
            with c2:
                st.date_input("Check-out", key="checkout")

            st.selectbox("Guests", ["1 guest", "2 guests", "3 guests", "4 guests"])
            st.button("Reserve", use_container_width=True)
            st.caption("You won't be charged yet")
            st.divider()

            for label, amount in [("€112 × 7 nights", "€784"), ("Cleaning fee", "€45"), ("Airbnb service fee", "€123")]:
                c1, c2 = st.columns([3, 1])
                with c1: st.write(label)
                with c2: st.write(amount)

            st.divider()
            c1, c2 = st.columns([3, 1])
            with c1: st.markdown("**Total before taxes**")
            with c2: st.markdown("**€952**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIBE CHECK
# ══════════════════════════════════════════════════════════════════════════════
with tab_vibecheck:

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Feature intro — reads like a real product feature, not a student demo
    col_intro, _ = st.columns([2, 1], gap="large")
    with col_intro:
        st.markdown("""
        <h3 style="font-size:1.3rem;font-weight:800;margin:0 0 10px;">
          Does this neighbourhood match your travel style?
        </h3>
        <p style="font-size:0.92rem;color:#444;line-height:1.7;margin:0;">
          Vibe Check analyses thousands of real guest reviews to build a detailed profile of every
          Barcelona neighbourhood across eight dimensions. Tell us what matters to you and we'll
          rank them by how well they match — so you can book knowing the area works for you,
          not just the flat.
        </p>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Preferences ────────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.95rem;font-weight:700;margin-bottom:4px;'>What matters to you on this trip?</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.83rem;color:#717171;margin-bottom:20px;'>Rate each factor from 1 (not important) to 5 (essential).</div>", unsafe_allow_html=True)

    user_prefs = {}
    col_a, col_b = st.columns(2, gap="large")
    for i, dim in enumerate(DIMENSIONS):
        with (col_a if i % 2 == 0 else col_b):
            user_prefs[dim] = st.slider(
                DIM_LABELS[dim],
                min_value=1, max_value=5, value=3,
                key=f"pref_{dim}",
            )

    st.divider()

    # ── Trip context ───────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.95rem;font-weight:700;margin-bottom:14px;'>About your trip</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        trip_type = st.selectbox(
            "Travelling as",
            ["Solo", "Couple", "Friends group", "Family with children", "Business"],
        )
    with c2:
        nights = st.number_input("Nights", min_value=1, max_value=90, value=7)
    with c3:
        pace = st.select_slider(
            "Travel pace",
            options=["Very relaxed", "Relaxed", "Balanced", "Active", "Non-stop"],
            value="Balanced",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run = st.button("Find my best neighbourhood match", use_container_width=True)

    # ── Results ────────────────────────────────────────────────────────────────
    # ── Results ────────────────────────────────────────────────────────────────
    if run:

        # This listing is in El Born (Sant Pere, Santa Caterina i la Ribera)
        LISTING_NEIGHBOURHOOD = "Sant Pere, Santa Caterina i la Ribera"

        bar = st.progress(0, text="Loading review data...")
        for pct, msg in [(25, "Building your preference profile..."),
                         (55, "Scoring El Born against your priorities..."),
                         (85, "Comparing with other neighbourhoods..."),
                         (100, "Done")]:
            time.sleep(0.2)
            bar.progress(pct, text=msg)
        time.sleep(0.25)
        bar.empty()

        # Run model across all neighbourhoods
        ranked = rank_neighbourhoods(user_prefs, scores_df)

        # Pull the listing's neighbourhood row specifically
        listing_row   = ranked[ranked["neighbourhood"] == LISTING_NEIGHBOURHOOD].iloc[0]
        listing_score = listing_row["fit_score"]
        listing_label, listing_color = fit_label(listing_score)
        listing_desc, listing_pros, listing_cons = NBHD_INFO.get(LISTING_NEIGHBOURHOOD, ("", [], []))
        listing_analysis = get_match_analysis(user_prefs, listing_row)
        listing_lat, listing_lon = NBHD_COORDS[LISTING_NEIGHBOURHOOD]

        # Rank position of this listing's neighbourhood
        listing_rank = ranked[ranked["neighbourhood"] == LISTING_NEIGHBOURHOOD].index[0] + 1
        total_nbhds  = len(ranked)
        rank_pct     = round((1 - (listing_rank - 1) / total_nbhds) * 100)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.7px;color:#717171;margin-bottom:18px;'>"
            f"{trip_type} &nbsp;·&nbsp; {nights} nights &nbsp;·&nbsp; {pace} pace</div>",
            unsafe_allow_html=True,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PRIMARY — How well does El Born match YOU
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:800;color:#222;margin-bottom:14px;'>"
            "How well does El Born match your travel style?</div>",
            unsafe_allow_html=True,
        )

        col_result, col_map = st.columns([1.4, 1], gap="large")

        with col_result:

            with st.container(border=True):
                score_col, name_col = st.columns([1, 3], gap="medium")
                with score_col:
                    st.markdown(f"""
                    <div style="text-align:center;padding:12px 0;">
                      <div style="font-size:3.5rem;font-weight:800;color:{listing_color};line-height:1;">{listing_score:.0f}</div>
                      <div style="font-size:0.75rem;color:#717171;margin-top:4px;">out of 100</div>
                      <div style="font-size:0.82rem;font-weight:700;color:{listing_color};margin-top:6px;">{listing_label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with name_col:
                    st.markdown(f"""
                    <div style="padding-top:8px;">
                      <div style="font-size:1.15rem;font-weight:800;margin-bottom:3px;">El Born</div>
                      <div style="font-size:0.8rem;color:#717171;margin-bottom:8px;">Sant Pere, Santa Caterina i la Ribera</div>
                      <div style="font-size:0.86rem;color:#444;line-height:1.6;">{listing_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Context: where this neighbourhood ranks vs the rest of the city
            st.markdown(f"""
            <div style="background:#F7F7F7;border-radius:8px;padding:11px 15px;
                        font-size:0.84rem;color:#555;margin:12px 0;line-height:1.6;">
              El Born ranks <strong style="color:#222;">#{listing_rank} out of {total_nbhds}</strong>
              Barcelona neighbourhoods for your profile — better than {rank_pct}% of areas in the city.
              &nbsp;{listing_analysis['model_note']}
            </div>
            """, unsafe_allow_html=True)

            # Strengths and frictions for this listing's neighbourhood
            if listing_analysis["strengths"] or listing_analysis["frictions"]:
                c_str, c_fri = st.columns(2, gap="medium")
                with c_str:
                    if listing_analysis["strengths"]:
                        st.markdown("<div style='font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#00A699;margin-bottom:8px;'>Works well for you</div>", unsafe_allow_html=True)
                        for d, _ in listing_analysis["strengths"]:
                            st.markdown(f"<div style='font-size:0.85rem;margin-bottom:5px;'>&#10003; {DIM_LABELS[d]}</div>", unsafe_allow_html=True)
                with c_fri:
                    if listing_analysis["frictions"]:
                        st.markdown("<div style='font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#FC642D;margin-bottom:8px;'>Worth considering</div>", unsafe_allow_html=True)
                        for d, _ in listing_analysis["frictions"]:
                            st.markdown(f"<div style='font-size:0.85rem;margin-bottom:5px;'>&#9651; {DIM_LABELS[d]}</div>", unsafe_allow_html=True)

            # Radar — your preferences vs El Born's profile
            st.markdown("<div style='font-size:0.88rem;font-weight:700;margin:18px 0 6px;'>Your priorities vs. El Born's profile</div>", unsafe_allow_html=True)
            st.plotly_chart(
                make_radar(user_prefs, listing_row),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Dimension bars
            st.markdown("<div style='font-size:0.88rem;font-weight:700;margin:4px 0 14px;'>Score by dimension</div>", unsafe_allow_html=True)
            for dim in DIMENSIONS:
                score = float(listing_row[dim])
                pref  = user_prefs[dim]
                lbl   = DIM_LABELS[dim]
                st.progress(int(score), text=f"{lbl} — {score:.0f}/100 (your priority: {pref}/5)")

        with col_map:
            st.markdown("<div style='font-size:0.88rem;font-weight:700;margin-bottom:8px;'>Location</div>", unsafe_allow_html=True)
            map_df = pd.DataFrame({"lat": [listing_lat], "lon": [listing_lon]})
            st.map(map_df, zoom=14, use_container_width=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Your fit score", f"{listing_score:.0f}/100")
            m2.metric("Walkability",    f"{float(listing_row['Walkability']):.0f}/100")
            m3, m4 = st.columns(2)
            m3.metric("Food scene", f"{float(listing_row['Food & Restaurants']):.0f}/100")
            m4.metric("Safety",     f"{float(listing_row['Safety']):.0f}/100")

            with st.expander("About El Born"):
                if listing_pros:
                    st.markdown("**What works well**")
                    for pro in listing_pros:
                        st.write(f"+ {pro}")
                if listing_cons:
                    st.markdown("**Worth knowing**")
                    for con in listing_cons:
                        st.write(f"- {con}")
                st.caption("Derived from aggregated guest review sentiment across Inside Airbnb Barcelona listings.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECONDARY — How other neighbourhoods compare
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:800;color:#222;margin-bottom:6px;'>"
            "How other Barcelona neighbourhoods compare</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.84rem;color:#717171;margin-bottom:16px;'>"
            "If El Born isn't the right fit, these are the areas worth exploring. "
            "Browse listings there to find a better match for your travel style.</div>",
            unsafe_allow_html=True,
        )

        for i, row in ranked.iterrows():
            name       = row["neighbourhood"]
            score      = row["fit_score"]
            label, _   = fit_label(score)
            is_listing = (name == LISTING_NEIGHBOURHOOD)

            if score >= 88:   score_cls = "score-good"
            elif score >= 68: score_cls = "score-mid"
            else:             score_cls = "score-low"

            card_cls   = "rank-row top" if is_listing else "rank-row"
            rank_cls   = "rank-num top" if is_listing else "rank-num"
            desc_short = NBHD_INFO.get(name, ("",))[0][:68] + "..."
            this_tag   = (
                "&nbsp;<span style='background:#FF385C;color:white;font-size:0.65rem;"
                "font-weight:700;padding:2px 7px;border-radius:8px;vertical-align:middle;"
                "'>This listing</span>"
            ) if is_listing else ""

            st.markdown(f"""
            <div class="{card_cls}">
              <div class="{rank_cls}">#{i+1}</div>
              <div style="flex:1;">
                <div class="rank-name">{name}{this_tag}</div>
                <div class="rank-desc">{desc_short}</div>
              </div>
              <div class="rank-score {score_cls}">{score:.0f}
                <span style="font-weight:400;color:#AAAAAA;font-size:0.8rem;"> · {label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<div style='font-size:0.82rem;color:#717171;margin-bottom:8px;'>Was this helpful?</div>", unsafe_allow_html=True)
        feedback = st.feedback("thumbs")
        if feedback is not None:
            st.success("Thanks for the feedback. We use this to improve the model.")
