# ==========================================================
# MovieIQ — Movie Success Intelligence
# Author  : Shubham Samarpit
# Stack   : Streamlit + scikit-learn + Plotly
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from streamlit_option_menu import option_menu

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# Design tokens — "midnight premiere"
# Charcoal-navy screening room, marquee gold, velvet crimson.
# ----------------------------------------------------------
BG       = "#12141C"
BG_ALT   = "#181B26"
CARD     = "#1D2130"
CARD_HI  = "#242940"
LINE     = "#2E3244"
GOLD     = "#E8B94D"
GOLD_DIM = "#8A6E2F"
CRIMSON  = "#C0392B"
TEXT     = "#F1EDE4"
MUTED    = "#8B8FA3"
GREEN    = "#4FAE8A"

GENRE_COLORS = {
    "Action": "#C0392B", "Adventure": "#E8B94D", "Animation": "#4FAE8A",
    "Comedy": "#D98E4A", "Drama": "#7C86B5", "Horror": "#8E2F3E",
    "Romance": "#C97C9E", "Science Fiction": "#4A8FD9", "Thriller": "#565C7A",
    "Unknown": "#5A5F70",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

.stApp {{
    background: {BG};
    color: {TEXT};
}}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
#MainMenu, footer, header {{visibility: hidden;}}

/* Sprocket-hole divider — the signature motif */
.reel-divider {{
    height: 18px;
    margin: 6px 0 28px 0;
    background-image: radial-gradient({BG} 3px, transparent 4px),
                       radial-gradient({BG} 3px, transparent 4px);
    background-size: 26px 18px;
    background-position: 0 0, 13px 0;
    background-color: {GOLD_DIM};
    border-radius: 2px;
    opacity: 0.55;
}}

/* Marquee hero title */
.marquee-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.2rem;
    letter-spacing: 0.06em;
    line-height: 1.05;
    color: {GOLD};
    text-shadow: 0 0 24px rgba(232,185,77,0.25);
    margin-bottom: 0;
}}
.marquee-sub {{
    font-family: 'Inter', sans-serif;
    color: {MUTED};
    font-size: 1.05rem;
    margin-top: 4px;
    letter-spacing: 0.02em;
}}

/* KPI cards */
.kpi-card {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}}
.kpi-label {{
    color: {MUTED};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    color: {TEXT};
    font-weight: 500;
}}
.kpi-accent {{ color: {GOLD}; }}

/* Section headers — "ticket stub" style */
.section-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(232,185,77,0.10);
    border: 1px solid rgba(232,185,77,0.4);
    color: {GOLD};
    padding: 4px 14px 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 12px;
}}
.section-eyebrow::before {{
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {GOLD};
    box-shadow: 0 0 6px {GOLD};
    flex-shrink: 0;
}}
.section-title-wrap {{
    margin-bottom: 22px;
}}
.section-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.7rem;
    letter-spacing: 0.03em;
    color: {TEXT};
    margin-top: 0;
    line-height: 1.05;
    text-shadow: 0 0 20px rgba(232,185,77,0.12);
}}
.section-underline {{
    width: 64px;
    height: 4px;
    margin-top: 10px;
    border-radius: 3px;
    background: linear-gradient(90deg, {GOLD}, {CRIMSON});
}}

/* Generic panel */
.panel {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 20px 22px;
}}

/* Verdict card in Prediction */
.verdict-hit {{
    background: linear-gradient(135deg, rgba(79,174,138,0.16), {CARD});
    border: 1px solid {GREEN};
    border-radius: 12px;
    padding: 26px;
    text-align: center;
}}
.verdict-miss {{
    background: linear-gradient(135deg, rgba(192,57,43,0.16), {CARD});
    border: 1px solid {CRIMSON};
    border-radius: 12px;
    padding: 26px;
    text-align: center;
}}
.verdict-label {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 0.05em;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {BG_ALT};
    border-right: 1px solid {LINE};
}}

/* Tables */
[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
    border-radius: 8px;
}}

/* Metric override (native st.metric not used, kept for safety) */
[data-testid="stMetricValue"] {{ color: {TEXT}; }}

hr {{ border-color: {LINE}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter"),
        title_font=dict(color=TEXT, size=16),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[GOLD, CRIMSON, "#4A8FD9", GREEN, "#D98E4A", "#7C86B5", "#8E2F3E", "#C97C9E"],
        margin=dict(t=48, l=10, r=10, b=10),
    )
)


def reel_divider():
    st.markdown('<div class="reel-divider"></div>', unsafe_allow_html=True)


def section_header(eyebrow: str, title: str):
    st.markdown(
        f'<div class="section-title-wrap">'
        f'<div class="section-eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-underline"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# Data & model loading
# ----------------------------------------------------------
DATA_PATH_CANDIDATES = [
    os.path.join("Clean Dataset", "movies_clean.csv"),
    "movies_clean.csv",
]
DATA_PATH = next((p for p in DATA_PATH_CANDIDATES if os.path.exists(p)), DATA_PATH_CANDIDATES[-1])

MODEL_PATH_CANDIDATES = [
    os.path.join("Models", "movie_success_model.pkl"),
    os.path.join("models", "movie_success_model.pkl"),
    "movie_success_model.pkl",
]
MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if os.path.exists(p)), MODEL_PATH_CANDIDATES[0])

ENCODER_PATH_CANDIDATES = [
    os.path.join("Models", "genre_encoder.pkl"),
    os.path.join("models", "genre_encoder.pkl"),
    "genre_encoder.pkl",
]
ENCODER_PATH = next((p for p in ENCODER_PATH_CANDIDATES if os.path.exists(p)), ENCODER_PATH_CANDIDATES[0])


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["primary_genre"] = df["primary_genre"].fillna("Unknown")
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


data_error = None
model_error = None
try:
    df = load_data()
except Exception as e:
    df = None
    data_error = str(e)

try:
    model, genre_encoder = load_model()
except Exception as e:
    model, genre_encoder = None, None
    model_error = str(e)

FEATURES = ["budget", "popularity", "runtime", "vote_average", "primary_genre"]


def encode_genre(genre_value: str) -> int:
    """Map a genre string to the encoder's integer code, falling back to 'Unknown'."""
    classes = list(genre_encoder.classes_)
    if genre_value not in classes:
        genre_value = "Unknown" if "Unknown" in classes else classes[0]
    return int(genre_encoder.transform([genre_value])[0])


@st.cache_data
def compute_stat_tests(data: pd.DataFrame):
    """Stage 3 of the brief: T-tests on numeric features + Chi-square on genre."""
    succ = data[data["success"] == 1]
    fail = data[data["success"] == 0]

    ttests = []
    for col, label in [
        ("popularity", "Popularity"),
        ("vote_average", "Vote average"),
        ("runtime", "Runtime"),
        ("budget", "Budget"),
    ]:
        t_stat, p_val = stats.ttest_ind(succ[col], fail[col], equal_var=False)
        ttests.append({
            "Feature": label,
            "Hit mean": succ[col].mean(),
            "Flop mean": fail[col].mean(),
            "t-statistic": t_stat,
            "p-value": p_val,
            "Significant (α=0.05)": "Yes" if p_val < 0.05 else "No",
        })
    ttest_df = pd.DataFrame(ttests)

    contingency = pd.crosstab(data["primary_genre"], data["success"])
    chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)

    return ttest_df, {"chi2": chi2, "p_value": chi2_p, "dof": dof}


# ----------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:2rem;'
        f'color:{GOLD};letter-spacing:0.05em;margin-bottom:0;">🎞️ MOVIEIQ</div>'
        f'<div style="color:{MUTED};font-size:0.85rem;margin-bottom:18px;">'
        f'Movie success intelligence</div>',
        unsafe_allow_html=True,
    )
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Home", "Dashboard", "Prediction", "Dataset", "About"],
        icons=["clipboard-data", "house", "bar-chart-line", "cpu", "table", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": BG_ALT},
            "icon": {"color": GOLD, "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "color": TEXT,
                "text-align": "left",
                "margin": "2px 0",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": CARD_HI, "color": GOLD},
        },
    )
    st.markdown('<div class="reel-divider" style="margin-top:20px;"></div>', unsafe_allow_html=True)
    if df is not None:
        st.caption(f"Dataset · {len(df):,} titles loaded")
    if model is not None:
        st.caption("Model · Random Forest · loaded")
    else:
        st.caption("Model · not found — run train_model.py")


# ==========================================================
# OVERVIEW
# ==========================================================
def show_overview():
    st.markdown('<div class="marquee-title" style="font-size:2.8rem;">OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="marquee-sub">The business case, the method, and what the data actually says.</div>',
        unsafe_allow_html=True,
    )
    reel_divider()

    if df is None:
        st.error(f"Couldn't load the dataset: {data_error}")
        return

    col1, col2 = st.columns(2)
    with col1:
        section_header("Stage 0", "Problem Statement")
        st.markdown(
            """
**Definition of success.** A movie is labeled a **success** when its
`revenue` exceeds its `budget` — a simple break-even rule, not a measure of
critical or cultural impact.

**Why it matters.** Greenlighting a film means committing tens of millions
of dollars before a single ticket sells.
- **Studios** use signals like this to decide which scripts get funded and
  how much to spend on marketing.
- **Investors and distributors** use it to size risk across a slate of
  films rather than betting on any single title.

**Objective.** Explore what's associated with box-office success, test
those associations statistically, train a classifier on the result, and
ship it as something a non-technical stakeholder can actually use.

**Why it's a classification problem.** The target, `success`, only takes
two values — 1 (hit) or 0 (flop). The model isn't predicting a revenue
number; it's predicting which of two bins a title falls into.
"""
        )

    with col2:
        section_header("Stages 1–5", "Approach")
        st.markdown(
            """
1. **Clean** the raw data — parse genres, flag zero/negative budget or
   revenue, remove duplicates.
2. **Engineer** the target and supporting metrics: `success`, `profit`,
   `roi`.
3. **Explore** relationships between budget, revenue, popularity, runtime,
   rating, and genre.
4. **Test statistically** whether those relationships are real or just
   noise (T-tests + a Chi-square test — see below).
5. **Model** success with a Random Forest, evaluate it honestly, and
   **ship** it in this Streamlit app for live scoring.
"""
        )

    reel_divider()
    section_header("Stage 3", "Statistical Testing — what's actually significant")
    ttest_df, chi2_result = compute_stat_tests(df)

    display_df = ttest_df.copy()
    display_df["Hit mean"] = display_df["Hit mean"].map(lambda v: f"{v:,.2f}")
    display_df["Flop mean"] = display_df["Flop mean"].map(lambda v: f"{v:,.2f}")
    display_df["t-statistic"] = display_df["t-statistic"].map(lambda v: f"{v:.3f}")
    display_df["p-value"] = display_df["p-value"].map(lambda v: f"{v:.4f}")
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.caption(
        f"Chi-square test — genre vs. success: χ² = {chi2_result['chi2']:.3f}, "
        f"dof = {chi2_result['dof']}, p = {chi2_result['p_value']:.4f}. "
        f"Null hypothesis: genre and success are independent. "
        f"{'Rejected' if chi2_result['p_value'] < 0.05 else 'Not rejected'} at α = 0.05."
    )
    st.markdown(
        "A p-value is the probability of seeing a difference this large (or larger) "
        "if there were truly no relationship. Below 0.05, we call it significant."
    )

    reel_divider()
    section_header("Findings", "Business recommendations")

    sig_features = display_df[ttest_df["p-value"] < 0.05]["Feature"].tolist()
    genre_sig = chi2_result["p_value"] < 0.05

    findings = []
    if sig_features:
        findings.append(
            f"**{', '.join(sig_features)}** show a statistically significant difference "
            f"between hits and flops — worth weighing in greenlight decisions, though the "
            f"effect sizes are small."
        )
    else:
        findings.append(
            "None of budget, popularity, runtime, or rating show a statistically "
            "significant difference between hits and flops on their own."
        )

    if genre_sig:
        findings.append("Genre is significantly associated with success — some genres are safer bets than others.")
    else:
        findings.append(
            "**Genre shows no significant association with success** (p ≈ "
            f"{chi2_result['p_value']:.2f}). Despite the intuitive pull of 'safe genres,' "
            "this data doesn't back that up — hit rates across genres sit within a few "
            "points of each other."
        )

    findings.append(
        "**Budget and revenue are strongly correlated (r ≈ 0.76)** — bigger productions "
        "scale everything up together, which is exactly why budget alone is a weak "
        "predictor of the *ratio* between the two."
    )
    findings.append(
        "**Recommendation:** treat budget, genre, and runtime as weak individual signals "
        "at best. A real greenlight model would need variables this dataset doesn't have — "
        "marketing spend, release-date competition, franchise/sequel status, and cast "
        "recognition are the more likely levers."
    )

    for f in findings:
        st.markdown(f"- {f}")

    st.info(
        "Open **Dashboard** to explore these relationships visually, or **Prediction** "
        "to score a hypothetical title with the trained model.",
        icon="📋",
    )


# ==========================================================
# HOME
# ==========================================================
def show_home():
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown('<div class="marquee-title">MOVIEIQ</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="marquee-sub">Reading the box office before opening night — '
            'budget, buzz, and runtime run through a trained model to size up a '
            'film\'s odds.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.write("")

    st.write("")
    reel_divider()

    if df is None:
        st.error(f"Couldn't load the dataset: {data_error}")
        return

    total = len(df)
    success_rate = df["success"].mean() * 100
    avg_budget = df["budget"].mean()
    avg_roi = df["roi"].mean()

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "Titles Tracked", f"{total:,}"),
        (c2, "Historical Hit Rate", f"{success_rate:.1f}%"),
        (c3, "Average Budget", f"${avg_budget/1e6:,.1f}M"),
        (c4, "Average ROI", f"{avg_roi:,.0f}%"),
    ]
    for col, label, value in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value kpi-accent">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        section_header("At a glance", "Where the hits come from")
        genre_stats = (
            df.groupby("primary_genre")
            .agg(count=("title", "count"), hit_rate=("success", "mean"))
            .reset_index()
            .sort_values("hit_rate", ascending=False)
        )
        fig = px.bar(
            genre_stats,
            x="hit_rate",
            y="primary_genre",
            orientation="h",
            color="primary_genre",
            color_discrete_map=GENRE_COLORS,
            labels={"hit_rate": "Hit rate", "primary_genre": ""},
            text=genre_stats["hit_rate"].map(lambda v: f"{v*100:.0f}%"),
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig, width='stretch', key="chart_1")

    with col2:
        section_header("Top performers", "Highest ROI titles")
        top_roi = df.nlargest(6, "roi")[["title", "primary_genre", "roi"]]
        for _, row in top_roi.iterrows():
            st.markdown(
                f'<div class="panel" style="margin-bottom:8px;padding:12px 16px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<div><b>{row["title"]}</b><br>'
                f'<span style="color:{MUTED};font-size:0.82rem;">{row["primary_genre"]}</span></div>'
                f'<div style="color:{GOLD};font-family:\'JetBrains Mono\',monospace;">'
                f'+{row["roi"]:.0f}%</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.info(
        "Use **Prediction** to score a hypothetical movie, or open **Dashboard** "
        "for the full exploratory analysis.",
        icon="🎬",
    )


# ==========================================================
# DASHBOARD
# ==========================================================
def show_dashboard():
    section_header("Exploratory analysis", "The Dashboard")

    if df is None:
        st.error(f"Couldn't load the dataset: {data_error}")
        return

    genres = ["All"] + sorted(df["primary_genre"].unique().tolist())
    f1, f2 = st.columns([1, 3])
    with f1:
        genre_pick = st.selectbox("Filter by genre", genres)
    view = df if genre_pick == "All" else df[df["primary_genre"] == genre_pick]

    reel_divider()

    row1a, row1b = st.columns(2)
    with row1a:
        st.markdown("**Budget vs. Revenue**")
        fig = px.scatter(
            view, x="budget", y="revenue", color="primary_genre",
            color_discrete_map=GENRE_COLORS, opacity=0.75,
            hover_data=["title", "success"],
            labels={"budget": "Budget ($)", "revenue": "Revenue ($)", "primary_genre": "Genre"},
        )
        max_val = max(view["budget"].max(), view["revenue"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color=MUTED, dash="dot"))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig, width='stretch', key="chart_2")
        st.caption("Points above the dotted line earned more than they cost.")

    with row1b:
        st.markdown("**ROI Distribution**")
        fig = px.histogram(
            view, x="roi", nbins=40, color_discrete_sequence=[GOLD],
            labels={"roi": "Return on investment (%)"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, bargap=0.05)
        st.plotly_chart(fig, width='stretch', key="chart_3")

    row2a, row2b = st.columns(2)
    with row2a:
        st.markdown("**Runtime vs. Rating**")
        fig = px.scatter(
            view, x="runtime", y="vote_average", color="success",
            color_discrete_map={0: CRIMSON, 1: GREEN},
            labels={"runtime": "Runtime (min)", "vote_average": "Rating", "success": "Success"},
            hover_data=["title"],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=360)
        st.plotly_chart(fig, width='stretch', key="chart_4")

    with row2b:
        st.markdown("**Feature Correlation**")
        numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "profit", "roi"]
        corr = view[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=["#1D2130", GOLD],
            aspect="auto",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=360)
        st.plotly_chart(fig, width='stretch', key="chart_5")

    reel_divider()
    st.markdown("**Genre share of the catalogue**")
    genre_counts = view["primary_genre"].value_counts().reset_index()
    genre_counts.columns = ["primary_genre", "count"]
    fig = px.pie(
        genre_counts, names="primary_genre", values="count", hole=0.55,
        color="primary_genre", color_discrete_map=GENRE_COLORS,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
    st.plotly_chart(fig, width='stretch', key="chart_6")

    reel_divider()
    with st.expander("📋 Statistical tests (T-test + Chi-square) — full results"):
        st.caption("Computed on the full dataset, not the genre filter above.")
        ttest_df, chi2_result = compute_stat_tests(df)
        display_df = ttest_df.copy()
        display_df["Hit mean"] = display_df["Hit mean"].map(lambda v: f"{v:,.2f}")
        display_df["Flop mean"] = display_df["Flop mean"].map(lambda v: f"{v:,.2f}")
        display_df["t-statistic"] = display_df["t-statistic"].map(lambda v: f"{v:.3f}")
        display_df["p-value"] = display_df["p-value"].map(lambda v: f"{v:.4f}")
        st.dataframe(display_df, width="stretch", hide_index=True)
        st.markdown(
            f"Chi-square (genre vs. success): χ² = {chi2_result['chi2']:.3f}, "
            f"dof = {chi2_result['dof']}, p = {chi2_result['p_value']:.4f}. "
            "See **Overview** for what these results mean for the business."
        )


# ==========================================================
# PREDICTION
# ==========================================================
def show_prediction():
    section_header("Score a title", "Prediction")

    if model is None:
        st.error(
            f"No trained model found. Run `train_model.py` first to generate "
            f"`Models/movie_success_model.pkl`.\n\nDetails: {model_error}"
        )
        return

    genres = [g for g in genre_encoder.classes_ if g != "Unknown"]

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input(
                "Budget ($)", min_value=100_000, max_value=400_000_000,
                value=50_000_000, step=1_000_000,
            )
            popularity = st.slider("Popularity score", 0.0, 150.0, 45.0, 0.5)
            runtime = st.slider("Runtime (minutes)", 60, 220, 115)
        with c2:
            vote_average = st.slider("Expected rating (0–10)", 0.0, 10.0, 6.5, 0.1)
            genre = st.selectbox("Primary genre", genres)

        submitted = st.form_submit_button("Run prediction", width='stretch')

    if not submitted:
        st.caption("Fill in the details above and run the prediction.")
        return

    genre_code = encode_genre(genre)
    input_row = pd.DataFrame(
        [[budget, popularity, runtime, vote_average, genre_code]],
        columns=["budget", "popularity", "runtime", "vote_average", "primary_genre"],
    )

    proba = model.predict_proba(input_row)[0]
    hit_prob = proba[1] * 100
    prediction = int(hit_prob >= 50)

    reel_divider()
    col1, col2 = st.columns([1, 1.4])
    with col1:
        if prediction == 1:
            st.markdown(
                f'<div class="verdict-hit"><div class="verdict-label" style="color:{GREEN};">'
                f'LIKELY HIT</div><div style="color:{MUTED};margin-top:6px;">'
                f'{hit_prob:.0f}% modeled probability of revenue &gt; budget</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-miss"><div class="verdict-label" style="color:{CRIMSON};">'
                f'AT RISK</div><div style="color:{MUTED};margin-top:6px;">'
                f'{hit_prob:.0f}% modeled probability of revenue &gt; budget</div></div>',
                unsafe_allow_html=True,
            )

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hit_prob,
            number={"suffix": "%", "font": {"color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": GOLD},
                "bgcolor": CARD,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(192,57,43,0.25)"},
                    {"range": [50, 100], "color": "rgba(79,174,138,0.2)"},
                ],
            },
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=220, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch', key="chart_7")

    st.write("")
    with st.expander("What drove this prediction?"):
        importance = pd.DataFrame({
            "Feature": ["Budget", "Popularity", "Runtime", "Rating", "Genre"],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig = px.bar(
            importance, x="Importance", y="Feature", orientation="h",
            color_discrete_sequence=[GOLD],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=260)
        st.plotly_chart(fig, width='stretch', key="chart_8")
        st.caption(
            "Feature importance is model-wide (from training), not specific to this "
            "single prediction."
        )

    st.warning(
        "Heads-up on model quality: 81% of titles in this dataset are already "
        "labeled 'success', and these five features carry very little signal "
        "for telling flops apart from hits — the probability above will move "
        "only a little as you change the inputs. Read it as a rough lean, not "
        "a verdict — see **About** for the full, honest breakdown.",
        icon="⚠️",
    )


# ==========================================================
# DATASET
# ==========================================================
def show_dataset():
    section_header("Explore the source data", "Dataset")

    if df is None:
        st.error(f"Couldn't load the dataset: {data_error}")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        genre_filter = st.multiselect(
            "Genre", sorted(df["primary_genre"].unique().tolist())
        )
    with f2:
        success_filter = st.selectbox("Outcome", ["All", "Success only", "Flop only"])
    with f3:
        budget_range = st.slider(
            "Budget range ($M)",
            float(df["budget"].min() / 1e6),
            float(df["budget"].max() / 1e6),
            (float(df["budget"].min() / 1e6), float(df["budget"].max() / 1e6)),
        )

    view = df.copy()
    if genre_filter:
        view = view[view["primary_genre"].isin(genre_filter)]
    if success_filter == "Success only":
        view = view[view["success"] == 1]
    elif success_filter == "Flop only":
        view = view[view["success"] == 0]
    view = view[
        (view["budget"] / 1e6 >= budget_range[0]) & (view["budget"] / 1e6 <= budget_range[1])
    ]

    st.caption(f"Showing {len(view):,} of {len(df):,} titles")
    st.dataframe(
        view[["title", "primary_genre", "budget", "revenue", "popularity",
              "runtime", "vote_average", "roi", "success"]],
        width='stretch',
        height=440,
    )

    st.download_button(
        "Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="movieiq_filtered.csv",
        mime="text/csv",
    )


# ==========================================================
# ABOUT
# ==========================================================
def show_about():
    section_header("The project", "About MovieIQ")

    st.markdown(
        f"""
<div class="panel">
MovieIQ estimates whether a movie is likely to earn back its budget, using a
Random Forest trained on budget, popularity, runtime, audience rating, and
genre. It started as a civil engineering background applied to a very
different kind of structural question: what holds up a film's box office
outcome.
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        section_header("Pipeline", "How it's built")
        st.markdown(
            f"""
1. **Clean** — raw TMDB-style data parsed for genre, deduplicated, and checked for
   zero/negative values (`data_cleaning.ipynb`).
2. **Engineer** — `success`, `profit`, and `roi` derived from budget and revenue.
3. **Explore** — distributions and correlations reviewed (`eda.ipynb`).
4. **Train** — a Random Forest (200 trees) fit on five features (`train_model.py`).
5. **Serve** — this Streamlit app loads the saved model for live scoring.
"""
        )
    with col2:
        section_header("Honest limits", "Model quality, plainly")
        st.markdown(
            """
- The dataset is imbalanced: about **81% of titles are labeled a success**,
  so a model guessing "hit" every time would already score ~81% accuracy.
- Trained plainly, the Random Forest **did exactly that** — it defaulted to
  predicting every title a hit. This build uses `class_weight="balanced"`
  plus a capped tree depth so it actually differentiates, but recall on
  flops is still weak (roughly 10%): budget, popularity, runtime, rating,
  and genre alone don't carry much signal for *this* outcome definition.
- **Missing genres** (previously left blank) are now coded as their own
  `Unknown` category rather than dropped or silently mismatched.
- Budget, popularity, and rating are known figures here — in a real
  pre-release scenario, some inputs would be forecasts, not facts.
- Treat predictions as a **rough lean**, not a verdict — and treat this
  page as a demonstration of the pipeline, not a production model.
"""
        )

    reel_divider()
    section_header("Stack", "Built with")
    tools = ["Python", "Pandas", "Scikit-learn", "Streamlit", "Plotly", "Joblib"]
    cols = st.columns(len(tools))
    for c, t in zip(cols, tools):
        with c:
            st.markdown(
                f'<div class="panel" style="text-align:center;padding:14px;">{t}</div>',
                unsafe_allow_html=True,
            )

    reel_divider()
    st.markdown(
        f'<div style="color:{MUTED};">Author · <span style="color:{TEXT};">'
        f'Shubham Samarpit</span> — Civil Engineering → Data Analytics</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# Router
# ==========================================================
PAGES = {
    "Overview": show_overview,
    "Home": show_home,
    "Dashboard": show_dashboard,
    "Prediction": show_prediction,
    "Dataset": show_dataset,
    "About": show_about,
}
PAGES[selected]()
