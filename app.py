import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="⚽",
    layout="wide"
)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    # Caching prevents reloading the heavy models on every click
    classifier = joblib.load("classifier.pkl")
    home_goal_model = joblib.load("home_goal_model.pkl")
    away_goal_model = joblib.load("away_goal_model.pkl")
    return classifier, home_goal_model, away_goal_model

try:
    classifier, home_goal_model, away_goal_model = load_models()
except Exception as e:
    st.error(f"Error loading machine learning models: {e}")
    st.stop()

# =====================================================
# LOAD FIFA RANKINGS & DICT CONFIG
# =====================================================
@st.cache_data
def load_data():
    rankings = pd.read_csv("fifa_rankings.csv")
    latest_rankings = rankings.sort_values("rank_date").drop_duplicates("country_full", keep="last")
    
    team_rankings = {}
    for _, row in latest_rankings.iterrows():
        team_rankings[row["country_full"]] = {
            "rank": row["rank"],
            "points": row["total_points"]
        }
    return team_rankings

try:
    team_rankings = load_data()
    teams = sorted(team_rankings.keys())
except Exception as e:
    st.error(f"Error loading system dataset: {e}")
    st.stop()

# =====================================================
# PREMIUM MODERN UI CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    /* Dark glassmorphism background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 70%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Global Card styling */
    .card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Match Day Layout Container */
    .match-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(34, 197, 94, 0.2);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        text-align: center;
        margin: 25px 0;
    }
    
    .match-team {
        flex: 2;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    .match-score-box {
        flex: 1;
        background: rgba(0, 0, 0, 0.3);
        padding: 15px 30px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .match-score {
        font-size: 75px;
        font-weight: 900;
        color: #22c55e;
        line-height: 1;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
    }
    
    .vs-label {
        font-size: 14px;
        color: #64748b;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Single Segmented Probability Bar */
    .prob-bar-container {
        display: flex;
        height: 24px;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        background-color: #334155;
        margin: 15px 0 30px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    .prob-home { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .prob-draw { background: linear-gradient(90deg, #64748b, #475569); }
    .prob-away { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    /* Stats labels */
    .stat-label {
        font-size: 15px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 700;
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO HEADER SECTION
# =====================================================
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='color:white; font-size:56px; font-weight: 900; margin-bottom:0px; letter-spacing:-1px;'>
        ⚽ FIFA WORLD CUP 2026
    </h1>
    <p style='color:#64748b; font-size:20px; font-weight:500; letter-spacing:1px; margin-top:5px;'>
        ADVANCED AI MATCH PREDICTION ENGINE
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR INFO
# =====================================================
st.sidebar.title("⚙️ Engine Metrics")
st.sidebar.markdown("""
<div class="card" style="padding:15px;">
    <h4 style='margin-top:0; color: white;'>Framework Architecture</h4>
    <p style='font-size:14px; color:#94a3b8; line-height:1.6;'>
        ⚡ <b>Random Forest Classifier</b> for match outcome probabilities.<br><br>
        🎯 <b>Random Forest Regressor</b> for distinct goals metrics computation.<br><br>
        📈 Structured against active real-time FIFA point indexes.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# TEAM SELECTORS (Clean Visual Cards)
# =====================================================
st.markdown("<h3 style='font-size:22px; font-weight:700;'>Select Competitors</h3>", unsafe_allow_html=True)
sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    team1 = st.selectbox("🏠 Home Team Venue Configuration", teams, key="home_select")
with sel_col2:
    # Set default away team securely to avoid array conflicts
    default_away = teams.index("Albania") if "Albania" in teams else min(1, len(teams)-1)
    team2 = st.selectbox("✈️ Away Team Venue Configuration", teams, index=default_away, key="away_select")

st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
predict = st.button("🚀 EXECUTE AI PREDICTION SIMULATION", use_container_width=True)
st.markdown("---")

# =====================================================
# PREDICTION SIMULATION ENGINE RUN
# =====================================================
if predict:
    if team1 == team2:
        st.error("Validation Error: Please configure two distinct nations to evaluate outcomes.")
    else:
        # Features Engineering Extraction
        team1_rank = team_rankings[team1]["rank"]
        team2_rank = team_rankings[team2]["rank"]
        team1_points = team_rankings[team1]["points"]
        team2_points = team_rankings[team2]["points"]

        rank_difference = float(team1_rank - team2_rank)
        points_difference = float(team1_points - team2_points)

        # Formatted explicitly as a 2D matrix array to ensure compliance with all scikit-learn models
        X = np.array([[rank_difference, points_difference]])

        # Running Models safely
        probabilities = classifier.predict_proba(X)[0]
        home_goals = max(0, int(round(home_goal_model.predict(X)[0])))
        away_goals = max(0, int(round(away_goal_model.predict(X)[0])))

        # Safe Tuple/Array Index Checks for Win Probabilities
        home_win_prob = probabilities[0]
        away_win_prob = probabilities[1]
        
        # If your classifier only outputs 2 classes (Win/Loss), set draw probability cleanly to 0.0
        if len(probabilities) >= 3:
            draw_prob = probabilities[2]
        else:
            draw_prob = 0.0

        # UI ELEMENT 1: PREMIUM LIVE-SCORE CARD LAYOUT
        st.markdown(f"""
        <div class="match-container">
            <div class="match-team">🏠 {team1}</div>
            <div class="match-score-box">
                <div class="match-score">{home_goals} : {away_goals}</div>
                <div class="vs-label">Predicted Score</div>
            </div>
            <div class="match-team">✈️ {team2}</div>
        </div>
        """, unsafe_allow_html=True)

        # UI ELEMENT 2: SINGLE LUXURY SPORTS MATCH SEGMENTED PROBABILITY BAR
        st.markdown("<h3 style='font-size:22px; font-weight:700; margin-bottom:5px;'>📊 Win Probability Spectrum</h3>", unsafe_allow_html=True)
        
        # Build clean custom CSS distribution strip based on runtime weights
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-home" style="width: {home_win_prob*100}%" title="Home Win"></div>
            <div class="prob-draw" style="width: {draw_prob*100}%" title="Draw"></div>
            <div class="prob-away" style="width: {away_win_prob*100}%" title="Away Win"></div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Metrics Display Boxes
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🔹 {team1} Win", f"{home_win_prob*100:.1f}%")
        m2.metric("⚪ Match Draw Event", f"{draw_prob*100:.1f}%")
        m3.metric(f"🔻 {team2} Win", f"{away_win_prob*100:.1f}%")

        # UI ELEMENT 3: NATION PERFORMANCE MATRIX DATA CARDS
        st.markdown("<br><h3 style='font-size:22px; font-weight:700;'>📌 Team Analytical Profiles</h3>", unsafe_allow_html=True)
        col_prof1, col_prof2 = st.columns(2)

        with col_prof1:
            st.markdown(f"""
            <div class="card">
                <h3 style='margin-top:0; color:#3b82f6;'>🏠 {team1}</h3>
                <hr style='border: 1px solid rgba(255,255,255,0.05); margin: 12px 0;'>
                <table style='width:100%; border-collapse: collapse;'>
                    <tr>
                        <td class="stat-label" style='padding: 8px 0;'>Current FIFA Ranking</td>
                        <td class="stat-value" style='text-align:right;'>#{int(team1_rank)}</td>
                    </tr>
                    <tr>
                        <td class="stat-label" style='padding: 8px 0;'>Total FIFA Index Points</td>
                        <td class="stat-value" style='text-align:right; color:#22c55e;'>{team1_points:,.1f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_prof2:
            st.markdown(f"""
            <div class="card">
                <h3 style='margin-top:0; color:#ef4444;'>✈️ {team2}</h3>
                <hr style='border: 1px solid rgba(255,255,255,0.05); margin: 12px 0;'>
                <table style='width:100%; border-collapse: collapse;'>
                    <tr>
                        <td class="stat-label" style='padding: 8px 0;'>Current FIFA Ranking</td>
                        <td class="stat-value" style='text-align:right;'>#{int(team2_rank)}</td>
                    </tr>
                    <tr>
                        <td class="stat-label" style='padding: 8px 0;'>Total FIFA Index Points</td>
                        <td class="stat-value" style='text-align:right; color:#22c55e;'>{team2_points:,.1f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)