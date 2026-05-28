import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ----------------------------------------------------------------
st.set_page_config(
    page_title="FIFA 2026 AI Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Global Dark UI Overrides
st.markdown("""
<style>
    .main { background-color: #0b0d16; color: #ffffff; }
    div[data-testid="stSidebar"] { background-color: #111625; border-right: 1px solid rgba(255,255,255,0.05); }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00FF66 0%, #00CC52 100%);
        color: #000000 !important;
        font-weight: 700 !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(0, 255, 102, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# 2. CORE MACHINE LEARNING MODEL LOADER
# ----------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    classifier = joblib.load("classifier.pkl")
    home_model = joblib.load("home_goal_model.pkl")
    away_model = joblib.load("away_goal_model.pkl")
    rankings = pd.read_csv("fifa_rankings.csv")
    return classifier, home_model, away_model, rankings

try:
    classifier, home_goal_model, away_goal_model, rankings_df = load_models_and_data()
    # Extract clean list of unique country names using the correct column 'country_full'
    available_teams = sorted(rankings_df['country_full'].unique())
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# ----------------------------------------------------------------
# 3. SIDEBAR ENGINE METRICS & ARCHITECTURE DISPLAY
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Engine Metrics")
    
    st.markdown("""
    <div style="background-color: #161c30; padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        <p style="font-weight:700; margin-bottom:8px; color:#8a99ad; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Framework Architecture</p>
        <p style="font-size:14px; margin-bottom:12px;">⚡ <b>Random Forest Classifier</b> for match outcome probabilities.</p>
        <p style="font-size:14px; margin-bottom:12px;">🎯 <b>Random Forest Regressor</b> for distinct goals metrics computation.</p>
        <p style="font-size:14px; margin-bottom:0px;">📊 Structured against active real-time FIFA point indexes.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------
# 4. MAIN USER INTERFACE HEADER
# ----------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-size: 42px; font-weight: 800; margin-bottom: 5px; letter-spacing: -0.5px;">⚽ FIFA WORLD CUP 2026</h1>
    <p style="font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #8a99ad;">Advanced AI Match Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

# Competitor Selection Columns
st.markdown("### Select Competitors")
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🏠 Home Team Venue Configuration", options=available_teams, index=0)
with col2:
    team2 = st.selectbox("✈️ Away Team Venue Configuration", options=available_teams, index=min(1, len(available_teams)-1))

st.markdown("<br>", unsafe_allow_html=True)
trigger_prediction = st.button("🚀 EXECUTE AI PREDICTION SIMULATION")
st.markdown("---")

# ----------------------------------------------------------------
# 5. MACHINE LEARNING INFERENCE & RESPONSIVE UI PROCESSING
# ----------------------------------------------------------------
if trigger_prediction:
    if team1 == team2:
        st.warning("⚠️ Invalid Selection: Please choose two separate countries to simulate a match.")
    else:
        # Fetching input baseline details using the correct column 'country_full'
        t1_data = rankings_df[rankings_df['country_full'] == team1].iloc[0]
        t2_data = rankings_df[rankings_df['country_full'] == team2].iloc[0]
        
        # Computing relative baseline differentials for the ML input vector (X)
        rank_diff = t1_data['rank'] - t2_data['rank']
        point_diff = t1_data['total_points'] - t2_data['total_points']
        
        # Build your features array matching your training shape
        X = np.array([[rank_diff, point_diff]])
        
        # 1. Run Probability Inference
        probabilities = classifier.predict_proba(X)[0] # [Home_Win, Away_Win, Draw]
        home_win_prob = probabilities[0]
        away_win_prob = probabilities[1]
        draw_prob = probabilities[2]
        
        # 2. Run Exact Score Regressions
        home_goals = max(0, int(round(home_goal_model.predict(X)[0])))
        away_goals = max(0, int(round(away_goal_model.predict(X)[0])))
        
        # ------------------------------------------------------------
        # UI COMPONENT: RESPONSIVE FLEXBOX LIVE-SCORECARD CARD
        # ------------------------------------------------------------
        # Outer static framework structure 
        card_top = """
        <div style="
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(145deg, #111625, #161c30);
            padding: 22px 15px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            max-width: 500px;
            margin: 0 auto 25px auto;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        ">
        """
        
        # Home block content injection
        home_block = f"""
            <div style="flex: 1; text-align: center; min-width: 70px;">
                <div style="font-size: 30px; margin-bottom: 4px;">🏠</div>
                <div style="font-size: 15px; font-weight: 700; color: #ffffff; white-space: normal; word-wrap: break-word;">{team1}</div>
            </div>
        """
        
        # Center Score Box content injection with fixed flex rows
        score_block = f"""
            <div style="
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: #0b0d16;
                padding: 12px 22px;
                border-radius: 12px;
                min-width: 120px;
                margin: 0 12px;
                border: 1px solid rgba(0, 255, 102, 0.15);
            ">
                <div style="display: flex; flex-direction: row; align-items: center; justify-content: center; gap: 12px;">
                    <span style="font-size: 40px; font-weight: 800; color: #00FF66; line-height: 1;">{home_goals}</span>
                    <span style="font-size: 28px; font-weight: 800; color: #00FF66; line-height: 1; opacity: 0.7; padding-bottom: 4px;">:</span>
                    <span style="font-size: 40px; font-weight: 800; color: #00FF66; line-height: 1;">{away_goals}</span>
                </div>
                <div style="font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad; margin-top: 8px; text-align: center;">Predicted Score</div>
            </div>
        """
        
        # Away block content injection
        away_block = f"""
            <div style="flex: 1; text-align: center; min-width: 70px;">
                <div style="font-size: 30px; margin-bottom: 4px;">✈️</div>
                <div style="font-size: 15px; font-weight: 700; color: #ffffff; white-space: normal; word-wrap: break-word;">{team2}</div>
            </div>
        """
        
        card_bottom = "</div>"
        
        # Assemble whole clean container and print safely to browser UI 
        full_scorecard_html = card_top + home_block + score_block + away_block + card_bottom
        st.markdown(full_scorecard_html, unsafe_allow_html=True)
        
        # ------------------------------------------------------------
        # UI COMPONENT: SINGLE LUXURY SPORTS MATCH DISTRIBUTION STRIP
        # ------------------------------------------------------------
        st.markdown("<h3 style='font-size:20px; font-weight:700; margin-bottom:15px;'>📊 Win Probability Spectrum</h3>", unsafe_allow_html=True)
        
        # Math percentage transformations
        p_home = home_win_prob * 100
        p_draw = draw_prob * 100
        p_away = away_win_prob * 100
        
        # Custom-engineered runtime segment alignment strip layout
        prob_bar_html = f"""
        <div style="
            width: 100%; 
            background-color: #111625; 
            border-radius: 50px; 
            display: flex; 
            overflow: hidden; 
            height: 24px;
            margin-bottom: 8px;
            border: 1px solid rgba(255,255,255,0.02);
        ">
            <div style="width: {p_home}%; background: linear-gradient(90deg, #00FF66, #00CC52); display: flex; align-items: center; justify-content: center; color: black; font-weight: 700; font-size: 11px;">{p_home:.1f}%</div>
            <div style="width: {p_draw}%; background-color: #2c354d; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px;">{p_draw:.1f}%</div>
            <div style="width: {p_away}%; background: linear-gradient(90deg, #0099FF, #0066FF); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 11px;">{p_away:.1f}%</div>
        </div>
        
        <div style="display: flex; justify-content: space-between; padding: 0 10px; font-size: 12px; font-weight: 600; color: #8a99ad;">
            <span>🟢 Home Win</span>
            <span>⚪ Draw Probability</span>
            <span>🔵 Away Win</span>
        </div>
        """
        st.markdown(prob_bar_html, unsafe_allow_html=True)