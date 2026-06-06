import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the expected features list
@st.cache_resource
def load_model_and_features():
    model = joblib.load('flight_price_rf_model.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

try:
    model, model_features = load_model_and_features()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Define dropdown categories based on standard dataset values (including the reference/dropped categories)
airlines = ['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara']
source_cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
destination_cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
classes = ['Business', 'Economy']

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="SkyFare | Flight Price Predictor", page_icon="✈️", layout="wide")

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- GLOBAL RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(30, 90, 255, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(0, 200, 180, 0.08) 0%, transparent 60%),
        #080c14 !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    padding: 2.5rem 3rem 4rem 3rem !important;
    max-width: 1300px !important;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1320; }
::-webkit-scrollbar-thumb { background: #1e3a6e; border-radius: 3px; }

/* ---- HERO HEADER ---- */
.hero-wrap {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-left { display: flex; flex-direction: column; gap: 0.4rem; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(30, 90, 255, 0.12);
    border: 1px solid rgba(30, 90, 255, 0.3);
    color: #6ea3ff;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    width: fit-content;
    margin-bottom: 0.5rem;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #4d90ff;
    border-radius: 50%;
    box-shadow: 0 0 6px #4d90ff;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: #ffffff !important;
    letter-spacing: -0.03em !important;
    margin: 0 !important;
}
.hero-title span { color: #4d90ff; }
.hero-sub {
    color: #6b7a99;
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 420px;
    line-height: 1.6;
}
.hero-stat {
    text-align: right;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.2rem;
}
.hero-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #4d90ff;
}
.hero-stat-label { font-size: 0.8rem; color: #4a5568; letter-spacing: 0.06em; }

/* ---- CARD PANELS ---- */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.8rem 1.6rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(77,144,255,0.5), transparent);
    border-radius: 20px 20px 0 0;
}

.panel-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 1rem;
}
.panel-icon-blue  { background: rgba(30,90,255,0.15); border: 1px solid rgba(30,90,255,0.25); }
.panel-icon-teal  { background: rgba(0,200,180,0.12); border: 1px solid rgba(0,200,180,0.22); }
.panel-icon-amber { background: rgba(255,170,30,0.12); border: 1px solid rgba(255,170,30,0.22); }

.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 1.2rem;
}

/* ---- STREAMLIT WIDGET OVERRIDES ---- */
label, .stSelectbox label, .stSlider label, [data-testid="stWidgetLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #8899bb !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(77,144,255,0.5) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #4d90ff !important;
    box-shadow: 0 0 0 3px rgba(77,144,255,0.15) !important;
}
[data-baseweb="select"] * { color: #e8eaf0 !important; font-family: 'DM Sans', sans-serif !important; }
[data-baseweb="popover"] {
    background: #0f1a2e !important;
    border: 1px solid rgba(77,144,255,0.25) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-baseweb="option"] { background: transparent !important; color: #c8d0e0 !important; }
[data-baseweb="option"]:hover { background: rgba(77,144,255,0.15) !important; color: #fff !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 4px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1e5aff, #4d90ff) !important;
    border-radius: 4px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #ffffff !important;
    border: 3px solid #4d90ff !important;
    box-shadow: 0 0 12px rgba(77,144,255,0.5) !important;
    width: 18px !important; height: 18px !important;
}
[data-testid="stSlider"] p, .stSlider span {
    color: #6ea3ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ---- PREDICT BUTTON ---- */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1e5aff 0%, #0d3fcc 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(30,90,255,0.35), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    text-transform: uppercase !important;
    margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2d6aff 0%, #1a50e0 100%) !important;
    box-shadow: 0 6px 32px rgba(30,90,255,0.5), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ---- RESULT CARD ---- */
.result-wrap {
    margin-top: 2.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(77,144,255,0.2);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(30,90,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.result-glyph {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    filter: drop-shadow(0 0 12px rgba(77,144,255,0.5));
}
.result-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.8rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
    text-shadow: 0 0 40px rgba(77,144,255,0.3);
}
.result-price sup { font-size: 1.5rem; font-weight: 600; color: #4d90ff; vertical-align: super; }
.result-note {
    margin-top: 1rem;
    font-size: 0.82rem;
    color: #3d4f6b;
}

/* ---- DIVIDER ---- */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 2.5rem 0 !important; }

/* ---- WARNING / ERROR ---- */
[data-testid="stAlert"] {
    background: rgba(255,170,30,0.07) !important;
    border: 1px solid rgba(255,170,30,0.25) !important;
    border-radius: 12px !important;
    color: #f0c040 !important;
}

/* ---- SPINNER ---- */
[data-testid="stSpinner"] { color: #4d90ff !important; }

/* ---- COLUMN GAPS ---- */
[data-testid="column"] { gap: 0 !important; }
[data-testid="stVerticalBlock"] > div { gap: 1rem !important; }

/* ---- FOOTER ---- */
.footer-bar {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.footer-brand { font-family: 'Syne', sans-serif; font-weight: 700; color: #2a3a5c; font-size: 0.9rem; }
.footer-copy { font-size: 0.78rem; color: #2a3a5c; }
</style>
""", unsafe_allow_html=True)


# ----------------- HERO HEADER -----------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-left">
        <div class="hero-badge">✦ AI-Powered Estimation</div>
        <div class="hero-title">Sky<span>Fare</span></div>
        <p class="hero-sub">Instant, data-driven flight price estimates across India — powered by machine learning trained on millions of fares.</p>
    </div>
    <div class="hero-stat">
        <div class="hero-stat-num">±3.2%</div>
        <div class="hero-stat-label">Average prediction error</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- THREE PANELS -----------------
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    st.markdown("""
    <div class="panel">
        <div class="panel-icon panel-icon-blue">🗺️</div>
        <div class="panel-title">Route Details</div>
    </div>
    """, unsafe_allow_html=True)
    source_city      = st.selectbox("Source City", source_cities, key="src")
    destination_city = st.selectbox("Destination City", destination_cities, key="dst")
    stops = st.slider("Number of Stops", min_value=0, max_value=2, value=0, step=1,
                      help="0 = Non-stop · 1 = One stop · 2 = Two+ stops", key="stops")

with col2:
    st.markdown("""
    <div class="panel">
        <div class="panel-icon panel-icon-teal">🕐</div>
        <div class="panel-title">Flight Schedule</div>
    </div>
    """, unsafe_allow_html=True)
    departure_time = st.selectbox("Departure Time", times, key="dep")
    arrival_time   = st.selectbox("Arrival Time", times, key="arr")
    duration = st.slider("Duration (hours)", min_value=0.5, max_value=50.0, value=2.0, step=0.5, key="dur")

with col3:
    st.markdown("""
    <div class="panel">
        <div class="panel-icon panel-icon-amber">✈️</div>
        <div class="panel-title">Airline & Class</div>
    </div>
    """, unsafe_allow_html=True)
    airline      = st.selectbox("Airline", airlines, key="air")
    flight_class = st.selectbox("Travel Class", classes, key="cls")
    days_left = st.slider("Days to Departure", min_value=1, max_value=50, value=15, step=1, key="days")


# ----------------- PREDICT BUTTON -----------------
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("⟶  Predict Price", type="primary", use_container_width=True)


# ----------------- PREDICTION LOGIC -----------------
if predict_clicked:
    if source_city == destination_city:
        st.warning("⚠️  Source and Destination cities cannot be the same. Please pick a different route.")
    else:
        input_data = {'stops': stops, 'duration': duration, 'days_left': days_left}
        for feature in model_features:
            if feature not in input_data:
                input_data[feature] = 0

        for key, val in [
            (f'airline_{airline}', 1),
            (f'source_city_{source_city}', 1),
            (f'destination_city_{destination_city}', 1),
            (f'departure_time_{departure_time}', 1),
            (f'arrival_time_{arrival_time}', 1),
            (f'class_{flight_class}', 1),
        ]:
            if key in input_data:
                input_data[key] = val

        input_df = pd.DataFrame([input_data])[model_features]

        with st.spinner("Calculating..."):
            prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-glyph">✈</div>
            <div class="result-label">Estimated Flight Price</div>
            <div class="result-price"><sup>₹</sup>{prediction:,.0f}</div>
            <p class="result-note">
                {source_city} → {destination_city} &nbsp;·&nbsp; {airline} &nbsp;·&nbsp; {flight_class}
                &nbsp;·&nbsp; {stops} stop{'s' if stops != 1 else ''} &nbsp;·&nbsp; {days_left}d before departure
            </p>
        </div>
        """, unsafe_allow_html=True)


# ----------------- FOOTER -----------------
st.markdown("""
<div class="footer-bar">
    <span class="footer-brand">✈ SkyFare</span>
    <span class="footer-copy">Predictions are estimates based on historical data and may vary from actual fares.</span>
</div>
""", unsafe_allow_html=True)