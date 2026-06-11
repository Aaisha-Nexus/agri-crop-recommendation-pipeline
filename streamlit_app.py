import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai

# -----------------------
# PAGE SETUP
# -----------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
columns_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")

# -----------------------
# LOAD MODEL FILES
# -----------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(columns_path)

crop_mapping = {
    0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea",
    4: "coconut", 5: "coffee", 6: "cotton", 7: "grapes",
    8: "jute", 9: "kidneybeans", 10: "lentil", 11: "maize",
    12: "mango", 13: "mothbeans", 14: "mungbean", 15: "muskmelon",
    16: "orange", 17: "papaya", 18: "pigeonpeas",
    19: "pomegranate", 20: "rice", 21: "watermelon"
}

crop_icons = {
    "rice": "🌾",
    "chickpea": "🌱",
    "maize": "🌽",
    "banana": "🍌",
    "mango": "🥭",
    "apple": "🍎",
    "coconut": "🥥",
    "coffee": "☕",
    "cotton": "☁️",
    "grapes": "🍇",
    "orange": "🍊",
    "watermelon": "🍉",
    "pomegranate": "🔴",
    "papaya": "🟠",
    "muskmelon": "🍈",
    "jute": "🌿",
    "lentil": "🌱",
    "mungbean": "🌱",
    "mothbeans": "🌱",
    "blackgram": "🌱",
    "kidneybeans": "🫘",
    "pigeonpeas": "🌱"
}



# -----------------------
# SESSION STATE
# -----------------------
# This keeps the latest prediction available for the AI assistant.
# If the user asks a question after prediction, the assistant can use this context.
if "latest_prediction_context" not in st.session_state:
    st.session_state.latest_prediction_context = None

def html(content):
    st.html(content)


# -----------------------
# CSS
# -----------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f1e5 0%, #e9f3d9 100%);
        color: #14210f;
    }

    .block-container {
        padding-top: 0rem;
        max-width: 1180px;
    }

    header {
        visibility: hidden;
    }

    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 24px 0 14px 0;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 19px;
        font-weight: 700;
        color: #14210f;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #285515;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 23px;
    }

    .nav-link {
        color: #4f6a3e;
        font-weight: 700;
        font-size: 16px;
        text-decoration: none;
    }

    .hero {
        position: relative;
        min-height: 520px;
        overflow: hidden;
        padding: 70px 48px 36px 48px;
        border-radius: 0 0 36px 36px;
        background:
            radial-gradient(circle at 18% 16%, rgba(255,255,255,0.85) 0 7%, transparent 18%),
            radial-gradient(circle at 42% 8%, rgba(255,255,255,0.65) 0 5%, transparent 12%),
            linear-gradient(105deg, #f8f3e8 0%, #eaf3d9 52%, #cfe7b5 100%);
        box-shadow: 0 20px 45px rgba(54, 78, 42, 0.12);
    }

    .hero:before {
        content: "";
        position: absolute;
        bottom: -55px;
        left: -100px;
        width: 120%;
        height: 210px;
        background: rgba(122, 160, 98, 0.20);
        border-radius: 50% 50% 0 0;
        transform: rotate(-2deg);
    }

    .hero:after {
        content: "";
        position: absolute;
        bottom: -70px;
        right: -160px;
        width: 75%;
        height: 260px;
        background: rgba(87, 137, 66, 0.35);
        border-radius: 50% 50% 0 0;
        transform: rotate(2deg);
    }

    .badge {
        display: inline-block;
        padding: 11px 24px;
        background: rgba(255,255,255,0.88);
        border: 1px solid #d8dec9;
        border-radius: 999px;
        color: #285515;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 34px;
        position: relative;
        z-index: 2;
    }

    .hero-title {
        font-family: Georgia, serif;
        font-size: 56px;
        line-height: 1.08;
        font-weight: 900;
        color: #203714;
        max-width: 590px;
        position: relative;
        z-index: 2;
        margin-bottom: 24px;
    }

    .hero-text {
        font-size: 18px;
        line-height: 1.8;
        color: #5d714e;
        max-width: 550px;
        position: relative;
        z-index: 2;
    }

    .hero-stats {
        display: flex;
        gap: 56px;
        margin-top: 72px;
        position: relative;
        z-index: 2;
    }

    .hero-stat-number {
        font-size: 30px;
        font-weight: 900;
        color: #111b0d;
    }

    .hero-stat-label {
        color: #5d714e;
        font-size: 14px;
        margin-top: 4px;
    }

    .windmill {
        position: absolute;
        right: 115px;
        top: 95px;
        width: 260px;
        height: 430px;
        z-index: 4;
    }

    .tower {
        position: absolute;
        left: 105px;
        top: 185px;
        width: 60px;
        height: 260px;
        background: linear-gradient(90deg, #415238, #778b62, #415238);
        clip-path: polygon(25% 0%, 75% 0%, 100% 100%, 0% 100%);
        border-radius: 12px;
    }

    .tower-base {
        position: absolute;
        left: 86px;
        top: 430px;
        width: 100px;
        height: 22px;
        background: rgba(48, 67, 40, 0.55);
        border-radius: 50%;
    }

    .hub-box {
        position: absolute;
        left: 91px;
        top: 163px;
        width: 88px;
        height: 36px;
        background: #5f7253;
        border-radius: 9px;
    }

    .hub-center {
        position: absolute;
        left: 118px;
        top: 153px;
        width: 39px;
        height: 39px;
        background: #285515;
        border-radius: 50%;
        z-index: 5;
    }

    .hub-center:after {
        content: "";
        position: absolute;
        left: 12px;
        top: 12px;
        width: 15px;
        height: 15px;
        background: #c98a24;
        border-radius: 50%;
    }

    .rotor {
        position: absolute;
        left: 137px;
        top: 172px;
        width: 0;
        height: 0;
        animation: rotateWindmill 6s linear infinite;
        transform-origin: 0 0;
        z-index: 4;
    }

    .blade {
        position: absolute;
        left: 0;
        top: -8px;
        width: 170px;
        height: 18px;
        background: #86af72;
        border-radius: 100% 10px 10px 100%;
        transform-origin: 0 50%;
    }

    .blade.b1 { transform: rotate(0deg); }
    .blade.b2 { transform: rotate(120deg); }
    .blade.b3 { transform: rotate(240deg); }

    @keyframes rotateWindmill {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .small-label {
        letter-spacing: 6px;
        text-transform: uppercase;
        color: #496b3a;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .section-title {
        font-family: Georgia, serif;
        font-size: 32px;
        color: #111b0d;
        font-weight: 900;
        margin-bottom: 24px;
    }

    div[data-testid="stNumberInput"] label {
        color: #16320f !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #eee8db !important;
        border: 1px solid #d5ceb9 !important;
        border-radius: 18px !important;
        color: #111b0d !important;
        height: 52px !important;
        font-size: 17px !important;
    }

    .hint {
        color: #5d714e;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    .stButton>button {
        background: #285515;
        color: white;
        border-radius: 18px;
        border: none;
        padding: 15px 22px;
        font-size: 18px;
        font-weight: 800;
        width: 100%;
        height: 58px;
        margin-top: 18px;
        box-shadow: 0 12px 25px rgba(40,85,21,0.25);
    }

    .stButton>button:hover {
        background: #1d3f0f;
        color: white;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #d9dec9;
        border-top: 9px solid #c98a24;
        border-radius: 30px;
        box-shadow: 0 20px 45px rgba(44, 63, 34, 0.15);
        padding: 44px 50px;
        margin-top: 46px;
    }

    .crop-header {
        display: flex;
        align-items: center;
        gap: 22px;
        margin: 4px 0 18px 0;
    }

    .crop-icon-box {
        width: 82px;
        height: 82px;
        border-radius: 22px;
        background: #fff3dc;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        border: 1px solid #eeddbb;
        flex-shrink: 0;
    }

    .crop-name {
        font-family: Georgia, serif;
        font-size: 56px;
        font-weight: 900;
        color: #111b0d;
        text-transform: capitalize;
        margin: 0;
    }

    .confidence-pill {
        display: inline-block;
        padding: 9px 16px;
        border-radius: 999px;
        background: #e9f3dc;
        border: 1px solid #d5e4c2;
        color: #285515;
        font-weight: 800;
    }

    .confidence-track {
        height: 10px;
        width: 420px;
        max-width: 100%;
        background: #e8e2d7;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 12px;
    }

    .confidence-fill {
        height: 10px;
        background: #c98a24;
        border-radius: 20px;
    }

    .reason-box {
        background: #fff3dc;
        border-radius: 22px;
        padding: 24px;
        margin-top: 24px;
        color: #26331d;
        font-size: 17px;
        line-height: 1.7;
    }

    .chip {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        border: 1px solid #d8dec9;
        color: #5d714e;
        background: white;
        margin-right: 8px;
        margin-top: 18px;
        font-size: 14px;
    }

    .section-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #d9dec9;
        border-radius: 28px;
        box-shadow: 0 18px 38px rgba(44, 63, 34, 0.12);
        padding: 34px 38px;
        margin-top: 28px;
        min-height: 360px;
    }

    .bars {
        display: flex;
        align-items: flex-end;
        justify-content: space-around;
        height: 170px;
        margin-top: 54px;
    }

    .bar-wrap {
        text-align: center;
        color: #5d714e;
        font-size: 14px;
    }

    .bar {
        width: 42px;
        border-radius: 8px 8px 0 0;
        margin: 0 auto 12px auto;
    }

    .bar.n { background: #285515; }
    .bar.p { background: #6aa14d; }
    .bar.k { background: #c98a24; }

    .vote-row {
        margin-bottom: 16px;
    }

    .vote-label {
        display: flex;
        justify-content: space-between;
        color: #5d714e;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .vote-track {
        height: 8px;
        background: #e8e2d7;
        border-radius: 20px;
        overflow: hidden;
    }

    .vote-fill {
        height: 8px;
        background: #6aa14d;
        border-radius: 20px;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 22px;
        margin-top: 22px;
    }

    .metric-box {
        background: #edf5df;
        border: 1px solid #dce8ca;
        padding: 26px;
        border-radius: 24px;
        text-align: center;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 900;
        color: #285515;
        margin-bottom: 5px;
    }

    .metric-label {
        color: #5d714e;
        font-size: 14px;
    }


    .radar-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2px;
    }

    .profile-list {
        margin-top: 18px;
        font-size: 15px;
        color: #5d714e;
        line-height: 2;
    }

    .footer-action {
        text-align: center;
        margin: 34px 0 10px 0;
        color: #5d714e;
        font-size: 17px;
        font-weight: 700;
    }

    .assistant-card {
        background: linear-gradient(135deg, #ffffff 0%, #fff8e8 100%);
        border: 1px solid #d9dec9;
        border-radius: 30px;
        box-shadow: 0 20px 45px rgba(44, 63, 34, 0.13);
        padding: 38px 44px;
        margin-top: 36px;
    }

    .assistant-title-row {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 12px;
    }

    .assistant-icon {
        width: 62px;
        height: 62px;
        border-radius: 20px;
        background: #e9f3dc;
        border: 1px solid #d5e4c2;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 31px;
    }

    .assistant-title {
        font-family: Georgia, serif;
        font-size: 32px;
        color: #111b0d;
        font-weight: 900;
        margin-bottom: 3px;
    }

    .assistant-subtitle {
        color: #5d714e;
        font-size: 15px;
        line-height: 1.7;
    }

    .assistant-context {
        background: #edf5df;
        border: 1px solid #dce8ca;
        border-radius: 20px;
        padding: 18px 22px;
        color: #355229;
        font-size: 14px;
        line-height: 1.8;
        margin: 22px 0;
    }

    .assistant-answer {
        background: #ffffff;
        border: 1px solid #eadfc9;
        border-left: 6px solid #c98a24;
        border-radius: 22px;
        padding: 24px 26px;
        color: #26331d;
        font-size: 16px;
        line-height: 1.8;
        margin-top: 22px;
        box-shadow: 0 10px 24px rgba(44, 63, 34, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------
# FEATURE HELPERS
# -----------------------
def get_rainfall_category(rainfall):
    if rainfall <= 100:
        return "Low"
    elif rainfall <= 200:
        return "Medium"
    return "High"


def get_temperature_category(temperature):
    if temperature < 20:
        return "Cool"
    elif temperature < 30:
        return "Moderate"
    return "Hot"


def get_humidity_category(humidity):
    if humidity < 40:
        return "Low"
    elif humidity < 70:
        return "Medium"
    return "High"


def get_season_type(temperature):
    if temperature < 20:
        return "Winter"
    elif temperature < 30:
        return "Moderate"
    return "Summer"


def create_model_input(N, P, K, temperature, humidity, ph, rainfall):
    row = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    rainfall_cat = get_rainfall_category(rainfall)
    temp_cat = get_temperature_category(temperature)
    humidity_cat = get_humidity_category(humidity)
    season = get_season_type(temperature)

    row.update({
        "rainfall_category_Low": 0,
        "rainfall_category_Medium": 0,
        "rainfall_category_High": 0,
        "temperature_category_Cool": 0,
        "temperature_category_Moderate": 0,
        "temperature_category_Hot": 0,
        "humidity_category_Low": 0,
        "humidity_category_Medium": 0,
        "humidity_category_High": 0,
        "season_type_Moderate": 0,
        "season_type_Summer": 0,
        "season_type_Winter": 0
    })

    row[f"rainfall_category_{rainfall_cat}"] = 1
    row[f"temperature_category_{temp_cat}"] = 1
    row[f"humidity_category_{humidity_cat}"] = 1
    row[f"season_type_{season}"] = 1

    input_df = pd.DataFrame([row])

    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    return input_df


def reason_text(crop, rainfall, humidity, temperature, ph):
    crop_title = crop.title()

    custom = {
        "Rice": "High humidity and substantial rainfall make this field profile suitable for rice cultivation. Rice generally performs well in moist and warm growing conditions.",
        "Chickpea": "Lower humidity, moderate rainfall, and cooler temperature conditions make this field profile suitable for chickpea cultivation.",
        "Mango": "Warm temperature and suitable rainfall conditions support mango cultivation, especially in tropical and subtropical growing regions.",
        "Cotton": "Cotton often performs well under warm conditions with balanced soil nutrients and moderate moisture availability.",
        "Coffee": "Coffee is generally suited to humid conditions with stable rainfall and moderate temperatures."
    }

    if crop_title in custom:
        return custom[crop_title]

    return (
        f"{crop_title} was recommended because the field profile shows "
        f"{get_rainfall_category(rainfall).lower()} rainfall, "
        f"{get_humidity_category(humidity).lower()} humidity, "
        f"{get_temperature_category(temperature).lower()} temperature, "
        f"and soil pH of {ph:.2f}."
    )


def chip_values(rainfall, humidity, temperature):
    return [
        f"{get_humidity_category(humidity)} humidity",
        f"{get_rainfall_category(rainfall)} rainfall",
        f"{get_temperature_category(temperature)} temperature"
    ]


# -----------------------
# AI CROP ASSISTANT HELPERS
# -----------------------
# This function uses Gemini API to answer crop/agriculture questions.
# It can answer general questions, and if a crop prediction exists, it also uses that context.
def build_assistant_context():
    context = st.session_state.latest_prediction_context

    if context is None:
        return "No crop prediction has been generated yet. Answer the user's agriculture question in a general and educational way."

    return f"""
Latest prediction context:
- Recommended crop: {context['crop']}
- Nitrogen: {context['N']}
- Phosphorus: {context['P']}
- Potassium: {context['K']}
- Temperature: {context['temperature']} °C
- Humidity: {context['humidity']} %
- Soil pH: {context['ph']}
- Rainfall: {context['rainfall']} mm
- Rainfall category: {context['rainfall_category']}
- Humidity category: {context['humidity_category']}
- Temperature category: {context['temperature_category']}
- Season type: {context['season_type']}
"""


def ask_gemini_agri_assistant(user_question):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return (
            "Gemini API key is not configured yet. Add GEMINI_API_KEY in "
            ".streamlit/secrets.toml locally and in Streamlit Cloud secrets after deployment."
        )

    context = build_assistant_context()

    prompt = f"""
You are an AI Crop Assistant inside a crop recommendation machine learning project.

Your role:
- Answer agriculture, soil, crop, nutrient, rainfall, humidity, and crop-care questions.
- Keep answers simple, practical, and beginner-friendly.
- Use the latest prediction context if it is relevant.
- If the user asks a general agriculture question, answer generally.
- Do not claim to replace professional agronomists or local agricultural experts.
- If a question depends on local climate, soil testing, pests, or regulations, advise checking local agricultural guidance.

Project/model context:
{context}

User question:
{user_question}

Answer in 4 to 7 concise bullet points or short paragraphs.
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"AI assistant error: {e}"


def radar_svg(N, P, K, temperature, humidity, rainfall):
    import math

    values = {
        "N": min(N / 140, 1),
        "P": min(P / 140, 1),
        "K": min(K / 200, 1),
        "Temp": min(temperature / 45, 1),
        "Humid": min(humidity / 100, 1),
        "Rain": min(rainfall / 300, 1),
    }

    cx, cy = 160, 138
    radius = 82
    labels = list(values.keys())
    angles = [-90, -30, 30, 90, 150, 210]

    def point(angle_deg, scale):
        angle = math.radians(angle_deg)
        return cx + radius * scale * math.cos(angle), cy + radius * scale * math.sin(angle)

    grid = []
    for scale in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{point(a, scale)[0]:.1f},{point(a, scale)[1]:.1f}" for a in angles)
        grid.append(f'<polygon points="{pts}" fill="none" stroke="#dce5d4" stroke-width="1" />')

    axes = []
    names = []
    for label, angle in zip(labels, angles):
        x, y = point(angle, 1)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#dce5d4" stroke-width="1" />')
        lx, ly = point(angle, 1.18)
        names.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="14" fill="#5d714e">{label}</text>')

    data_pts = " ".join(
        f"{point(angle, values[label])[0]:.1f},{point(angle, values[label])[1]:.1f}"
        for label, angle in zip(labels, angles)
    )

    return (
        '<svg width="320" height="270" viewBox="0 0 320 270" xmlns="http://www.w3.org/2000/svg">'
        + ''.join(grid)
        + ''.join(axes)
        + f'<polygon points="{data_pts}" fill="rgba(106,161,77,0.23)" stroke="#6aa14d" stroke-width="3" />'
        + ''.join(names)
        + '</svg>'
    )


# -----------------------
# NAV
# -----------------------
html("""
<div class="top-nav">
    <div class="brand">
        <div class="brand-icon">🌾</div>
        <div>Crop Recommendation System</div>
    </div>
    <a class="nav-link" href="https://github.com/Aaisha-Nexus" target="_blank">GitHub</a>
</div>
""")


# -----------------------
# HERO
# -----------------------
html("""
<section class="hero">
    <div class="badge">ML-Powered Agriculture Intelligence</div>

    <div class="hero-title">
        Welcome to the<br>
        Crop Recommendation<br>
        System
    </div>

    <div class="hero-text">
        Enter soil composition and climate data to receive a machine learning based crop recommendation tailored to field conditions.
    </div>

    <div class="hero-stats">
        <div>
            <div class="hero-stat-number">22</div>
            <div class="hero-stat-label">Crop Classes</div>
        </div>
        <div>
            <div class="hero-stat-number">7</div>
            <div class="hero-stat-label">Core Features</div>
        </div>
        <div>
            <div class="hero-stat-number">99%</div>
            <div class="hero-stat-label">Model Accuracy</div>
        </div>
    </div>

    <div class="windmill">
        <div class="rotor">
            <div class="blade b1"></div>
            <div class="blade b2"></div>
            <div class="blade b3"></div>
        </div>
        <div class="hub-box"></div>
        <div class="hub-center"></div>
        <div class="tower"></div>
        <div class="tower-base"></div>
    </div>
</section>
""")


# -----------------------
# FORM
# -----------------------
html('<div class="small-label" style="margin-top:42px;">Soil & Climate Inputs</div>')
html('<div class="section-title">Enter Field Parameters</div>')

with st.form("crop_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=90.0, step=1.0)
        st.markdown('<div class="hint">Essential for leaf and stem growth</div>', unsafe_allow_html=True)

        K = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0, step=1.0)
        st.markdown('<div class="hint">Improves drought resistance</div>', unsafe_allow_html=True)

        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        st.markdown('<div class="hint">Relative air humidity</div>', unsafe_allow_html=True)

        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=202.93, step=0.1)
        st.markdown('<div class="hint">Average rainfall measurement</div>', unsafe_allow_html=True)

    with col2:
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=42.0, step=1.0)
        st.markdown('<div class="hint">Supports root development</div>', unsafe_allow_html=True)

        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.87, step=0.1)
        st.markdown('<div class="hint">Average growing temperature</div>', unsafe_allow_html=True)

        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.50, step=0.1)
        st.markdown('<div class="hint">Common optimal range: 6.0 to 7.5</div>', unsafe_allow_html=True)

    predict_button = st.form_submit_button("Recommend Crop  →")


# -----------------------
# PREDICTION
# -----------------------
if predict_button:
    model_input = create_model_input(N, P, K, temperature, humidity, ph, rainfall)

    prediction = model.predict(model_input)[0]
    crop = crop_mapping[int(prediction)]
    crop_icon = crop_icons.get(crop, "🌱")

    confidence = 0
    top_predictions = []

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        confidence = probabilities[int(prediction)] * 100

        top_indices = probabilities.argsort()[-3:][::-1]
        top_predictions = [
            (crop_mapping[int(i)], probabilities[int(i)] * 100)
            for i in top_indices
        ]

    chips = chip_values(rainfall, humidity, temperature)
    confidence_width = min(confidence, 100)

    # Save the latest prediction context for the AI Crop Assistant.
    # This lets the assistant answer questions related to the current recommendation.
    st.session_state.latest_prediction_context = {
        "crop": crop,
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "rainfall_category": get_rainfall_category(rainfall),
        "humidity_category": get_humidity_category(humidity),
        "temperature_category": get_temperature_category(temperature),
        "season_type": get_season_type(temperature)
    }


    html(f"""
    <div class="result-card">
        <div class="small-label">Recommended Crop</div>

        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:20px;">
            <div>
                <div class="crop-header">
                    <div class="crop-icon-box">{crop_icon}</div>
                    <div class="crop-name">{crop}</div>
                </div>

                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{confidence_width:.0f}%;"></div>
                </div>

                <div style="color:#5d714e; margin-top:10px;">
                    Model vote confidence: {confidence:.2f}%
                </div>
            </div>

            <div class="confidence-pill">High model agreement</div>
        </div>

        <div class="reason-box">
            {reason_text(crop, rainfall, humidity, temperature, ph)}
        </div>

        <div>
            <span class="chip">{chips[0]}</span>
            <span class="chip">{chips[1]}</span>
            <span class="chip">{chips[2]}</span>
        </div>
    </div>
    """)

    card1, card2 = st.columns(2, gap="medium")

    with card1:
        max_npk = max(N, P, K, 1)
        n_h = max(45, (N / max_npk) * 145)
        p_h = max(45, (P / max_npk) * 145)
        k_h = max(45, (K / max_npk) * 145)

        html(f"""
        <div class="section-card">
            <div class="small-label">Nutrient Profile</div>
            <div class="section-title">NPK Summary</div>

            <div class="bars">
                <div class="bar-wrap">
                    <div class="bar n" style="height:{n_h}px;"></div>
                    Nitrogen<br><b>{N:.0f}</b>
                </div>

                <div class="bar-wrap">
                    <div class="bar p" style="height:{p_h}px;"></div>
                    Phosphorus<br><b>{P:.0f}</b>
                </div>

                <div class="bar-wrap">
                    <div class="bar k" style="height:{k_h}px;"></div>
                    Potassium<br><b>{K:.0f}</b>
                </div>
            </div>
        </div>
        """)

    with card2:
        radar = radar_svg(N, P, K, temperature, humidity, rainfall)

        # Streamlit can sometimes sanitize inline SVG when using normal markdown/html.
        # components.html renders the radar chart reliably inside this card.
        components.html(
            f"""
            <div style="
                background: rgba(255,255,255,0.94);
                border: 1px solid #d9dec9;
                border-radius: 28px;
                box-shadow: 0 18px 38px rgba(44, 63, 34, 0.12);
                padding: 30px 34px;
                min-height: 360px;
                box-sizing: border-box;
                font-family: Arial, sans-serif;
            ">
                <div style="
                    letter-spacing: 6px;
                    text-transform: uppercase;
                    color: #496b3a;
                    font-size: 13px;
                    font-weight: 800;
                    margin-bottom: 8px;
                ">
                    Feature Contribution
                </div>

                <div style="
                    font-family: Georgia, serif;
                    font-size: 32px;
                    color: #111b0d;
                    font-weight: 900;
                    margin-bottom: 10px;
                ">
                    Environmental Profile
                </div>

                <div style="display:flex; justify-content:center; align-items:center; margin-top:6px;">
                    {radar}
                </div>
            </div>
            """,
            height=395,
            scrolling=False
        )

    vote_html = ""
    for crop_name, prob in top_predictions:
        vote_html += f"""
        <div class="vote-row">
            <div class="vote-label">
                <span>{crop_name.title()}</span>
                <span>{prob:.2f}%</span>
            </div>
            <div class="vote-track">
                <div class="vote-fill" style="width:{min(prob, 100):.0f}%;"></div>
            </div>
        </div>
        """

    html(f"""
    <div class="section-card" style="min-height:auto;">
        <div class="small-label">Prediction Details</div>
        <div class="section-title">Top Model Votes</div>

        {vote_html}

        <hr style="border:none; border-top:1px solid #d9dec9; margin:22px 0;">

        <div class="profile-list">
            Rainfall category: <b>{get_rainfall_category(rainfall)}</b><br>
            Humidity category: <b>{get_humidity_category(humidity)}</b><br>
            Temperature category: <b>{get_temperature_category(temperature)}</b><br>
            Season type: <b>{get_season_type(temperature)}</b>
        </div>
    </div>
    """)

    html(f"""
    <div class="section-card">
        <div class="small-label">Environmental Conditions</div>

        <div class="insight-grid">
            <div class="metric-box">
                <div class="metric-value">{temperature:.1f}°C</div>
                <div class="metric-label">Temperature</div>
            </div>

            <div class="metric-box">
                <div class="metric-value">{humidity:.1f}%</div>
                <div class="metric-label">Humidity</div>
            </div>

            <div class="metric-box">
                <div class="metric-value">{ph:.2f}</div>
                <div class="metric-label">Soil pH</div>
            </div>
        </div>
    </div>

    <div class="footer-action">↻ Try different parameters</div>
    """)


# -----------------------
# AI CROP ASSISTANT SECTION
# -----------------------
# This section adds an LLM-powered assistant.
# Users can ask general agriculture questions or questions about the latest predicted crop.
html("""
<div class="assistant-card">
    <div class="assistant-title-row">
        <div class="assistant-icon">🤖</div>
        <div>
            <div class="small-label">AI Crop Assistant</div>
            <div class="assistant-title">Ask Agriculture Questions</div>
            <div class="assistant-subtitle">
                Ask about crops, soil nutrients, rainfall, humidity, pH, planting care, or the latest recommendation.
            </div>
        </div>
    </div>
</div>
""")

current_context = st.session_state.latest_prediction_context

if current_context is not None:
    html(f"""
    <div class="assistant-context">
        <b>Current prediction context:</b><br>
        Recommended crop: <b>{current_context['crop'].title()}</b> |
        Rainfall: <b>{current_context['rainfall_category']}</b> |
        Humidity: <b>{current_context['humidity_category']}</b> |
        Temperature: <b>{current_context['temperature_category']}</b> |
        Season: <b>{current_context['season_type']}</b>
    </div>
    """)
else:
    html("""
    <div class="assistant-context">
        <b>No crop prediction selected yet.</b><br>
        You can still ask general agriculture questions, or generate a crop recommendation first for more context-aware answers.
    </div>
    """)

example_question = st.selectbox(
    "Choose a sample question or write your own below",
    [
        "Why was this crop recommended?",
        "How can I improve soil pH naturally?",
        "What does high nitrogen mean for crops?",
        "Which crops generally prefer high rainfall?",
        "How does humidity affect crop growth?",
        "Give basic care tips for the recommended crop."
    ]
)

user_question = st.text_area(
    "Ask the AI Crop Assistant",
    value="",
    placeholder="Example: Why is rice suitable for high rainfall and high humidity?",
    height=110
)

ask_button = st.button("Ask AI Crop Assistant")

if ask_button:
    final_question = user_question.strip() if user_question.strip() else example_question

    with st.spinner("Generating AI answer..."):
        ai_answer = ask_gemini_agri_assistant(final_question)

    st.markdown(
        f"""
        <div class="assistant-answer">
            <b>Question:</b> {final_question}<br><br>
            {ai_answer}
        </div>
        """,
        unsafe_allow_html=True
    )
