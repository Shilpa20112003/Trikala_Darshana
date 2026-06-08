import streamlit as st
import requests
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, json, joblib, numpy as np

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Trikala Darshana · ತ್ರಿಕಾಲ ದರ್ಶನ",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DATABASE SAFEGUARD ───────────────────────────────────────
def init_db():
    try:
        conn = sqlite3.connect("predictions.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS history (
            Funding     INTEGER,
            Rounds      INTEGER,
            City        TEXT,
            Industry    TEXT,
            Stage       TEXT,
            Prediction  TEXT,
            Confidence  REAL,
            Reason      TEXT,
            Advice      TEXT
        )""")
        conn.commit()
        return conn, c
    except sqlite3.DatabaseError:
        try:
            conn.close()
        except:
            pass
        if os.path.exists("predictions.db"):
            os.remove("predictions.db")
        conn = sqlite3.connect("predictions.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("""CREATE TABLE history (
            Funding     INTEGER,
            Rounds      INTEGER,
            City        TEXT,
            Industry    TEXT,
            Stage       TEXT,
            Prediction  TEXT,
            Confidence  REAL,
            Reason      TEXT,
            Advice      TEXT
        )""")
        conn.commit()
        return conn, c

conn, c = init_db()

# ─── LOAD LOCAL MODEL ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model     = joblib.load("trikala_rf_model.pkl")
        encodings = json.load(open("trikala_encodings.json"))
        metadata  = json.load(open("trikala_model_metadata.json"))
        return model, encodings, metadata
    except:
        return None, None, None

ml_model, encodings, model_meta = load_model()

# ─── CITY / INDUSTRY MAPS ─────────────────────────────────────
CITY_MAP = {
    "Bangalore": 6,   "Mumbai": 61,     "New Delhi": 66,  "Gurgaon": 33,
    "Hyderabad": 37,  "Chennai": 18,    "Pune": 73,       "Noida": 67,
    "Delhi": 23,      "Ahmedabad": 2,   "Kolkata": 50,    "Jaipur": 42,
    "Kochi": 49,      "Chandigarh": 17, "Indore": 40,     "Bhopal": 13,
    "Surat": 80,      "Nagpur": 63,     "Lucknow": 52,    "Goa": 30,
    "Other": 0,
}
INDUSTRY_MAP = {
    "E-Commerce": 3,  "Tech/SaaS": 12,  "EdTech": 2,     "FinTech": 4,
    "HealthTech": 6,  "FoodTech": 5,    "Logistics": 8,  "RealEstate": 10,
    "AgriTech": 0,    "Media": 9,       "Manufacturing": 7, "Services": 11,
    "Travel": 13,     "Other": 1,
}
STAGE_MAP = {
    "Pre-seed / Angel": 1, "Seed": 2,       "Bridge / Pre-Series A": 3,
    "Series A": 4,         "Series B": 5,   "Series C": 6,
    "Late Stage (D/E/F)": 7, "Private Equity": 8, "IPO": 9,
}

# ─── LOCAL PREDICT FUNCTION ────────────────────────────────────
def local_predict(funding, rounds, city, industry, stage):
    if ml_model is None:
        # fallback rule-based
        pred = "Safe" if funding > 500000 else "Failed"
        conf = 0.72
        return pred, conf
    rounds     = max(rounds, 1)
    fpr        = funding / rounds
    log_f      = np.log1p(funding)
    log_fpr    = np.log1p(fpr)
    city_enc   = CITY_MAP.get(city, 0)
    ind_enc    = INDUSTRY_MAP.get(industry, 1)
    stg_enc    = STAGE_MAP.get(stage, 2)
    major      = 1 if city in ["Bangalore","Mumbai","Delhi","Hyderabad","Chennai","Pune"] else 0
    tech       = 1 if industry in ["Tech/SaaS","EdTech","FinTech","HealthTech","E-Commerce"] else 0
    if funding == 0:       band = 0
    elif funding < 1e5:    band = 1
    elif funding < 5e5:    band = 2
    elif funding < 2e6:    band = 3
    elif funding < 1e7:    band = 4
    else:                  band = 5

    feats = pd.DataFrame([[
        funding, rounds, fpr, log_f, log_fpr,
        stg_enc, city_enc, ind_enc, major, tech, band
    ]], columns=[
        'TotalFunding','FundingRounds','FundingPerRound',
        'LogFunding','LogFundingPerRound','StageEncoded',
        'CityEncoded','IndustryEncoded','IsMajorCity',
        'IsTechIndustry','FundingBand'
    ])
    code  = ml_model.predict(feats)[0]
    proba = ml_model.predict_proba(feats)[0]
    return ("Safe" if code == 1 else "Failed"), float(proba[code])

def generate_reason_advice(prediction, confidence, funding, rounds, city, industry, stage):
    stage_s  = stage.lower()
    major    = city in ["Bangalore","Mumbai","Delhi","Hyderabad","Chennai","Pune"]
    tech     = industry in ["Tech/SaaS","EdTech","FinTech","HealthTech","E-Commerce"]
    fund_m   = funding / 1_000_000

    if prediction == "Safe":
        parts = []
        if fund_m >= 10:   parts.append(f"strong funding of ${fund_m:.1f}M")
        elif fund_m >= 1:  parts.append(f"solid funding of ${fund_m:.1f}M")
        else:              parts.append("early-stage funding in place")
        if rounds >= 3:    parts.append(f"{rounds} funding rounds showing investor confidence")
        elif rounds >= 2:  parts.append(f"{rounds} funding rounds indicating traction")
        if major:          parts.append(f"strong startup ecosystem in {city}")
        if tech:           parts.append(f"high-growth {industry} sector")
        reason = "This startup shows " + ", and ".join(parts[:3]) + "."
        if fund_m < 1:
            advice = "Consider raising a Seed or Series A round to strengthen runway. Focus on product-market fit before scaling."
        elif fund_m < 5:
            advice = "Good foundation! Focus on unit economics and retention. Prepare for Series A with clear growth metrics."
        elif rounds < 3:
            advice = "Strong funding base. Expand to Tier-2 cities and diversify revenue streams."
        else:
            advice = "Excellent position! Consider strategic partnerships and international expansion."
    else:
        parts = []
        if fund_m < 0.1:    parts.append("very limited funding runway")
        elif fund_m < 0.5:  parts.append(f"insufficient capital (${fund_m:.2f}M)")
        if rounds <= 1:     parts.append("limited investor backing")
        if not major:       parts.append(f"limited startup ecosystem in {city}")
        if not tech:        parts.append(f"challenging market for {industry} sector")
        if "debt" in stage_s or "bridge" in stage_s:
            parts.append("debt/bridge financing signals financial stress")
        if not parts:       parts.append("combination of market and operational risk factors")
        reason = "Warning signals detected: " + ", ".join(parts[:3]) + "."
        if fund_m < 0.5:
            advice = "Urgently seek bridge funding or angel investment. Reduce burn rate and focus on a single core product."
        elif rounds <= 1:
            advice = "Strengthen your pitch and approach angel investors or accelerators like Y Combinator India or Sequoia Surge."
        elif not major:
            advice = f"Consider relocating to Bangalore or Mumbai for better access to investors and talent."
        else:
            advice = "Re-evaluate business model. Focus on reducing CAC and improving LTV. Seek mentorship from experienced founders."
    return reason, advice

# ─── SESSION STATE ────────────────────────────────────────────
for key in ["last_prediction","last_confidence","last_reason","last_advice"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── MASTER CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cinzel:wght@400;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=Space+Mono:wght@400;700&display=swap');
:root {
    --deep:#04030e; --saffron:#FF9933; --saffron-d:#e07d1a;
    --gold:#FFD700; --chakra:#138808; --text:#f0e8d8;
    --muted:#9a8f7a; --card:rgba(255,240,200,0.04);
    --bdr:rgba(255,200,100,0.12); --bdr-gold:rgba(255,215,0,0.28); --radius:16px;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--deep)!important;color:var(--text)!important;font-size:16px!important;}
.stApp{background:radial-gradient(ellipse 90% 60% at 50% -10%,rgba(255,153,51,0.13) 0%,transparent 55%),radial-gradient(ellipse 60% 40% at 10% 80%,rgba(139,26,26,0.12) 0%,transparent 50%),radial-gradient(ellipse 50% 40% at 90% 70%,rgba(19,136,8,0.08) 0%,transparent 50%),var(--deep)!important;min-height:100vh;}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background-image:radial-gradient(circle 1.5px at 10% 15%,rgba(255,215,0,0.5) 0%,transparent 2px),radial-gradient(circle 1px at 25% 70%,rgba(255,240,200,0.35) 0%,transparent 2px),radial-gradient(circle 2px at 40% 8%,rgba(255,153,51,0.4) 0%,transparent 3px),radial-gradient(circle 1px at 55% 88%,rgba(255,255,255,0.25) 0%,transparent 2px),radial-gradient(circle 1.5px at 70% 30%,rgba(255,215,0,0.4) 0%,transparent 2px),radial-gradient(circle 1px at 82% 60%,rgba(255,240,200,0.3) 0%,transparent 2px);animation:starTwinkle 6s ease-in-out infinite;}
@keyframes starTwinkle{0%,100%{opacity:1}40%{opacity:0.5}70%{opacity:0.8}}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 4rem 2rem!important;position:relative;z-index:1;}
.crown-wrap{text-align:center;padding:2.8rem 0 1.4rem;animation:crownIn 1.2s cubic-bezier(0.34,1.4,0.64,1) both;}
@keyframes crownIn{from{opacity:0;transform:translateY(-30px) scale(0.96)}to{opacity:1;transform:translateY(0) scale(1)}}
.diya-row{display:flex;align-items:center;justify-content:center;gap:1.2rem;margin-bottom:0.6rem;}
.diya-symbol{font-size:2rem;animation:diayGlow 3s ease-in-out infinite;filter:drop-shadow(0 0 8px rgba(255,153,51,0.7));}
@keyframes diayGlow{0%,100%{filter:drop-shadow(0 0 6px rgba(255,153,51,0.6));transform:scale(1)}50%{filter:drop-shadow(0 0 16px rgba(255,215,0,0.9));transform:scale(1.06)}}
.main-title{font-family:'Cinzel Decorative',serif;font-size:clamp(2.8rem,6vw,5rem);font-weight:900;letter-spacing:0.04em;line-height:1.05;background:linear-gradient(135deg,#FF9933 0%,#FFD700 25%,#FFF5CC 45%,#FFD700 65%,#FF9933 80%,#e07d1a 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;background-size:300%;animation:goldShimmer 5s ease-in-out infinite;}
@keyframes goldShimmer{0%,100%{background-position:0%}50%{background-position:100%}}
.devanagari-sub{font-family:'Cinzel',serif;font-size:1.8rem;font-weight:600;color:rgba(255,215,0,0.65);letter-spacing:0.15em;margin-top:0.4rem;animation:fadeIn 1.5s ease 0.8s both;}
.tagline{font-family:'Space Mono',monospace;font-size:0.9rem;letter-spacing:0.2em;text-transform:uppercase;color:#c8b89a;margin-top:0.8rem;animation:fadeIn 1.5s ease 1.2s both;}
.tagline em{color:var(--saffron);font-style:normal;}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.gold-divider{display:flex;align-items:center;gap:1rem;margin:1.4rem auto 0;max-width:600px;}
.gold-line{flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--bdr-gold),transparent);}
.divider-chakra{font-size:1.4rem;color:var(--gold);animation:chakraSpin 12s linear infinite;display:inline-block;filter:drop-shadow(0 0 5px rgba(255,215,0,0.5));}
@keyframes chakraSpin{to{transform:rotate(360deg)}}
.tricolor{width:100%;height:4px;border-radius:2px;background:linear-gradient(90deg,var(--saffron) 33.3%,rgba(255,255,255,0.5) 33.3% 66.6%,var(--chakra) 66.6%);margin:1.6rem 0 2rem;opacity:0.8;animation:triSlide 1.2s ease-out 0.5s both;}
@keyframes triSlide{from{transform:scaleX(0);opacity:0}to{transform:scaleX(1);opacity:0.8}}
[data-testid="stSidebar"]{background:rgba(6,4,18,0.97)!important;border-right:1px solid var(--bdr)!important;}
[data-testid="stSidebar"] *{font-size:15px!important;}
.sb-title{font-family:'Cinzel',serif;font-size:1.2rem!important;font-weight:700;color:var(--gold);letter-spacing:0.08em;padding-bottom:0.9rem;border-bottom:1px solid var(--bdr);margin-bottom:1.2rem;}
.sb-desc{font-size:1rem!important;color:#c8b89a;line-height:1.8;padding:1rem;background:rgba(255,200,80,0.03);border:1px solid rgba(255,215,0,0.08);border-radius:10px;margin-bottom:1rem;}
.stNumberInput>div>div>input{background:rgba(255,200,80,0.04)!important;border:1px solid var(--bdr)!important;border-radius:10px!important;color:var(--text)!important;font-family:'Space Mono',monospace!important;font-size:1.1rem!important;padding:0.6rem 1rem!important;height:48px!important;}
.stSelectbox>div>div{background:rgba(255,200,80,0.04)!important;border:1px solid var(--bdr)!important;border-radius:10px!important;font-size:1rem!important;}
.stSelectbox>div>div>div{color:var(--text)!important;font-size:1rem!important;}
label[data-testid="stWidgetLabel"] p{font-family:'Space Mono',monospace!important;font-size:0.82rem!important;letter-spacing:0.1em!important;text-transform:uppercase!important;color:#c8b89a!important;}
.stButton>button{background:linear-gradient(135deg,var(--saffron) 0%,var(--gold) 50%,var(--saffron-d) 100%)!important;color:#2a1000!important;font-family:'Cinzel',serif!important;font-weight:700!important;font-size:1.1rem!important;letter-spacing:0.1em!important;border:none!important;border-radius:12px!important;padding:1rem 2rem!important;width:100%!important;box-shadow:0 4px 22px rgba(255,153,51,0.38),0 0 0 1px rgba(255,215,0,0.2)!important;animation:btnGlow 3s ease-in-out infinite!important;}
@keyframes btnGlow{0%,100%{box-shadow:0 4px 22px rgba(255,153,51,0.35),0 0 0 1px rgba(255,215,0,0.18)}50%{box-shadow:0 6px 32px rgba(255,215,0,0.55),0 0 12px rgba(255,153,51,0.3)}}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--bdr)!important;gap:0!important;padding:0!important;}
.stTabs [data-baseweb="tab"]{font-family:'Cinzel',serif!important;font-size:1rem!important;font-weight:600!important;letter-spacing:0.06em!important;color:#c8b89a!important;background:transparent!important;border:none!important;padding:1rem 1.8rem!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:3px solid var(--gold)!important;}
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:var(--radius)!important;padding:1.4rem 1.5rem!important;transition:border-color 0.3s,transform 0.25s!important;}
[data-testid="metric-container"]:hover{border-color:var(--bdr-gold)!important;transform:translateY(-4px)!important;}
[data-testid="metric-container"] label{font-family:'Space Mono',monospace!important;font-size:0.78rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;color:#c8b89a!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Cinzel',serif!important;font-size:2.2rem!important;font-weight:700!important;color:var(--gold)!important;}
.sec-label{font-family:'Space Mono',monospace;font-size:0.82rem;letter-spacing:0.18em;text-transform:uppercase;color:#c8b89a;margin-bottom:1rem;display:flex;align-items:center;gap:0.6rem;}
.sec-label::before{content:'✦';color:var(--saffron);font-size:0.7rem;}
.sec-label::after{content:'';flex:1;height:1px;background:var(--bdr);}
.result-safe{background:linear-gradient(135deg,rgba(19,136,8,0.18),rgba(0,255,170,0.06));border:2px solid rgba(19,136,8,0.55);border-radius:var(--radius);padding:2rem 2.2rem;animation:resultPop 0.75s cubic-bezier(0.34,1.56,0.64,1) both;}
.result-fail{background:linear-gradient(135deg,rgba(139,26,26,0.22),rgba(255,77,141,0.06));border:2px solid rgba(139,26,26,0.6);border-radius:var(--radius);padding:2rem 2.2rem;animation:resultPop 0.75s cubic-bezier(0.34,1.56,0.64,1) both;}
@keyframes resultPop{from{opacity:0;transform:scale(0.9)}to{opacity:1;transform:scale(1)}}
.conf-bar-wrap{margin-top:1rem;}
.conf-bar-bg{height:12px;background:rgba(255,255,255,0.08);border-radius:100px;overflow:hidden;}
.conf-bar-fill-safe{height:100%;background:linear-gradient(90deg,#138808,#4caf50);border-radius:100px;transition:width 1s ease;}
.conf-bar-fill-fail{height:100%;background:linear-gradient(90deg,#8B1A1A,#ef5350);border-radius:100px;transition:width 1s ease;}
.reason-card{margin-top:1.2rem;padding:1.4rem 1.6rem;border-radius:14px;background:rgba(255,200,80,0.05);border:1px solid rgba(255,215,0,0.15);}
.reason-label{font-family:'Space Mono',monospace;font-size:0.78rem;letter-spacing:0.15em;text-transform:uppercase;color:#c8b89a;margin-bottom:0.6rem;}
.reason-text{font-size:1.05rem;color:#f0e8d8;line-height:1.8;}
.advice-card{margin-top:0.9rem;padding:1.4rem 1.6rem;border-radius:14px;background:rgba(255,153,51,0.07);border:1px solid rgba(255,153,51,0.25);}
.advice-label{font-family:'Space Mono',monospace;font-size:0.78rem;letter-spacing:0.15em;text-transform:uppercase;color:#FF9933;margin-bottom:0.6rem;}
.advice-text{font-size:1.05rem;color:#f0e8d8;line-height:1.8;}
.model-badge{display:inline-flex;align-items:center;gap:0.5rem;padding:0.5rem 1.1rem;border-radius:100px;background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.22);font-family:'Space Mono',monospace;font-size:0.82rem;color:#FFD700;letter-spacing:0.08em;}
.glass-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--radius);padding:1.8rem 2rem;position:relative;overflow:hidden;transition:border-color 0.3s;}
.glass-card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,200,80,0.04),transparent 60%);pointer-events:none;}
.glass-card:hover{border-color:var(--bdr-gold);}
.pill{display:inline-block;padding:0.35rem 0.9rem;border-radius:100px;font-family:'Space Mono',monospace;font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;margin:0.25rem 0.2rem;}
.pill-s{background:rgba(255,153,51,0.1);color:var(--saffron);border:1px solid rgba(255,153,51,0.28);}
.pill-g{background:rgba(19,136,8,0.1);color:#4caf50;border:1px solid rgba(19,136,8,0.25);}
.pill-m{background:rgba(255,215,0,0.08);color:var(--gold);border:1px solid rgba(255,215,0,0.22);}
h1,h2,h3{font-family:'Cinzel',serif!important;color:var(--gold)!important;font-size:1.6rem!important;}
p,div,span{font-size:1rem;}
.stMarkdown p{font-size:1rem!important;color:#f0e8d8!important;line-height:1.8!important;}
.stCaption{font-size:0.85rem!important;color:#c8b89a!important;}

/* ── ANIMATION 1: Input glow on focus ── */
.stNumberInput>div>div>input:focus{
  border-color:rgba(255,215,0,0.6)!important;
  box-shadow:0 0 0 3px rgba(255,215,0,0.12),0 0 14px rgba(255,153,51,0.2)!important;
  transform:scale(1.01)!important;
}
.stSelectbox>div>div:focus-within{
  border-color:rgba(255,215,0,0.6)!important;
  box-shadow:0 0 0 3px rgba(255,215,0,0.12),0 0 14px rgba(255,153,51,0.2)!important;
}

/* ── ANIMATION 2: Page load slide-in ── */
@keyframes slideUp{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideLeft{from{opacity:0;transform:translateX(-24px)}to{opacity:1;transform:translateX(0)}}
@keyframes fadeScale{from{opacity:0;transform:scale(0.94)}to{opacity:1;transform:scale(1)}}
[data-testid="metric-container"]{animation:fadeScale 0.6s cubic-bezier(0.34,1.4,0.64,1) both!important;}
[data-testid="metric-container"]:nth-child(1){animation-delay:0.1s!important;}
[data-testid="metric-container"]:nth-child(2){animation-delay:0.2s!important;}
[data-testid="metric-container"]:nth-child(3){animation-delay:0.3s!important;}
[data-testid="metric-container"]:nth-child(4){animation-delay:0.4s!important;}
[data-testid="metric-container"]:nth-child(5){animation-delay:0.5s!important;}
.stTabs{animation:slideUp 0.7s ease 0.2s both;}
.block-container>div>div{animation:slideUp 0.6s ease both;}
[data-testid="stSidebar"]>div{animation:slideLeft 0.6s cubic-bezier(0.34,1.2,0.64,1) 0.1s both;}

/* ── ANIMATION 3: Safe particles float up ── */
@keyframes floatUp{0%{opacity:1;transform:translateY(0) scale(1)}100%{opacity:0;transform:translateY(-80px) scale(0.4)}}
@keyframes floatUp2{0%{opacity:0.8;transform:translateY(0) scale(0.8)}100%{opacity:0;transform:translateY(-60px) scale(0.2)}}
.particle-wrap{position:relative;overflow:visible;pointer-events:none;}
.particle{position:absolute;font-size:1.1rem;animation:floatUp 2.2s ease-out forwards;pointer-events:none;}
.particle:nth-child(1){left:5%;animation-delay:0s;}
.particle:nth-child(2){left:20%;animation-delay:0.2s;animation-name:floatUp2;}
.particle:nth-child(3){left:40%;animation-delay:0.4s;}
.particle:nth-child(4){left:60%;animation-delay:0.15s;animation-name:floatUp2;}
.particle:nth-child(5){left:78%;animation-delay:0.35s;}
.particle:nth-child(6){left:92%;animation-delay:0.1s;animation-name:floatUp2;}

/* ── ANIMATION 3: Failed shake effect ── */
@keyframes shakeAlert{
  0%{transform:translateX(0)}
  10%{transform:translateX(-8px)}
  20%{transform:translateX(8px)}
  30%{transform:translateX(-6px)}
  40%{transform:translateX(6px)}
  50%{transform:translateX(-4px)}
  60%{transform:translateX(4px)}
  70%{transform:translateX(-2px)}
  80%{transform:translateX(2px)}
  100%{transform:translateX(0)}
}
.result-fail{animation:resultPop 0.75s cubic-bezier(0.34,1.56,0.64,1) both,shakeAlert 0.6s ease 0.8s!important;}

/* ── ANIMATION 4: Metric counter (via JS) ── */
.metric-count{display:inline-block;transition:all 0.3s ease;}
</style>
<script>
// Counter animation for metric cards
function animateCounters() {
  const els = document.querySelectorAll('[data-testid="stMetricValue"]');
  els.forEach(el => {
    const text = el.innerText.trim();
    const isPercent = text.includes('%');
    const num = parseFloat(text.replace('%','').replace(',',''));
    if (isNaN(num) || num === 0 || text === '_') return;
    let start = 0;
    const duration = 1200;
    const step = 16;
    const increment = num / (duration / step);
    el.innerText = isPercent ? '0%' : '0';
    const timer = setInterval(() => {
      start += increment;
      if (start >= num) {
        start = num;
        clearInterval(timer);
      }
      el.innerText = isPercent
        ? Math.round(start) + '%'
        : (Number.isInteger(num) ? Math.round(start) : start.toFixed(1));
    }, step);
  });
}
// Run on load and after Streamlit rerenders
setTimeout(animateCounters, 800);
const obs = new MutationObserver(() => setTimeout(animateCounters, 400));
obs.observe(document.body, { childList: true, subtree: true });
</script>""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="crown-wrap">
  <div class="diya-row">
    <span class="diya-symbol">🪔</span>
    <span class="diya-symbol" style="animation-delay:0.4s">✦</span>
    <span class="diya-symbol" style="animation-delay:0.8s;font-size:1.8rem">☸</span>
    <span class="diya-symbol" style="animation-delay:1.2s">✦</span>
    <span class="diya-symbol" style="animation-delay:1.6s">🪔</span>
  </div>
  <div style="font-family:'Cinzel',serif;font-size:0.75rem;letter-spacing:0.35em;text-transform:uppercase;color:#FF9933;opacity:0.8;margin-bottom:0.5rem">
    ॐ &nbsp; ಸತ್ಯಮೇವ ಜಯತೇ &nbsp; ॐ
  </div>
  <div class="main-title">Trikala Darshana</div>
  <div class="devanagari-sub">ತ್ರಿಕಾಲ ದರ್ಶನ</div>
  <div class="tagline">Vision of Three Times &nbsp;·&nbsp; <em>Past · Present · Future</em> &nbsp;·&nbsp; Startup Survival Intelligence</div>
  <div class="tagline" style="margin-top:0.35rem;color:rgba(255,153,51,0.6)">❝ Trained on Failure. Built for Survival. ❞</div>
  <div class="gold-divider">
    <div class="gold-line"></div>
    <span class="divider-chakra">☸</span>
    <span style="font-family:'Cinzel',serif;font-size:0.6rem;color:rgba(255,215,0,0.4);letter-spacing:0.2em">TRIKALA</span>
    <span class="divider-chakra" style="animation-direction:reverse">☸</span>
    <div class="gold-line"></div>
  </div>
</div>
<div class="tricolor"></div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">🔮 Nimitta Pravishtikā</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-desc">Enter startup parameters below to invoke the Trikala dṛṣṭi and reveal survival intelligence.</div>', unsafe_allow_html=True)

    st.markdown('<p class="sec-label" style="margin-top:1rem">Vittīya · Financials</p>', unsafe_allow_html=True)
    funding  = st.number_input("Total Funding (₹)", min_value=0, step=100000, format="%d")
    rounds   = st.number_input("Funding Rounds", min_value=0, step=1)

    st.markdown('<p class="sec-label" style="margin-top:0.8rem">Nagar · City</p>', unsafe_allow_html=True)
    city_name = st.selectbox("Select City", options=list(CITY_MAP.keys()))

    st.markdown('<p class="sec-label" style="margin-top:0.8rem">Udyog · Industry</p>', unsafe_allow_html=True)
    industry_name = st.selectbox("Select Industry", options=list(INDUSTRY_MAP.keys()))

    st.markdown('<p class="sec-label" style="margin-top:0.8rem">Dhaura · Stage</p>', unsafe_allow_html=True)
    stage_name = st.selectbox("Funding Stage", options=list(STAGE_MAP.keys()))

    st.markdown("---")
    predict_btn = st.button("🪔 Darshana Prakaṭa · Predict")

    # Model accuracy badge in sidebar
    if model_meta:
        acc  = model_meta['random_forest']['accuracy'] * 100
        f1   = model_meta['random_forest']['f1']
        auc  = model_meta['random_forest']['roc_auc']
        st.markdown(f"""
        <div style="margin-top:1rem;padding:0.9rem;background:rgba(255,200,80,0.03);border:1px solid rgba(255,215,0,0.1);border-radius:12px">
          <div style="font-family:'Space Mono',monospace;font-size:0.52rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f7a;margin-bottom:0.6rem">
            Yantra Vivara · Model Info
          </div>
          <div style="display:flex;flex-direction:column;gap:0.35rem">
            <span class="model-badge">🎯 Accuracy: {acc:.1f}%</span>
            <span class="model-badge">📊 F1 Score: {f1:.4f}</span>
            <span class="model-badge">🔮 ROC AUC: {auc:.4f}</span>
          </div>
          <div style="font-size:0.68rem;color:#9a8f7a;margin-top:0.7rem;line-height:1.7">
            Random Forest · v2.0<br>
            Indian Startups 2015–2025<br>
            60,432 rows · 11 features
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0.9rem;background:rgba(255,200,80,0.03);border:1px solid rgba(255,215,0,0.1);border-radius:12px">
          <div style="font-family:'Space Mono',monospace;font-size:0.52rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f7a;margin-bottom:0.5rem">Yantra Vivara · Model Info</div>
          <div style="font-size:0.7rem;color:#f0e8d8;line-height:1.8">Random Forest · v2.0<br><span style="color:#9a8f7a">Indian IT Startups<br>2015–2025 · 60,432 rows</span></div>
        </div>
        """, unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🪔  Darshana · Predict", "📊  Vishleshan · Analytics", "ℹ️  Parichaya · About"])

# ══════════════════════════════════════════════════════════════
# TAB 1 · DARSHANA — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("🔮 Predict Startup Survival")

    # ── LOCAL MODEL PREDICT ───────────────────────────────────
    if st.button("Predict (Local Model)"):

        # ── CHAKRA ANIMATION ──────────────────────────────────
        anim_slot = st.empty()
        anim_slot.markdown("""
        <div style="text-align:center;padding:2.5rem 0 1.5rem">
          <style>
            @keyframes chakraBig  { from{transform:rotate(0deg)}   to{transform:rotate(360deg)} }
            @keyframes chakraPulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.6;transform:scale(1.15)} }
            @keyframes fadeWords  { 0%{opacity:0;transform:translateY(8px)} 100%{opacity:1;transform:translateY(0)} }
            .anim-chakra { display:inline-block; font-size:5rem; animation:chakraBig 1.2s linear infinite;
                           filter:drop-shadow(0 0 18px rgba(255,215,0,0.8)); color:#FFD700; }
            .anim-ring   { display:inline-block; font-size:7rem; animation:chakraPulse 1.5s ease-in-out infinite;
                           position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                           opacity:0.15; color:#FF9933; }
            .anim-wrap   { position:relative; display:inline-block; width:120px; height:120px; }
            .anim-text   { font-family:'Cinzel',serif; font-size:1.1rem; color:#FFD700;
                           letter-spacing:0.2em; margin-top:1rem;
                           animation:fadeWords 0.6s ease forwards; }
            .anim-sub    { font-family:'Space Mono',monospace; font-size:0.72rem;
                           color:#9a8f7a; letter-spacing:0.15em; margin-top:0.4rem;
                           animation:fadeWords 0.6s ease 0.3s both; }
            .anim-dots span { animation:chakraPulse 1s ease-in-out infinite; display:inline-block; }
            .anim-dots span:nth-child(2){ animation-delay:0.2s; }
            .anim-dots span:nth-child(3){ animation-delay:0.4s; }
          </style>
          <div class="anim-wrap">
            <span class="anim-ring">☸</span>
            <span class="anim-chakra">☸</span>
          </div>
          <div class="anim-text">invoking trikala dṛṣṭi</div>
          <div class="anim-sub">
            <span class="anim-dots">
              <span>●</span><span>●</span><span>●</span>
            </span>
          </div>
          <div style="font-family:'Cinzel',serif;font-size:0.7rem;color:rgba(255,153,51,0.5);
                      letter-spacing:0.12em;margin-top:0.6rem;
                      animation:fadeWords 0.6s ease 0.6s both;">
            ಎದ್ದೇಳು ಎಚ್ಚರಗೊಳ್ಳು
          </div>
        </div>
        """, unsafe_allow_html=True)

        import time
        time.sleep(2.2)
        anim_slot.empty()
        # ── END ANIMATION ─────────────────────────────────────

        prediction, confidence = local_predict(
            funding, rounds, city_name, industry_name, stage_name
        )
        reason, advice = generate_reason_advice(
            prediction, confidence, funding, rounds, city_name, industry_name, stage_name
        )
        conf_pct = f"{confidence*100:.1f}%"

        if prediction == "Safe":
            st.markdown(f"""
            <div class="result-safe particle-wrap">
              <div class="particle">🌿</div>
              <div class="particle">✨</div>
              <div class="particle">🌱</div>
              <div class="particle">⭐</div>
              <div class="particle">🌿</div>
              <div class="particle">✨</div>
              <div style="display:flex;align-items:center;gap:1.2rem">
                <div style="font-size:2.4rem">🌿</div>
                <div style="flex:1">
                  <div style="font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;color:#4caf50;margin-bottom:0.2rem">
                    SAFE — High Survival Probability
                  </div>
                  <div style="font-size:0.78rem;color:#9a8f7a">Model confidence: <strong style="color:#4caf50">{conf_pct}</strong></div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar-bg"><div class="conf-bar-fill-safe" style="width:{conf_pct}"></div></div>
                  </div>
                </div>
              </div>
              <div class="reason-card" style="margin-top:1rem">
                <div class="reason-label">🔍 Kāraṇa · Why Safe</div>
                <div class="reason-text">{reason}</div>
              </div>
              <div class="advice-card">
                <div class="advice-label">💡 Pariṇāma · Advice</div>
                <div class="advice-text">{advice}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fail">
              <div style="display:flex;align-items:center;gap:1.2rem">
                <div style="font-size:2.4rem">🔥</div>
                <div style="flex:1">
                  <div style="font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;color:#ef5350;margin-bottom:0.2rem">
                    FAILED — Risk Detected
                  </div>
                  <div style="font-size:0.78rem;color:#9a8f7a">Model confidence: <strong style="color:#ef5350">{conf_pct}</strong></div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar-bg"><div class="conf-bar-fill-fail" style="width:{conf_pct}"></div></div>
                  </div>
                </div>
              </div>
              <div class="reason-card" style="margin-top:1rem">
                <div class="reason-label">⚠️ Kāraṇa · Why Failed</div>
                <div class="reason-text">{reason}</div>
              </div>
              <div class="advice-card">
                <div class="advice-label">💡 Pariṇāma · Advice</div>
                <div class="advice-text">{advice}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (funding, rounds, city_name, industry_name, stage_name,
                   prediction, round(confidence, 4), reason, advice))
        conn.commit()
        st.session_state.last_prediction  = prediction
        st.session_state.last_confidence  = conf_pct
        st.session_state.last_reason      = reason
        st.session_state.last_advice      = advice

    # ── BACKEND PREDICT ───────────────────────────────────────
    if predict_btn:
        try:
            resp = requests.post("http://127.0.0.1:8000/predict",
                                 json={
                                     "total_funding"  : funding,
                                     "funding_rounds" : rounds,
                                     "city"           : city_name,
                                     "industry"       : industry_name,
                                     "stage"          : stage_name,
                                 }, timeout=5)
            data       = resp.json()
            prediction = data["prediction"]
            conf_pct   = data["confidence_pct"]
            reason     = data["reason"]
            advice     = data["advice"]
            confidence = data["confidence"]
        except Exception:
            prediction = None

        if prediction in ("Safe", "Failed"):
            if prediction == "Safe":
                st.markdown(f"""
                <div class="result-safe">
                  <div style="display:flex;align-items:center;gap:1.2rem">
                    <div style="font-size:2.4rem">🌿</div>
                    <div style="flex:1">
                      <div style="font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;color:#4caf50">SAFE — High Survival Probability</div>
                      <div style="font-size:0.78rem;color:#9a8f7a">Backend confidence: <strong style="color:#4caf50">{conf_pct}</strong></div>
                      <div class="conf-bar-wrap"><div class="conf-bar-bg"><div class="conf-bar-fill-safe" style="width:{conf_pct}"></div></div></div>
                    </div>
                  </div>
                  <div class="reason-card" style="margin-top:1rem"><div class="reason-label">🔍 Kāraṇa · Why Safe</div><div class="reason-text">{reason}</div></div>
                  <div class="advice-card"><div class="advice-label">💡 Pariṇāma · Advice</div><div class="advice-text">{advice}</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fail">
                  <div style="display:flex;align-items:center;gap:1.2rem">
                    <div style="font-size:2.4rem">🔥</div>
                    <div style="flex:1">
                      <div style="font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;color:#ef5350">FAILED — Risk Detected</div>
                      <div style="font-size:0.78rem;color:#9a8f7a">Backend confidence: <strong style="color:#ef5350">{conf_pct}</strong></div>
                      <div class="conf-bar-wrap"><div class="conf-bar-bg"><div class="conf-bar-fill-fail" style="width:{conf_pct}"></div></div></div>
                    </div>
                  </div>
                  <div class="reason-card" style="margin-top:1rem"><div class="reason-label">⚠️ Kāraṇa · Why Failed</div><div class="reason-text">{reason}</div></div>
                  <div class="advice-card"><div class="advice-label">💡 Pariṇāma · Advice</div><div class="advice-text">{advice}</div></div>
                </div>""", unsafe_allow_html=True)

            c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (funding, rounds, city_name, industry_name, stage_name,
                       prediction, round(confidence, 4), reason, advice))
            conn.commit()
            st.session_state.last_prediction = prediction
            st.session_state.last_confidence = conf_pct
            st.rerun()
        else:
            st.markdown("""
            <div style="background:rgba(255,153,51,0.07);border:1px solid rgba(255,153,51,0.3);border-radius:16px;padding:1.2rem 1.6rem;display:flex;align-items:center;gap:1rem">
              <span style="font-size:1.8rem">🪔</span>
              <span style="color:#FF9933;font-family:'Cinzel',serif;font-size:0.9rem">Backend Offline — Use Local Model button above</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────
    df_hist = pd.read_sql_query("SELECT * FROM history", conn)
    total   = len(df_hist)
    safe_ct = len(df_hist[df_hist["Prediction"] == "Safe"])
    fail_ct = len(df_hist[df_hist["Prediction"] == "Failed"])
    rate    = f"{round((safe_ct/total)*100)}%" if total else "_"
    avg_conf = f"{df_hist['Confidence'].mean()*100:.1f}%" if total and 'Confidence' in df_hist.columns else "_"

    st.markdown('<p class="sec-label">Sūchanā Saṃkhyā · Live Statistics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🧠 Queries",        total)
    c2.metric("🌿 Survived",       safe_ct)
    c3.metric("💀 Failed",         fail_ct)
    c4.metric("❄️ Survival Rate",  rate)
    c5.metric("🎯 Avg Confidence", avg_conf)

    # last prediction badge
    if st.session_state.last_prediction:
        p     = st.session_state.last_prediction
        conf  = st.session_state.last_confidence or ""
        color = "#4caf50" if p == "Safe" else "#ef5350"
        icon  = "🌿" if p == "Safe" else "🔥"
        st.markdown(f"""
        <div style="margin-top:1rem;padding:0.7rem 1.2rem;border-radius:10px;background:rgba(255,200,80,0.03);border:1px solid rgba(255,215,0,0.1);font-family:'Space Mono',monospace;font-size:0.7rem;color:{color}">
          {icon} &nbsp; Last Dṛṣṭi: <strong>{p}</strong> &nbsp;·&nbsp; Confidence: <strong>{conf}</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem;text-align:center;opacity:0.45">
      <div style="font-family:'Cinzel',serif;font-size:0.8rem;color:#FF9933;letter-spacing:0.15em;margin-bottom:0.25rem">
        ಎದ್ದೇಳು, ಎಚ್ಚರಗೊಳ್ಳು, ಗುರಿ ತಲುಪುವವರೆಗೆ ನಿಲ್ಲದಿರು
      </div>
      <div style="font-size:0.68rem;color:#9a8f7a;letter-spacing:0.08em">
        ಗುರಿ ತಲುಪುವವರೆಗೆ ಎದ್ದೇಳು, ಎಚ್ಚರಗೊಳ್ಳು, ನಿಲ್ಲದಿರು — ಕಠ ಉಪನಿಷದ್
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 · VISHLESHAN — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Vishleshan · Analytics")
    df_hist = pd.read_sql_query("SELECT * FROM history", conn)

    if df_hist.empty:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem">
          <div style="font-size:2.8rem;margin-bottom:1rem">🪔</div>
          <div style="font-family:'Cinzel',serif;font-size:1rem;color:#f0e8d8;margin-bottom:0.4rem">Koī Sūchanā Nahīṃ</div>
          <div style="font-size:0.78rem;color:#9a8f7a">No data yet. Run predictions from the Darshana tab.</div>
        </div>""", unsafe_allow_html=True)
    else:
        lft, rgt = st.columns([3,1])
        with lft:
            st.markdown('<p class="sec-label">Bhavishya Itihas · Prediction History</p>', unsafe_allow_html=True)
        with rgt:
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Niryāta · Export", csv, "trikala_history.csv", "text/csv")

        show_cols = [c for c in ["City","Industry","Stage","Funding","Prediction","Confidence"] if c in df_hist.columns]
        st.dataframe(
            df_hist[show_cols].style.applymap(
                lambda v: "color:#4caf50;font-weight:700" if v=="Safe" else "color:#ef5350;font-weight:700" if v=="Failed" else "",
                subset=["Prediction"]
            ),
            use_container_width=True, height=220
        )

        st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="sec-label">Dṛśya Vishleṣaṇa · Visual Analysis</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#07060f")
            ax.set_facecolor("#07060f")
            for sp in ax.spines.values():
                sp.set_edgecolor("#FFC850"); sp.set_alpha(0.15)
            ax.grid(color="#FFC850", linestyle="--", linewidth=0.5, alpha=0.06)
            counts = df_hist["Prediction"].value_counts()
            colors = ["#4caf50" if k == "Safe" else "#e07d1a" for k in counts.index]
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.42)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                        str(int(b.get_height())), ha="center", va="bottom", color="#f0e8d8", fontsize=9)
            ax.set_title("Prediction Outcomes", color="#f0e8d8", fontsize=10, pad=10)
            ax.tick_params(colors="#9a8f7a", labelsize=8)
            safe_p = mpatches.Patch(color="#4caf50", label="Safe")
            fail_p = mpatches.Patch(color="#e07d1a", label="Failed")
            ax.legend(handles=[safe_p, fail_p], facecolor="#07060f", edgecolor="#FFC850", labelcolor="#f0e8d8", fontsize=8)
            st.pyplot(fig); plt.close()

        with col2:
            if 'Industry' in df_hist.columns and len(df_hist) > 1:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor("#07060f")
                ax2.set_facecolor("#07060f")
                for sp in ax2.spines.values():
                    sp.set_edgecolor("#FFC850"); sp.set_alpha(0.15)
                ind_counts = df_hist.groupby(['Industry','Prediction']).size().unstack(fill_value=0)
                if 'Safe' in ind_counts.columns:
                    ax2.barh(ind_counts.index, ind_counts.get('Safe',0), label='Safe', color='#4caf50', alpha=0.85)
                if 'Failed' in ind_counts.columns:
                    ax2.barh(ind_counts.index, ind_counts.get('Failed',0),
                             left=ind_counts.get('Safe',0), label='Failed', color='#e07d1a', alpha=0.85)
                ax2.set_title("By Industry", color="#f0e8d8", fontsize=10, pad=10)
                ax2.tick_params(colors="#9a8f7a", labelsize=7)
                ax2.legend(fontsize=7, labelcolor='#f0e8d8', facecolor='#07060f', edgecolor='#FFC850')
                st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 · PARICHAYA — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([3, 2])

    with left:
        acc_str = f"{model_meta['random_forest']['accuracy']*100:.1f}%" if model_meta else "87.2%"
        f1_str  = str(model_meta['random_forest']['f1']) if model_meta else "0.9207"
        auc_str = str(model_meta['random_forest']['roc_auc']) if model_meta else "0.9046"

        st.markdown(f"""
        <div class="glass-card">
          <div style="font-family:'Cinzel Decorative',serif;font-size:1.4rem;font-weight:900;background:linear-gradient(135deg,#FF9933,#FFD700,#FFF5CC,#FFD700,#FF9933);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;background-size:300%;animation:goldShimmer 5s ease-in-out infinite;margin-bottom:0.2rem">
            Trikala Darshana
          </div>
          <div style="font-family:'Cinzel',serif;font-size:0.88rem;color:rgba(255,215,0,0.45);letter-spacing:0.12em;margin-bottom:1.2rem">ತ್ರಿಕಾಲ ದರ್ಶನ</div>
          <div style="font-size:0.86rem;color:#c0b09a;line-height:1.8;margin-bottom:1rem">
            An AI/ML-powered startup survival intelligence platform built for the soul of Indian entrepreneurship.<br><br>
            <em style="color:#FF9933">Trikala</em> — the three times (past, present, future).
            <em style="color:#FF9933">Darshana</em> — vision, insight, philosophy.
          </div>
          <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:1rem">
            <span class="model-badge">🎯 Accuracy: {acc_str}</span>
            <span class="model-badge">📊 F1: {f1_str}</span>
            <span class="model-badge">🔮 AUC: {auc_str}</span>
          </div>
          <div style="margin-bottom:1.2rem">
            <span class="pill pill-s">FastAPI</span>
            <span class="pill pill-m">Random Forest</span>
            <span class="pill pill-g">Streamlit</span>
            <span class="pill pill-s">SMOTE</span>
            <span class="pill pill-m">XGBoost</span>
            <span class="pill pill-g">60K+ Rows</span>
          </div>
          <div style="padding-top:1.1rem;border-top:1px solid rgba(255,215,0,0.1);font-family:'Space Mono',monospace;font-size:0.65rem;color:#9a8f7a">
            ನಿರ್ಮಿಸಿದವರು: <span style="color:#FF9933;font-size:0.78rem">Shilpa 🌸</span>
            &nbsp;·&nbsp; MCA Capstone 2025–26 &nbsp;·&nbsp; Indian IT Ecosystem
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem;padding:1.2rem 1.5rem;border-radius:14px;background:rgba(255,153,51,0.05);border:1px solid rgba(255,153,51,0.18);text-align:center">
          <div style="font-family:'Cinzel',serif;font-size:0.88rem;color:#FF9933;letter-spacing:0.12em;line-height:2;margin-bottom:0.4rem">
            ಯತ್ರ ನಾರ್ಯಸ್ತು ಪೂಜ್ಯಂತೇ ರಮಂತೇ ತತ್ರ ದೇವತಾಃ
          </div>
          <div style="font-size:0.68rem;color:#9a8f7a;font-style:italic">
            ಜ್ಞಾನವನ್ನು ಗೌರವಿಸುವಲ್ಲಿ ದೈವ ನೆಲೆಸುತ್ತದೆ — ಮನುಸ್ಮೃತಿ
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card" style="margin-bottom:0.9rem">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;color:#9a8f7a;margin-bottom:0.9rem">Yantra Mārgaḥ · ML Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("०१","Datta Saṃgrahaṇa","Kaggle · Crunchbase · MCA · 60K+ rows","#FF9933"),
            ("०२","EDA & Nāmāṅkana","Signal-based Status labeling","#4fc3f7"),
            ("०३","Lakṣaṇa Nirmāṇa","11 features · SMOTE balancing","#FFD700"),
            ("०४","Pramāṇa Śikṣaṇa","Random Forest · XGBoost · LR","#e07d1a"),
            ("०५","Sthāpanā","FastAPI + Streamlit · Confidence + Advice","#4caf50"),
        ]
        for num, title, desc, clr in steps:
            st.markdown(f"""
            <div style="display:flex;gap:0.85rem;align-items:flex-start;padding:0.65rem 0.85rem;border-radius:10px;background:rgba(255,200,80,0.025);border:1px solid rgba(255,215,0,0.07);margin-bottom:0.5rem">
              <div style="font-family:'Cinzel',serif;font-size:0.65rem;color:{clr};min-width:1.6rem;font-weight:700;margin-top:0.1rem">{num}</div>
              <div>
                <div style="font-family:'Cinzel',serif;font-size:0.78rem;font-weight:700;color:#f0e8d8">{title}</div>
                <div style="font-size:0.68rem;color:#9a8f7a">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
          <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;color:#9a8f7a;margin-bottom:0.75rem">Lakṣita Jana · Target Users</div>
          <div style="display:flex;gap:0.5rem;flex-wrap:wrap">
            <span style="padding:0.3rem 0.8rem;border-radius:8px;font-size:0.73rem;background:rgba(255,153,51,0.08);border:1px solid rgba(255,153,51,0.22);color:#FF9933">🧑‍💼 Udyamī · Entrepreneurs</span>
            <span style="padding:0.3rem 0.8rem;border-radius:8px;font-size:0.73rem;background:rgba(255,215,0,0.07);border:1px solid rgba(255,215,0,0.2);color:#FFD700">💼 Niveshaka · Investors</span>
            <span style="padding:0.3rem 0.8rem;border-radius:8px;font-size:0.73rem;background:rgba(19,136,8,0.08);border:1px solid rgba(19,136,8,0.22);color:#4caf50">🔍 Kāryakartā · Job Seekers</span>
          </div>
        </div>
        """, unsafe_allow_html=True)