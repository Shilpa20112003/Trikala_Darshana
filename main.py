# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   TRIKALA DARSHANA · त्रिकाल दर्शन                                      ║
# ║   Step 4+5: FastAPI Backend — Predict + Confidence + Reason + Advice    ║
# ║   Trained on Failure. Built for Survival.                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Trikala Darshana API · त्रिकाल दर्शन",
    description = "Startup Survival Intelligence — Trained on Failure. Built for Survival.",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL + ENCODINGS
# ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

model   = joblib.load(BASE / "trikala_rf_model.pkl")
scaler  = joblib.load(BASE / "trikala_scaler.pkl")

with open(BASE / "trikala_encodings.json") as f:
    encodings = json.load(f)

with open(BASE / "trikala_model_metadata.json") as f:
    metadata = json.load(f)

CITY_MAP     = encodings["city_map"]
INDUSTRY_MAP = encodings["industry_map"]
FEATURES     = encodings["features"]

# ─────────────────────────────────────────────────────────────────
# CITY / INDUSTRY NAME → CLEAN CATEGORY MAPPINGS
# ─────────────────────────────────────────────────────────────────
def get_industry_category(industry_name: str) -> str:
    ind = industry_name.lower()
    if any(x in ind for x in ['edtech','education','learning']):
        return 'EdTech'
    elif any(x in ind for x in ['fintech','finance','financial','payment','banking']):
        return 'FinTech'
    elif any(x in ind for x in ['health','medical','pharma','biotech','wellness']):
        return 'HealthTech'
    elif any(x in ind for x in ['ecommerce','e-commerce','retail','commerce','trading']):
        return 'E-Commerce'
    elif any(x in ind for x in ['saas','software','technology','tech','internet']):
        return 'Tech/SaaS'
    elif any(x in ind for x in ['food','restaurant','delivery']):
        return 'FoodTech'
    elif any(x in ind for x in ['logistic','transport','supply']):
        return 'Logistics'
    elif any(x in ind for x in ['real estate','proptech','construction']):
        return 'RealEstate'
    elif any(x in ind for x in ['agri','agriculture']):
        return 'AgriTech'
    elif any(x in ind for x in ['media','entertainment','content']):
        return 'Media'
    elif any(x in ind for x in ['manufacturing','machinery']):
        return 'Manufacturing'
    else:
        return 'Services'

def encode_stage(stage: str) -> int:
    s = stage.lower()
    if any(x in s for x in ['pre-seed','angel']):    return 1
    elif 'seed' in s:                                 return 2
    elif any(x in s for x in ['pre-series','bridge']):return 3
    elif 'series a' in s:                             return 4
    elif 'series b' in s:                             return 5
    elif 'series c' in s:                             return 6
    elif any(x in s for x in ['series d','series e']):return 7
    elif 'private equity' in s:                       return 8
    elif 'ipo' in s:                                  return 9
    else:                                             return 2

# ─────────────────────────────────────────────────────────────────
# REASON + ADVICE GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_reason_advice(
    prediction: str,
    confidence: float,
    funding: float,
    rounds: int,
    city: str,
    industry_clean: str,
    stage: str,
) -> dict:

    stage_s    = stage.lower()
    major_cities = ['Bangalore','Mumbai','Delhi','Hyderabad','Chennai','Pune']
    is_major   = city in major_cities
    is_tech    = industry_clean in ['EdTech','FinTech','HealthTech','Tech/SaaS','E-Commerce']
    funding_m  = funding / 1_000_000  # in millions

    if prediction == "Safe":
        # Build reason
        reasons = []
        if funding_m >= 10:
            reasons.append(f"strong funding of ${funding_m:.1f}M")
        elif funding_m >= 1:
            reasons.append(f"solid funding of ${funding_m:.1f}M")
        else:
            reasons.append(f"early-stage funding in place")

        if rounds >= 3:
            reasons.append(f"{rounds} funding rounds showing investor confidence")
        elif rounds >= 2:
            reasons.append(f"{rounds} funding rounds indicating traction")

        if is_major:
            reasons.append(f"strong startup ecosystem in {city}")
        if is_tech:
            reasons.append(f"high-growth {industry_clean} sector")
        if any(x in stage_s for x in ['series b','series c','series d','private equity']):
            reasons.append("late-stage funding indicating maturity")

        reason = "This startup shows " + ", and ".join(reasons[:3]) + "."

        # Build advice
        if funding_m < 1:
            advice = "Consider raising a Seed or Series A round to strengthen runway. Focus on achieving product-market fit before scaling."
        elif funding_m < 5:
            advice = "Good foundation! Focus on unit economics and customer retention. Prepare for Series A with clear growth metrics."
        elif rounds < 3:
            advice = "Strong funding base. Expand to Tier-2 cities and diversify revenue streams to reduce risk."
        else:
            advice = "Excellent position! Consider strategic partnerships and international expansion. Profitability roadmap should be prioritized."

    else:  # Failed
        reasons = []
        if funding_m < 0.1:
            reasons.append("very limited funding runway")
        elif funding_m < 0.5:
            reasons.append(f"insufficient capital (${funding_m:.2f}M)")

        if rounds <= 1:
            reasons.append("limited investor backing")
        if not is_major:
            reasons.append(f"limited startup ecosystem support in {city}")
        if not is_tech:
            reasons.append(f"challenging market conditions for {industry_clean}")
        if any(x in stage_s for x in ['debt','bridge','unknown']):
            reasons.append("debt/bridge financing signals financial stress")

        if not reasons:
            reasons.append("combination of market and operational risk factors")

        reason = "Warning signals detected: " + ", ".join(reasons[:3]) + "."

        advice = ""
        if funding_m < 0.5:
            advice = "Urgently seek bridge funding or angel investment. Reduce burn rate immediately and focus on a single core product."
        elif rounds <= 1:
            advice = "Strengthen your pitch deck and approach angel investors or accelerators like Y Combinator India or Sequoia Surge."
        elif not is_major:
            advice = f"Consider relocating operations to Bangalore or Mumbai for better access to investors and talent."
        else:
            advice = "Re-evaluate business model and pivot if needed. Focus on reducing CAC and improving LTV. Seek mentorship from experienced founders."

    return {"reason": reason, "advice": advice}

# ─────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    total_funding  : float
    funding_rounds : int
    city           : str
    industry       : str
    stage          : str = "Seed"

class PredictResponse(BaseModel):
    prediction  : str
    confidence  : float
    confidence_pct: str
    reason      : str
    advice      : str
    model_accuracy: str
    city        : str
    industry    : str
    stage       : str

# ─────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────
def build_features(req: PredictRequest) -> np.ndarray:
    industry_clean  = get_industry_category(req.industry)
    stage_encoded   = encode_stage(req.stage)

    major_cities    = ['Bangalore','Mumbai','Delhi','Hyderabad','Chennai','Pune']
    tech_industries = ['EdTech','FinTech','HealthTech','Tech/SaaS','E-Commerce']

    city_encoded    = CITY_MAP.get(req.city, 0)
    industry_encoded= INDUSTRY_MAP.get(industry_clean, 0)
    is_major_city   = 1 if req.city in major_cities else 0
    is_tech         = 1 if industry_clean in tech_industries else 0
    rounds          = max(req.funding_rounds, 1)
    funding_per_rnd = req.total_funding / rounds
    log_funding     = np.log1p(req.total_funding)
    log_fpr         = np.log1p(funding_per_rnd)

    if req.total_funding == 0:     band = 0
    elif req.total_funding < 1e5:  band = 1
    elif req.total_funding < 5e5:  band = 2
    elif req.total_funding < 2e6:  band = 3
    elif req.total_funding < 1e7:  band = 4
    else:                          band = 5

    return np.array([[
        req.total_funding,
        rounds,
        funding_per_rnd,
        log_funding,
        log_fpr,
        stage_encoded,
        city_encoded,
        industry_encoded,
        is_major_city,
        is_tech,
        band,
    ]])

# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"     : "🪔 Trikala Darshana API — Startup Survival Intelligence",
        "tagline"     : "Trained on Failure. Built for Survival.",
        "version"     : "2.0.0",
        "model"       : "Random Forest",
        "accuracy"    : f"{metadata['random_forest']['accuracy']*100:.1f}%",
        "f1_score"    : metadata['random_forest']['f1'],
        "endpoints"   : ["/predict", "/health", "/model-info"],
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/model-info")
def model_info():
    return {
        "model_name"  : "Random Forest Classifier",
        "accuracy"    : f"{metadata['random_forest']['accuracy']*100:.1f}%",
        "f1_score"    : metadata['random_forest']['f1'],
        "roc_auc"     : metadata['random_forest']['roc_auc'],
        "precision"   : metadata['random_forest']['precision'],
        "recall"      : metadata['random_forest']['recall'],
        "cv_mean_f1"  : metadata['cv_mean_f1'],
        "train_rows"  : metadata['train_rows'],
        "features"    : FEATURES,
        "dataset"     : "Indian Startups 2015–2025 · 60,432 rows",
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = build_features(req)

    prediction_code = model.predict(features)[0]
    proba           = model.predict_proba(features)[0]
    confidence      = float(proba[prediction_code])
    prediction      = "Safe" if prediction_code == 1 else "Failed"

    industry_clean  = get_industry_category(req.industry)
    reason_advice   = generate_reason_advice(
        prediction, confidence,
        req.total_funding, req.funding_rounds,
        req.city, industry_clean, req.stage
    )

    return PredictResponse(
        prediction    = prediction,
        confidence    = round(confidence, 4),
        confidence_pct= f"{confidence*100:.1f}%",
        reason        = reason_advice["reason"],
        advice        = reason_advice["advice"],
        model_accuracy= f"{metadata['random_forest']['accuracy']*100:.1f}%",
        city          = req.city,
        industry      = industry_clean,
        stage         = req.stage,
    )

# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
