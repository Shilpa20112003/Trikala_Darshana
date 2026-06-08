from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("trikala_model.pkl")

@app.post("/predict")
def predict(total_funding: float, funding_rounds: int, city_encoded: int, industry_encoded: int):
    data = pd.DataFrame([[total_funding, funding_rounds, city_encoded, industry_encoded]],
                        columns=["TotalFunding","FundingRounds","CityEncoded","IndustryEncoded"])
    prediction = model.predict(data)[0]
    return {"prediction": "Safe" if prediction == 0 else "Failed"}
