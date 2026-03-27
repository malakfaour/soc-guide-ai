from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from xgboost import XGBClassifier

app = FastAPI()

# Load model once
model = XGBClassifier()
model.load_model("models/xgboost_model.json")


class InputData(BaseModel):
    features: list  # 44 features


@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)

    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))

    return {
        "prediction": pred,
        "probabilities": proba.tolist()
    }