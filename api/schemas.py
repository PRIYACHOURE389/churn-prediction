from pydantic import BaseModel
from pydantic import ConfigDict
from typing import List


# ---------- Base ----------
class BaseSchema(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )


# ---------- Input ----------
class CustomerFeatures(BaseSchema):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    TechSupport: str
    PaperlessBilling: str


# ---------- Output ----------
class PredictionResponse(BaseSchema):
    churn_probability: float
    risk_segment: str
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseSchema):
    predictions: List[PredictionResponse]
