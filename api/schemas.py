from pydantic import BaseModel, Field

class ChurnRequest(BaseModel):
    tenure: int = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=850.0)
    Contract: str = Field(..., example="Month-to-month")
    InternetService: str = Field(..., example="Fiber optic")
    PaymentMethod: str = Field(..., example="Electronic check")
    SeniorCitizen: int = Field(..., example=0)
    Dependents: str = Field(..., example="No")


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    version: str
