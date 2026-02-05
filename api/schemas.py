from pydantic import BaseModel, Field
from typing import List

class ChurnRequest(BaseModel):
    tenure: int = Field(..., example=12, description="Number of months customer stayed")
    monthly_charges: float = Field(..., example=70.5)
    total_charges: float = Field(..., example=850.0)
    contract: str = Field(..., example="Month-to-month")
    internet_service: str = Field(..., example="Fiber optic")
    payment_method: str = Field(..., example="Electronic check")
    senior_citizen: int = Field(..., example=0)
    dependents: str = Field(..., example="No")

class BatchChurnRequest(BaseModel):
    customers: List[ChurnRequest]

class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., example=0.72)
    risk_level: str = Field(..., example="High Risk")
    message: str = Field(..., example="Customer is likely to leave")
    version: str = Field(..., example="v1.0")
