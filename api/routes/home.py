from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def home():
    return {
        "service": "Telco Customer Churn Prediction API",
        "status": "running",
        "version": "v1"
    }
