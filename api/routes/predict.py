from typing import List
import pandas as pd
from fastapi import APIRouter
from api.schemas import CustomerFeatures, BatchPredictionResponse
from api.services.predictor import predict_single

router = APIRouter()


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: List[CustomerFeatures]):

    predictions = []

    for customer in customers:
        result = predict_single(customer)
        predictions.append(result)

    return {"predictions": predictions}
