from api.model import predict_churn

@app.post("/predict")
def predict(customer: dict):
    return predict_churn(customer)
