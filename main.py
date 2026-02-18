from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, health, metadata

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="v1"
)
from api.routes import explain

app.include_router(explain.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router)
app.include_router(metadata.router, tags=["Metadata"])


@app.get("/")
def root():
    return {
        "service": "Telco Customer Churn Prediction API",
        "status": "running",
        "version": "v1"
    }
