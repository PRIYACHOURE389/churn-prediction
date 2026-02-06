from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io

from api.inference import predict_dataframe

app = FastAPI(
    title="Churn Prediction Platform",
    description="Business-friendly churn prediction system",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/batch-predict", response_class=HTMLResponse)
async def batch_predict(
    request: Request,
    file: UploadFile = File(...)
):
    df = pd.read_csv(file.file)

    results = predict_dataframe(df)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "tables": results.head(20).to_html(index=False),
        }
    )


@app.post("/download")
async def download_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    results = predict_dataframe(df)

    buffer = io.StringIO()
    results.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=churn_predictions.csv"
        }
    )
