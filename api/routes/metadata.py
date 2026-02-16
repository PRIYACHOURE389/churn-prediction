from fastapi import APIRouter

router = APIRouter()

@router.get("/metadata")
def metadata():
    return {
        "model": "ChurnClassifier",
        "version": "v1",
        "framework": "scikit-learn"
    }
