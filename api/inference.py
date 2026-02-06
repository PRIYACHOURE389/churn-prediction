# api/inference.py

import pandas as pd
from typing import Dict, List, Union

from src.inference import predict_dataframe, predict_single


# --------------------------------------------------
# Batch prediction (used by API endpoints)
# --------------------------------------------------
def predict_payload(
    payload: Union[Dict, List[Dict]]
) -> Union[Dict, List[Dict]]:
    """
    Accepts:
    - single record dict
    - list of record dicts

    Returns:
    - prediction dict
    - list of prediction dicts
    """

    # Single record
    if isinstance(payload, dict):
        return predict_single(payload)

    # Batch records
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
        result_df = predict_dataframe(df)

        return result_df[
            ["churn_probability", "churn_prediction"]
        ].to_dict(orient="records")

    raise ValueError("Payload must be a dict or list of dicts")
