import pickle
import pandas as pd
import numpy as np
import sklearn
import uvicorn
from fastapi import FastAPI,Body
import uvicorn
from typing import Dict, Any
app = FastAPI(title="churn-prediction")


with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
def predict_single(customer: Dict[str,Any]):
    result=pipeline.predict_proba(customer)[0,1]
    return float(result)
@app.post("/predict")    
def predict(customer:Dict[str, Any] = Body(...)):
    churn = predict_single(customer)
    return {"churn_probability":churn, 
         "churn":bool(churn>=0.5)
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)    
