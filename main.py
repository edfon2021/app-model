from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Inicializar la app de modelos predictivos y el modulo de clusterizacion
app = FastAPI(title="API Modelos")

# Cargar modelos previamente entrenados
rf_model = joblib.load("random_forest_model.pkl")
kproto_model = joblib.load("kprototypes_model.pkl")

# Definir el esquema de entrada con Pydantic
class InputData(BaseModel):
    features: list  # lista de valores numéricos o mixtos según el modelo

# Endpoint para RandomForest
@app.post("/predict/regresor")
def predict_regresor(data: InputData):
    X = np.array(data.features).reshape(1, -1)  # convertir a array 2D
    prediction = rf_model.predict(X)[0]
    return {"input": data.features, "prediction": float(prediction)}

# Endpoint para KPrototypes
@app.post("/predict/cluster")
def predict_cluster(data: InputData):
    X = np.array(data.features, dtype=object).reshape(1, -1)  # KPrototypes maneja datos mixtos

    cat_cols_index = [4, 5, 6]

    cluster = kproto_model.predict(X, categorical=cat_cols_index)[0]
    return {"input": data.features, "cluster": int(cluster)}
