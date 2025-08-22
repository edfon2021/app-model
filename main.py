from fastapi import FastAPI
import numpy as np
# No necesitamos pandas para esta prueba si no lo usamos
# import pandas as pd 
# import joblib # Comentado para la prueba
from pydantic import BaseModel

# Inicializar la app
app = FastAPI(title="API Modelos (Modo Prueba)")

# # Cargar modelos previamente entrenados - ESTO ES LO QUE ESTAMOS DESACTIVANDO
# rf_model = joblib.load("random_forest_model.pkl")
# kproto_model = joblib.load("kprototypes_model.pkl")

# Definir el esquema de entrada con Pydantic
class InputData(BaseModel):
    features: list

# Endpoint para RandomForest (versión de prueba)
@app.post("/predict/regresor")
def predict_regresor(data: InputData):
    print("Recibida petición en /predict/regresor") # Añadimos un print para ver en los logs
    return {"message": "PRUEBA EXITOSA: El endpoint de regresión funciona", "input": data.features}

# Endpoint para KPrototypes (versión de prueba)
@app.post("/predict/cluster")
def predict_cluster(data: InputData):
    print("Recibida petición en /predict/cluster") # Añadimos un print para ver en los logs
    return {"message": "PRUEBA EXITOSA: El endpoint de cluster funciona", "input": data.features}