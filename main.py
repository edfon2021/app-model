from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# --- 1. CARGA DE TODOS LOS ARTEFACTOS ---
# Asegúrate de que todos estos archivos .pkl estén en la misma carpeta que main.py
print("Cargando artefactos...")
try:
    rf_model = joblib.load("random_forest_model.pkl")
    kproto_model = joblib.load("kprototypes_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("standar_escaler.pkl") # Corregí el nombre a 'standarD_escaler.pkl'
    print("Todos los artefactos cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error fatal al iniciar: No se encontró el archivo {e.filename}.")
    # En un caso real, la app no podría funcionar.
    rf_model, kproto_model, encoders, scaler = None, None, None, None

# --- 2. INICIALIZACIÓN DE LA API ---
app = FastAPI(title="API Modelos Inmobiliarios")

# --- 3. ESQUEMA DE DATOS DE ENTRADA (DTO) ---
# Ahora definimos cada campo explícitamente. ¡Esto es mucho más robusto!
class ViviendaData(BaseModel):
    # Lista aquí TODAS las columnas que tu modelo necesita, en su formato original
    # Ejemplo:
    Superficie_Construida: float
    Numero_Habitaciones: int
    Numero_Pisos: int
    Habitaciones_mt2: float
    Tipo: str
    Zona: str
    Obra: str
    # ... y el resto de tus columnas ...
    
    class Config:
        json_schema_extra = {
            "example": {
                "Superficie_Construida": 89.474213,"Numero_Habitaciones": 6,"Numero_Pisos": 3,
                "Habitaciones_mt2": 0.06, "Tipo": "Vivienda Lujo","Zona": "Urbana","Obra": "Completa"
            }
        }

# --- 4. ENDPOINTS DE LA API ---

@app.post("/predict/regresor")
def predict_regresor(data: ViviendaData):
    """
    Recibe los datos de una vivienda, los preprocesa y devuelve la predicción del precio.
    """
    if not all([rf_model, encoders, scaler]):
        raise HTTPException(status_code=503, detail="El servicio de regresión no está disponible.")

    # 1. Convertir datos de entrada a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # --- PREPROCESAMIENTO ---
    try:
        # a) Codificar variables categóricas
        for column, encoder in encoders.items():
            # Asegurarse de que la columna existe en el DataFrame de entrada
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[[column]])
        
        # b) Escalar variables numéricas
        # ¡Importante! Debes tener una lista de los nombres de las columnas numéricas
        columnas_numericas = ['Superficie_Construida', 'Numero_Habitaciones', 'Numero_Pisos'] 
        input_df[columnas_numericas] = scaler.transform(input_df[columnas_numericas])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el preprocesamiento: {e}")

    # 2. Asegurar el orden correcto de las columnas para el modelo
    # ¡CRÍTICO! Esta lista debe tener el orden EXACTO con el que se entrenó el modelo.
    orden_columnas_modelo = ['Superficie_Construida', 'Numero_Habitaciones', 'Numero_Pisos', 'Habitaciones_mt2', 'Tipo', 'Zona','Obra'] 
    
    try:
        input_df_ordenado = input_df[orden_columnas_modelo]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Error interno: Falta la columna {e} después de procesar.")
    
    # 3. Realizar la predicción
    prediction = rf_model.predict(input_df_ordenado)
    
    # Devolver la entrada original y la predicción
    return {"input_original": data.dict(), "prediction": float(prediction[0])}


@app.post("/predict/cluster", summary="Asigna el Clúster a una Vivienda")
def predict_cluster(data: ViviendaData):
    """
    Recibe los datos de una vivienda, PREPROCESA los datos numéricos, 
    y devuelve el clúster al que pertenece.
    """
    if not all([kproto_model, scaler]):
        raise HTTPException(status_code=503, detail="El servicio de clustering no está disponible.")
    
    # 1. Convertir datos de entrada a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # --- PREPROCESAMIENTO NUMÉRICO ---
    try:
        # ¡EL PASO QUE FALTABA! Escalamos las variables numéricas.
        columnas_numericas_a_escalar = ['Superficie_Construida', 'Numero_Habitaciones', 'Numero_Pisos'] 
        input_df[columnas_numericas_a_escalar] = scaler.transform(input_df[columnas_numericas_a_escalar])
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al escalar los datos numéricos: {e}")

    # 2. Reconstruir el array de entrada en el orden correcto que espera el modelo.
    # Ahora contendrá las numéricas ESCALADAS y las categóricas como TEXTO.
    # ¡CRÍTICO! El orden aquí debe ser el mismo que usaste para entrenar K-Prototypes.
    try:
        input_array = np.array([[
            # Columnas Categóricas (se mantienen como texto)
            input_df['Tipo'].iloc[0],
            input_df['Zona'].iloc[0],
            input_df['Obra'].iloc[0],
            
            # Columnas Numéricas (ahora están escaladas)
            input_df['Superficie_Construida'].iloc[0],
            input_df['Numero_Habitaciones'].iloc[0],
            input_df['Numero_Pisos'].iloc[0],
            input_df['Habitaciones_mt2'].iloc[0]
        ]], dtype=object) # <--- ¡VERIFICA ESTE ORDEN!
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error creando el array de entrada final: {e}")

    # 3. Definir los índices de las columnas categóricas
    # Basado en el 'input_array' de arriba: Tipo(0), Zona(1), Obra(2) son categóricas.
    cat_cols_index = [0, 1, 2] # <--- ¡VERIFICA ESTOS ÍNDICES!

    # 4. Realizar la predicción del clúster
    try:
        cluster = kproto_model.predict(input_array, categorical=cat_cols_index)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción del clúster: {e}")

    return {"input_original": data.dict(), "cluster": int(cluster)}