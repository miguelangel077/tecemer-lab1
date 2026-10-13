import datetime as dt
import json
import os
import pickle
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------
# Paso 1.1 — Instalar dependencias y descargar dataset
# ---------------------------------------------------------
LATITUD, LONGITUD = -12.0654, -75.2049  # Huancayo

fin = dt.date.today() - dt.timedelta(days=5)
inicio = fin - dt.timedelta(days=400)

url = "https://archive-api.open-meteo.com/v1/archive"
parametros = {
    "latitude": LATITUD,
    "longitude": LONGITUD,
    "start_date": inicio.isoformat(),
    "end_date": fin.isoformat(),
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
    "timezone": "America/Lima",
}

print("Consultando API de Open-Meteo...")
respuesta = requests.get(url, params=parametros, timeout=30)
respuesta.raise_for_status()
datos = respuesta.json()["daily"]

df = pd.DataFrame(datos)
df = df.rename(
    columns={
        "time": "fecha",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precipitacion",
        "windspeed_10m_max": "viento_max",
    }
)
df = df.dropna().reset_index(drop=True)
print(f"Dataset descargado correctamente: {len(df)} filas.")

# ---------------------------------------------------------
# Paso 1.2 — Variable objetivo (sin fuga de información)
# ---------------------------------------------------------
if "precipitacion" in df.columns:
    df["lluvia_mañana"] = (df["precipitacion"].shift(-1) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=["precipitacion"])

COLUMNAS_CARACTERISTICAS = ["temp_max", "temp_min", "viento_max"]
print("\nDistribución de la variable objetivo 'lluvia_mañana':")
print(df["lluvia_mañana"].value_counts())
# ---------------------------------------------------------
# Paso 1.3 — Preparar los datos y entrenar el modelo base (v1)
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

X = df[COLUMNAS_CARACTERISTICAS].values
y = df["lluvia_mañana"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)
X_test_esc = escalador.transform(X_test)

print(f"\nEntrenamiento: {X_train_esc.shape} | Prueba: {X_test_esc.shape}")

def construir_modelo(unidades_ocultas=8):
    modelo = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(unidades_ocultas, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo

print("Entrenando Modelo v1 (8 unidades ocultas)...")
modelo_v1 = construir_modelo(unidades_ocultas=8)
modelo_v1.fit(X_train_esc, y_train, epochs=60, batch_size=16, validation_split=0.2, verbose=0)
print("Entrenamiento v1 completo.")

# ---------------------------------------------------------
# Paso 1.4 y 2.1 — Evaluar y guardar Versión 1
# ---------------------------------------------------------
y_pred_v1 = (modelo_v1.predict(X_test_esc, verbose=0) > 0.5).astype(int).flatten()
reporte_v1 = classification_report(y_test, y_pred_v1, output_dict=True)

metricas_v1 = {
    "accuracy": reporte_v1["accuracy"],
    "precision_clase_1": reporte_v1["1"]["precision"],
    "recall_clase_1": reporte_v1["1"]["recall"],
    "f1_clase_1": reporte_v1["1"]["f1-score"]
}

with open("metricas_v1.json", "w", encoding="utf-8") as f:
    json.dump(metricas_v1, f, indent=2)

TIMESTAMP_V1 = dt.datetime.now().strftime("%Y%m%d_%H%M")
NOMBRE_V1 = f"modelo_lluvia_v1_{TIMESTAMP_V1}.keras"

modelo_v1.save(NOMBRE_V1)
with open("escalador_v1.pkl", "wb") as f:
    pickle.dump(escalador, f)

metadatos_v1 = {
    "version": "v1",
    "archivo_modelo": NOMBRE_V1,
    "fecha_entrenamiento": TIMESTAMP_V1,
    "caracteristicas": COLUMNAS_CARACTERISTICAS,
    "hiperparametros": {"unidades_ocultas": 8, "epochs": 60, "batch_size": 16},
    "metricas": metricas_v1,
}

with open("version_info_v1.json", "w", encoding="utf-8") as f:
    json.dump(metadatos_v1, f, indent=2)

print(f"Modelo v1 guardado como: {NOMBRE_V1}")
print("Métricas v1:")
print(json.dumps(metricas_v1, indent=2))
# ---------------------------------------------------------
# Paso 2.2 — Entrenar una segunda versión (v2)
# ---------------------------------------------------------
import shutil

print("\nEntrenando Modelo v2 (24 unidades ocultas)...")
modelo_v2 = construir_modelo(unidades_ocultas=24)
modelo_v2.fit(X_train_esc, y_train, epochs=60, batch_size=16, validation_split=0.2, verbose=0)
print("Entrenamiento v2 completo.")

y_pred_v2 = (modelo_v2.predict(X_test_esc, verbose=0) > 0.5).astype(int).flatten()
reporte_v2 = classification_report(y_test, y_pred_v2, output_dict=True)

metricas_v2 = {
    "accuracy": reporte_v2["accuracy"],
    "precision_clase_1": reporte_v2["1"]["precision"],
    "recall_clase_1": reporte_v2["1"]["recall"],
    "f1_clase_1": reporte_v2["1"]["f1-score"]
}

with open("metricas_v2.json", "w", encoding="utf-8") as f:
    json.dump(metricas_v2, f, indent=2)

TIMESTAMP_V2 = dt.datetime.now().strftime("%Y%m%d_%H%M")
NOMBRE_V2 = f"modelo_lluvia_v2_{TIMESTAMP_V2}.keras"

modelo_v2.save(NOMBRE_V2)
with open("escalador_v2.pkl", "wb") as f:
    pickle.dump(escalador, f)

metadatos_v2 = {
    "version": "v2",
    "archivo_modelo": NOMBRE_V2,
    "fecha_entrenamiento": TIMESTAMP_V2,
    "caracteristicas": COLUMNAS_CARACTERISTICAS,
    "hiperparametros": {"unidades_ocultas": 24, "epochs": 60, "batch_size": 16},
    "metricas": metricas_v2,
}

with open("version_info_v2.json", "w", encoding="utf-8") as f:
    json.dump(metadatos_v2, f, indent=2)

print(f"Modelo v2 guardado como: {NOMBRE_V2}")
print("Métricas v2:")
print(json.dumps(metricas_v2, indent=2))

# ---------------------------------------------------------
# Paso 2.3 — Comparar y promover a producción
# ---------------------------------------------------------
f1_v1 = metricas_v1["f1_clase_1"]
f1_v2 = metricas_v2["f1_clase_1"]

print(f"\nComparación F1-score -> v1: {f1_v1:.4f} | v2: {f1_v2:.4f}")

ganadora = metadatos_v2 if f1_v2 > f1_v1 else metadatos_v1
print(f"Versión promovida a producción: {ganadora['version']}")

shutil.copy(ganadora["archivo_modelo"], "modelo_produccion.keras")
shutil.copy(f"escalador_{ganadora['version']}.pkl", "escalador_produccion.pkl")
print("Archivos 'modelo_produccion.keras' y 'escalador_produccion.pkl' listos en producción.")