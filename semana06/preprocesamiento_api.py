import numpy as np

COLUMNAS_CARACTERISTICAS = ["temp_max", "temp_min", "viento_max"]


def extraer_caracteristicas(datos_json):
    for col in COLUMNAS_CARACTERISTICAS:
        if col not in datos_json:
            raise ValueError(f"Falta el parámetro requerido: {col}")

    valores = [datos_json[col] for col in COLUMNAS_CARACTERISTICAS]
    return np.array([valores], dtype=float)