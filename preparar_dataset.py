import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def calcular_dia_lluvioso(precipitacion):
    """Devuelve 1 si hubo precipitación registrada (>0), 0 en caso contrario."""
    return int(precipitacion > 0)

if __name__ == "__main__":
    # 1. Cargar dataset
    df = pd.read_csv("pronostico_huancayo.csv")
    print("Dimensiones iniciales:", df.shape)

    # 2. Crear nuevas variables
    df["dia_lluvioso"] = df["precipitacion"].apply(calcular_dia_lluvioso)
    df["amplitud_termica"] = df["temp_max"] - df["temp_min"]

    columnas_features = ["temp_max", "temp_min", "amplitud_termica"]
    X = df[columnas_features].copy()
    y = df["dia_lluvioso"].copy()

    # 3. Limpieza de datos nulos
    datos_completos = pd.concat([X, y], axis=1).dropna()
    X = datos_completos[columnas_features]
    y = datos_completos["dia_lluvioso"]

    # 4. División Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Escalado de características
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    # 6. Guardar arreglos procesados
    np.savez(
        "dataset_preparado.npz",
        X_train=X_train_esc,
        X_test=X_test_esc,
        y_train=y_train.values,
        y_test=y_test.values,
    )
    print("Dataset procesado y guardado en dataset_preparado.npz")