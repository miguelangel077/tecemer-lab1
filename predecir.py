import numpy as np
from tensorflow import keras

# 1. Cargar el modelo guardado
modelo = keras.models.load_model("modelo_lluvia.keras")

# 2. Nuevas observaciones de prueba (datos normalizados)
# Ejemplo 1: Temperatura max/min normalizadas y amplitud térmica
# Ejemplo 2: Condiciones con menores temperaturas y menor amplitud
observaciones_nuevas = np.array([
    [0.5, 0.3, 0.1],
    [-1.2, -0.8, -0.4],
])

# 3. Realizar predicciones de probabilidad
probabilidades = modelo.predict(observaciones_nuevas)
predicciones = (probabilidades > 0.5).astype(int)

# 4. Mostrar resultados
print("\n--- Resultados de las Predicciones ---")
for i, (prob, pred) in enumerate(zip(probabilidades, predicciones)):
    etiqueta = "lluvioso" if pred[0] == 1 else "no lluvioso"
    print(f"Observación {i+1}: Probabilidad={prob[0]:.3f} -> Predicción: Día {etiqueta}")