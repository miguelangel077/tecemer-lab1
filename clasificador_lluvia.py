import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# 1. Cargar el dataset preparado
datos = np.load("dataset_preparado.npz")
X_train, X_test = datos["X_train"], datos["X_test"]
y_train, y_test = datos["y_train"], datos["y_test"]

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("Proporción de días lluviosos en entrenamiento:", round(y_train.mean(), 3))

# 2. Modelo Baseline (Regresión Logística)
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)
precision_baseline = accuracy_score(y_test, pred_baseline)
print(f"\nPrecisión del Baseline (Regresión Logística): {precision_baseline:.4f}")

# 3. Red Neuronal Multicapa (MLP)
modelo = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

historial = modelo.fit(
    X_train, y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# 4. Evaluación y comparación
perdida_test, precision_test = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Resultados Finales ---")
print(f"Precisión del Baseline: {precision_baseline:.4f}")
print(f"Precisión de la Red Neuronal (MLP): {precision_test:.4f}")

# 5. Generar y guardar gráfico
plt.figure(figsize=(6, 4))
plt.plot(historial.history["loss"], label="entrenamiento")
plt.plot(historial.history["val_loss"], label="validación")
plt.title("Pérdida - Clasificador de lluvia (Huancayo)")
plt.xlabel("Época")
plt.legend()
plt.tight_layout()
plt.savefig("curvas_entrenamiento_lluvia.png", dpi=100)
print("\nGráfico guardado en curvas_entrenamiento_lluvia.png")

# 6. Guardar el modelo entrenado
modelo.save("modelo_lluvia.keras")
print("Modelo guardado en modelo_lluvia.keras")