import matplotlib
matplotlib.use("Agg")  # Guarda la imagen directamente en archivo sin abrir ventana
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# 1. Generar conjunto de datos sintéticos
X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)

# 2. Definir arquitectura de la red neuronal (MLP)
modelo = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
modelo.summary()

# 3. Compilar el modelo
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4. Entrenar el modelo
historial = modelo.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# 5. Generar y guardar curvas de entrenamiento
fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
ejes[0].plot(historial.history["loss"], label="entrenamiento")
ejes[0].plot(historial.history["val_loss"], label="validación")
ejes[0].set_title("Pérdida (loss)")
ejes[0].set_xlabel("Época")
ejes[0].legend()

ejes[1].plot(historial.history["accuracy"], label="entrenamiento")
ejes[1].plot(historial.history["val_accuracy"], label="validación")
ejes[1].set_title("Precisión (accuracy)")
ejes[1].set_xlabel("Época")
ejes[1].legend()

plt.tight_layout()
plt.savefig("curvas_entrenamiento_sintetico.png", dpi=100)
print("\nGráfico guardado en curvas_entrenamiento_sintetico.png")

# 6. Evaluar con el conjunto de prueba
perdida_test, precision_test = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\nPérdida en test: {perdida_test:.4f}")
print(f"Precisión en test: {precision_test:.4f}")