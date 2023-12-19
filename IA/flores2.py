import numpy as np
from sklearn.naive_bayes import GaussianNB

# Definir el conjunto de datos
# Cada fila representa (Color de Pétalos, Espinas, Clase)
# Donde 1 representa "Sí" y 0 representa "No"
datos_flores = np.array([
    [1, 1, 'A'],  # Flor A
    [1, 0, 'B'],  # Flor B
    [0, 1, 'B'],  # Flor B
    [0, 1, 'C'],  # Flor C
    [0, 0, 'C'],  # Flor C
    [1, 0, 'A']   # Flor A
])

X_flores = datos_flores[:, :-1].astype(int)
y_flores = datos_flores[:, -1]

# Inicializar el clasificador Naive Bayes
nb_classifier_flores = GaussianNB()

# Entrenar el clasificador
nb_classifier_flores.fit(X_flores, y_flores)

# Función para realizar la predicción
def predecir_flores(color_petalo, espinas):
    resultado = nb_classifier_flores.predict([[color_petalo, espinas]])[0]
    return f"Flor {resultado}"

# Clasificar un ejemplo
color_petalo = 1  # 1 representa "Rojo"
espinas = 1  # 1 representa "Sí"

resultado = predecir_flores(color_petalo, espinas)
print(f"Para una flor con pétalos rojos y espinas, se predice: {resultado}")
