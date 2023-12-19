import numpy as np
from sklearn.naive_bayes import GaussianNB

# Definir el conjunto de datos
# Cada fila representa (Diámetro, Mancha, Clase)
# Donde 1 representa "Sí" y 0 representa "No"
datos = [
    [1, 1, 1],  # Manzana
    [1, 0, 1],  # Manzana
    [0, 1, 1],  # Naranja
    [0, 1, 1],  # Naranja
    [0, 1, 0],  # Naranja
    [1, 0, 0]   # Manzana
]

# Dividir los datos en características (X) y etiquetas (y)
X = np.array([d[:2] for d in datos])
y = np.array([d[2] for d in datos])

# Inicializar el clasificador Naive Bayes
nb_classifier = GaussianNB()

# Entrenar el clasificador
nb_classifier.fit(X, y)

# Función para realizar la predicción
def predecir(diametro, mancha):
    resultado = nb_classifier.predict([[diametro, mancha]])[0]
    return "Manzana" if resultado == 1 else "Naranja"

# Clasificar un ejemplo
diametro = 1  # 1 representa "Pequeño"
mancha = 1  # 1 representa "Sí"

resultado = predecir(diametro, mancha)
print(f"Para un fruto con diámetro 'Pequeño' y mancha 'Sí', se predice: {resultado}")
