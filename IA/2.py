import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB

# Datos de entrenamiento
X_train = np.array([
    ["Soleado", "Alta", "Alta", "No"],
    ["Soleado", "Alta", "Alta", "Si"],
    ["Nublado", "Alta", "Alta", "No"],
    ["Lluvioso", "Media", "Alta", "No"],
    ["Lluvioso", "Baja", "Normal", "No"],
    ["Lluvioso", "Baja", "Normal", "Si"],
    ["Nublado", "Baja", "Normal", "Si"],
    ["Soleado", "Media", "Alta", "No"],
    ["Soleado", "Baja", "Normal", "Si"],
    ["Lluvioso", "Media", "Normal", "Si"],
    ["Soleado", "Media", "Normal", "Si"],
    ["Nublado", "Media", "Alta", "Si"],
    ["Nublado", "Alta", "Normal", "Si"],
    ["Lluvioso", "Media", "Alta", "No"]
])

y_train = np.array(["No", "No", "Si", "Si", "Si", "No", "Si", "No", "Si", "Si", "Si", "Si", "Si", "No"])

# Crear un codificador One-Hot
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = encoder.fit_transform(X_train)

# Inicializar el clasificador Naive Bayes
nb_classifier = GaussianNB()

# Entrenar el clasificador
nb_classifier.fit(X_train_encoded, y_train)

# Ejemplo de clasificación
test_example = ["Soleado", "Baja", "Alta", "Si"]
test_example_encoded = encoder.transform([test_example])
prediccion = nb_classifier.predict(test_example_encoded)
print(f"La predicción es: {prediccion}")
