import numpy as np

# Funciones para calcular probabilidades condicionales y de clases

def calcular_probabilidades_condicionales(X_train, y_train):
    probs_condicionales = {}
    for col in range(X_train.shape[1]):
        for val in np.unique(X_train[:, col]):
            for clase in np.unique(y_train):
                mask = (X_train[:, col] == val) & (y_train == clase)
                probs_condicionales[(col, val, clase)] = (np.sum(mask) + 1) / (np.sum(y_train == clase) + len(np.unique(X_train[:, col])))
    return probs_condicionales

def calcular_probabilidades_clases(y_train):
    probs_clases = {}
    for clase in np.unique(y_train):
        probs_clases[clase] = (np.sum(y_train == clase) + 1) / (len(y_train) + len(np.unique(y_train)))
    return probs_clases

# Función para predecir una clase dada una muestra de test

def predecir_ejemplo(test_example, probs_clases, probs_condicionales):
    prob = {}
    for clase in probs_clases:
        prob[clase] = np.log(probs_clases[clase])
        for i, valor in enumerate(test_example):
            prob[clase] += np.log(probs_condicionales[(i, valor, clase)])
    return max(prob, key=prob.get)

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

# Calcular probabilidades
probs_condicionales = calcular_probabilidades_condicionales(X_train, y_train)
probs_clases = calcular_probabilidades_clases(y_train)

# Ejemplo de clasificación
test_example = ["Soleado", "Baja", "Alta", "Si"]
prediccion = predecir_ejemplo(test_example, probs_clases, probs_condicionales)
print(f"La predicción es: {prediccion}")
