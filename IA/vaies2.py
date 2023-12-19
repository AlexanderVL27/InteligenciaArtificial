import numpy as np

# Definimos nuestros datos de entrenamiento
X_train = np.array([[1.2, 0.7], [2.3, 1.9], [0.5, 1.5], [3.0, 3.1]])
y_train = np.array([0, 1, 0, 1])

# Definir la función de probabilidad gaussiana
def calcular_probabilidad(x, media, desviacion):
    exponente = -((x - media)**2) / (2 * (desviacion**2))
    return (1 / (desviacion * (2 * 3.14159265359)**0.5)) * 2.718281828459045**exponente

# Calcular media y desviación estándar para cada característica y clase
def calcular_estadisticas(datos, etiquetas, clase):
    muestras_clase = datos[etiquetas == clase]
    media = muestras_clase.mean(axis=0)
    desviacion = muestras_clase.std(axis=0)
    return media, desviacion

# Entrenar el clasificador Naive Bayes
def entrenar_naive_bayes(datos, etiquetas):
    clases = np.unique(etiquetas)
    medias = []
    desviaciones = []
    for clase in clases:
        media, desviacion = calcular_estadisticas(datos, etiquetas, clase)
        medias.append(media)
        desviaciones.append(desviacion)
    return medias, desviaciones

# Calcular probabilidades a posteriori para una muestra
def calcular_probabilidades_posteriori(muestra, medias, desviaciones):
    probabilidades = []
    for i in range(len(medias)):
        clase_prob = len(X_train[y_train == i]) / len(y_train)  # Cambio aquí
        prob = np.prod([calcular_probabilidad(muestra[j], medias[i][j], desviaciones[i][j]) for j in range(len(muestra))])
        probabilidades.append(clase_prob * prob)
    return probabilidades

# Predecir la clase de una muestra
def predecir_clase(muestra, medias, desviaciones):
    probabilidades = calcular_probabilidades_posteriori(muestra, medias, desviaciones)
    return np.argmax(probabilidades)

# Entrenar el clasificador
medias, desviaciones = entrenar_naive_bayes(X_train, y_train)

# Predecir la clase de una muestra de prueba
muestra_de_prueba = np.array([1.7, 1.8])
clase_predicha = predecir_clase(muestra_de_prueba, medias, desviaciones)
print(f"Clase predicha: {clase_predicha}")
