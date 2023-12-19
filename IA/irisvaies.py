import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Función para calcular la probabilidad condicional P(X|C)
def prob_condicional(x, media, desviacion):
    return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-((x - media) ** 2) / (2 * desviacion ** 2))

# Función para entrenar el clasificador Naive Bayes
def entrenar_naive_bayes(X, y):
    num_muestras, num_caracteristicas = X.shape
    clases = np.unique(y)
    num_clases = len(clases)
    
    # Inicializar diccionarios para almacenar medias y desviaciones estándar
    medias = {}
    desviaciones = {}
    for clase in clases:
        medias[clase] = np.zeros(num_caracteristicas)
        desviaciones[clase] = np.zeros(num_caracteristicas)
    
    # Calcular medias y desviaciones estándar por clase
    for clase in clases:
        X_clase = X[y == clase]
        medias[clase] = np.mean(X_clase, axis=0)
        desviaciones[clase] = np.std(X_clase, axis=0)
    
    return medias, desviaciones

# Función para predecir la clase de un ejemplo de prueba
def predecir_ejemplo(x, medias, desviaciones, probs_clases):
    clases = list(medias.keys())
    num_caracteristicas = len(x)
    probs = {clase: np.log(probs_clases[clase]) for clase in clases}
    
    for clase in clases:
        for i in range(num_caracteristicas):
            prob = prob_condicional(x[i], medias[clase][i], desviaciones[clase][i])
            probs[clase] += np.log(prob)
    
    clase_predicha = max(probs, key=probs.get)
    return clase_predicha

# Calcular probabilidades a priori P(C)
probs_clases = {clase: np.mean(y_train == clase) for clase in np.unique(y_train)}

# Entrenar el clasificador Naive Bayes
medias, desviaciones = entrenar_naive_bayes(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = []
for ejemplo in X_test:
    clase_predicha = predecir_ejemplo(ejemplo, medias, desviaciones, probs_clases)
    y_pred.append(clase_predicha)

# Calcular precisión
accuracy = np.mean(y_pred == y_test)
print(f"Precisión del clasificador Naive Bayes: {accuracy:.4f}")
