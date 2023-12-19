import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el clasificador Naive Bayes
nb_classifier = GaussianNB()

# Entrenar el clasificador
nb_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = nb_classifier.predict(X_test)

# Calcular precisión
accuracy = np.mean(y_pred == y_test)
print(f"Precisión del clasificador Naive Bayes: {accuracy:.4f}")
