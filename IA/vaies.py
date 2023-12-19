from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Definimos nuestros datos de entrenamiento y prueba
X_train = [[1.2, 0.7], [2.3, 1.9], [0.5, 1.5], [3.0, 3.1]]
y_train = [0, 1, 0, 1]
X_test = [[1.7, 1.8]]

# Inicializamos el clasificador Naive Bayes
nb_classifier = GaussianNB()

# Entrenamos el clasificador
nb_classifier.fit(X_train, y_train)

# Predecimos la clase de la muestra de prueba
y_pred = nb_classifier.predict(X_test)

print(f"Clase predicha: {y_pred[0]}")
