import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Generar datos de entrenamiento con distribuciones gaussianas
np.random.seed(0)

# Clase A
num_muestras_a = 25
media_a = [7, 15]
cov_a = [[3, 0], [0, 3]]
X_entrenamiento_a = np.random.multivariate_normal(media_a, cov_a, num_muestras_a)
y_entrenamiento_a = np.array(['A'] * num_muestras_a)

# Clase B
num_muestras_b = 25
media_b = [13, 5]
cov_b = [[2, 0], [0, 2]]
X_entrenamiento_b = np.random.multivariate_normal(media_b, cov_b, num_muestras_b)
y_entrenamiento_b = np.array(['B'] * num_muestras_b)

X_entrenamiento = np.vstack((X_entrenamiento_a, X_entrenamiento_b))
y_entrenamiento = np.concatenate((y_entrenamiento_a, y_entrenamiento_b))

# Crear y entrenar un clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_entrenamiento, y_entrenamiento)

# Graficar los datos de entrenamiento y la frontera de decisión
plt.figure(figsize=(10, 6))

# Graficar datos de entrenamiento
plt.scatter(X_entrenamiento_a[:, 0], X_entrenamiento_a[:, 1], label='Clase A', color='red')
plt.scatter(X_entrenamiento_b[:, 0], X_entrenamiento_b[:, 1], label='Clase B', color='blue')

# Crear una malla de puntos para visualizar la frontera de decisión
xx, yy = np.meshgrid(np.linspace(0, 20, 500), np.linspace(0, 20, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificador de Árbol de Decisión')
plt.legend()
plt.grid()
plt.show()
