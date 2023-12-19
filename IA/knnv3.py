import numpy as np
import matplotlib.pyplot as plt

# Generar datos de entrenamiento aleatorios
# np.random.seed(42)
num_samples = 50
X_train = np.random.rand(num_samples, 2) * 20
y_train = np.random.choice(['A', 'B'], size=num_samples)

# Nuevo punto aleatorio a predecir
new_point = np.random.rand(2) * 20

# Número de vecinos a considerar
k = 5

# Implementación del algoritmo k-NN
def k_nearest_neighbors(X, y, new_point, k):
    distances = [np.linalg.norm(x - new_point) for x in X] #La función np.linalg.norm calcula la longitud de un vector, que en este caso es la distancia entre dos puntos en un plano.
    k_nearest_indices = np.argsort(distances)[:k] #La función np.argsort devuelve los índices que ordenarían el arreglo en orden ascendente.
    k_nearest_labels = [y[i] for i in k_nearest_indices] #Aquí estamos utilizando NumPy para crear una lista de las etiquetas correspondientes a los k puntos más cercanos. Estamos utilizando los índices que obtuvimos antes para acceder a las etiquetas en el arreglo y.
    predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return predicted_label

# Predecir la etiqueta del nuevo punto
predicted_label = k_nearest_neighbors(X_train, y_train, new_point, k)
print("Etiqueta predicha:", predicted_label)

# Graficar los datos de entrenamiento, el nuevo punto y los k vecinos más cercanos
plt.figure(figsize=(10, 6))

# Graficar datos de entrenamiento
for label in ['A', 'B']:
    indices = np.where(y_train == label)
    plt.scatter(X_train[indices, 0], X_train[indices, 1], label=f'Clase {label}')

# Graficar nuevo punto
plt.scatter(new_point[0], new_point[1], color='red', marker='x', label='Nuevo Punto')

# Graficar k vecinos más cercanos
k_nearest_indices = np.argsort(np.linalg.norm(X_train - new_point, axis=1))[:k]
for i in k_nearest_indices:
    if y_train[i] == 'A':
        plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', marker='o')
    elif y_train[i] == 'B':
        plt.scatter(X_train[i, 0], X_train[i, 1], color='orange', marker='o')

# Calcular el radio del círculo que cubre los vecinos cercanos
radius = max(np.linalg.norm(X_train[k_nearest_indices] - new_point, axis=1))

# Agregar círculo que cubre los vecinos cercanos
circle = plt.Circle(new_point, radius, color='purple', fill=False)
plt.gca().add_patch(circle)

# Agregar etiqueta predicha al gráfico
plt.text(new_point[0] + 0.1, new_point[1] + 0.1, f'Predicción: {predicted_label}', color='Black')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Algoritmo k-NN')
plt.legend()
plt.grid()
plt.show()
