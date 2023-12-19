import numpy as np
import matplotlib.pyplot as plt

# Generar datos de entrenamiento aleatorios
#np.random.seed(42)
num_samples = 50
X_train = np.random.rand(num_samples, 2) * 20
y_train = np.random.choice(['A', 'B'], size=num_samples)

# Nuevo punto aleatorio a predecir
new_point = np.random.rand(2) * 20

# Número de vecinos a considerar
k = 5

# Implementación del algoritmo k-NN
def k_nearest_neighbors(X, y, new_point, k):
    distances = [np.linalg.norm(x - new_point) for x in X]
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y[i] for i in k_nearest_indices]
    predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return predicted_label

# Predecir la etiqueta del nuevo punto
predicted_label = k_nearest_neighbors(X_train, y_train, new_point, k)
print("Etiqueta predicha:", predicted_label)

# Graficar los datos de entrenamiento, el nuevo punto y los k vecinos más cercanos
plt.figure(figsize=(10, 6))

# Graficar datos de entrenamiento
for i, label in enumerate(['A', 'B']):
    indices = np.where(y_train == label)
    plt.scatter(X_train[indices, 0], X_train[indices, 1], label=f'Clase {label}')

# Graficar nuevo punto
plt.scatter(new_point[0], new_point[1], color='red', marker='x', label='Nuevo Punto')

# Graficar k vecinos más cercanos
#k_nearest_points = X_train[np.argsort(np.linalg.norm(X_train - new_point, axis=1))[:k]]
#plt.scatter(k_nearest_points[:, 0], k_nearest_points[:, 1], color='purple', marker='o', label='K Vecinos Cercanos')

# Agregar etiqueta predicha al gráfico
plt.text(new_point[0] + 0.1, new_point[1] + 0.1, f'Predicción: {predicted_label}', color='blue')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Algoritmo k-NN')
plt.legend()
plt.grid()
plt.show()
