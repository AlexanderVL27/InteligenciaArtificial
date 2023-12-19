import numpy as np
import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

k = 5

def k_nearest_neighbors(X, y, new_point, k):
    distances = [np.linalg.norm(np.array(x) - np.array(new_point)) for x in X]
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y[i] for i in k_nearest_indices]
    predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return predicted_label

predicted_label = k_nearest_neighbors(list(zip(x, y)), classes, new_point[0], k)
print("Etiqueta predicha:", predicted_label)

plt.figure(figsize=(10, 6))

for label in [0, 1]:
    indices = [i for i, x in enumerate(classes) if x == label]
    plt.scatter([x[i] for i in indices], [y[i] for i in indices], label=f'Clase {label}')

plt.scatter(new_x, new_y, color='red', marker='x', label='Nuevo Punto')

k_nearest_indices = np.argsort([np.linalg.norm(np.array(x) - np.array(new_point)) for x in zip(x, y)])[:k]
for i in k_nearest_indices:
    if classes[i] == 0:
        plt.scatter(x[i], y[i], color='blue', marker='o')
    elif classes[i] == 1:
        plt.scatter(x[i], y[i], color='orange', marker='o')

radius = max([np.linalg.norm(np.array([x[i], y[i]]) - np.array(new_point)) for i in k_nearest_indices])

circle = plt.Circle((new_x, new_y), radius, color='purple', fill=False)
plt.gca().add_patch(circle)

plt.text(new_x + 0.1, new_y + 0.1, f'Predicción: {predicted_label}', color='Black')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Algoritmo k-NN')
plt.legend()
plt.grid()
plt.show()
