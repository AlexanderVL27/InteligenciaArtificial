import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

# Ejemplo de datos de entrenamiento
datos = {
    'Color': ['Rojo', 'Rojo', 'Verde', 'Verde', 'Azul', 'Rojo', 'Verde', 'Azul', 'Rojo', 'Verde'],
    'Forma': ['Circular', 'Cuadrado', 'Circular', 'Cuadrado', 'Circular', 'Cuadrado', 'Circular', 'Cuadrado', 'Circular', 'Cuadrado'],
    'Tamaño': ['Pequeño', 'Pequeño', 'Grande', 'Pequeño', 'Grande', 'Pequeño', 'Grande', 'Pequeño', 'Grande', 'Grande'],
    'Etiqueta': ['No', 'Sí', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No', 'Sí']
}

df = pd.DataFrame(datos)

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(-prob * np.log2(prob) for prob in probabilities)
    return entropy

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / sum(counts)) * entropy(data.where(data[feature] == val).dropna()[target])
        for i, val in enumerate(values)
    )
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_best_attribute(data, features, target):
    information_gains = [information_gain(data, feature, target) for feature in features]
    best_attribute_index = np.argmax(information_gains)
    return features[best_attribute_index]

def build_tree(data, features, target):
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    elif len(features) == 0:
        return np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
    else:
        best_attribute = find_best_attribute(data, features, target)
        tree = {best_attribute: {}}
        remaining_features = [f for f in features if f != best_attribute]
        
        for value in np.unique(data[best_attribute]):
            sub_data = data.where(data[best_attribute] == value).dropna()
            subtree = build_tree(sub_data, remaining_features, target)
            tree[best_attribute][value] = subtree
            
        return tree

def load_data():
    target = 'Etiqueta'
    features = df.columns[df.columns != target].tolist()
    tree = build_tree(df, features, target)
    
    # Tabla de frecuencias
    freq_label.config(text=str(df.describe(include='all')))
    
    # Desarrollo de las fórmulas
    formulas_text.delete(1.0, tk.END)  # Limpiar contenido previo
    formulas_text.insert(tk.END, "Fórmula de Entropía:\n")
    formulas_text.insert(tk.END, "Entropy(column) = - Σ (P(xi) * log2(P(xi)))\n\n")
    formulas_text.insert(tk.END, "Fórmula de Ganancia de Información:\n")
    formulas_text.insert(tk.END, "Information_Gain(data, feature, target) = Entropía_Total - Entropía_Ponderada\n\n")
    formulas_text.insert(tk.END, "Donde:\n")
    formulas_text.insert(tk.END, "- Entropía_Total es la entropía del conjunto de datos completo.\n")
    formulas_text.insert(tk.END, "- Entropía_Ponderada es la entropía ponderada del conjunto de datos tras dividirlo por el atributo específico.\n")
    
    # Reglas de decisión
    rules_text.delete(1.0, tk.END)  # Limpiar contenido previo
    rules_text.insert(tk.END, str(tree))  # Mostrar el árbol de decisión

root = tk.Tk()
root.title("Algoritmo ID3")

load_button = tk.Button(root, text="Ejecutar ID3 con Datos Cargados", command=load_data)
load_button.pack()

# Sección para mostrar la tabla de frecuencias
freq_label = tk.Label(root, text="Tabla de Frecuencias:", wraplength=300)
freq_label.pack()

# Sección para mostrar el desarrollo de las fórmulas
formulas_label = tk.Label(root, text="Desarrollo de las Fórmulas:")
formulas_label.pack()
formulas_text = tk.Text(root, height=10, width=50)
formulas_text.pack()

# Sección para mostrar las reglas de decisión
rules_label = tk.Label(root, text="Reglas de Decisión:")
rules_label.pack()
rules_text = tk.Text(root, height=10, width=50)
rules_text.pack()

root.mainloop()
