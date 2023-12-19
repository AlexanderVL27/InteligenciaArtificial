# Definir el conjunto de datos
# Cada fila representa (Color de Pétalos, Espinas, Clase)
# Donde 1 representa "Sí" y 0 representa "No"
datos_flores = [
    (1, 1, 'A'),  # Flor A
    (1, 0, 'B'),  # Flor B
    (0, 1, 'B'),  # Flor B
    (0, 1, 'C'),  # Flor C
    (0, 0, 'C'),  # Flor C
    (1, 0, 'A')   # Flor A
]

# Definir las clases
clases_flores = {
    'A': "Flor A",
    'B': "Flor B",
    'C': "Flor C"
}

# Función para calcular probabilidades
def calcular_probabilidades_flores(datos, caracteristica, valor, clase):
    # Filtrar datos que cumplan con el valor de la característica
    subconjunto = [d for d in datos if d[caracteristica] == valor]
    
    # Contar cuántos de estos pertenecen a la clase especificada
    pertenecen_a_clase = [d for d in subconjunto if d[2] == clase]
    
    # Calcular la probabilidad
    return len(pertenecen_a_clase) / len(subconjunto)

# Función para realizar la predicción
def predecir_flores(color_petalo, espinas):
    probabilidades_clases = {}
    for clase in clases_flores.keys():
        prob = 1  # Inicializar con probabilidad 1
        for i, valor in enumerate([color_petalo, espinas]):
            prob *= calcular_probabilidades_flores(datos_flores, i, valor, clase)
        probabilidades_clases[clase] = prob
    
    # Obtener la clase con la mayor probabilidad
    resultado = max(probabilidades_clases, key=probabilidades_clases.get)
    return clases_flores[resultado]

# Clasificar un ejemplo
color_petalo = 1  # 1 representa "Rojo"
espinas = 1  # 1 representa "Sí"

resultado = predecir_flores(color_petalo, espinas)
print(f"Para una flor con pétalos rojos y espinas, se predice: {resultado}")
