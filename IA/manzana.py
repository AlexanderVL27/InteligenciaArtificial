# Definir el conjunto de datos
# Cada fila representa (Diámetro, Mancha, Clase)
# Donde 1 representa "Sí" y 0 representa "No"
datos = [
    (1, 1, 1),  # Manzana
    (1, 0, 1),  # Manzana
    (0, 1, 1),  # Naranja
    (0, 1, 1),  # Naranja
    (0, 1, 0),  # Naranja
    (1, 0, 0)   # Manzana
]

# Definir las clases
clases = {
    0: "Naranja",
    1: "Manzana"
}

# Función para calcular probabilidades
def calcular_probabilidades(datos, caracteristica, valor, clase):
    # Filtrar datos que cumplan con el valor de la característica
    subconjunto = [d for d in datos if d[caracteristica] == valor]
    
    # Contar cuántos de estos pertenecen a la clase especificada
    pertenecen_a_clase = [d for d in subconjunto if d[2] == clase]
    
    # Calcular la probabilidad
    return len(pertenecen_a_clase) / len(subconjunto)

# Función para realizar la predicción
def predecir(diametro, mancha):
    probabilidades_clases = {}
    for clase in clases.keys():
        prob = 1  # Inicializar con probabilidad 1
        for i, valor in enumerate([diametro, mancha]):
            prob *= calcular_probabilidades(datos, i, valor, clase)
        probabilidades_clases[clase] = prob
    
    # Obtener la clase con la mayor probabilidad
    resultado = max(probabilidades_clases, key=probabilidades_clases.get)
    return clases[resultado]

# Clasificar un ejemplo
diametro = 1  # 1 representa "Pequeño"
mancha = 1  # 1 representa "Sí"

resultado = predecir(diametro, mancha)
print(f"Para un fruto con diámetro 'Pequeño' y mancha 'Sí', se predice: {resultado}")
