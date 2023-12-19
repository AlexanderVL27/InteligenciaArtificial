import random

# Definir la función de evaluación
def evaluar(x):
    return (x - 2)**2

# Función para generar una población inicial aleatoria
def generarpoblacion(poblacion_size, min_value, max_value):
    return [random.uniform(min_value, max_value) for _ in range(poblacion_size)]

# Función para evaluar la aptitud de la población
def evaluarpo(poblacion):
    return [(x, evaluar(x)) for x in poblacion]

# Función para seleccionar padres mediante torneo binario
def selec(poblacion, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(poblacion, 2)
        parents.append(min(tournament, key=lambda x: x[1])[0])
    return parents

# Función principal del algoritmo genético
def al_genetico(poblacion_size, min_value, max_value, num_gene, mutacion):
    population = generarpoblacion(poblacion_size, min_value, max_value)
    for generation in range(num_gene):
        evaluarpob = evaluarpo(population)
        mejor = min(evaluarpob, key=lambda x: x[1])
        if mejor[1] == evaluar(2):
            return generation, "Se encontró el máximo en x=2", evaluarpob[:10]
        parents = selec(evaluarpob, poblacion_size//2)
        nuevap = []
        while len(nuevap) < poblacion_size:
            p1, p2 = random.sample(parents, 2)
            hijo1 = (p1 + p2) / 2  # Cruce simple
            hijo1 += random.uniform(-mutacion, mutacion)  # Mutación simple
            nuevap.append(hijo1)
        population = nuevap
    return num_gene, "Límite de generaciones alcanzado", evaluarpob[:10]

# Parámetros
poblacion_size = 100
min_value = -10
max_value = 10
num_gene = 39
mutacion = 0.1

# Ejecutar el algoritmo genético
generacion, stop, top_10_individuos = al_genetico(poblacion_size, min_value, max_value, num_gene, mutacion)

# Imprimir resultados
print(f"### Experimento 2b: Cruce Uniforme y 39 Generaciones ###")
print(f"Total de generaciones: {generacion}")
print(f"Condición de paro: {stop}")
print("Top 10 mejores respuestas:")
for ind, val in top_10_individuos:
    print(f"Individuo: {ind}, Valor: {val}")
