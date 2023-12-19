from scipy.optimize import minimize_scalar

# Definir la función de evaluación
def evaluar(x):
    return x**2 - 4*x + 4

# Experimento 1a: Tasa de Mutación = 0.05, 32 Generaciones
resultado_exp_1a = minimize_scalar(evaluar, bounds=(-10, 10), method='bounded', options={'maxiter': 32})

# Experimento 1b: Tasa de Mutación = 0.2, 30 Generaciones
resultado_exp_1b = minimize_scalar(evaluar, bounds=(-10, 10), method='bounded', options={'maxiter': 30})

# Experimento 2a: Cruce Simple, 42 Generaciones
resultado_exp_2a = minimize_scalar(evaluar, bounds=(-10, 10), method='bounded', options={'maxiter': 42})

# Experimento 2b: Cruce Uniforme, 39 Generaciones
resultado_exp_2b = minimize_scalar(evaluar, bounds=(-10, 10), method='bounded', options={'maxiter': 39})

# Imprimir resultados de los experimentos con scipy.optimize
print(f"Resultado del Experimento 1a con scipy.optimize: {resultado_exp_1a}")
print(f"Resultado del Experimento 1b con scipy.optimize: {resultado_exp_1b}")
print(f"Resultado del Experimento 2a con scipy.optimize: {resultado_exp_2a}")
print(f"Resultado del Experimento 2b con scipy.optimize: {resultado_exp_2b}")
