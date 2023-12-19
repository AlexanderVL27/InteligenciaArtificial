import pygame
import heapq

# Definir la función heurística (en este caso, distancia Manhattan)
def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# Definir el algoritmo A*
def astar(start, goal, obstacles):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, (0, start, []))

    visited_points = []  # Lista para almacenar los puntos visitados

    while open_list:
        _, current, path = heapq.heappop(open_list)

        if current == goal:
            return path + [current], visited_points

        closed_set.add(current)

        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                         (current[0], current[1]+1), (current[0], current[1]-1)]:
            if neighbor not in obstacles and neighbor not in closed_set:
                new_path = path + [current]
                heapq.heappush(open_list, (len(new_path) + heuristic(neighbor, goal), neighbor, new_path))
                visited_points.append(neighbor)  # Registrar el punto visitado

# Inicializar pygame
pygame.init()

# Definir colores
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Definir dimensiones del mapa y tamaño de celda
MAP_WIDTH, MAP_HEIGHT = 800, 600
CELL_SIZE = 20

# Crear la pantalla
screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
pygame.display.set_caption("A* Pathfinding")

# Función para visualizar el mapa
def draw_map(start, goal, obstacles, path, current_pos, visited_points):
    screen.fill(WHITE)

    # Dibujar el mapa
    for obstacle in obstacles:
        pygame.draw.rect(screen, RED, (obstacle[0]*CELL_SIZE, obstacle[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Dibujar el inicio y el objetivo
    pygame.draw.rect(screen, GREEN, (start[0]*CELL_SIZE, start[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, BLUE, (goal[0]*CELL_SIZE, goal[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Dibujar el camino
    if path:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, BLUE, (path[i][0]*CELL_SIZE + CELL_SIZE//2, path[i][1]*CELL_SIZE + CELL_SIZE//2),
                             (path[i+1][0]*CELL_SIZE + CELL_SIZE//2, path[i+1][1]*CELL_SIZE + CELL_SIZE//2), 5)

    # Dibujar los puntos visitados
    for point in visited_points:
        pygame.draw.circle(screen, (0, 255, 0), (point[0]*CELL_SIZE + CELL_SIZE//2, point[1]*CELL_SIZE + CELL_SIZE//2), 2)

    # Dibujar la posición actual
    pygame.draw.rect(screen, GREEN, (current_pos[0]*CELL_SIZE, current_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()

# Definir el mapa y obstáculos
obstacles = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (5, 2), (5, 3), (5, 4), (5, 5), 
             (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12)]  # Obstáculos más grandes
start = (0, 0)
goal = (15, 10)

# Ejecutar el algoritmo
current_pos = start
path = []
visited_points = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if current_pos != goal:
        path, new_visited_points = astar(current_pos, goal, obstacles)
        if len(path) > 1:
            current_pos = path[1]

        visited_points += new_visited_points  # Agregar los nuevos puntos visitados

    draw_map(start, goal, obstacles, path, current_pos, visited_points)
    pygame.time.wait(500)  # Esperar un poco antes de actualizar la pantalla

pygame.quit()
