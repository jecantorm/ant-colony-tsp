import math

from aco import ACO, Graph
from plot import plot

# Función que retorna la distancia entre el nodo y el nodo j
def distance(city1, city2):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def main():
    cities = []
    points = []
    # Se lee el archivo .txt se encuentran en orden: el índice de la ciudad, su coordenada en X y su coordenada en Y
    with open('/home/juancm/Desktop/Octavo/Tesis/ProyectoGrado/Metaheuristicas/ant-colony-tsp/ant-colony-tsp/data/chn31.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
            points.append((int(city[1]), int(city[2])))

    # Se genera la matriz de costos a partir de los nodos y las coordenadas leídas
    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)
    # Se instancia ACO, en donde se envía como parámetro: la cantidad de ants, el número de generaciones, alpha, beta, rho, Q, Estrategia para calccular T(i,j)
    aco = ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    print('cost: {}, path: {}'.format(cost, path))
    plot(points, path)

if __name__ == '__main__':
    main()
