import math

from aco import ACO, Graph
from plot import plot

#Cargar datos de matrices imports
import cargarDatosMatrices as datos

# Función que retorna la distancia entre el nodo y el nodo j
def distance(city1, city2):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def main():
    nodes = []
    points = []
    
    # Se elige un rango de horas y el ciclo académico a utilizar, en este caso Ciclo 1 de Lunes de 6:20 - 6:30
    # En este arreglo se guardaran los valores de Strings que se usarán para consultar en la db de Firebase    
    h = ['1','L','620-630']

    matrix_ct, matrix_ci,dic_edificios_nodos, numNodes = datos.cargarDatosMatrices(h[0],h[1],h[2])

    with open('./data/chn31.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            nodes.append(dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
            points.append((int(city[1]), int(city[2])))

    # Se genera lad matrices de las dos funciones
    cost_f1 = []
    cost_f2 = []
    rank = len(nodes)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(nodes[i], nodes[j]))
        cost_matrix.append(row)
    # Se instancia ACO, en donde se envía como parámetro: la cantidad de ants, el número de generaciones, alpha, beta, rho, Q, Estrategia para calccular T(i,j)
    aco = ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    print('cost: {}, path: {}'.format(cost, path))
    plot(points, path)

if __name__ == '__main__':
    main()
