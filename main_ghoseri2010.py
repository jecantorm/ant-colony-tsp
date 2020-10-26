import math

from aco_ghoseri2010 import ACO, Graph
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

    # f1 : Función Costo de Transporte
    # f2 : Función Costo de Interacción 

    f1_cost_matrix, f2_cost_matrix,dic_edificios_nodos, numNodes = datos.cargarDatosMatrices(h[0],h[1],h[2])

    Q = 0.99
    phi = 0.99
    rho = 0.99
    beta = 10.0
    alpha = 1.0
    ant_count = 10
    generations = 100
    update_strategy = 2

    # Se instancia ACO, en donde se envía como parámetro: la cantidad de ants, el número de generaciones, alpha, beta, rho, Q, Estrategia para calcular T(i,j)
    aco = ACO(ant_count, generations, alpha, beta, phi, rho, Q, update_strategy)
    graph = Graph(f1_cost_matrix,f2_cost_matrix, numNodes)
    path, cost = aco.solve(graph, dic_edificios_nodos)
    print('cost: {}, path: {}'.format(cost, path))
    plot(points, path)

if __name__ == '__main__':
    main()