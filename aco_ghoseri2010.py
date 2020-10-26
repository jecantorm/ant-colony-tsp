# Script que implementa Ant-Colony bi-objetivo propuesto por [Ghoseri et al 2010 ], el cual esta basado en [Dorigo 1997]
# Así mismo, se van a utilizar las funciones de costo transporte y costo interacción de ProyectoGrado
# Para este escenario de prueba, se utilizara el edificio SD y el edificio Ga como nodo origen y destino respectivamente

import random


class Graph(object):
    def __init__(self, f1_cost_matrix,f2_cost_matrix, numNodes):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix (Number of nodes, or in this case, number of cities)
        """
        self.f1_matrix = f1_cost_matrix
        self.f2_matrix = f2_cost_matrix
        self.numNodes = numNodes
        # noinspection PyUnusedLocal

        # f1: Función Costo de Transporte
        # f2: Función Costo de Interacción

        self.f1_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]
        self.f2_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]


class ACO(object):
    def __init__(self, ant_count, generations, alpha, beta, phi, rho, q0,
                 strategy):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = random.uniform(0,1)
        self.q0 = q0
        self.phi = phi
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def local_update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.f1_pheromone):
            for j, col in enumerate(row):

                graph.f1_pheromone[i][j] *= self.phi
                graph.f2_pheromone[i][j] *= self.phi

    def global_update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):

                graph.f1_pheromone[i][j] *= self.rho
                graph.f2_pheromone[i][j] *= self.rho

                graph.f1_pheromone[i][j] = min(1, graph.f1_pheromone[i][j] + (graph.numNodes/graph.f1_matrix[i][j]) )
                graph.f2_pheromone[i][j] = min(1, graph.f2_pheromone[i][j] + (graph.numNodes/graph.f2_matrix[i][j]) )

    # noinspection PyProtectedMember
    def solve(self, graph, dic_edificios_nodos):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            # 0. Preparing a new colony
            ants = [_Ant(self, graph,dic_edificios_nodos) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    # 2.0 Choose next node by applying the state transition rule 
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu

                # Apply local pheromone update
                ant.local_update_pheromone_delta()

            # 5. Apply global pheromone update
            self.global_update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco, graph,dic_edificios_nodos):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list List[ T(i,j) ]
        self.f1_pheromone_delta = []  # the local increase of pheromone (Delta_T(i,j))
        self.f2_pheromone_delta = []  # the local increase of pheromone (Delta_T(i,j))
        self.allowed = [i for i in range(graph.numNodes)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.numNodes)] for i in
                    range(graph.rank)]  # heuristic information (N(i,j))

        # 1. Position each ant in a starting node
        source = dic_edificios_nodos['SD']  # start from Source Node
        self.destiny = dic_edificios_nodos['GA']
        self.tabu.append(source)
        self.current = source
        self.allowed.remove(source)

    #2.1 Choose next node by applying the state transition rule
    def _select_next(self):

        denominator = 0
        for i in self.allowed:
            
            # Eq 3b
            denominator += self.graph.f1_pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
        # noinspection PyUnusedLocal
        # Se inicializa una lista vacía (0's) de probabilidades de que la hormiga estando en el nodo i pueda ir al nodo j
        probabilities = [0 for i in range(self.graph.numNodes)]  # probabilities for moving to a node in the next step

        # Verifica si el nodo actual es el nodo destino
        if self.current == self.destiny:
            pass

        else:
            for i in range(self.graph.numNodes):
                try:
                    #Verifica si la hormiga puede ir al nodo j
                    self.allowed.index(i)  # test if allowed list contains i

                    # Eq 3
                    probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                        self.eta[self.current][i] ** self.colony.beta / denominator

                except ValueError:
                    pass  # do nothing

        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break

        # La hormiga se mueve al nodo j
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
