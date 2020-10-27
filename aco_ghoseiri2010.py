# Script que implementa Ant-Colony bi-objetivo propuesto por [ghoseiri et al 2016 ], el cual esta basado en [Dorigo 1997]
# Así mismo, se van a utilizar las funciones de costo transporte y costo interacción de ProyectoGrado
# Para este escenario de prueba, se utilizara el edificio SD y el edificio Ga como nodo origen y destino respectivamente

import random


class Graph(object):
    def __init__(self, f1_cost_matrix,f2_cost_matrix, numNodes):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix (Number of nodes, or in this case, number of cities)
        """
        self.f1_cost = f1_cost_matrix
        self.f2_cost = f2_cost_matrix
        self.numNodes = numNodes
        # noinspection PyUnusedLocal

        # f1: Función Costo de Transporte
        # f2: Función Costo de Interacción

        self.f1_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]
        self.f2_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]


class ACO(object):
    def __init__(self, ant_count, generations, alpha, beta, phi, rho, Q, q0, a, b, epsilon,
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
        self.Q = Q
        self.q = random.uniform(0,1)
        self.q0 = q0
        self.phi = phi
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy


    def global_update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):

                graph.f1_pheromone[i][j] *= self.rho
                graph.f2_pheromone[i][j] *= self.rho

                for ant in ants:

                    graph.f1_pheromone[i][j] += ant.f1_local_pheromone[i][j]
                    graph.f2_pheromone[i][j] += ant.f2_local_pheromone[i][j]

            graph.f1_pheromone[i][j] = min(1, graph.f1_pheromone[i][j] + (graph.numNodes/graph.f1_cost[i][j]) )
            graph.f2_pheromone[i][j] = min(1, graph.f2_pheromone[i][j] + (graph.numNodes/graph.f2_cost[i][j]) )

    # noinspection PyProtectedMember

    def get_max_values_costs():

        max_f1 = -1
        max_f2 = -1

        for i in range(numNodes):
            for j in range(numNodes):

                if max_f1 < self.graph.f1_cost[i][j]:
                    max_f1 = self.graph.f1_cost[i][j]
                
                if max_f2 < self.graph.f2_cost[i][j]:
                    max_f2 = self.graph.f2_cost[i][j]

        return max_f1,max_f2

    def get_min_values_costs():

        min_f1 = 999
        min_f2 = 999

        for i in range(numNodes):
            for j in range(numNodes):

                if min_f1 > self.graph.f1_cost[i][j]:
                    min_f1 = self.graph.f1_cost[i][j]
                
                if min_f2 > self.graph.f2_cost[i][j]:
                    min_f2 = self.graph.f2_cost[i][j]

        return min_f1,min_f2
        
    def solve(self, graph, dic_edificios_nodos):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            # 0. Preparing a new colony
            ants = [_Ant(self, i, graph,dic_edificios_nodos, self.epsilon) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.numNodes - 1):
                    # 2.0 Choose next node by applying the state transition rule 
                    ant._select_next(self.Q, self.q0, self.a, self.b)
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu

                # Apply local pheromone update
                ant.local_update_pheromone()

            # 5. Apply global pheromone update
            self.global_update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, index, aco, graph,dic_edificios_nodos, epsilon):
        self.index = index
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list List[ T(i,j) ]
        self.f1_local_pheromone = []  # the local increase of pheromone (Delta_T(i,j))
        self.f2_local_pheromone = []  # the local increase of pheromone (Delta_T(i,j))
        self.allowed = [i for i in range(graph.numNodes)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.numNodes)] for i in
                    range(graph.rank)]  # heuristic information (N(i,j))

        self.epsilon = epsilon

        # 1. Position each ant in a starting node
        source = dic_edificios_nodos['SD']  # start from Source Node
        self.destiny = dic_edificios_nodos['GA']
        self.tabu.append(source)
        self.current = source
        self.allowed.remove(source)
    
    # Se calcula el valor de lambda para resolver la probabilidad P(i,j), según lo enunciado en [ghoseiri2010]
    def _calculate_Lambda(self,a,b):

        if h <= a:
            return 0
        elif a < h and h < b:
            return i/(b-a) - a/(b-a)
        
        elif h >= b:
            return 1

    def _generate_first_heuristic_parameter(self,f,i,j):

        max_f1_cost, max_f2_cost = self.graph.get_max_values_costs
        min_f1_cost, min_f2_cost = self.graph.get_min_values_costs

        if (f == 1):
            return min(1, (max_f1_cost - self.graph.f1_cost[i][j])/(max_f1_cost - min_f1_cost) + self.epsilon)
        
        else:
            return min(1, (max_f2_cost - self.graph.f2_cost[i][j])/(max_f2_cost - min_f2_cost) + self.epsilon)

    
    def _generate_second_heuristic_parameter(self):
        
        #l = []
        #label = [float('inf') for i in self.graph.numNodes]
        #label = [0 for i in self.graph.numNodes]
        return

        
        

    #2.1 Choose next node by applying the state transition rule
    def _select_next(self, q, q0, a, b, h):

        denominator = 0
        for i in self.allowed:
            
            first_term_denominator = ((((self.f1_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
            second_term_denominator = ((((self.f2_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
            denominator += first_term_denominator*second_term_denominator*0.5
                                                                                            
        # noinspection PyUnusedLocal
        # Se inicializa una lista vacía (0's) de probabilidades de que la hormiga estando en el nodo i pueda ir al nodo j
        probabilities = [0 for i in range(self.graph.numNodes)]  # probabilities for moving to a node in the next step

        # Verifica si el nodo actual es el nodo destino
        if self.current == self.destiny:
            pass

        else:

            ant_lambda = _calculate_Lambda(a,b)

            if q <= q0:
                   
                for j in self.allowed:

                    max_p = 0 

                    for i in self.allowed:

                        first_term = ((((self.f1_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
                        second_term = ((((self.f2_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
                        numerator = first_term * second_term * 0.5

                        if numerator > max_p:
                            max_p = numerator
                    
                    if j == max_p:
                        probabilities[i] = max_p
                    else:
                        probabilities[i] = 0
            
            else:

                for i in self.allowed:

                    first_term_numerator = ((((self.f1_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
                    second_term_numerator = ((((self.f2_local_pheromone[self.current][i])**self.colony.alpha) * (_generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
                    numerator = first_term_numerator * second_term_numerator * 0.5

                    probabilities = numerator / denominator

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


    def local_update_pheromone(self):

        self.f1_local_pheromone = [[0 for j in range(self.graph.numNodes)] for i in range(self.graph.numNodes)]
        self.f2_local_pheromone = [[0 for j in range(self.graph.numNodes)] for i in range(self.graph.numNodes)] 

        for i, row in enumerate(self.graph.f1_pheromone):
            for j, col in enumerate(row):

                self.f1_local_pheromone[i][j] = self.colony.Q / self.total_cost
                self.f2_local_pheromone[i][j] = self.colony.Q / self.total_cost

                self.f1_local_pheromone[i][j] *= self.phi
                self.f2_local_pheromone[i][j] *= self.phi

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
