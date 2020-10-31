# Script que implementa Ant-Colony bi-objetivo propuesto por [ghoseiri et al 2016 ], el cual esta basado en [Dorigo 1997]
# Así mismo, se van a utilizar las funciones de costo transporte y costo interacción de ProyectoGrado
# Para este escenario de prueba, se utilizara el edificio SD y el edificio Ga como nodo origen y destino respectivamente

import random

# Testing Purposes
import matplotlib.pyplot as plt

class Graph(object):

    def get_max_values_costs(self):

        max_f1 = -1
        max_f2 = -1

        for i in range(self.numNodes):
            for j in range(self.numNodes):

                if max_f1 < self.f1_cost[i][j]:
                    max_f1 = self.f1_cost[i][j]
                
                if max_f2 < self.f2_cost[i][j]:
                    max_f2 = self.f2_cost[i][j]

        return max_f1,max_f2

    def get_min_values_costs(self):

        min_f1 = 999
        min_f2 = 999

        for i in range(self.numNodes):
            for j in range(self.numNodes):

                if min_f1 > self.f1_cost[i][j]:
                    min_f1 = self.f1_cost[i][j]
                
                if min_f2 > self.f2_cost[i][j]:
                    min_f2 = self.f2_cost[i][j]

        return min_f1,min_f2

    def __init__(self, f1_cost_matrix,f2_cost_matrix, coorX, coorY, numNodes):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix (Number of nodes, or in this case, number of cities)
        """
        self.numNodes = numNodes
        self.coorX = coorX
        self.coorY = coorY
        self.f1_cost = f1_cost_matrix
        self.f2_cost = f2_cost_matrix
        self.f1_min, self.f2_min = self.get_min_values_costs()
        self.f1_max, self.f2_max = self.get_max_values_costs()
        # noinspection PyUnusedLocal

        # f1: Función Costo de Transporte
        # f2: Función Costo de Interacción

        self.f1_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]
        self.f2_pheromone = [[1 / (numNodes * numNodes) for j in range(numNodes)] for i in range(numNodes)]
    

class ACO(object):

    def __init__(self, ant_count, generations, alpha, beta, phi, rho, Q, q0, a, b, epsilon, delta,
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
        self.delta = delta
        self.a = a
        self.b = b
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

        self.In = []
        self.Out = []

        self.second_heuristic_parameter = []

    def global_update_pheromone(self, graph, ants):
        for i, row in enumerate(graph.f1_pheromone):
            for j, col in enumerate(row):

                for ant in ants:

                    graph.f1_pheromone[i][j] += ant.f1_local_pheromone[i][j]
                    graph.f2_pheromone[i][j] += ant.f2_local_pheromone[i][j]
                
                graph.f1_pheromone[i][j] *= self.rho
                graph.f2_pheromone[i][j] *= self.rho

            graph.f1_pheromone[i][j] = min(1, graph.f1_pheromone[i][j] + (graph.numNodes/graph.f1_cost[i][j]) )
            graph.f2_pheromone[i][j] = min(1, graph.f2_pheromone[i][j] + (graph.numNodes/graph.f2_cost[i][j]) )

    # noinspection PyProtectedMember

    def _generate_set_incoming_and_outgoing_nodes(self, graph):
        In = [[] for i in range(graph.numNodes)]
        Out = [[] for i in range(graph.numNodes)]
        for i in range(graph.numNodes):
            for j in range(graph.numNodes):

                if (graph.f1_cost[i][j] <= 0.00012):

                    In[j].append(i)
                    Out[i].append(j)

        self.In = In
        self.Out = Out


    def _generate_second_heuristic_parameter(self, graph, dic_edificios_nodos):

        l = [j for j in range(graph.numNodes)]
        label = [99999 for i in range(graph.numNodes)]
        #label = [0 for i in range(graph.numNodes)]
        n_2 = [0 for j in range(graph.numNodes)]

        #label[dic_edificios_nodos['GA']] = 0

        while len(l) > 0:
            j = l.pop(0)
            #print("\n","INFO: Incoming nodes for node %s: %d"%(j,len(self.In[j])))
            for k in self.In[j]:
                #print("\n","INFO: Incoming node: %s" %k)
                label[k] = min(label[j] + 1, label[k])

        for i in range(graph.numNodes):

            if i == dic_edificios_nodos['GA']:
                pass

            else:
                n_2[i] = 1/label[i]
                #print("INFO: n_2[i] assigned: %s" %n_2[i])
        
        return n_2
            
       
    def solve(self, graph, dic_edificios_nodos):
        """
        :param graph:
        """
        best_f1_cost = float('inf')
        best_f1_solution = []

        best_f2_cost = float('inf')
        best_f2_solution = []

        self._generate_set_incoming_and_outgoing_nodes(graph)
        self.second_heuristic_parameter = self._generate_second_heuristic_parameter(graph,dic_edificios_nodos)

        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            # 0. Preparing a new colony

            print("\n",'STATUS: Preparing %s colony' %gen)

            ants = [_Ant(self, i, graph,dic_edificios_nodos, self.epsilon, self.phi) for i in range(self.ant_count)]
            ant_index = 0
            for ant in ants:
                ant_index +=1
                print("\n",'STATUS: For %s colony preparing %d ant' %(gen,ant_index))

                while ant.current != ant.destiny:
                    # 2.0 Choose next node by applying the state transition rule 
                    ant._select_next(self.Q, self.q0, self.a, self.b)

                    # Apply local pheromone update
                    ant.local_update_pheromone()
                    
                ant.f1_total_cost += graph.f1_cost[ant.f1_tabu[-1]][ant.f1_tabu[0]]
                ant.f2_total_cost += graph.f2_cost[ant.f2_tabu[-1]][ant.f2_tabu[0]]

                if ant.f1_total_cost < best_f1_cost and ant.f2_total_cost < best_f2_cost:
                    best_f1_cost = ant.f1_total_cost
                    best_f2_cost = ant.f2_total_cost
                    best_f1_solution = [] + ant.f2_tabu

                
                # Apply local pheromone update
                #ant.local_update_pheromone()

            # 5. Apply global pheromone update
            self.global_update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_f1_solution, best_f1_cost


class _Ant(object):
    def __init__(self, aco, index, graph,dic_edificios_nodos, epsilon, phi):

        self.index = index
        self.colony = aco
        self.graph = graph
        self.f1_total_cost = 0.0
        self.f2_total_cost = 0.0
        self.f1_tabu = []  # tabu list List[ T(i,j) ]
        self.f2_tabu = []  # tabu list List[ T(i,j) ]
        self.f1_local_pheromone = [[0 for j in range(self.graph.numNodes)] for i in range(self.graph.numNodes)]  # the local increase of pheromone (Delta_T(i,j))
        self.f2_local_pheromone = [[0 for j in range(self.graph.numNodes)] for i in range(self.graph.numNodes)]  # the local increase of pheromone (Delta_T(i,j))
        self.allowed = [i for i in range(graph.numNodes)]  # nodes which are allowed for the next selection
        self.selected_nodes = []
        #self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.numNodes)] for i in
        #            range(graph.numNodes)]  # heuristic information (N(i,j))

        self.epsilon = epsilon
        self.phi = phi

        # 1. Position each ant in a starting node
        
        self.source = dic_edificios_nodos['SD']  # start from Source Node
        self.destiny = dic_edificios_nodos['GA']
        self.f1_tabu.append(self.source)
        self.f2_tabu.append(self.source)
        self.current = self.source
        self.allowed.remove(self.source)
    
    # Se calcula el valor de lambda para resolver la probabilidad P(i,j), según lo enunciado en [ghoseiri2010]
    def _calculate_Lambda(self,a,b):

        if self.index <= a:
            return 0
        elif a < self.index and self.index < b:
            return self.index/(b-a) - a/(b-a)
        
        elif self.index >= b:
            return 1

    def _generate_first_heuristic_parameter(self,f,i,j):

        # f: Objetivo del parametro heuristíco que se quiere obtener, 1 para f1, d.l.c f2

        max_f1_cost = self.graph.f1_max
        max_f2_cost = self.graph.f2_max

        min_f1_cost = self.graph.f1_min
        min_f2_cost = self.graph.f2_min

        if (f == 1):
            return min(1, (max_f1_cost - self.graph.f1_cost[i][j])/(max_f1_cost - min_f1_cost) + self.epsilon)
        
        else:
            return min(1, (max_f2_cost - self.graph.f2_cost[i][j])/(max_f2_cost - min_f2_cost) + self.epsilon)

    #2.1 Choose next node by applying the state transition rule
    def _select_next(self, q, q0, a, b):

        # Verifica si el nodo actual es el nodo destino
        if self.current == self.destiny:
            print("\n","STATUS: Destination Node Found") 
            return

        ant_lambda = self._calculate_Lambda(a,b)
        denominator = 0
        #print("\n","STATUS: Calculating denominator") 
        for i in self.allowed:

            #print("\n","STATUS: ... for %s index node" %i) 
            # ((self.colony.second_heuristic_parameter[self.current])**self.colony.delta)

            #if self.graph.f1_cost[self.current][i] <= 0.012:

            first_term_denominator = ((((self.graph.f1_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
            second_term_denominator = ((((self.graph.f2_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
            denominator += first_term_denominator*second_term_denominator#*(self.colony.second_heuristic_parameter[i]**self.colony.delta)
                                                                                            
        # noinspection PyUnusedLocal
        # Se inicializa una lista vacía (0's) de probabilidades de que la hormiga estando en el nodo i pueda ir al nodo j
        #print("\n","STATUS: Initializing Probabilities List") 
        #probabilities = [0 for i in range(len(self.colony.Out[self.current]))]  # probabilities for moving to a node in the next step
        probabilities = [0 for i in range(self.graph.numNodes)]
        #print("\n","INFO: Probabilities Length: %s" %(len(probabilities)))

        if q <= q0:

            max_p = 0 
            
            for i in range(self.graph.numNodes):

                try:

                    self.allowed.index(i)
                
                    #print("\n","INFO: Outgoing nodes: %s" %str(self.colony.Out[self.current]))

                    #if self.graph.f1_cost[self.current][i] <= 0.012:

                    first_term = ((((self.graph.f1_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
                    second_term = ((((self.graph.f2_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
                    numerator = first_term * second_term#*(self.colony.second_heuristic_parameter[i]**self.colony.delta)

                    if numerator > max_p:
                        max_p = numerator
                
                except ValueError:
                    pass

              
            for i in range(self.graph.numNodes):

                try:
                    #self.allowed.index(i)
            
                    #print("\n","INFO: Outgoing nodes: %s" %str(self.colony.Out[self.current]))

                    #if self.graph.f1_cost[self.current][i] <= 0.012:

                    if i == max_p:

                        probabilities[i] = 1
                            #probabilities.append(1)
                            #print("/n","STATUS: Probability Assigned: %s" %probabilities[i])
                    else:

                        probabilities[i] = 0
                            #probabilities.append(0)
                            
                            #print("/n","STATUS: Probability Assigned: %s" %probabilities[i])
                    
                except ValueError:
                    pass
            
        else:
            
            for i in range(self.graph.numNodes):

                try:

                    if i in self.allowed:

                        self.allowed.index(i)

                        #print("\n","INFO: Outgoing nodes: %s" %str(self.colony.Out[self.current]))

                        #if self.graph.f1_cost[self.current][i] <= 0.012:

                            #print("\n","INFO: Outgoing nodes length: %s" %len(self.colony.Out[self.current]))

                        first_term_numerator = ((((self.graph.f1_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(1,self.current,i)**self.colony.beta))**ant_lambda)
                        second_term_numerator = ((((self.graph.f2_pheromone[self.current][i])**self.colony.alpha) * (self._generate_first_heuristic_parameter(2,self.current,i)**self.colony.beta))**( 1 - ant_lambda))
                        numerator = first_term_numerator * second_term_numerator#*(self.colony.second_heuristic_parameter[i]**self.colony.delta)

                        probabilities[i] = numerator / denominator
                            #probabilities.append(numerator / denominator)
                            #print("/n","STATUS: Probability Assigned: %s" %probabilities[i])
                    
                    else:
                        probabilities[i] = 0

                except ValueError:
                    pass

        # select next node by probability roulette
        #selected = 0
        #rand = random.random()
        #for i, probability in enumerate(probabilities):
        #    #
        #    print(i,probability)
        #    rand -= probability
        #    if rand <= 0:
        #        selected = i
        #        break
        
        selected = 0
        max_p = 0
        for i, probability in enumerate(probabilities):
            #print(i,probability)
            if probability > max_p:
                max_p = probability
                selected = i

        #print("\n","STATUS: Selected Node: %s" %selected)        
        # La hormiga se mueve al nodo j

        #print("INFO: Number allowed nodes to visit: %s" %len(self.allowed))
        #print("INFO: Selected Node: %s" %selected)

        self.allowed.remove(selected)
        self.selected_nodes.append(selected)

        self.f1_tabu.append(selected)
        self.f2_tabu.append(selected)
        self.f1_total_cost += self.graph.f1_cost[self.current][selected]
        self.f2_total_cost += self.graph.f1_cost[self.current][selected]
        self.current = selected
    


    def local_update_pheromone(self):

        for _ in range(1, len(self.f1_tabu)):
            i = self.f1_tabu[_ - 1]
            j = self.f1_tabu[_]

            k = self.f2_tabu[_ - 1]
            l = self.f2_tabu[_]

            self.f1_local_pheromone[i][j] = self.colony.Q / self.f1_total_cost
            self.f2_local_pheromone[k][l] = self.colony.Q / self.f2_total_cost

            self.f1_local_pheromone[i][j] *= self.phi
            self.f2_local_pheromone[k][l] *= self.phi
