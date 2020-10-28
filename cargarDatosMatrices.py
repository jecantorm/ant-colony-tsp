"""
Script que carga de Firebase los origenes y destinos de los estudiantes.
Retorna la matriz de costo de transporte, la matriz de costo de interacción y el número de nodos de la red.

@author: Juan Cantor
"""

# OS,Sys Imports
import os
import sys

#Datetime Imports
import datetime

#Dijkstra Shortest Path Imports (Camino de costo minímo para estudiantes)
import math
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.sparse import csr_matrix

#GeoJSON Imports
import json
import matplotlib.pyplot as plt

#Pickle Imports (Escribir y leer objetos, como la matriz de costo transporte y la matriz de costo de interaccion)
import pickle

#Firebase Imports (Base de datos)
import firebase_admin
from firebase_admin import credentials, firestore


# Se declara de antemano el diccionario que servirá para guardar el nombre del edificio con su respectivo indíce
dic_edificios_nodos = {}

# Se declara el radio de cobertura que deben haber entre los nodos
RC = 0.00012

# Se declaran los arrays en los cuales se van a guardar las coordenas X y Y de los nodos
coorX = []
coorY = []

# Función que calcula la matriz de costos de transporte (distancia entre nodos) de los nodos cargados en el GeoJSON
def calcularMatrizCostoTransporte(coorX,coorY):
    

    #Se define un radio de cobertura entre cada nodo y la matriz donde se van guardar los costos de transporte
    RC = 0.00012
    numNodes = len(coorX)
    matrix_ct = 99999*np.ones((numNodes,numNodes)) 
    
    i=-1
    for eachCordX in coorX:
        i=i+1
        j=-1
        for eachCordY in coorY:
            j=j+1
            dij = math.sqrt((coorX[i]-coorX[j])**2+(coorY[i]-coorY[j])**2)
            
            #print(i,j,dij)
            if dij <= RC and i!=j:
                
                matrix_ct[i][j]=dij
                plt.plot([coorX[i],coorX[j]],[coorY[i],coorY[j]],'k--',markersize=1 )
    
    return matrix_ct

# Función que lee el JSON descargado de GeoJSON y plotea los puntos con matplotlib
def plotGeoJSON():
    
    f = open('/home/juancm/Desktop/Octavo/Tesis/ProyectoGrado/Modelos Matematicos/puntos_edificios_nodos_universidad.json','r')
    geojson = json.loads(f.read())
    
    print('\n','STATUS: Reading GeoJSON JSON')
    
    for i in geojson['features']:
        coords = i['geometry']["coordinates"]
        coorX.append(coords[0])
        coorY.append(coords[1])
        
        if(bool(i['properties'])):
            str_name = i['properties']['name']
            
            #Se obtienen los nombres de los edificios y el número de nodos donde quedaron guardados
            if 'edificio' in str_name:
                dic_edificios_nodos[str_name.split('_')[1]] = len(coorX) - 1
            
            elif "t_" in str_name: 
                dic_edificios_nodos[str_name.split('_')[1]] = len(coorX) - 1
          
                                     
    # Se plotean los nodos
    print('\n','STATUS: Plotting GeoJSON points')
    plt.plot(coorX, coorY, 'ko', label='Nodes',markersize=2)
    
    # Se plotean el número de cada nodo
    # cont=-1
    # for eachCoorX in coorX:
    #     cont=cont+1
    #     x=coorX[cont]
    #     y=coorY[cont]
    #     textPosX=x; textPosY=y
    #     offsetX=0.000075; offsetY=0.000025
    #     texto =str(cont)
        #plt.text(textPosX + offsetX, textPosY + offsetY, texto, rotation=0, size=1)
    #end
    
    print('\n','STATUS: Printing Edificios - Num. Nodos dictionary')
    print('\n',dic_edificios_nodos)
    
    #Retorna como valores las matrices de las coordenadas X y Y, así como el diccionario del nombre de los edificios con sus respectivos indíces
    return coorX, coorY, dic_edificios_nodos
            
    

# Se elige un rango de horas y el ciclo académico a utilizar, en este caso Ciclo 1 de Lunes de 6:20 - 6:30
# En este arreglo se guardaran los valores de Strings que se usarán para consultar en la db de Firebase

# Funcion que consulta Firestore y que obtiene los origenes y destinos de los estudiantes que
# van a transitar durante el rango de horas establecido
def get_SyD_Firestore(ciclo,dia,horario):
    
    #Métodos y variables para iniciar las instancias de Firebase
    
    if not firebase_admin._apps:
        cred = credentials.Certificate("/home/juancm/Desktop/Octavo/Tesis/ProyectoGrado/ETL/proyectogrado-b1000-firebase-adminsdk-mclb7-5a0e16b08e.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    rango_document = db.collection(u'cursos_partes_v2').document(u'' + ciclo).collection(u'' + dia).document(u'' + horario).get()
    
    if rango_document.exists:
        
        print('Returning Source and Destiny Dictionary')
        return list(rango_document.to_dict().values())
        
    else:
        print('Firestore: Document Not Found')


#Función que calcula el camino de costo minímo de los estudiantes que van a transitar
# Y asi mismo, guarda los valores de los parámetros de los costos de interaccion de los caminos que los estudiantes utilizarán
def darMatrizCantidadPersonasCaminos(coorX,coorY,matrix_ct,dic_edificios_nodos,list_dict_syd):
    
    numNodes = len(coorX)
    
    # Matriz que guardara la cantidad de estudiantes que han pasado por los caminos
    matrix_cantidad_E = np.zeros((numNodes,numNodes))
    
    # Se define una función que retorne la matriz de nodos intermedios entre el origen y el destino a partir de la matriz total de nodos precursores
    def get_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path[::-1]
    
        
    matrix_ct_csr = csr_matrix(matrix_ct)
    
    # Se obtienen los nodos precursos de la función shortest_path importada de la libreria SciPy
    sp, pr = shortest_path(csgraph = matrix_ct_csr, directed=False, method='D', return_predecessors = True)
    
    
    # TEST: Se declara un contador de cuantos caminos de costo minímo se deben calcular. Esto se hace con el fin de hacer mas rápidas las computaciones
    
    countEstudiantes = 0
    maxEstudiantes = len(list_dict_syd)
    
    for i in list_dict_syd:
        
        if countEstudiantes < maxEstudiantes:
        
            source = dic_edificios_nodos[i['s']]
            destiny = dic_edificios_nodos[i['d']]
            
            sp_indexes = get_path(pr,source,destiny)
            
            # Se cuentan la cantidad de estudiantes que transitaron por los caminos de costo minímo transitados
            # para que después se puedan usar estos datos en la función objetivo del costo de interacción
            # Ademas, se dibuja con una línea azul el camino de costo minímo entre dos nodos 
            
            i=0

            while i < len(sp_indexes) - 1:
                matrix_cantidad_E[sp_indexes[i]][sp_indexes[i+1]] += 1
                
                plt.plot([coorX[sp_indexes[i]],coorX[sp_indexes[i+1]]],[coorY[sp_indexes[i]],coorY[sp_indexes[i+1]]],'b', linewidth=2)
                
                i = i+1
            
            if len(sp_indexes) == 2 and matrix_ct[source, destiny] > RC:
                print("\nNo se ha encontrado camino")
                print(sp_indexes)
                
                
                print('STATUS: Calculating Dijkstra Shortest Path on %s' %(i))
                
            countEstudiantes +=1
        
        else:
            break
                
    return matrix_cantidad_E,numNodes

# Funcion que carga las matrices de costo si ya se habían guardado previamente
def cargarDatosMatrices(ciclo,dia,horario):
        
    coorX, coorY, dic_edificios_nodos = plotGeoJSON()
    
    matrix_ct = calcularMatrizCostoTransporte(coorX,coorY)
    
    list_dict_syd = get_SyD_Firestore(ciclo,dia,horario)
    
    matrix_ci,numNodes = darMatrizCantidadPersonasCaminos(coorX,coorY,matrix_ct,dic_edificios_nodos,list_dict_syd)
       
    print('INFO: numNodes', numNodes)
    
    return coorX, coorY, matrix_ct, matrix_ci,dic_edificios_nodos, numNodes

#Guardar en una imagen .eps la red de nodos 
#plt.savefig('cargarDatosMatrices_redNodosUniversidad_%s.eps' %str(datetime.datetime.now()), format='eps')
                
## FIN DE LOS MÉTODOS DE CONFIGURACION DE PARAMETROS