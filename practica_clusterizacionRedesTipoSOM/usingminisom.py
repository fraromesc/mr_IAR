#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:03:47 2021

@author: OscarReinosoGarcia
"""

# Importamos los datos de iris_dataset
from minisom import MiniSom
import numpy as np
import pandas as pd

# Leemos los datos y los dividimos en la entrada (data) y salida (target)
columns = ['longitudSepalo', 'anchuraSepalo', 'longitudPetalo', 'anchuraPetalo',
           'clase']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                   sep=',', names=columns, header=None, engine='python')
target = data['clase'].values
#Convertimos las etiquetas a numeros
targetValues=pd.factorize(target)[0]
data = data[data.columns[:-1]]

# Normalizamos los datos
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values


# Creamos la red SOM y entrenamos
n = 8
m = 8
som = MiniSom(n, m, data.shape[1], sigma=1.5, learning_rate =0.5, 
              neighborhood_function='gaussian', 
              activation_distance='euclidean', random_seed=0)
som.pca_weights_init(data)
som.train(data,1000,verbose=True)


# Visualizamos algunos resultados
import matplotlib.pyplot as plt

# En primer lugar representamos la distancia euclidea entre los pesos con las
# celdas vecinas
plt.figure(figsize=(9,9))
plt.pcolor(som.distance_map().T, cmap='Oranges')
plt.colorbar()
plt.title('Distancia de los pesos entre neuronas')

# # En segundo lugar representamos la respuesta de la red a cada patron de entrada
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(data):
    win = som.winner(xx) # obtenemos la neurona ganadora
    plt.plot(win[0]+0.5, win[1]+0.5, markers[targetValues[cnt]], 
              markerfacecolor='None', 
              markeredgecolor=colors[targetValues[cnt]], markersize=12, 
              markeredgewidth=2)
    
plt.show()

# Ahora dibujamos las neuronas que han sido ganadoras para cada entrada
# Se añade un offset aleatorio para evitar solapamientos al pintarlas con objeto
# de que podamos ver las veces que ha sido ganadora cada celda
# Creamos los valores (x,y) de las neuronas ganadaoras para cada dato de entrada
w_x, w_y = zip(*[som.winner(d) for d in data])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()


for c in np.unique(targetValues):
    idx_target = targetValues==c
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[c], label=np.unique(target)[c])
plt.legend(loc='upper right')
plt.grid()
plt.title('Neuronas ganadoras y clase de salida')
plt.show()

# Por último podemos ver cuales son las neuronas mas ganadoras (activadas) ante
# los datos presentados
plt.figure(figsize=(8,8))
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues')
plt.colorbar()
plt.title('Neuronas más activadas')
plt.show()



