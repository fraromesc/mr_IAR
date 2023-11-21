#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:36:00 2021

@author: OscarReinosoGarcia
"""

import numpy as np
import matplotlib.pyplot as plt

# Funcion que devuelve la celda mas cercana a la entrada
# Parametros de entrada: array 3D con la red SOM y valor de entrada (2D)
# Parametros de salida: coordenadas de la celda mas cercana al valor x
def find_BMU(SOM,x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
    



# Actualizacion de los pesos de las neuronas de la red SOM a partir de
# la neurona ganadora
def update_weights(SOM, train_ex, learn_rate, radius_sq, 
                    BMU_coord, step=3):
    g, h = BMU_coord
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])   
    return SOM    




# Rutina principal para el entrenamiento de la red SOM. 
def train_SOM(SOM, train_data, learn_rate = .1, radius_sq = 1, 
              lr_decay = .1, radius_decay = .1, epochs = 10):    
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)      
        for train_ex in train_data:
            g, h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex, 
                                  learn_rate, radius_sq, (g,h))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)            
    return SOM


###### CREAMOS LA MISMA RED QUE EN MATLAB POR COMPARAR
m=8
n=8
# Leemos los datos de entrenamiento
with open('data.csv', 'r') as f:
    data = np.loadtxt(f, delimiter=",")
    
train_data=np.transpose(data)   

# print(train_data)
    
# Construimos la red SOM: mxnx2 con valores tipo float aleatorios
rand = np.random.RandomState(0)
red_som = np.random.random_sample((m,n,2))

# print(red_som)

# Construida la red SOM junto con los datos de entrenamiento, 
# se realiza el entrenamiento de la red, con objeto de que las celdas se agrupen
red_som_entrenada = train_SOM(red_som, train_data, epochs=10)

# Dibujamos la red una vez está entrenada. Representamos los pesos
plt.scatter(red_som[:,:,0],red_som[:,:,1])
plt.scatter(train_data[:,0], train_data[:,1], s=0.2, marker='.')
plt.title('Distribucion de las neuronas (8x8)')
plt.show()








# # Creamos los DATOS DE ENTRENAMIENTO: 3000 valores aleatorios
# n_x = 3000
# rand = np.random.RandomState(0)
# # Inicializamos los datos de entrenamiento: cada dato es un color RGB
# train_data = rand.randint(0, 255, (n_x, 3))

# # COMPROBACION DE LA RED SOM
# # Creamos una red SOM de 10x10 celdas
# m = 10
# n = 10
# # Construimos la red SOM: mxnx3 con valores RGB tipo float aleatorios
# red_som = rand.randint(0, 255, (m, n, 3)).astype(float)

# # Representamos tanto los datos de entrenamiento (array de 3000 valores)
# # como los pesos iniciales de cada una de las celdas de la red (mxn)
# fig, ax = plt.subplots(
#     nrows=1, ncols=2, figsize=(12, 3.5), 
#     subplot_kw=dict(xticks=[], yticks=[]))
# ax[0].imshow(train_data.reshape(50, 60, 3))
# ax[0].title.set_text('Datos de entrenamiento')
# ax[1].imshow(red_som.astype(int))
# ax[1].title.set_text('Pesos iniciales en la red SOM')


# # Construida la red SOM junto con los datos de entrenamiento, 
# # se realiza el entrenamiento de la red, con objeto de que las celdas se agrupen
# # Se representa el resultado con 1, 5, 10 y 20 epochs para comprobar la evolución de los
# # pesos de cada celda en función de las iteraciones del entrenamiento
# fig, ax = plt.subplots(
#     nrows=1, ncols=4, figsize=(15, 3.5), 
#     subplot_kw=dict(xticks=[], yticks=[]))
# total_epochs = 0
# for epochs, i in zip([1, 4, 5, 10], range(0,4)):
#     total_epochs += epochs
#     red_som = train_SOM(red_som, train_data, epochs=epochs)
#     ax[i].imshow(red_som.astype(int))
#     ax[i].title.set_text('Epochs = ' + str(total_epochs))






