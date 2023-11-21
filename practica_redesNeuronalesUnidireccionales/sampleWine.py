#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:15:00 2021

@author: oscarreinoso
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
#from tensorflow.keras.utils import plot_model

# Se leen los datos del fichero
data=np.loadtxt('./wineData.txt', delimiter=",")



# Aleatorizamos las filas de la matriz
dataRandom=data
np.random.shuffle(dataRandom)

## Datos de entreanmiento 60% - Test 40%
#numFil=int(0.6*len(dataRandom))
#trainData = dataRandom[0: numFil]
#testData = dataRandom[numFil:]
#
## Se extraen los datos de entrada/salida para el entrenamiento
#x = trainData[:, 1:].astype(float)
#y = trainData[:, 0].astype(int)

# Se extraen los datos de entrada/salida 
x = dataRandom[:, 1:].astype(float)
y = dataRandom[:, 0].astype(int)


# Se normalizan los datos de entrada
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Se codifican las clases de salida: 1 (1 0 0), 2 (0 1 0), 3(0 0 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
#La siguiente funcion necesita un array bidimensional como entrada
y_reshaped = np.array(y).reshape(-1,1) 
y = ohe.fit_transform(y_reshaped).toarray()


# Dividimos los datos en un conjunto de entrenamiento y otro de test
# En este caso dejamos un 10% como test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)

# Se crea la red neuronal con una capa oculta de 10 neuronas
# capa de entrada de 13 neuronas, y de salida de 3 neuronas
# una vez que se ha modificado la salida
model = Sequential() 
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Sacamos por pantalla el modelo que hemos construido
model.summary()

# Se dibuja la arquitectura de red a fichero
#plot_model(model, to_file='./model1CapaOculta.png')


# Ajustes del modelo para entrenar
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=50)
# Se dibujan los resultados
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('Precision del entrenamiento con 100 iteraciones')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# Se evalua el modelo con los datos de test
y_pred=model.predict(x_test)

# Convertimos las predicciones hechas por la red a las etiquetas 0, 1 ó 2
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
    
# Convertimos los valores deseados y_test a las etiquetas 0, 1 o 2
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
    
from sklearn.metrics import accuracy_score
accIteraciones = accuracy_score(pred,test)
print('Precisión tras ITERACIONES:', accIteraciones*100)

print('Numero de valores de test:', len(test))
import matplotlib.pyplot as plt
plt.plot(test[:], 'bo')
plt.plot(pred[:], 'r+')
plt.title('Valores deseados vs Valores salida por la red \n Entrenamiento por ITERACIONES')
plt.ylabel('Etiqueta de salida')
plt.xlabel('Entrada')
plt.legend(['Valor deseado', 'Valor salida'], loc='best')
plt.show()



# Otra alternativa es entrenar mediante el metodo de validacion cruzada
history2 = model.fit(x_train, y_train,validation_data = (x_test,y_test), 
                      epochs=100)

# Se dibujan los resultados
import matplotlib.pyplot as plt
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model accuracy Validacion Cruzada')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plt.plot(history2.history['loss']) 
# plt.plot(history2.history['val_loss']) 
# plt.title('Model loss') 
# plt.ylabel('Loss') 
# plt.xlabel('Epoch') 
# plt.legend(['Train', 'Test'], loc='upper left') 
# plt.show()


# Se evalua el modelo con los datos de test
y_pred2=model.predict(x_test)

# Convertimos las predicciones hechas por la red a las etiquetas 0, 1 ó 2
pred2 = list()
for i in range(len(y_pred)):
    pred2.append(np.argmax(y_pred2[i]))
    
accValidacion = accuracy_score(pred2,test)
print('Precisión tras VALIDACION:', accValidacion*100)



import matplotlib.pyplot as plt
plt.plot(test[:], 'bo')
plt.plot(pred2[:], 'r+')
plt.title('Valores deseados vs Valores salida por la red \n Entrenamiento por VALIDACION CRUZADA')
plt.ylabel('Etiqueta de salida')
plt.xlabel('Entrada')
plt.legend(['Valor deseado', 'Valor salida'], loc='best')
plt.show()

