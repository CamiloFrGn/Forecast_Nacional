# -*- coding: utf-8 -*-
"""
Created on Sun May 10 00:26:38 2020

@author: jsdelgadoc
"""

# Parte 1 - Preprocesado de los datos

# Importación de las librerías
import numpy as np
import pandas as pd
from keras.models import load_model
import modulo_conn_sql as mcq
import datetime 

def conectarSQL():
    conn = mcq.ConexionSQL()
    cursor = conn.getCursor()
    return cursor



def obtenerDataTrain(pais, inicio, fin):
    #Conectar con base sql y ejecutar consulta
    cursor = conectarSQL()
    try:
        cursor.execute("{CALL SCAC_AP4_Serie_VolumenDiario (?,?,?)}" , (pais, inicio, fin) )
        #obtener nombre de columnas
        names = [ x[0] for x in cursor.description]
        
        #Reunir todos los resultado en rows
        rows = cursor.fetchall()
        resultadoSQL = []
            
        #Hacer un array con los resultados
        while rows:
            resultadoSQL.append(rows)
            if cursor.nextset():
                rows = cursor.fetchall()
            else:
                rows = None
                
        #Redimensionar el array para que quede en dos dimensiones
        resultadoSQL = np.array(resultadoSQL)
        resultadoSQL = np.reshape(resultadoSQL, (resultadoSQL.shape[1], resultadoSQL.shape[2]) )
    finally:
            if cursor is not None:
                cursor.close()
    return pd.DataFrame(resultadoSQL, columns = names)

def obtenerDataAux_Test(pais, dias):
    #Conectar con base sql y ejecutar consulta
    cursor = conectarSQL()
    try:
        cursor.execute("{CALL SCAC_AP4_Serie_VolumenDiario_AuxFecha (?,?)}" , (pais, dias) )
        #obtener nombre de columnas
        names = [ x[0] for x in cursor.description]
        
        #Reunir todos los resultado en rows
        rows = cursor.fetchall()
        resultadoSQL = []
            
        #Hacer un array con los resultados
        while rows:
            resultadoSQL.append(rows)
            if cursor.nextset():
                rows = cursor.fetchall()
            else:
                rows = None
                
        #Redimensionar el array para que quede en dos dimensiones
        resultadoSQL = np.array(resultadoSQL)
        resultadoSQL = np.reshape(resultadoSQL, (resultadoSQL.shape[1], resultadoSQL.shape[2]) )
    finally:
            if cursor is not None:
                cursor.close()
    return pd.DataFrame(resultadoSQL, columns = names)


# Importar el dataset de entrenamiento
#dataset_train = pd.read_excel("VolumenDiarioDespachado.xlsx")

pais = 'Colombia'      
inicioHistoria = datetime.datetime(2013, 5, 1) #'2013-05-01'
finHistoria = datetime.datetime.today() #fecha actual

dataset_train = obtenerDataTrain( pais, inicioHistoria.strftime("%Y-%m-%d"), finHistoria.strftime("%Y-%m-%d") )
training_set  = dataset_train.iloc[:, 2:4].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

timesteps = 20

# Crear una estructura de datos con 60 timesteps y 1 salida
X_train = []
X_train_vol = []
X_train_diaSemana = []
y_train = []
for i in range(timesteps, len(training_set)):
    #columna 0 -> Volmen
    X_train_vol.append( training_set_scaled[i-timesteps:i, 0])
    #columna 1 -> DiaSemana
    X_train_diaSemana.append( training_set_scaled[i-timesteps:i, 1])
    
    y_train.append(training_set_scaled[i, 0])
    
X_train_vol, X_train_diaSemana, y_train = np.array(X_train_vol), np.array(X_train_diaSemana), np.array(y_train)

#Agregamos una nueva dimension a las variables
X_train_vol_reshape = np.reshape(X_train_vol, (X_train_vol.shape[0], X_train_vol.shape[1], 1 ))
X_train_diaSemana_reshape = np.reshape(X_train_diaSemana, (X_train_diaSemana.shape[0], X_train_diaSemana.shape[1], 1 ))

# Creamos el array que resultará en una matriz del tamaño ej:(1198, 60, 3)
X_train = np.append(X_train_vol_reshape, (X_train_diaSemana_reshape), axis = 2 )


# Parte 2 - Construcción de la RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicialización del modelo
regressor = Sequential()

# Añadir la primera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2] ) ))
regressor.add(Dropout(0.2))

# Añadir la segunda capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la quinta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la septima capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 600))
regressor.add(Dropout(0.2))

# Añadir la capa de salida
regressor.add(Dense(units = 1))

# Compilar la RNR
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Ajustar la RNR al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 550, batch_size = 512)


#Guardo la RNR entrenada para futuros usos
##########regressor.save("../datos/forecastNacional_" +  pais + "Escenario26"  + "_" + str(timesteps) + "Timesteps_600celdas_550epochs_8layers.h5")

# Parte 3 - Ajustar las predicciones y visualizar los resultados

regressor = load_model('../datos/forecastNacional_Colombia2020-10-09_20Timesteps_100-200celdas_400epochs.h5')


data_aux_fecha = obtenerDataAux_Test(pais, 0)
fecha_prediccion = data_aux_fecha.iloc[:, 3:4]
data_aux_fecha  = data_aux_fecha.iloc[:, 0:1].values.astype(float)

sc2 = MinMaxScaler(feature_range = (0, 1))
data_aux_fecha_scaled = sc2.fit_transform(data_aux_fecha)

inputs = training_set_scaled[len(training_set_scaled) - timesteps: len(training_set_scaled)]

for j in range (0, data_aux_fecha.shape[0]):
    
#se crea x_test con la arquitectura para ingresar datos a la RNR
#es decir por cada prediccion necesitamos la informacion de los timesteps anteriores
    X_test = []
    X_test_vol = []
    X_test_diaSemana = []
    
    for i in range(timesteps, inputs.shape[0] + 1):
        X_test_vol.append(inputs[i-timesteps:i, 0])
        X_test_diaSemana.append(inputs[i-timesteps:i, 1])
    
    X_test_vol, X_test_diaSemana = np.array(X_test_vol), np.array(X_test_diaSemana) 
    
    X_test_vol_reshaped = np.reshape(X_test_vol, (X_test_vol.shape[0], X_test_vol.shape[1], 1))
    X_test_diaSemana_reshaped = np.reshape(X_test_diaSemana, (X_test_diaSemana.shape[0], X_test_diaSemana.shape[1], 1))
    
    X_test = np.append(X_test_vol_reshaped, (X_test_diaSemana_reshaped), axis=2)
    
    #por fin ejecutar prediccion
    vol_demanda_prediccion = regressor.predict(X_test)
    
    vol_demanda_prediccion = np.append(vol_demanda_prediccion, ( data_aux_fecha_scaled[ : len(vol_demanda_prediccion) , :1] ), axis = 1)
    
    inputs = training_set_scaled[len(training_set_scaled) - timesteps:]
    
    inputs = np.append(inputs, (vol_demanda_prediccion), axis = 0 )
    
    #print(j)

#la prediccion esta escalada, se hace la operacion inversa para tener los valores legibles
prediccion =  sc.inverse_transform(vol_demanda_prediccion)

df = pd.DataFrame(prediccion)
df = pd.concat([fecha_prediccion, df], axis=1)
df.to_excel("../datos/Resultado" + pais +  "Escenario27" + ".xlsx")










