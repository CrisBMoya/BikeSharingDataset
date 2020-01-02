%config Completer.use_jedi = False
import keras
import numpy as np
import zipfile as zip
import pandas as pd
import tensorflow as tf
import os
import re

#Set Working Directory
os.chdir(re.sub(pattern='Python', repl='', string=os.getcwd()))

#Load Train DF
BikeDF=pd.read_table('Python/Temp_X.txt')
BikeDF.shape
BikeDF.drop(axis=1, labels=['casual','registered'], inplace=True)
BikeDF.shape

#Load dependent var
BikeDF_Y=pd.read_table('Python/Temp_Y.txt')
BikeDF_Y.shape
#
TrainDF=np.array(BikeDF)
TrainDF=TrainDF.reshape((-1,3*2,4))
TrainDF_Y=np.array(BikeDF_Y)

TrainDF.shape
TrainDF_Y.shape


###########
model=keras.Sequential()
model.add(keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=1))
model.add(keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=1))


#Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#Fit model to Data
model.fit(TrainDF, TrainDF_Y, epochs=100, batch_size=16)
