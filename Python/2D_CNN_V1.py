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
BikeDF=pd.read_csv(zip.ZipFile("Data/bike-sharing-demand.zip").open('train.csv'), parse_dates=["datetime"])

#Split into Test and Train
TestDF=BikeDF[:2000]
TrainDF=BikeDF[2000:]

#Subset TrainDF
TrainDF_Y=np.array(TrainDF[['count']])
TrainDF=np.array(TrainDF[['temp', 'atemp', 'humidity', 'windspeed']])

#Reshape the data
TrainDF=TrainDF.reshape((-1,2,2,1))

#Create Keras 2D CNN Model
model=keras.Sequential()
model.add(keras.layers.Conv2D(filters=10, kernel_size=[2,2], padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=15, kernel_size=[2,2], padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=20, kernel_size=[3,3], padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=[2,2], strides=1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=30, activation='relu'))
model.add(keras.layers.Dense(units=20, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='relu'))
model.add(keras.layers.Dense(units=1))

#Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#Fit model to Data
model.fit(TrainDF, TrainDF_Y, epochs=100)

#Subset TestDF
TestDF_Y=np.array(TestDF[['count']])
TestDF=np.array(TestDF[['temp', 'atemp', 'humidity', 'windspeed']])

#Reshape the data
TestDF=TestDF.reshape((-1,2,2,1))

#Evaluate on Test
EvalRes=model.evaluate(TestDF, TestDF_Y)
print('Test data Loss: ', EvalRes[0])
print('Test data Accuracy: ', EvalRes[1])

#Load Data to predict
BikeDF_Test=pd.read_csv(zip.ZipFile("Data/bike-sharing-demand.zip").open('test.csv'), parse_dates=["datetime"])
BikeDF_Test[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']]
