%config Completer.use_jedi = False
import keras
import numpy as np
import pandas as pd

#Read dta
KaggleBike="https://raw.githubusercontent.com/CrisBMoya/BikeSharingDataset/master/Data/train.csv"
TrainDF=pd.read_csv(KaggleBike, parse_dates=["datetime"])

#Subset
X_Train=TrainDF[2000:]

#Subset TrainDF
YVal=np.array(X_Train[["count"]])
XVal=np.array(X_Train[["temp", "atemp", "humidity", "windspeed","casual","registered"]])

#Data reshape
YVal=YVal.reshape((8886,1))
XVal=XVal.reshape((8886,2,3,1))

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
model.fit(XVal, YVal, epochs=30)

