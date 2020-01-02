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
BikeDF.drop(axis=1, labels=['registered','casual','datetime'], inplace=True)

#Get response value
TrainDF_Y=np.array(BikeDF['count'])

#Apply Sliding Window
SlidingSize=12

#Function to apply the sliding window to a DF
def ApplySliding(DF, Size):
    
    #Create Ranges table
    SlidingRange=pd.DataFrame(columns=['A','B'])
    SlidingRange['A']=range(0,DF.shape[0]-(SlidingSize-1))
    SlidingRange['B']=range(SlidingSize, DF.shape[0]+1)
    
    #Iterate and populate array
    XVal=np.empty(shape=(DF.shape[0]-SlidingSize+1, SlidingSize, (DF.shape[1]-1)))
    YVal=np.empty(shape=(DF.shape[0]-SlidingSize+1, 1))
    
    for i in range(0, SlidingRange.shape[0]):
        Temp=DF.iloc[SlidingRange.iloc[i][0]:SlidingRange.iloc[i][1]]
        XVal[i]=Temp.drop(columns='count', inplace=False)
        YVal[i]=Temp['count'].iloc[SlidingSize-1]
    pass
    
    #Return array
    return(XVal, YVal)
pass

#Array with sliding applied
TrainDF_X, TrainDF_Y=ApplySliding(DF=BikeDF, Size=SlidingSize)

#Subset responses to match last value in each sliding size
#TrainDF_Y=TrainDF_Y[(SlidingSize-1):TrainDF_Y.shape[0]]

#Check shapes
TrainDF_X.shape
TrainDF_Y.shape

# LSTM

## Define Feature number, which are column number
FeatureNumber=TrainDF_X.shape[2]

##
StepNumer=SlidingSize

## Define the model
LSTMModel=keras.Sequential()
LSTMModel.add(keras.layers.LSTM(50, activation='relu', input_shape=(StepNumer, FeatureNumber)))
LSTMModel.add(keras.layers.Dense(1))
LSTMModel.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

## Dit model
LSTMModel.fit(TrainDF_X, TrainDF_Y, epochs=30)



model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(TrainDF_X.shape[1], TrainDF_X.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
keras.optimizers.Adam(learning_rate=0.005)

history = model.fit(
    TrainDF_X, TrainDF_Y,
    epochs=30,
    batch_size=36,
    validation_split=0.1,
    shuffle=False
)