%config Completer.use_jedi = False
import tensorflow.keras as keras
import numpy as np
import zipfile as zip
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler

#Set Working Directory
os.chdir(re.sub(pattern='Python', repl='', string=os.getcwd()))

#Load Train DF
BikeDF=pd.read_csv(zip.ZipFile("Data/bike-sharing-demand.zip").open('train.csv'), parse_dates=["datetime"])
BikeDF.drop(axis=1, labels=['registered','casual','datetime'], inplace=True)
ColNames=BikeDF.columns

#Get response value
#TrainDF_Y=np.array(BikeDF['count'])

#Scale - TEMP
#scaler = StandardScaler()
#ScaleFit=scaler.fit(BikeDF)
#BikeDF=scaler.transform(BikeDF)
#BikeDF=pd.DataFrame(BikeDF, columns=ColNames)

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

#Split Data
TestDF_X=TrainDF_X[9788:TrainDF_X.shape[0]]
TestDF_Y=TrainDF_Y[9788:TrainDF_Y.shape[0]]

TrainDF_X=TrainDF_X[0:9788]
TrainDF_Y=TrainDF_Y[0:9788]


# LSTM

## Define Feature number, which are column number
FeatureNumber=TrainDF_X.shape[2]

##
StepNumer=SlidingSize

## Define the model
LSTMModel=keras.Sequential()
LSTMModel.add(keras.layers.LSTM(50, activation='relu', input_shape=(StepNumer, FeatureNumber)))
LSTMModel.add(keras.layers.Dense(1))
LSTMModel.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

## Fit model
LSTMModel.fit(TrainDF_X, TrainDF_Y, epochs=10, batch_size=48, verbose=2, shuffle=False)
#LSTMModel.fit(TrainDF_X, TrainDF_Y, epochs=10, batch_size=48, verbose=2, shuffle=False, validation_data=(TestDF_X, TestDF_Y))

##########
##########

#10%
BikeDF.shape[0]-1088

#New Split
TrainDF_X=BikeDF[0:9798]
TrainDF_Y=TrainDF_X['count']

TestDF_X=BikeDF[9798:BikeDF.shape[0]]
TestDF_Y=TestDF_X['count']

#Drop
TrainDF_X.drop(columns='count', inplace=True)
TestDF_X.drop(columns='count', inplace=True)

#Reshape
TrainDF_X=np.array(TrainDF_X).reshape((TrainDF_X.shape[0],1,TrainDF_X.shape[1]))
TestDF_X=np.array(TestDF_X).reshape((TestDF_X.shape[0],1,TestDF_X.shape[1]))
TrainDF_Y=np.array(TrainDF_Y).reshape((-1))
TestDF_Y=np.array(TestDF_Y).reshape((-1))

TrainDF_X.shape
TrainDF_Y.shape
TestDF_X.shape
TestDF_Y.shape

#####
## Define the model
LSTMModel=keras.Sequential()
LSTMModel.add(keras.layers.LSTM(50, input_shape=(TrainDF_X.shape[1], TrainDF_X.shape[2])))
LSTMModel.add(keras.layers.Dense(1))
LSTMModel.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

## Fit model
LSTMModel.fit(TrainDF_X, TrainDF_Y, epochs=10, batch_size=50, verbose=2, shuffle=False, validation_data=(TestDF_X, TestDF_Y))


########
# define model
model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(12, activation='relu'), input_shape=(TrainDF_X.shape[1], TrainDF_X.shape[2])))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# fit model
model.fit(TrainDF_X, TrainDF_Y, epochs=10, batch_size=50, verbose=2, shuffle=False, validation_data=(TestDF_X, TestDF_Y))

###########
###########
