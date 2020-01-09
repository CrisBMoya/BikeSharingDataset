%config Completer.use_jedi = False
import tensorflow.keras as keras
import numpy as np
import zipfile as zipped
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import boxcox, inv_boxcox
from datetime import datetime
import xgboost as xg

#Set Working Directory
os.chdir(re.sub(pattern='Python', repl='', string=os.getcwd()))

#Load Train DF
train_df=pd.read_csv(zipped.ZipFile("Data/bike-sharing-demand.zip").open('train.csv'))

train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

new_df=train_df

new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)

final_df=new_df.drop(['datetime','temp','windspeed','casual','registered','day'], axis=1)

weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)
                     
final_df=final_df.join(weather_df)
final_df=final_df.join(year_df)
final_df=final_df.join(month_df)                     
final_df=final_df.join(hour_df)
final_df=final_df.join(season_df)
                     
final_df.columns
X=final_df.iloc[:,final_df.columns!='count'].values
Y=final_df.iloc[:,6].values

xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)
xgr.fit(X,Y)

#TEST DATA
new_df=pd.read_csv(zipped.ZipFile("Data/bike-sharing-demand.zip").open('test.csv'))
new_df['datetime']=new_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
DateSave=new_df['datetime']

new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)


new_df=new_df.drop(['datetime','temp','windspeed','day'], axis=1)

#adding dummy varibles to categorical variables
weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
yr_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)


new_df=new_df.join(weather_df)
new_df=new_df.join(yr_df)
new_df=new_df.join(month_df)                     
new_df=new_df.join(hour_df)
new_df=new_df.join(season_df)
                     

X_test=new_df.iloc[:,:].values


y_output=xgr.predict(X_test)
y_output[y_output<0]=0

TEMP=pd.DataFrame(list(zip(DateSave, y_output)))
TEMP.columns=['datetime','count']
TEMP.to_clipboard(index=False, sep=',')

#############
#############

TrainX=final_df.iloc[:,final_df.columns!='count'].values
TrainX=TrainX.reshape((-1,1,50))
TrainX.shape

TrainY=final_df.iloc[:,6].values
TrainY.shape

## Define the model
LSTMModel=keras.Sequential()
LSTMModel.add(keras.layers.LSTM(50, activation='relu', input_shape=(1, 50)))
LSTMModel.add(keras.layers.Dense(1))
LSTMModel.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

## Fit model
LSTMModel.fit(TrainX, TrainY, epochs=100, batch_size=12, verbose=2, shuffle=False)


TestX=X_test.reshape((-1,1,50))

#THIS MODEL, AS IS, HAS AN SCORE 0F 0.86
LSTMOutput=LSTMModel.predict(TestX)
LSTMOutput[LSTMOutput<0]=0

#
np.corrcoef(x=y_output, y=LSTMOutput.reshape((-1)))


TEMP=pd.DataFrame(list(zip(DateSave, LSTMOutput.reshape((-1)))))
TEMP.columns=['datetime','count']
TEMP.to_clipboard(index=False, sep=',')




## Define the model
LSTMModel=keras.Sequential()
LSTMModel.add(keras.layers.LSTM(50, activation='relu', input_shape=(1, 50)))
LSTMModel.add(keras.layers.Dense(1))
LSTMModel.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

#Re do with prediction
TestY=LSTMModel.predict(TestX).reshape((-1))
TestY.shape
TestX.shape
TrainX.shape
TrainY.shape
TEMP=LSTMModel.fit(TrainX, TrainY, epochs=10, batch_size=12, verbose=2, shuffle=False, validation_data=(TestX, TestY))
TEMP.history['acc'][0]*100

