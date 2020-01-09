#Clean environment
rm(list=ls())

#Load libraries
library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(lubridate)
library(dummies)
library(xgboost)

#Use python
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))


#Load Data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')
TrainDF$casual=NULL
TrainDF$registered=NULL
TestDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='test.csv'), delim=',')

# Objective: to subset Train DF in such a way that it matches TestDF data.
# Context: while playing with LSTM with this data, a simple EDA (creating dummy variables) resulted in 
# an score on Kaggle of 0.86, while the accuracy during training was no higher than 0.095.
# This means that the LSTM network is doing validations on something useless while correclty training the net.
# So what if we test it on real data.


## Extract data from datetime
TrainDF$Hour=hour(x=TrainDF$datetime)
TrainDF$Month=month(x=TrainDF$datetime)
TrainDF$Year=year(x=TrainDF$datetime)

## Create dummy variables and fuse with original DF
Train_X=as_tibble(dummy.data.frame(data=as.data.frame(TrainDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Train_X=bind_cols(TrainDF, Train_X)

## Split
SplitDFSeason=createDataPartition(y=Train_X$season, times=1, p=0.1)
SplitDFYear=createDataPartition(y=Train_X$Year, times=1, p=0.1)
Split=unique(unlist(SplitDFSeason$Resample1),unlist(SplitDFYear$Resample1))

## Test
Dev_Y=Train_X[Split,c('count')]
Dev_X=Train_X[Split,colnames(Train_X)[!colnames(Train_X) %in% c('datetime','temp','windspeed','count')]]

## Train
Train_Y=Train_X[-Split,c('count')]
Train_X=Train_X[-Split,colnames(Train_X)[!colnames(Train_X) %in% c('datetime','temp','windspeed','count')]]

## Check dimension
dim(Dev_X)
dim(Train_X)
dim(Dev_Y)
dim(Train_Y)

## Reshape
Train_XReshape=array_reshape(x=as.matrix(Train_X), dim=c(nrow(Train_X), 1, ncol(Train_X)))
Train_YReshape=array_reshape(x=as.matrix(Train_Y), dim=c(nrow(Train_Y), 1))

Dev_XReshape=array_reshape(x=as.matrix(Dev_X), dim=c(nrow(Dev_X), 1, ncol(Dev_X)))
Dev_YReshape=array_reshape(x=as.matrix(Dev_Y), dim=c(nrow(Dev_Y), 1))

## Check dimension
dim(Dev_XReshape)
dim(Dev_YReshape)

dim(Train_XReshape)
dim(Train_YReshape)

## Model
LSTMModel=keras_model_sequential()
LSTMModel %>%
  layer_lstm(units=50, activation='relu', input_shape=c(1, ncol(Train_X))) %>%
  layer_dense(units=1)
  
LSTMModel %>% compile(loss="msle", optimizer=optimizer_adam())

Results=LSTMModel %>% fit(Train_XReshape, Train_YReshape, epochs = 100, batch_size=12, shuffle=FALSE)

## Evaluate
PredictTest=LSTMModel %>% predict(x=Dev_XReshape)
cor(x=Dev_Y[,1], y=PredictTest[,1])
RMSE(pred=PredictTest[,1], Dev_YReshape[,1])

# Same treatment for Test
## Extract data from datetime
TestDF$Hour=hour(x=TestDF$datetime)
TestDF$Month=month(x=TestDF$datetime)
TestDF$Year=year(x=TestDF$datetime)

## Create dummy variables and fuse with original DF
Test_X=as_tibble(dummy.data.frame(data=as.data.frame(TestDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Test_X=bind_cols(TestDF[,c('season','holiday','workingday','weather','atemp','humidity','Hour','Month','Year')], Test_X)

## Reshape
Test_XReshape=array_reshape(x=as.matrix(Test_X), dim=c(nrow(Test_X), 1, ncol(Test_X)))
dim(Test_XReshape)

## Predict data
LSTMPredictions=LSTMModel %>% predict(x=Test_XReshape)
LSTMPredictions[LSTMPredictions<0]=0

#Save
SubV=data.frame('datetime'=TestDF$datetime, 'count'=LSTMPredictions[,1])
write.table(x=SubV, file='R/Sub_LSTM_V3.csv', sep=',', row.names=FALSE, quote=FALSE)

#Submit
RE=FALSE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c bike-sharing-demand -f R/Sub_LSTM_V3.csv -m "Submission from API - LSTM Minimal EDA"')
}

# XGBoost

## Model
XGBoostTrain=xgboost(data=as.matrix(Train_X), label=as.matrix(Train_Y),
  params=list('max_depth'=8, 'min_child_weight'=6, 'gamma'=0.4, 'colsample_bytree'=0.6,'subsample'=0.6), 
  nrounds=1000)

## Evaluate
Pred=predict(object=XGBoostTrain, newdata=as.matrix(Dev_X))

##
cor(x=Dev_Y[,1], y=Pred)
RMSE(pred=as.numeric(Pred), obs=as.numeric(Dev_Y$count))

## Precit
XGBoostPredictions=predict(object=XGBoostTrain, newdata=as.matrix(Test_X))
XGBoostPredictions[XGBoostPredictions<0]=0

TEMP=read.table(file='clipboard', header=TRUE, sep=',')
cor(x=TEMP$count, y=XGBoostPredictions)
RMSE(pred=as.numeric(XGBoostPredictions), obs=as.numeric(TEMP$count))

colnames(Train_X)

#Save
SubV=data.frame('datetime'=TestDF$datetime, 'count'=PredReal)
write.table(x=SubV, file='R/Sub_XGBoost_V7.csv', sep=',', row.names=FALSE, quote=FALSE)

#Submit
RE=FALSE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c bike-sharing-demand -f R/Sub_XGBoost_V7.csv -m "From API - XGBoost Minimal EDA"')
}

# Compare XGBoost against LSTM -- So far, XGBoost gives better predictions
cor(x=XGBoostPredictions, y=LSTMPredictions)
RMSE(pred=LSTMPredictions, obs=XGBoostPredictions)

# Compare LSTM against best Prediction
cor(x=TEMP$count, y=LSTMPredictions)
RMSE(pred=LSTMPredictions, obs=TEMP$count)
