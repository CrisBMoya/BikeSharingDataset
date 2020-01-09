#Clean environment
rm(list=ls())

#Load libraries
library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(lubridate)
library(dummies)

#Use python
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))


#Load Data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')
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
Train_X=bind_cols(TrainDF[,c('holiday','workingday','atemp','humidity')], Train_X)

## Create Y DF
Train_Y=TrainDF[,c('count')]

## Reshape
Train_X=array_reshape(x=as.matrix(Train_X), dim=c(nrow(Train_X), 1, 50))
Train_Y=array_reshape(x=as.matrix(Train_Y), dim=c(nrow(Train_Y), 1))

## Model
LSTMModel=keras_model_sequential()
LSTMModel %>%
  layer_lstm(units=50, activation='relu', input_shape=c(1, 50)) %>%
  layer_dense(units=1)
  
LSTMModel %>% compile(loss='mse', optimizer=optimizer_adam(), metrics=c('accuracy'))

Results=LSTMModel %>% fit(Train_X, Train_Y, epochs = 100, batch_size=12, shuffle=FALSE)


# Same treatment for Test
## Extract data from datetime
TestDF$Hour=hour(x=TestDF$datetime)
TestDF$Month=month(x=TestDF$datetime)
TestDF$Year=year(x=TestDF$datetime)

## Create dummy variables and fuse with original DF
Test_X=as_tibble(dummy.data.frame(data=as.data.frame(TestDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Test_X=bind_cols(TestDF[,c('holiday','workingday','atemp','humidity')], Test_X)

## Reshape
Test_X=array_reshape(x=as.matrix(Test_X), dim=c(nrow(Test_X), 1, 50))

## Predict data
Predictions=LSTMModel %>% predict(x=Test_X)
Predictions[Predictions<0]=0

1-Results$metrics$acc

