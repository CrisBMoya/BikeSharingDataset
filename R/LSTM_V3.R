#Clean environment
rm(list=ls())

#Load libraries
library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(lubridate)
library(dummies)

#Apply the sliding window
ApplySliding=function(DF, Sliding){
  SlidingRange=tibble('from'=1:(nrow(DF)-(Sliding-1)), 'to'=Sliding:nrow(DF))
  
  #Separate by sliding window and add unique row to merge later
  Res=lapply(X=1:nrow(SlidingRange), FUN=function(x){
    Temp=DF[SlidingRange[x,]$from:SlidingRange[x,]$to,]
    #Temp$UniqueRow=1:nrow(Temp)
    #Temp$UniqueTable=paste0('Table_',x)
    return(Temp)
  })
  
  return(Res)
}

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

## Apply Sliding Window
Train_X=ApplySliding(DF=Train_X, Sliding=12)
Train_Y=lapply(X=Train_X, FUN=function(x){
  x$count[nrow(x)]
  })

Train_X=lapply(X=Train_X, FUN=function(x){
  x$count=NULL
  x$datetime=NULL
  x
  })

# ## Split
# SplitDFSeason=createDataPartition(y=Train_X$season, times=1, p=0.1)
# SplitDFYear=createDataPartition(y=Train_X$Year, times=1, p=0.1)
# Split=unique(unlist(SplitDFSeason$Resample1),unlist(SplitDFYear$Resample1))
# 
# ## Test
# Dev_Y=Train_X[Split,c('count')]
# Dev_X=Train_X[Split,colnames(Train_X)[!colnames(Train_X) %in% c('datetime','temp','windspeed','count')]]
# 
# ## Train
# Train_Y=Train_X[-Split,c('count')]
# Train_X=Train_X[-Split,colnames(Train_X)[!colnames(Train_X) %in% c('datetime','temp','windspeed','count')]]

# ## Check dimension
# dim(Dev_X)
# dim(Train_X)
# dim(Dev_Y)
# dim(Train_Y)

## Reshape
Train_XReshape=array_reshape(x=unlist(Train_X), dim=c(length(Train_X), nrow(Train_X[[1]]), ncol(Train_X[[1]])))
Train_YReshape=array_reshape(x=unlist(Train_Y), dim=c(length(Train_Y), 1))

# Dev_XReshape=array_reshape(x=as.matrix(Dev_X), dim=c(nrow(Dev_X), 1, ncol(Dev_X)))
# Dev_YReshape=array_reshape(x=as.matrix(Dev_Y), dim=c(nrow(Dev_Y), 1))

## Check dimension
dim(Dev_XReshape)
dim(Dev_YReshape)

dim(Train_XReshape)
dim(Train_YReshape)

## Model
LSTMModel=keras_model_sequential()
LSTMModel %>%
  layer_lstm(units=50, activation='relu', input_shape=c(nrow(Train_X[[1]]), ncol(Train_X[[1]]))) %>%
  layer_dense(units=1)
  
LSTMModel %>% compile(loss="msle", optimizer=optimizer_adam())

Results=LSTMModel %>% fit(Train_XReshape, Train_YReshape, epochs = 100, batch_size=12, shuffle=FALSE)

## Evaluate
PredictTest=LSTMModel %>% predict(x=Train_XReshape)
cor(x=Dev_Y[,1], y=PredictTest[,1])
RMSE(pred=PredictTest[,1], Dev_YReshape[,1])

# Same treatment for Test
## Extract data from datetime
TestDF$Hour=hour(x=TestDF$datetime)
TestDF$Month=month(x=TestDF$datetime)
TestDF$Year=year(x=TestDF$datetime)

## Create dummy variables and fuse with original DF
Test_X=as_tibble(dummy.data.frame(data=as.data.frame(TestDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Test_X=bind_cols(TestDF[,c('season','holiday','workingday','weather','temp','atemp','humidity','windspeed','Hour','Month','Year')], Test_X)

##
Test_X=ApplySliding(DF=Test_X, Sliding=12)
colnames(Train_X[[1]])==colnames(Test_X[[1]])
## Reshape
Test_XReshape=array_reshape(x=unlist(Test_X), dim=c(length(Test_X), nrow(Test_X[[1]]), ncol(Test_X[[1]])))
dim(Test_XReshape)

## Predict data
LSTMPredictions=LSTMModel %>% predict(x=Test_XReshape)
LSTMPredictions=LSTMPredictions*-1
LSTMPredictions[LSTMPredictions<0]=0

#Save
SubV=data.frame('datetime'=TestDF$datetime, 'count'=c(rep(x=0, 11),LSTMPredictions[,1]))
write.table(x=SubV, file='R/Sub_LSTM_V4.csv', sep=',', row.names=FALSE, quote=FALSE)

#Submit
RE=TRUE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c bike-sharing-demand -f R/Sub_LSTM_V4.csv -m "Submission from API - LSTM With Sliding"')
}
