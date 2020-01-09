## Clean environment
rm(list=ls())

#Load libraries
library(tidyverse)
library(caret)
library(lubridate)
library(dummies)
library(h2o)

## Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

## Load Data
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
TrainDF$Hour=lubridate::hour(x=TrainDF$datetime)
TrainDF$Month=lubridate::month(x=TrainDF$datetime)
TrainDF$Year=lubridate::year(x=TrainDF$datetime)

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

# AutoML
## Start H2O
h2o.init(nthreads=-1, ip='localhost', port=8079)

## Test H2O
Dev_XH2O=as.h2o(Dev_X)

## Train H2O
Train_XH2O=as.h2o(cbind(Train_X, Train_Y))

## Run Autm ML
AutoML=h2o.automl(x=setdiff(colnames(Train_X), 'count'), 
  y='count',
  training_frame=Train_XH2O,
  max_models=10,
  max_runtime_secs_per_model=60,
  seed=1)

## AutoML Leaderboard
BestRes=AutoML@leaderboard

## Optionally edd extra model information to the leaderboard
BestRes=h2o.get_leaderboard(AutoML, extra_columns = "ALL")

## Print all rows (instead of default 6 rows)
print(BestRes, n = nrow(BestRes))

## Predict
Predict=h2o.predict(AutoML@leader, Dev_XH2O)

## Check
cor(as.data.frame(x=Predict)[,1], Dev_Y$count)
RMSE(pred=as.data.frame(x=Predict)[,1], obs=Dev_Y$count)

## NewData
# Same treatment for Test
## Extract data from datetime
TestDF$Hour=lubridate::hour(x=TestDF$datetime)
TestDF$Month=lubridate::month(x=TestDF$datetime)
TestDF$Year=lubridate::year(x=TestDF$datetime)

## Create dummy variables and fuse with original DF
Test_X=as_tibble(dummy.data.frame(data=as.data.frame(TestDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Test_X=bind_cols(TestDF[,c('season','holiday','workingday','weather','atemp','humidity','Hour','Month','Year')], Test_X)
Test_XH2O=as.h2o(Test_X)

## Preduict
TestPredict=h2o.predict(object=AutoML@leader, Test_XH2O)
TestPredict=as.data.frame(x=TestPredict)

## Shutdown
h2o.shutdown(prompt=FALSE)

## Format results deliver
TestPredict=TestPredict[,1]
TestPredict[TestPredict<0]=0

#Save
SubV=data.frame('datetime'=TestDF$datetime, 'count'=TestPredict)
write.table(x=SubV, file='R/Sub_AutoML_V1.csv', sep=',', row.names=FALSE, quote=FALSE)

#Submit
RE=TRUE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c bike-sharing-demand -f R/Sub_AutoML_V1.csv -m "Submission from API - AutoML"')
}
