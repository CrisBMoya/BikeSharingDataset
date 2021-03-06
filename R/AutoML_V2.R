## Clean environment
rm(list=ls())

#Load libraries
library(tidyverse)
library(caret)
library(lubridate)
library(dummies)
library(h2o)
library(ggplot2)
library(dplyr)
library(plyr)
library(gtools)

# Purpose:
## To use AutoML in as a model recommendation system and to extract importance of variables.
## To check wether AutoML produces better results without any EDA or feature engineering.

## Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

## Load Data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')
TrainDF$datetime=as.character.Date(TrainDF$datetime)
TrainDF$casual=NULL
TrainDF$registered=NULL
TestDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='test.csv'), delim=',')
TestDF$datetime=as.character.Date(TestDF$datetime)


## Create train df
Train_X=TrainDF

## Split
Split=createDataPartition(y=Train_X$season, times=1, p=0.1)
Split=Split$Resample1

## Test
Dev_Y=Train_X[Split,c('count')]
Dev_X=Train_X[Split,colnames(Train_X)[!colnames(Train_X) %in% c('count')]]

## Train
Train_Y=Train_X[-Split,c('count')]
Train_X=Train_X[-Split,colnames(Train_X)[!colnames(Train_X) %in% c('count')]]

ReDo=TRUE
if(ReDo){
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
    max_models=30,
    max_runtime_secs_per_model=180,
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
  
  ## Inspect model
  unlist(lapply(AutoML@leader@allparameters$base_models, FUN=function(x){
    x$name
  }))
  
  ## Extract all Ids
  ModelId=as.data.frame(AutoML@leaderboard)$model_id
  
  ## Extract models
  ModelList=lapply(X=ModelId, FUN=function(x){
    h2o.getModel(model_id=x)
  })
  
  ## Extract important variables
  ModelVarImp=lapply(X=ModelList, FUN=function(x){
    suppressWarnings(h2o.varimp(object=x))
  })
  names(ModelVarImp)=ModelId
  
  ## Turn list into dataframe, sort them and create factors for ggplot to respect positions
  ModelVarImpDF=as_tibble(ldply(.data=ModelVarImp, .fun=as.data.frame))
  ModelVarImpDF=ModelVarImpDF[order(ModelVarImpDF$scaled_importance, decreasing=FALSE),]
  ModelVarImpDF$scaled_importance=factor(x=ModelVarImpDF$scaled_importance, levels=unique(ModelVarImpDF$scaled_importance))
  
  ## Plot important variables
  P1=ggplot(data=ModelVarImpDF, aes(x=variable, y=scaled_importance, fill=.id)) + geom_col(position='dodge2') +
    theme_minimal() + theme(legend.position='none', axis.text.y=element_blank())
  pdf(file='R/AutoML_V2_Implot.pdf', width=8, height=6)
  print(P1)
  dev.off()
  
  ## Save best non ensemble model
  h2o.saveModel(object=h2o.getModel(model_id=ModelId[3]), path='R/AutoMLModel_V2.h2o', force=TRUE)
  
  ## Shut down H2O
  h2o.shutdown(prompt=FALSE)
}

# Load model
## Start H2O
h2o.init(nthreads=-1, ip='localhost', port=8079)

## Load Model
AutoML=h2o.loadModel(path='R/AutoMLModel_V2.h2o/GBM_2_AutoML_20200114_130405')

## Test H2O
Dev_XH2O=as.h2o(Dev_X)

## Predict
Predict=h2o.predict(AutoML, Dev_XH2O)

## Check
cor(as.data.frame(x=Predict)[,1], Dev_Y$count)
RMSE(pred=as.data.frame(x=Predict)[,1], obs=Dev_Y$count)

# Improve model

## Learn about hyperparameters
AutoML@parameters

## Check important variables
h2o.varimp(object=AutoML)

## Discrete quantization
ggplot(data=TrainDF, aes(y=count, x=datetime, colour=atemp)) + geom_point()

TrainDF$QuantAtemp=quantcut(x=TrainDF$atemp, q=4)
ggplot(data=TrainDF, aes(y=count, x=datetime, colour=QuantAtemp)) + geom_point()

TrainDF$QuantTemp=quantcut(x=TrainDF$temp, q=4)
##

## Load Data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')
TrainDF$casual=NULL
TrainDF$registered=NULL

## Extract data from datetime
TrainDF$Hour=lubridate::hour(x=TrainDF$datetime)
TrainDF$Month=lubridate::month(x=TrainDF$datetime)
TrainDF$Year=lubridate::year(x=TrainDF$datetime)

## Create dummy variables and fuse with original DF
Train_X=as_tibble(dummy.data.frame(data=as.data.frame(TrainDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Train_X=bind_cols(TrainDF, Train_X)

## Split
Split=createDataPartition(y=Train_X$season, times=1, p=0.1)
Split=Split$Resample1

## Test
Dev_Y=Train_X[Split,c('count')]
Dev_X=Train_X[Split,colnames(Train_X)[!colnames(Train_X) %in% c('count','datetime')]]

## Train
Train_Y=Train_X[-Split,c('count')]
Train_X=Train_X[-Split,colnames(Train_X)[!colnames(Train_X) %in% c('count','datetime')]]

## Test H2O
Dev_XH2O=as.h2o(Dev_X)

## Train H2O
Train_XH2O=as.h2o(cbind(Train_X, Train_Y))

## Train model
AutoML@parameters

NewModel=h2o.gbm(nfolds=5, keep_cross_validation_models=FALSE, keep_cross_validation_predictions=FALSE, score_tree_interval=5, fold_assignment='Modulo', ntrees=53, max_depth=7, stopping_metric='deviance', stopping_tolerance=0.01010359, seed=4, distribution='gaussian', sample_rate=0.8, col_sample_rate=0.8, x=setdiff(colnames(Train_X), 'count'), y='count', training_frame=Train_XH2O)

## Predict
Predict=h2o.predict(NewModel, Dev_XH2O)

## Check
cor(as.data.frame(x=Predict)[,1], Dev_Y$count)
RMSE(pred=as.data.frame(x=Predict)[,1], obs=Dev_Y$count)

# New Data
TestDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='test.csv'), delim=',')

## Extract data from datetime
TestDF$Hour=lubridate::hour(x=TestDF$datetime)
TestDF$Month=lubridate::month(x=TestDF$datetime)
TestDF$Year=lubridate::year(x=TestDF$datetime)

Test_X=as_tibble(dummy.data.frame(data=as.data.frame(TestDF[,c('weather','Year','Month','Hour','season')]), dummy.class='ALL'))
Test_X=bind_cols(TestDF, Test_X)
Test_X$datetime=NULL

Test_XH2O=as.h2o(Test_X)

## Predict
Predict=h2o.predict(NewModel, Test_XH2O)
Predict=as.data.frame(Predict)[,1]
Predict[Predict<0]=0

#Save
SubV=data.frame('datetime'=TestDF$datetime, 'count'=Predict)
write.table(x=SubV, file='R/Sub_AutoML_V2.csv', sep=',', row.names=FALSE, quote=FALSE)

#Submit
RE=TRUE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c bike-sharing-demand -f R/Sub_AutoML_V2.csv -m "Submission from API - AutoML"')
}


## Shut down H2O
h2o.shutdown(prompt=FALSE)
