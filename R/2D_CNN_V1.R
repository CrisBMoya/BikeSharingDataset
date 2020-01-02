rm(list=ls())


#Library
library(tidyverse)
library(plyr)
library(keras)

#
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

#Set seed
set.seed(101)

#Read data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')

#Subset
X_Train=TrainDF[2000:nrow(TrainDF),c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]
Y_Train=as.matrix(TrainDF[2000:nrow(TrainDF),c('count')])
X_Test=TrainDF[1:1999,c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]
Y_Test=as.matrix(TrainDF[1:1999,c('count')])

#Create all possible combinations of dimensions
Combination=c(1,2,3,nrow(X_Train))
CombinationGrid=expand.grid(Combination, Combination,Combination,Combination)
CombinationGrid=CombinationGrid[which(apply(X=CombinationGrid, MARGIN=1, FUN=function(x){
  length(unique(as.numeric(x)))
})==4),]
FOrd=CombinationGrid
FOrd$Order='F'
COrd=CombinationGrid
COrd$Order='C'
CombinationGrid=rbind(FOrd, COrd)

#YVal
YVal=array_reshape(x=as.matrix(Y_Train), dim=c(8887, 1))

#For loop and try all combinations
Results=list()
for(i in 1:nrow(CombinationGrid)){

  #Reshape using all possible combinations
  XVal=array_reshape(x=as.matrix(X_Train), dim=CombinationGrid[i,1:4], order=CombinationGrid[i,]$Order)
  
  #Keras Model
  model=keras_model_sequential() 
  model %>% 
    layer_conv_2d(filters=10, kernel_size=c(2,2), padding='same', activation='relu') %>%
    layer_conv_2d(filters=15, kernel_size=c(2,2), padding='same', activation='relu') %>%
    layer_conv_2d(filters=20, kernel_size=c(3,3), padding='same') %>%
    layer_max_pooling_2d(pool_size=c(2,2), strides=1) %>%
    layer_flatten() %>%
    layer_dense(units=30, activation='relu') %>%
    layer_dense(units=20, activation='relu') %>%
    layer_dense(units=10, activation='relu') %>%
    layer_dense(units=1)
  
  #Compile model
  model %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(),
    metrics = c('accuracy'))
  
  #Train model
  Hist=tryCatch({
    model %>% fit(XVal, YVal, epochs = 100)
  },error=function(e){
    Hist=list('metrics'=list('loss'=NA, 'acc'=NA))
  })
  
  Results[[i]]=list('Loss'=Hist$metrics$loss[length(Hist$metrics$loss)], 'Acc'=Hist$metrics$acc[length(Hist$metrics$acc)])
  
}
Results=ldply(.data=Results, .fun=as.data.frame)
Results=cbind(CombinationGrid, Results)
write.table(x=Results, file='C:/Users/Tobal/Google Drive/Github/RKerasResult.txt', quote=FALSE, sep='\t', row.names=FALSE)

sessionInfo()

