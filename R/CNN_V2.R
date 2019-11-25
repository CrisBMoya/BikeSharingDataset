rm(list=ls())

#Library
library(tidyverse)
library(Amelia)
library(ggplot2)
library(plyr)
library(lubridate)
library(keras)
library(gtools)

#Apply the sliding window
ApplySliding=function(DF, Sliding){
  SlidingRange=tibble('from'=1:(nrow(DF)-(Sliding-1)), 'to'=Sliding:nrow(DF))
  
  #Separate by sliding window and add unique row to merge later
  Res=lapply(X=1:nrow(SlidingRange), FUN=function(x){
    Temp=DF[SlidingRange[x,]$from:SlidingRange[x,]$to,]
    Temp$UniqueRow=1:nrow(Temp)
    Temp$UniqueTable=paste0('Table_',x)
    return(Temp)
  })
  
  return(Res)
}

#
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

#Set seed
set.seed(101)

#Read data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')
TrainDF=TrainDF[order(TrainDF$datetime, decreasing=FALSE),]

#Apply sliding window of 3 hours
ResX=ApplySliding(DF=TrainDF, Sliding=12)

#Create ResY
ResY=lapply(X=1:length(ResX), FUN=function(x){
  ResX[[x]][,c('count')]
  })

#Edit lists
# ResX=lapply(X=1:length(ResX), FUN=function(x){
#    ResX[[x]][,!(colnames(ResX[[x]]) %in% c('count','datetime','UniqueTable','UniqueRow'))]
#   })
ResX=lapply(X=1:length(ResX), FUN=function(x){
   ResX[[x]][,c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]
  })


#Set up arrays
XVal=array(data=unlist(ResX), dim=c(12,6,length(ResX)))
YVal=t(matrix(data=unlist(ResY), nrow=12, ncol=length(ResY)))


#
YVal=(matrix(data=unlist(ResY), nrow=12, ncol=length(ResY)))
YVal[,1]
ResY[1]
#
dim(XVal)
dim(YVal)

#Define
n_timestep=dim(XVal)[1]
n_feature=dim(XVal)[2]
n_outputs=dim(YVal)[2]
n_timestep
n_feature
n_outputs


#Keras Model
model <- keras_model_sequential() 
model %>% 
  layer_conv_1d(filters=10, kernel_size=1, activation='relu', input_shape=c(n_timestep,n_feature)) %>%
  layer_conv_1d(filters=15, kernel_size=1, activation='relu') %>%
  layer_max_pooling_1d(pool_size=1) %>%
  layer_conv_1d(filters=20, kernel_size=1, activation='relu') %>%
  layer_max_pooling_1d(pool_size=1) %>%
  layer_flatten() %>%
  layer_dense(units=10, activation='relu') %>%
  layer_dense(units=n_outputs)

#Compile model
model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

#Train model
history <- model %>% fit(
  XVal, YVal, 
  epochs = 100, batch_size = 15, 
  validation_split = 0.2
)

plot(history)
