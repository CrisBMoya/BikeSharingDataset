rm(list=ls())


#Library
library(tidyverse)
library(plyr)
library(keras)
library(reticulate)
#
use_python("C:/Program Files/Python37/")
np=import("numpy", convert=FALSE)
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


#XVal
XVal=np$array(object=X_Train, dtype=np$float32)
XVal=np$reshape(XVal, c(8887L,2L,3L,1L))
#YVal
YVal=np$array(object=Y_Train)

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

model %>% fit(XVal, YVal, epochs = 100)

Results=ldply(.data=Results, .fun=as.data.frame)
Results=cbind(CombinationGrid, Results)
write.table(x=Results, file='C:/Users/Tobal/Google Drive/Github/RKerasResult.txt', quote=FALSE, sep='\t', row.names=FALSE)

sessionInfo()

