rm(list=ls())

#Library
library(tidyverse)
library(keras)
library(RCurl)

#Use python
use_python("C:/Program Files/Python37/")

#Read data
KaggleBike="https://raw.githubusercontent.com/CrisBMoya/BikeSharingDataset/master/Data/train.csv"
TrainDF=read_delim(file=getURL(KaggleBike), delim=",")

#Subset
Y_Train=as.matrix(TrainDF[2001:nrow(TrainDF),c('count')])
X_Train=TrainDF[2001:nrow(TrainDF),c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]

#Data reshape
YVal=array_reshape(x=as.matrix(Y_Train), dim=c(8886, 1))
XVal=array_reshape(x=as.matrix(X_Train), dim=c(8886,2,3,1))

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
model %>% fit(XVal, YVal, epochs = 30)

