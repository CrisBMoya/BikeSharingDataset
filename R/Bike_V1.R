rm(list=ls())

#Library
library(tidyverse)
library(Amelia)
library(ggplot2)
library(plyr)
library(lubridate)
library(keras)

#
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

#Set seed
set.seed(101)

#Read data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')

#Check NA with Missmap function from Amelia package
missmap(TrainDF)

#Take year only
TrainDF$year=year(TrainDF$datetime)
TrainDF$month=months.POSIXt(TrainDF$datetime)
TrainDF$day=day(TrainDF$datetime)

#Some Exploratory Data Analysis EDA
ggplot(data=TrainDF, aes(x=datetime, y=count, colour=temp)) + geom_point() +
  scale_colour_distiller(palette='YlOrRd', direction=1) +
  facet_grid(rows=year~.)

dim(TrainDF)


#Coded TimeStamp
TrainDF$LabelRow=1:nrow(TrainDF)

#Separate DF
X_Train=as.matrix(TrainDF[,c('LabelRow','temp','atemp')])
Y_Train=as.matrix(TrainDF[,c('count')])
#Expand X
#X_Train=k_expand_dims(x=X_Train, axis=-1)


#Model
model <- keras_model_sequential() 
# model %>% 
#   layer_dense(units = 256, activation = 'relu', input_shape = c(ncol(X_Train))) %>% 
#   layer_dropout(rate = 0.4) %>% 
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dropout(rate = 0.3) %>%
#   layer_dense(units = 978, activation = 'softmax')
dim(X_Train)
model %>%
  layer_reshape(target_shape=c(nrow(X_Train),ncol(X_Train)),  input_shape=c(prod(dim(X_Train)),1)) %>%
  layer_conv_1d(kernel_size=10, filters=2, activation='relu', input_shape=c(prod(dim(X_Train)),1)) %>%
  #layer_conv_1d(kernel_size=10, filters=100, activation='relu') %>%
  #layer_max_pooling_1d(pool_size=2) %>%
  #layer_conv_1d(kernel_size=10, filters=160, activation='relu') %>%
  #layer_conv_1d(kernel_size=10, filters=160, activation='relu') %>%
  #layer_global_average_pooling_1d() %>%
  #layer_dropout(rate=0.5) %>%
  layer_dense(units=978, activation='softmax')
model

#Compile model
model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#Train model


###
X_Train
X_Train=k_expand_dims(x=X_Train, axis=1)

history <- model %>% fit(
  X_Train, Y_Train, 
  epochs = 30, batch_size = 128, steps_per_epoch=2,
  validation_split = 0.2
)
dim(X_Train)

################
X_Train

#
model <- keras_model_sequential() 

model %>%
  layer_conv_1d(filters=1, kernel_size=5, input_shape=c(10886,3))
model

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  X_Train, Y_Train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

#############
#Table: each count
#Rows: each time
#Cols: each feature

Temp=sample(x=0, size=prod(c(nrow(TrainDF),1,3)), replace=TRUE)
dim(Temp)=c(1,3,nrow(TrainDF))

for(i in 1:nrow(TrainDF)){
  
  Temp[,1,i]=TrainDF$datetime[i]
  Temp[,2,i]=TrainDF$temp[i]
  Temp[,3,i]=TrainDF$atemp[i]
  
  
}

model <- keras_model_sequential() 
model %>%
  layer_conv_1d(filters=512, kernel_size=1, input_shape=c(1,3)) %>%
  layer_activation(activation='relu') %>%
  layer_flatten() %>%
  layer_dense(units=2048, activation='relu') %>%
  layer_dense(units=1024, activation='relu') %>%
  layer_dense(units=nrow(TrainDF)) %>%
  layer_activation(activation='softmax')
model

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  Temp, Y_Train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

#################
#################

Temp=sample(x=0, size=prod(c(nrow(TrainDF),1,3)), replace=TRUE)
dim(Temp)=c(nrow(TrainDF),1,3)
dim(Temp)
for(i in 1:nrow(TrainDF)){
  
  Temp[i,,1]=TrainDF$datetime[i]
  Temp[i,,2]=TrainDF$temp[i]
  Temp[i,,3]=TrainDF$atemp[i]
  
  
}

model <- keras_model_sequential() 
model %>%
  layer_conv_1d(filters=512, kernel_size=1, input_shape=c(1,3)) %>%
  layer_activation(activation='relu') %>%
  #layer_flatten() %>%
  layer_max_pooling_1d(1) %>%
  layer_dense(units=2048, activation='relu') %>%
  layer_dense(units=1024, activation='relu') %>%
  layer_dense(units=nrow(TrainDF)) %>%
  layer_activation(activation='softmax')
model

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  Temp, Y_Train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

#################
#################
TrainDF
Temp=sample(x=0, size=prod(c(nrow(TrainDF),1,3)), replace=TRUE)
dim(Temp)=c(nrow(TrainDF),1,3)

for(i in 1:nrow(TrainDF)){
  
  Temp[i,,1]=TrainDF$datetime[i]
  Temp[i,,2]=TrainDF$temp[i]
  Temp[i,,3]=TrainDF$atemp[i]
  
  
}

model <- keras_model_sequential() 
model %>%
  layer_reshape(c(2,3,1), input_shape=prod(c(2,3,1)))
  layer_conv_1d(filters=100, kernel_size=1, activation='relu', input_shape=c(1,3)) %>%
  layer_conv_1d(filters=100, kernel_size=1, activation='relu') %>%
  layer_max_pooling_1d(1) %>%
  layer_conv_1d(filters=160, kernel_size=1, activation='relu') %>%
  layer_conv_1d(filters=160, kernel_size=1, activation='relu') %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units=nrow(TrainDF)) %>%
  layer_activation(activation='softmax')

model

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  Temp, Y_Train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)


##################
##################
Temp=sample(x=rnorm(n=prod(5,5,3,2), mean=5, sd=5), size=prod(5,5,3,2), replace=TRUE)

#nrow as number of tables as units
#Two dataframes per unit
#three rows per dataframe
#Two table per unit
dim(Temp)=c(5,5,3,2)

Temp[5,,,]
Temp[,,]
Temp[,,,1]


Temp=sample(x=rnorm(n=prod(5,5,3,2), mean=5, sd=5), size=prod(5,5,3,2), replace=TRUE)
x <- array(rnorm(n=prod(12), mean=5, sd=5), dim = list(3, 1, 2,3))
x

length(c(9, 9, 7, 9, 6, 5, 4, 6, 2, 1, 3, 2))
