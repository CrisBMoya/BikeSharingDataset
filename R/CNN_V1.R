rm(list=ls())

#Library
library(tidyverse)
library(Amelia)
library(ggplot2)
library(plyr)
library(lubridate)
library(keras)
library(gtools)
#
use_python("C:/Program Files/Python37/")

#Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/BikeSharingDataset', x=getwd()))

#Set seed
set.seed(101)

#Read data
TrainDF=read_delim(file=unz(description='Data/bike-sharing-demand.zip', filename='train.csv'), delim=',')

#Complete data
TrainDF$year=year(TrainDF$datetime)
TrainDF$month=months.POSIXt(TrainDF$datetime)
TrainDF$day=day(TrainDF$datetime)
TrainDF$Hours=hour(TrainDF$datetime)

#Date without year
TrainDF$DateNonYear=paste0(TrainDF$month,'-',TrainDF$day,'-',TrainDF$Hours)

#Drop non equal dates
TrainDF=TrainDF[!(TrainDF$DateNonYear %in% names(which(table(TrainDF$DateNonYear)==min(table(TrainDF$DateNonYear))))),]

#Subset
# X_Train=TrainDF[2000:nrow(TrainDF),c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]
# Y_Train=as.matrix(TrainDF[2000:nrow(TrainDF),c('count')])
# 
# X_Test=TrainDF[1:1999,c('temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered')]
# Y_Test=as.matrix(TrainDF[1:1999,c('count')])


#Reorganize data
#Task: 
# 1 - Separate 2011 data, sort by date, ascending (older first). Take 90% of data as Trainning data.
# 2 - Take first 3 hours of data, create a single dataframe with it.
# 3 - Take the second 3 hours, starting from hour 2 (from 1 to 3, then from 2 to 4, then from 3 to 5, and so on.)
# The above line means to apply a SLIDING WINDOW OF 3 HOURS on de data.
# 4 - Do the same for year 2012.
# 5 - Mix 2012 dataframes with 2012, so 2012 works as a replicate. This can be done sorting by a unique row number.
# The row number mentioned above can be inserted during a for loop when passing the sliding window

#Take 2011 data only
DF11=TrainDF[TrainDF$year=='2011',]
DF12=TrainDF[TrainDF$year=='2012',]

#Sort increasing by date
DF11=DF11[order(DF11$datetime, decreasing=FALSE),]
DF12=DF12[order(DF12$datetime, decreasing=FALSE),]

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

SlDF11=ApplySliding(DF=DF11, Sliding=3)
SlDF12=ApplySliding(DF=DF12, Sliding=3)

#Mashup dataframes
DF=as_tibble(bind_rows(ldply(SlDF11, as.data.frame), ldply(SlDF12, as.data.frame)))
DF=DF[mixedorder(DF$UniqueTable, decreasing=FALSE),]

#Check, days seems to be too different
#ggplot(data=DF, aes(x=DateNonYear, y=count, color=year)) + geom_point()



#Apply the sliding window again, but this time is double since we put together the two years
ResX=list()
ResY=list()
for(i in 1:length(unique(DF$UniqueTable))){
  TabName=unique(DF$UniqueTable)
  
  TEMP=DF[DF$UniqueTable==TabName[i],]
  TEMP=TEMP[order(TEMP$UniqueRow),]
  ResY[[i]]=TEMP[,c('count')]
  TEMP[,c('count','datetime','UniqueTable','UniqueRow','DateNonYear','month','year','day','Hours')]=NULL
  ResX[[i]]=TEMP
}
XVal=array(data=unlist(ResX), dim=c(length(ResX),3,ncol(ResX[[1]])))
YVal=t(matrix(data=unlist(ResY), nrow=3, ncol=length(ResY)))

#Define
n_timestep=dim(XVal)[2]
n_feature=dim(XVal)[3]
n_outputs=dim(YVal)[2]

#Keras Model
model <- keras_model_sequential() 
model %>% 
  layer_conv_1d(filters=32, kernel_size=1, activation='relu', input_shape=c(n_timestep,n_feature)) %>%
  layer_conv_1d(filters=32, kernel_size=1, activation='relu') %>%
  layer_max_pooling_1d(pool_size=1) %>%
  layer_conv_1d(filters=16, kernel_size=1, activation='relu') %>%
  layer_max_pooling_1d(pool_size=1) %>%
  layer_flatten() %>%
  layer_dense(units=100, activation='relu') %>%
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
  epochs = 1000, batch_size = 16, 
  validation_split = 0.2
)

plot(history)
