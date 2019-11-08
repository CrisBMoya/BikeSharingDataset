rm(list=ls())

#Library
library(tidyverse)
library(Amelia)
library(ggplot2)
library(plyr)
library(lubridate)

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

