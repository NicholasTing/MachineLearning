#Data Preprocessing

#Importing dataset
dataset = read.csv('Data.csv')
# dataset = dataset[,2:3]

# Splitting the training and test set
install.packages('caTools')
library(caTools)
set.seed(123)

# TRUE if it is in the training set
# FALSE if it is in the test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

