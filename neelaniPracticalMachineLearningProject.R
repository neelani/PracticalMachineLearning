############################################################################################################ 
## Project     : Coursera Practical Machine Learning Project
## Author      : Neelani
## Created Date: 2017-05-21
## Program     : neelaniPracticalMachineLearning.R 
########################################################################################################## 

### Load the required libraries 

library(foreach)
library(lattice)
library(ggplot2)
library(MASS)
library(rpart)
library(parallel)
library(iterators)
library(doParallel) 
library(caret)
library(randomForest)

### Set the environment for parallelism

## Clear the Global Environment
rm(list=ls())
wd <- getwd()

# Calculate the number of cores to run the process in parallel mode
no_cores <- detectCores() - 1

### Download the required files

downloadFile <- "pml_training.csv"
if (!file.exists(downloadFile)) {
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(url, destfile = downloadFile)
}
trainData <- read.csv(downloadFile, na.strings = c("NA","#DIV/0!",""))

downloadFile <- "pml_testing.csv"
if (!file.exists(downloadFile)) {
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(url, destfile = downloadFile)
}
testData <- read.csv(downloadFile, na.strings = c("NA","#DIV/0!",""))


### Cleanup the training data by removing NA, near zero and no-value-add attributes

trainData <- trainData[, -(1:7)]
testData  <- testData[, -(1:7)]

## Remove near zero value columns                      
nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
trainData <- trainData[, -nzv$nzv == FALSE]

# remove variables that are almost always NA
nas <- sapply(trainData, function(x) mean(is.na(x))) > 0.95
trainData <- trainData[, nas==FALSE]


## Align the columns of test data similar to train data set
cnames <- colnames(trainData)
testData <- testData[, cnames[-53]]

### Split the training set data for cross validation

# Divide the training data into a two sets for trainig and cross validation
set.seed(12345) 
splitData <- createDataPartition(trainData$classe, p=0.7,list=FALSE)
trainDataSet1 <- as.data.frame(trainData[splitData,])
trainDataSet2 <- as.data.frame(trainData[-splitData,])

# Set the trainControl parameters for the 5 fold cross validation
control <- trainControl(method = "cv", number = 5)

### Build Linear Discriminant Analysis (LDA) Model

# Create the Linear Discriminant Analysis (LDA) in Paralel mode

# Initiate cluster
cl <- makeCluster(no_cores)
registerDoParallel(cl)

ldaModelFit <- train(classe ~ ., 
                     data = trainDataSet1,
                     method = "lda",
                     trcontrol = control)
stopCluster(cl)


# Save the LDA Model 
save(ldaModelFit, file = "LDA_modelFit.RData")

# Apply the LDA to training Data Set 1 and validate it using confusion matrix
predictSet1 <- predict(ldaModelFit, trainDataSet1)
confusionMatrix(predictSet1, trainDataSet1$classe)

### Build Desision Tree Model

# Create the Decision Tree Model in Paralel mode

# Initiate cluster
cl <- makeCluster(no_cores)
registerDoParallel(cl)

dtModelFit <- train(classe ~ ., 
                    data = trainDataSet1,
                    method = "rpart")
stopCluster(cl)


# Save the Decision Tree Model 
save(dtModelFit, file = "DecisionTree_modelFit.RData")

# Apply the Decision Tree Model to training Data Set 1 and validate it using confusion matrix
predictSet1 <- predict(dtModelFit, trainDataSet1)
confusionMatrix(predictSet1, trainDataSet1$classe)

### Build Random Forest Model

# Create the Random Forest Model in Paralel mode

modelFile <- "RandonForest_modelFit.RData"

if (!file.exists(modelFile)) {
  # Initiate cluster
  cl <- makeCluster(no_cores)
  registerDoParallel(cl)
  
  rfModelFit <- train(classe ~ ., 
                      data = trainDataSet1,
                      method = "rf",
                      trcontrol = control)
  stopCluster(cl)
  
  # Save the Random Forest Model 
  save(rfModelFit, file = "RandonForest_modelFit.RData")
} else {
  rfModelFit <-get(load(paste(wd, modelFile, sep = "/")))
}

# Apply the Random Forest Model to training Data Set 1 and validate it using confusion matrix
predictSet1 <- predict(rfModelFit, trainDataSet1)
confusionMatrix(predictSet1, trainDataSet1$classe)

# Apply the Random Forest Model to training Data Set 2 and validate it using confusion matrix
predictSet2 <- predict(rfModelFit, trainDataSet2)
confusionMatrix(predictSet2, trainDataSet2$classe)

### Conclusion

## Test the model with the given test data 
predictTestData <- predict(rfModelFit, testData)
predictTestData

# create function to write predictions to files
write_files <- function(x) {
  n <- length(x)
  for(i in 1:n) {
    filename <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
  }
}

# create prediction files to submit
write_files(predictTestData)
