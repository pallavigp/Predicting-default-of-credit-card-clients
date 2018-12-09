#--------------------------------------Begin--------------------------------------#

# load libraries
library(dplyr)
library(ggplot2)
library(caret)
library(plyr)
library(C50)
library(kernlab)
library(reshape)
library(MLmetrics)
library(caretEnsemble)
library(naivebayes)
library(neuralnet)
library(readxl)
library(VIM)
library(splitstackshape)
library(randomForest)
library(rpart)
library(gbm)

# load traning dataset
bankdata <- read_excel("Desktop/620 - Data Mining/Project/default of credit card clients.xlsx")

# summary of dataset
summary(bankdata)
# view top rows of dataset
head(bankdata)
# check class of dataset
class(bankdata)
# variables in the dataset
names(bankdata)
#variable types of dataset
str(bankdata)

#rename last column as class
colnames(bankdata)[25] <- "class"

#convert into dataframe
bankdata <- as.data.frame(bankdata[])
View(bankdata)

# check for missing values
summary(aggr(bankdata, plot =  TRUE))

#---------------Preparing Data for Classification----------------#
#do stratified sampling to take a subset of the data
bankdata <- stratified(bankdata, group = "class", size = 0.15, 
                       select = NULL, replace = FALSE,
                       bothSets = FALSE)
View(bankdata)

# write the sampled data to a csv file
write.csv(bankdata,"Desktop/620 - Data Mining/Project/sampled data.csv")

# Convert nominal variables to factor but first copy in a new dataframe
bankdata1 <- bankdata
str(bankdata1)

bankdata1[,c(3:5,7:12,25)] <- lapply(bankdata1[,c(3:5,7:12,25)], as.factor)
str(bankdata1)

#convert positive class 1 as yes and negative class as no in the dataset
bankdata1$class <- revalue(bankdata1$class, c("1"="Yes"))
bankdata1$class <- revalue(bankdata1$class, c("0"="No"))

View(bankdata1)

# class summary for dataset
summary(bankdata1$class)

#drop the customerid column
bankdata1$ID <- NULL

# plot the class distribution
qplot(class, data=bankdata1, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 

qplot(bankdata1$LIMIT_BAL, bins = 5, margins = TRUE)
qplot(bankdata1$SEX)
boxplot(bankdata1$LIMIT_BAL ~ bankdata1$class, bankdata1)

#create partition in data for training and testing
tdi <- createDataPartition(bankdata1$class, p=0.67, list = FALSE)

# Create Training Data as subset 
trainingdata <- bankdata1[tdi,]
View(trainingdata)

# write the training data to a csv file
write.csv(trainingdata,"Desktop/620 - Data Mining/Project/training data.csv")

# Everything else not in training is test data. Note the - (minus)sign
testdata <- bankdata1[-tdi,]
View(testdata)

# write the test data to a csv file
write.csv(testdata,"Desktop/620 - Data Mining/Project/test data.csv")

# relevel training data to get "yes" as positive class
trainingdata$class <- relevel(trainingdata$class, "Yes")
summary(trainingdata$class)

# relevel test data to get "yes" as positive class
testdata$class <- relevel(testdata$class, "Yes")
summary(testdata$class)

#-------------------Building Classifiers---------------------#

# we will train and evaluate the model using 10 fold cross validation
trainingparameter <- trainControl(method = "repeatedcv", number = 10, sampling = "up")

#-------------C5.0 algorithm for Decision Tree Classifier-------------#
DecTree <- train(trainingdata[,-24], trainingdata$class, 
             method = "C5.0", 
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter,
             na.action = na.omit)
DecTree

# make predictions on test set & see results
DecTreePred <- predict(DecTree, testdata, na.action = na.pass)
DecTreePred

# create confusion matrix & see results
ConfMatDT <-confusionMatrix(DecTreePred, testdata$class, mode = "everything")
ConfMatDT
t(ConfMatDT$table)

#-----------------------Naïve Bayes Classifer-----------------------#
NaiveBayes <- train(trainingdata[,-24], trainingdata$class, 
             method = "nb", 
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter,
             na.action = na.omit)
NaiveBayes

# make predictions on test set & see results
NaiveBayesPred <-predict(NaiveBayes, testdata, na.action = na.pass)
NaiveBayesPred
# create confusion matrix & see results
ConfMatNB <-confusionMatrix(NaiveBayesPred, testdata$class, mode = "everything")
ConfMatNB
t(ConfMatNB$table)

#---------------------Random Forest Classifier----------------------#
RF <- train(trainingdata[,-24], trainingdata$class,
             method = "rf",
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter)
RF

# make predictions on test set & see results
RFPred <-predict(RF, testdata)
RFPred
# create confusion matrix & see results
ConfMatRF <-confusionMatrix(RFPred, testdata$class, mode = "everything")
ConfMatRF
t(ConfMatRF$table)

#---------------------------GBM Classifier---------------------------#
GBM <- train(trainingdata[,-24], trainingdata$class,
            method = "gbm",
            preProcess = c("nzv", "corr"),
            trControl= trainingparameter)
GBM

# make predictions on test set & see results
GBMPred <-predict(GBM, testdata)
GBMPred

# create confusion matrix & see results
ConfMatGBM <-confusionMatrix(GBMPred, testdata$class, mode = "everything")
ConfMatGBM
t(ConfMatGBM$table)

#-----------Comparing Models through Weighted F Measure---------------#

# Decision Tree model F measure #score get worse for recall
F1_Score(testdata$class, DecTreePred)
FBeta_Score(testdata$class, DecTreePred, beta = 0.5)
FBeta_Score(testdata$class, DecTreePred, beta = 0.1)
FBeta_Score(testdata$class, DecTreePred, beta = 1.1)
FBeta_Score(testdata$class, DecTreePred, beta = 1.5)

# Naive Bayes model F measure #score gets better for recall
F1_Score(testdata$class, NaiveBayesPred)
FBeta_Score(testdata$class, NaiveBayesPred, beta = 0.5)
FBeta_Score(testdata$class, NaiveBayesPred, beta = 0.1)
FBeta_Score(testdata$class, NaiveBayesPred, beta = 1.1)
FBeta_Score(testdata$class, NaiveBayesPred, beta = 1.5)

# Random Forest model F measure #score get worse for recall
F1_Score(testdata$class, RFPred)
FBeta_Score(testdata$class, RFPred, beta = 0.5)
FBeta_Score(testdata$class, RFPred, beta = 0.1)
FBeta_Score(testdata$class, RFPred, beta = 1.1)
FBeta_Score(testdata$class, RFPred, beta = 1.5)

# GBM model F measure, #score get worse for recall
F1_Score(testdata$class, GBMPred)
FBeta_Score(testdata$class, GBMPred, beta = 0.5)
FBeta_Score(testdata$class, GBMPred, beta = 0.1)
FBeta_Score(testdata$class, GBMPred, beta = 1.1)
FBeta_Score(testdata$class, GBMPred, beta = 1.5)

# header for classifier list
classifier <- c("C5.0","nb","rf", "gbm")

# since we want to give more weightage to recall the Beta should be greater than 1
# we consider the Beta = 1.1 since it gives higher value of FMeasure between 1.1 and 1.5
FMeasure <- c(FBeta_Score(testdata$class, DecTreePred, beta = 1.1),
            FBeta_Score(testdata$class, NaiveBayesPred, beta = 1.1),
            FBeta_Score(testdata$class, RFPred, beta = 1.1),
            FBeta_Score(testdata$class, GBMPred, beta = 1.1))

# create dataframe to view classifiers with respective FMeasures for Beta = 1.1
FMeasureDF <- data.frame(classifier,FMeasure) 
FMeasureDF

#--------------Variable Importance----------------#

vimpdt <- plot(varImp(DecTree, scale = F), main = "Decision Tree")
vimpnb <- plot(varImp(NaiveBayes, scale = F), main = "Naive Bayes")
vimprf <- plot(varImp(RF, scale = F), main = "Random Forest")
vimpgbm <- plot(varImp(GBM, scale = F), main = "GBM")
grid.arrange(vimpdt, vimpnb, vimprf, vimpgbm)

#------------Finding Model Correlation and Building Ensemble Models--------------#

# create training control
EnsTrainParam <- trainControl(method="cv", number=5, summaryFunction = twoClassSummary, 
                              savePredictions=TRUE, classProbs=TRUE)

# create train of models for finding correlation
AllModels <- caretList(class ~., data=trainingdata,
                       methodList=c("C5.0", "nb", "rf", "gbm"),
                       trControl = EnsTrainParam)

AllModels
ModResults <- resamples(AllModels)
ModResults$values
ModResults$metrics
summary(ModResults)
dotplot(ModResults)

# find model correlation
ModCorr <-modelCor(ModResults)
ModCorr

# Plot model correlation
splom(ModResults)
ModResults

#----------------Stacked Models-------------------#

#create stacked models for all models
EnsStackModel <- caretStack(AllModels, method = "C5.0", metric = "Sens",
                            trControl = trainControl(number = 5, summaryFunction = twoClassSummary, classProbs = TRUE))
print(EnsStackModel)

#predict & create confusion matrix
EnsStackPred <- predict(EnsStackModel, testdata)
ConfMatStack <-confusionMatrix(EnsStackPred, testdata$class, 
                               mode="everything")
ConfMatStack
t(ConfMatStack$table)

#create stacked models for Random Forest and Gradient boosting as they are the least correlated
LowCorr <- caretList(class ~., data=trainingdata,
                     methodList=c("rf", "gbm"),
                     trControl = EnsTrainParam)

EnsStackModel2 <- caretStack(LowCorr, method = "C5.0", metric="Sens", 
                             trControl = trainControl(number = 5, 
                                                      summaryFunction = twoClassSummary,
                                                      classProbs = TRUE))
print(EnsStackModel2)

#predict & create confusion matrix
EnsStackPred2 <-predict(EnsStackModel2, testdata, na.action = na.omit)
ConfMatStack2 <-confusionMatrix(EnsStackPred2, testdata$class, 
                                mode="everything")
ConfMatStack2
t(ConfMatStack2$table)

#create stacked models for all models with up sampling
EnsStackModelUp <- caretStack(AllModels, method = "C5.0", metric="Sens", 
                             trControl = trainControl(number = 5, 
                                                      summaryFunction = twoClassSummary,
                                                      classProbs = TRUE, sampling = "up"))
print(EnsStackModelUp)

#predict & create confusion matrix
EnsStackPredUp <-predict(EnsStackModelUp, testdata, na.action = na.omit)
ConfMatStackUp <-confusionMatrix(EnsStackPredUp, testdata$class, 
                                mode="everything")
ConfMatStackUp
t(ConfMatStackUp$table)

#-------------------Ensemble Models------------------------#

#create ensemble model with sensitivity as optimization metric for all models
EnsModelSens <- caretEnsemble(AllModels, metric = "Sens",
                              trControl = trainControl(number = 5, 
                                                       summaryFunction = twoClassSummary,
                                                       classProbs = TRUE))

summary(EnsModelSens)

#predict & create confusion matrix
EnsPredSens <-predict(EnsModelSens, testdata, na.action = na.omit)
ConfMatEMS <-confusionMatrix(EnsPredSens, testdata$class, mode="everything")
ConfMatEMS
t(ConfMatEMS$table)

#create ensemble model with sensitivity as optimization metric for least correlated models
EnsModelSens2 <- caretEnsemble(LowCorr, metric = "Sens",
                              trControl = trainControl(number = 5, 
                                                       summaryFunction = twoClassSummary,
                                                       classProbs = TRUE))

summary(EnsModelSens2)

#predict & create confusion matrix
EnsPredSens2 <-predict(EnsModelSens2, testdata, na.action = na.omit)
ConfMatEMS2 <-confusionMatrix(EnsPredSens2, testdata$class, mode="everything")
ConfMatEMS2
t(ConfMatEMS2$table)

#improve with up sampling the ensemble model with sensitivity as optimization metric for all models
EnsModelSensUp <- caretEnsemble(AllModels, metric = "Sens",
                              trControl = trainControl(number = 5, 
                                                       summaryFunction = twoClassSummary,
                                                       classProbs = TRUE, sampling = "up"))

summary(EnsModelSensUp)

#predict & create confusion matrix
EnsPredSensUp <-predict(EnsModelSensUp, testdata, na.action = na.omit)
ConfMatEMSup <-confusionMatrix(EnsPredSensUp, testdata$class, mode="everything")
ConfMatEMSup
t(ConfMatEMSup$table)

#----------------------------Cost Analysis-----------------------------#

# create cost matrix
CostMatrix <- cbind(c(0,10), c(1,0))
t(CostMatrix)

# cost analysis for C5.0 algorithm
CostDecTree <- C5.0(class ~., trials = 5, data=trainingdata, cost=CostMatrix)
CostDecTree
summary(CostDecTree)

CostPred <- predict(CostDecTree, testdata)
CostPred

CostConfMat <-confusionMatrix(CostPred, testdata$class, mode = "everything")
CostConfMat
t(CostConfMat$table)

# multiply cost and confusion matrix
TotCostDT <- CostMatrix*CostConfMat$table
sum(TotCostDT)

# cost analysis for Naive Bayes classifier
CostNaiveBayes <- naive_bayes(class ~., trials = 5, data=trainingdata, cost=CostMatrix)
CostNaiveBayes
summary(CostNaiveBayes)

CostPredNB <- predict(CostNaiveBayes, testdata)
CostPredNB

CostConfMatNB <-confusionMatrix(CostPredNB, testdata$class, mode = "everything")
CostConfMatNB
t(CostConfMatNB$table)

#multiply cost and confusion matrix
TotCostNB <- CostMatrix*CostConfMatNB$table
sum(TotCostNB)

# cost analysis for Random Forest classifier
CostRndmFrst <- randomForest(class ~., trials = 5, data=trainingdata, cost=CostMatrix)
CostRndmFrst
summary(CostRndmFrst)

CostPredRF <- predict(CostRndmFrst, testdata)
CostPredRF

CostConfMatRF <-confusionMatrix(CostPredRF, testdata$class, mode = "everything")
CostConfMatRF
t(CostConfMatRF$table)

#multiply cost and confusion matrix
TotCostRF <- CostMatrix*CostConfMatRF$table
sum(TotCostRF)

#----------------------------End-----------------------------#