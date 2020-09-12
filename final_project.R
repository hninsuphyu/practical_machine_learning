# Load libraries

library(caret)
library(tidyverse)
library(data.table)
library(cowplot)
library(RColorBrewer)
library(scales)

# Function to display confusion5 matrices

conMatPlot = function(cfm, model) {
        plot = ggplot(data = as.data.frame(cfm$table),
               aes(x = Reference, y = Prediction)) +
                geom_tile(aes(fill = log(Freq)), color = "white") +
                scale_fill_gradient(low = "white", high = "steelblue") +
                geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
                theme(legend.position = "none") +
                ggtitle(paste(model, "Accuracy", percent_format()(cfm$overall[1]),
                              "Kappa", percent_format()(cfm$overall[2])))
        return(plot)
}

# Read data

training = 
        fread("/home/clay/Skole/JHU_Data_Science/Practical_Machine_Learning/Final_Project/pml-training.csv")

testing = 
        fread("/home/clay/Skole/JHU_Data_Science/Practical_Machine_Learning/Final_Project/pml-testing.csv")
```
## Exploratory Data Analysis

set.seed(2276)
newPart = createDataPartition(training$classe, p = 0.7, list = FALSE)
newTrain = training[newPart]
tempPart = training[-newPart]
splitTempPart = createDataPartition(tempPart$classe, p = 0.5, list = FALSE)
newTest = tempPart[splitTempPart]
valid = tempPart[-splitTempPart]

newTest = newTest %>% select(-V1, -user_name, -contains("timestamp"),
-contains("window"), -contains("kurtosis"), -contains("skewness"),
-contains("max"), -contains("min"), -contains("amplitude"), -contains("avg"),
-contains("stddev"), -contains("var"))
newTrain = newTrain %>% select(-V1, -user_name, -contains("timestamp"),
-contains("window"), -contains("kurtosis"), -contains("skewness"),
-contains("max"), -contains("min"), -contains("amplitude"), -contains("avg"),
-contains("stddev"), -contains("var"))
valid = valid %>% select(-V1, -user_name, -contains("timestamp"),
-contains("window"), -contains("kurtosis"), -contains("skewness"),
-contains("max"), -contains("min"), -contains("amplitude"), -contains("avg"),
-contains("stddev"), -contains("var"))

### Principle Component Analysis

preProc = preProcess(newTrain[,-53], method="pca", pcaComp=5)
trainPCA = predict(preProc, newTrain[,-53])
trainPCA = cbind(trainPCA, as.factor(newTrain$classe))
trainPCA = trainPCA %>% rename(classe = V2)
plot5 = ggplot(trainPCA, aes(x = PC1,  y = PC2, color = classe)) +
        geom_point() +
        scale_color_brewer(palette = "Paired")
plot5

### Highly Correlated Predictors

correlations = cor(newTrain[,-53], method = "pearson")
highCorCols = colnames(newTrain)[findCorrelation(correlations, cutoff = 0.75,
                                              verbose = FALSE)]
print(noquote("Highly correlated variables:"))
highCorCols

newTrain = newTrain %>% select(-accel_belt_z, -roll_belt, -accel_belt_y,
                         -accel_arm_y, -total_accel_belt, -accel_dumbbell_z,
                         -accel_belt_x, -pitch_belt, -magnet_dumbbell_x,
                         -accel_dumbbell_y, -magnet_dumbbell_y, -accel_arm_x,
                         -accel_dumbbell_x, accel_arm_z, -magnet_arm_y,
                         -magnet_belt_z, -accel_forearm_y, -gyros_forearm_y,
                         -gyros_dumbbell_x, -gyros_dumbbell_z,-gyros_arm_x)
newTest = newTest %>% select(-accel_belt_z, -roll_belt, -accel_belt_y,
                         -accel_arm_y, -total_accel_belt, -accel_dumbbell_z,
                         -accel_belt_x, -pitch_belt, -magnet_dumbbell_x,
                         -accel_dumbbell_y, -magnet_dumbbell_y, -accel_arm_x,
                         -accel_dumbbell_x, accel_arm_z, -magnet_arm_y,
                         -magnet_belt_z, -accel_forearm_y, -gyros_forearm_y,
                         -gyros_dumbbell_x, -gyros_dumbbell_z,-gyros_arm_x)
valid = valid %>% select(-accel_belt_z, -roll_belt, -accel_belt_y,
                         -accel_arm_y, -total_accel_belt, -accel_dumbbell_z,
                         -accel_belt_x, -pitch_belt, -magnet_dumbbell_x,
                         -accel_dumbbell_y, -magnet_dumbbell_y, -accel_arm_x,
                         -accel_dumbbell_x, accel_arm_z, -magnet_arm_y,
                         -magnet_belt_z, -accel_forearm_y, -gyros_forearm_y,
                         -gyros_dumbbell_x, -gyros_dumbbell_z,-gyros_arm_x)
newTest = newTest[1:2942]

### Cross-Validation

set.seed(2261)
tC = trainControl(method="cv", number=10, savePredictions = "all", 
                  classProbs=TRUE)
rfFit = train(classe ~ ., data = newTrain,  method="rf", trControl=tC)
gbmFit = train(classe ~ ., data = newTrain, method = "gbm", trControl = tC, 
               verbose = FALSE)
tbagFit = train(classe ~ ., data = newTrain, method = "treebag", trControl = tC)
nbFit = train(classe ~ ., data = newTrain, method = "nb", trControl = tC)
knnFit = train(classe ~ ., data = newTrain, method = "knn", trControl = tC)

results = resamples(list(GBM = gbmFit, KNN = knnFit, NB = nbFit, RF = rfFit,
                         TBAG = tbagFit))
summary(results)

modelCor(results)

rfPredict = predict(rfFit, newdata = valid)
gbmPredict = predict(gbmFit, newdata = valid)
tbagPredict = predict(tbagFit, newdata = valid)

predDF = data.frame(rfPredict, gbmPredict, tbagPredict, classe =
                            as.factor(valid$classe))
stackFit = train(classe ~ ., method="rf", data = predDF)

testRfPredict = predict(rfFit, newdata = newTest)
testGbmPredict = predict(gbmFit, newdata = newTest)
testTbagPredict = predict(tbagFit, newdata = newTest)

testPredDF = data.frame(testRfPredict, testGbmPredict, testTbagPredict,
                        classe = as.factor(newTest$classe))
stackPredict = predict(stackFit, testPredDF)

testRfConfusion = confusionMatrix(testRfPredict, as.factor(newTest$classe))
testGbmConfusion = confusionMatrix(testGbmPredict, as.factor(newTest$classe))
testTbagConfusion = confusionMatrix(testTbagPredict, as.factor(newTest$classe))
testStackConfusion = confusionMatrix(stackPredict, as.factor(newTest$classe))

con1 = conMatPlot(testRfConfusion, "RF: ")
con2 = conMatPlot(testGbmConfusion, "GBM: ")
con3 = conMatPlot(testTbagConfusion, "TBAG: ")
con4 = conMatPlot(testStackConfusion, "Stacked: ")

plot_grid(con1, con2, con3, con4)

## Estimating Out-of-Sample Error

lev = c("A", "B", "C", "D", "E")
dat1 = data.frame(obs = as.factor(newTest$classe), pred = testRfPredict)
multiClassSummary(dat1, lev = lev)

dat2 = data.frame(obs = as.factor(newTest$classe), pred = stackPredict)
multiClassSummary(dat2, lev= lev)

### Testing

rfFinalPredict = predict(rfFit, testing)
rfFinalPredict

