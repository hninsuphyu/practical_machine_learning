---
title: "Final Project: Practical Machine Learning"
author: "Clay Glad"
date: "10 September 2020"
output:
    html_document: 
      theme: cerulean
      keep_md: yes
---


## The Question

It is common to use wearable devices to measure how much of an activity someone  
engages in. The Human Activity Recognition Project of Groupware@LES has  
expanded these metrics to measure *how well* certain activities are performed.  
In particular, Velloso *et al.* [Velloso, E.; Bulling, A.; Gellersen, H.;  
Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting   
Exercises. Proceedings of 4th International Conference in Cooperation with  
SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.] have  
defined the quality of  execution in certain weightlifting exercises and used  
on-body sensing and ambient sensing in order to provide feedback to users on  
their technique. They classified the results into four categories. Category A  
indicates correct execution of the exercise, while categories B-E are those of  
specific errors in execution.
<br><br>
Their dataset is robust: nearly 20,000 observations of 160 variables. We propose  
to develop a machine learning model that when given observations on some  
subset of these variables can predict into which of the five categories any  
instance of execution falls.


## Exploratory Data Analysis

(Please see final_project.R for the full code.)

Training and testing data sets are given by the HAR project, but the test set  
is only 20 observations. We'll use this to test our final model, but we'll need  
to split the training set in order to build the model. We create a data part-  
ition from the training data set and call these newTrain and newTest in order to  
avoid confusion. Further, since we may want to ensemble our models, we create a  
third partition (valid) for validation. We give newTrain 70% of the training  
data and 15% each to newTest and valid.
<br><br>
A first look at the data shows that most of the variables are statistical  
rather than observational. We subset out the statistical data from all three  
partitions so that we are working only with the observational data.



### Principle Component Analysis
We've reduced the number of variables from 160 to 53. We now try principle com-  
ponent analysis as a way of getting an accurate but more parsimonious data set.
<br>  
Unfortunately, the results are not helpful.

![](index_files/figure-html/chunk3-1.png)<!-- -->

### Highly Correlated Predictors

PCA provides no separation between the categories. It may be that we will need  
all 53 observational variables, but if at all possible we'd like to reduce  
computational complexity before building our models. We check correlations  
between the variables. 


```
## [1] Highly correlated variables:
```

```
##  [1] "accel_belt_z"      "roll_belt"         "accel_arm_y"      
##  [4] "accel_belt_y"      "yaw_belt"          "accel_dumbbell_z" 
##  [7] "accel_belt_x"      "pitch_belt"        "magnet_dumbbell_x"
## [10] "accel_dumbbell_y"  "magnet_dumbbell_y" "accel_dumbbell_x" 
## [13] "accel_arm_x"       "accel_arm_z"       "magnet_arm_y"     
## [16] "magnet_belt_z"     "accel_forearm_y"   "gyros_arm_x"
```

And in fact 18 of the variables are correlated at .75 or greater. We remove  
these from our data sets and now work with a somewhat more manageable set of 32  
predictors.



## Model Building

### Cross-Validation
We've chosen to use 10-fold cross validation as a compromise between bias and  
variance. We could also have used 5-fold, but we have some concern with over-  
fitting and hope to avoid this with more robust cross-validation.

### Choice of Algorithm
While we have no particular reason to choose one algorithm over all the others,  
some are more typically used for classification, and we'll fit several models  
for purposes of comparison. We'll look at (alphabetically) Gradient Boosting,   
K-nearest Neighbors, Naive Bayes, Random Forest, and Treebagging.  

Summary of results:

```
## 
## Call:
## summary.resamples(object = results)
## 
## Models: GBM, KNN, NB, RF, TBAG 
## Number of resamples: 10 
## 
## Accuracy 
##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## GBM  0.9367273 0.9426331 0.9453549 0.9453307 0.9475601 0.9534207    0
## KNN  0.8426803 0.8452451 0.8496540 0.8496746 0.8539694 0.8566230    0
## NB   0.7134545 0.7214130 0.7264112 0.7271624 0.7322670 0.7458121    0
## RF   0.9868996 0.9890770 0.9901675 0.9906095 0.9923581 0.9941818    0
## TBAG 0.9708667 0.9739907 0.9785303 0.9772145 0.9794469 0.9818049    0
## 
## Kappa 
##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## GBM  0.9199297 0.9274104 0.9308721 0.9308270 0.9336543 0.9410824    0
## KNN  0.8010360 0.8041664 0.8096313 0.8098887 0.8155216 0.8189651    0
## NB   0.6404224 0.6497592 0.6565581 0.6574421 0.6638395 0.6809260    0
## RF   0.9834255 0.9861816 0.9875587 0.9881198 0.9903321 0.9926411    0
## TBAG 0.9631463 0.9671150 0.9728356 0.9711774 0.9740116 0.9769785    0
```

K-nearest Neighbors and Naive Bayes are poor performers; we won't consider them  
further.  

The accuracy for the random forest model is > .99. It's hard to imagine that we  
will gain much by stacking, but let's look at the correlations between the  
models.


```
##              GBM         KNN          NB         RF       TBAG
## GBM   1.00000000 -0.29678570 -0.08874747 -0.3635370 -0.2393675
## KNN  -0.29678570  1.00000000  0.04375038  0.1507642  0.5239197
## NB   -0.08874747  0.04375038  1.00000000 -0.4308880  0.1603090
## RF   -0.36353701  0.15076424 -0.43088799  1.0000000  0.2012209
## TBAG -0.23936753  0.52391973  0.16030897  0.2012209  1.0000000
```

The correlations between the three best-performing algorithms is quite low.  
Stacking may provide even greater accuracy. We combine the models and create a  
new one using random forest, first creating a level-one data set from the  
single-model predictions on the valid data set in order to train the stacked  
model. Once this is done we can examine confusion matrices for all four models.

![](index_files/figure-html/chunk9-1.png)<!-- -->

## Estimating Out-of-Sample Error

Using caret's multiclass summary function we can estimate out-of-sample error and  
compare other measures:

Random Forest:


```
##               Accuracy                  Kappa                Mean_F1 
##              0.9918423              0.9896784              0.9912620 
##       Mean_Sensitivity       Mean_Specificity    Mean_Pos_Pred_Value 
##              0.9908047              0.9979372              0.9917297 
##    Mean_Neg_Pred_Value         Mean_Precision            Mean_Recall 
##              0.9980158              0.9917297              0.9908047 
##    Mean_Detection_Rate Mean_Balanced_Accuracy 
##              0.1983685              0.9943710
```

Stacked Model:

```
##               Accuracy                  Kappa                Mean_F1 
##              0.9908226              0.9883914              0.9899423 
##       Mean_Sensitivity       Mean_Specificity    Mean_Pos_Pred_Value 
##              0.9899337              0.9977419              0.9899555 
##    Mean_Neg_Pred_Value         Mean_Precision            Mean_Recall 
##              0.9977477              0.9899555              0.9899337 
##    Mean_Detection_Rate Mean_Balanced_Accuracy 
##              0.1981645              0.9938378
```

With so little difference between the two models, we choose the simpler random  
forest as our final model.

### Testing

Finally, we apply our model against the testing data set of 20 observations.


```r
rfFinalPredict = predict(rfFit, testing)
rfFinalPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
