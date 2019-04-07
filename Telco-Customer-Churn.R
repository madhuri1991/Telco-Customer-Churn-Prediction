title: "Churn Prediction - 

Logistic Regression, Decision Tree and Random Forest"
output:
  html_document: default
pdf_document: default
word_document: default
---
  
  ## Data Overview
  The data was downloaded from IBM Sample Data Sets for customer retention programs. The goal of this project is to predict behaviors of churn or not churn to help retain customers. 
Each row represents a customer, each column contains a customer's attribute.

Customers who left within the last month - the column is called Churn  
Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies   
Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges  
Demographic info about customers - gender, age range, and if they have partners and dependents  

## Library
```{r echo=TRUE, warning = FALSE, message=FALSE}
library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(caret)
library(rms)
library(MASS)
library(e1071)
library(ROCR)
library(gplots)
library(pROC)
library(rpart)
library(randomForest)
library(ggpubr)
```

## Explore Data
telco = read.csv("Telco-Customer-Churn.csv")
```{r echo= FALSE, warning = FALSE, message=FALSE}
telco = read.csv("Telco-Customer-Churn.csv")
telco <- data.frame(telco)
```
```{r echo= TRUE, warning = FALSE, message=FALSE}
str(telco)
summary(telco)
```

#### Observations with Missing Values
Based on the summary, there are 11 missing values in the TotalCharges column, which account for only 0.16% of the total 
number of observations. So I remove those 11 rows with missing values. 

```{r echo= FALSE, warning = FALSE, message=FALSE}
telco <- telco[complete.cases(telco),] 
```

#### Continuous Variables
For continuous variables, let's check for distributions.

```{r echo= FALSE, warning = FALSE, message=FALSE}
ggplot(data = telco, aes(MonthlyCharges, color = Churn))+
geom_freqpoly(binwidth = 5, size = 1)
```

The number of current customers with MonthlyCharges below $25 is extremly high. For the customers with Monthlycharges greater than $30, 
the distributions are similar between who churned and who did not churn.

```{r echo= FALSE, warning = FALSE, message=FALSE}
ggplot(data = telco, aes(TotalCharges, color = Churn))+
geom_freqpoly(binwidth = 200, size = 1)
```

The distribution of TotalCharges is highly positive skew for all customers no matter whether they churned or not. 

```{r echo= FALSE, warning = FALSE, message=FALSE}
ggplot(data = telco, aes(tenure, colour = Churn))+
geom_freqpoly(binwidth = 5, size = 1)
```

The distributions for tenure are very different between customers who churned and who didn't churn. For customers
who churned, the distribution is positve skew, which means customers who churned are more likely to cancel the service 
in the first couple of months. For current customers who didn't churn, there are two spikes. The second spike is much more
drastic than the first one, which means a large group of current customers have been using the service more than 5 years.

No obvious outliers for 3 numeric variables. Then let's check for correlations.

```{r echo= TRUE, warning = FALSE, message=FALSE}
telco %>%
  dplyr::select (TotalCharges, MonthlyCharges, tenure) %>%
  cor() %>%
  corrplot.mixed(upper = "circle", tl.col = "black", number.cex = 0.7)
```

The plot shows high correlations between Totalcharges & tenure and between TotalCharges & MonthlyCharges. 
Pay attention to these variables while training models later. Multicollinearity does not 
reduce the predictive power or reliability of the model as a whole, at least within the sample data set. 
But it affects calculations regarding individual predictors.

The tenure represents time period in months. To better find patterns with time, I change it to a factor with 5 
levels, with each level represents a bin of tenure in years. 

```{r echo= FALSE, warning = FALSE, message=FALSE}
telco %>%
  mutate(tenure_year = case_when(tenure <= 12 ~ "0-1 year",
                                 tenure > 12 & tenure <= 24 ~ "1-2 years",
                                 tenure > 24 & tenure <= 36 ~ "2-3 years",
                                 tenure > 36 & tenure <= 48 ~ "3-4 years",
                                 tenure > 48 & tenure <= 60 ~ "4-5 years",
                                 tenure > 60 & tenure <= 72 ~ "5-6 years")) -> telco
telco$tenure <-NULL
table(telco$tenure_year)
```

#### Categorical Variables
I found that there is a column called Phone Service. And in the MultipleLines, some rows have the value of "No Phone Service".
Are they related?

```{r echo= TRUE, warning = FALSE, message=FALSE}
table(telco[, c("PhoneService","MultipleLines")])
```

When the value of Phone Service is "No", the value of Multiplelines shows "No Phone Service." The"No Phone Service" 
value in the Multiplelines column actually does not have any predicting power. 

The same problem appeared between Internet Service and Online Security, OnlineBackup, DeviceProtection, TechSupport, 
StreamingTV and StreamingMovies. When the value of Internet Service is "No", the values of the following 6 columns show "No Internet Service."

```{r echo= FALSE, warning = FALSE, message=FALSE}
table(telco[, c("InternetService", "OnlineSecurity")])
table(telco[, c("InternetService", "OnlineBackup")])
table(telco[, c("InternetService", "DeviceProtection")])
table(telco[, c("InternetService", "TechSupport")])
table(telco[, c("InternetService", "StreamingTV")])
table(telco[, c("InternetService", "StreamingMovies")])
```

I will address this problem later in the data preparation. Now I will check the distributions of churn by the levels of yes or no 
for the above 7 variables. I will remove the rows with "No phone service" and "No internet service" in the plot.  

```{r echo= FALSE, warning = FALSE, message=FALSE}
telco %>%
  mutate(SeniorCitizen = ifelse(SeniorCitizen == 0, "No", "Yes")) -> categorical

categorical %>%
  dplyr::select(gender:Dependents, PhoneService:PaymentMethod, Churn) -> categorical 

categorical %>%
  dplyr::select(MultipleLines, OnlineSecurity:StreamingMovies, Churn) %>%
  filter(MultipleLines != "No phone service" &
           OnlineSecurity != "No internet service") -> c2

gather(c2, columns, value, -Churn) -> c3

ggplot(c3)+
  geom_bar(aes(x = value, fill = Churn), position = "fill", stat = "count")+
  facet_wrap(~columns)+ 
  xlab("Attributes")
```

The customers who subscribe the service of DeviceProtection, OnlineBackup, OnlineSecurity and TechSupport have lower 
churn rate compared to the customers who don't. However, the churn rates do not have big difference between customers 
who have the service of MultipleLines, StreamingMovies and StreamingTV or not. 

```{r echo= FALSE, warning = FALSE, message=FALSE}
categorical %>%
dplyr::select(Contract:Churn) -> c4

ggplot(c4) +
geom_bar(aes(x = Contract, fill = Churn), position = "fill", stat = "count", 
show.legend = F) -> p7

ggplot(c4) +
geom_bar(aes(x = PaperlessBilling, fill = Churn), position = "fill", stat = "count", 
show.legend = T) -> p8

ggplot(c4) +
geom_bar(aes(x = PaymentMethod, fill = Churn), position = "fill", stat = "count", 
show.legend = F) +
scale_x_discrete(labels = c("Bank transfer", "Credit card", "Electronic check", "Mail check"))+
theme(axis.text= element_text(size=7)) -> p9

ggarrange(p7,p8,p9, ncol = 2, nrow = 2)
```

The customers who sign longer contract have lower churn rate (Two year < One year < Month-to-month).  
The customers who choose paperlessbilling have higher churn rate.   
The customers who pay with electronic check have higher churn rate than customers who pay with other methods.  


Lastly, I will check if churn rates are different among the attributes about customers' basic information. 

```{r echo= FALSE, warning = FALSE, message=FALSE}
categorical %>%
  dplyr::select(gender:Dependents, PhoneService, InternetService, Churn) %>%
  mutate(Gender_male = ifelse(gender =="Male", "Yes", "No")) -> c1 

c1$gender <- NULL

ggplot(c1) +
  geom_bar(aes(x = Gender_male, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p1
ggplot(c1) +
  geom_bar(aes(x = SeniorCitizen, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p2
ggplot(c1) +
  geom_bar(aes(x = Partner, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p3    
ggplot(c1) +
  geom_bar(aes(x = Dependents, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p4  
ggplot(c1) +
  geom_bar(aes(x = PhoneService, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p5
ggplot(c1) +
  geom_bar(aes(x = InternetService, fill = Churn), position = "fill", stat = "count", 
           show.legend = F) -> p6

ggarrange(p1,p2,p3,p4,p5,p6, ncol = 3, nrow = 2)
```

The churn rates are not changed by genders and phone service.   
The senior customers have higher churn rate.   
The customers who have partners or dependents have lower churn rate.


#### Check Churn Rate for the full dataset
```{r echo= FALSE, warning = FALSE, message=FALSE}
telco %>%
  summarise(Total = n(), n_Churn = sum(Churn == "Yes"), p_Churn = n_Churn/Total)
```
There are 26.6% of customers churn.


# Logistic Regression Model

### Data Preparation
To prepare the data for logistic regression, I modify binomial charactors to (0,1) and change the SeniorCitizen column from int to num.

```{r echo= TRUE, warning = FALSE, message=FALSE}
telco_lr <- telco
```
```{r echo= FALSE, warning = FALSE, message=FALSE}
telco_lr %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) -> telco_lr
telco_lr %>%
  mutate(gender = ifelse(gender == "Female", 1, 0)) -> telco_lr
telco_lr %>%
  mutate(Partner = ifelse(Partner == "Yes", 1, 0)) -> telco_lr
telco_lr %>%
  mutate(PhoneService = ifelse(PhoneService == "Yes", 1, 0)) -> telco_lr
telco_lr %>%
  mutate(Dependents = ifelse(Dependents == "Yes", 1, 0)) -> telco_lr
telco_lr %>%
  mutate(PaperlessBilling = ifelse(PaperlessBilling == "Yes", 1, 0)) -> telco_lr
```

I delete the customerID and make one-hot coding to create dummy variables for all charactor variables.

```{r echo= TRUE, warning = FALSE, message=FALSE}
telco_lr$customerID <- NULL
dmy <- dummyVars(" ~ .", data = telco_lr)
dmy <- data.frame(predict(dmy, newdata = telco_lr))
str(dmy)
```

Then, I remove the variables with "No Phone Service" because they don't have any predicting power 

```{r echo= TRUE, warning = FALSE, message=FALSE}
dmy$MultipleLinesNo.phone.service <- NULL
dmy$OnlineSecurityNo.internet.service <- NULL
dmy$OnlineBackupNo.internet.service <- NULL
dmy$DeviceProtectionNo.internet.service <- NULL
dmy$TechSupportNo.internet.service <- NULL
dmy$StreamingTVNo.internet.service <- NULL
dmy$StreamingMoviesNo.internet.service <- NULL
```

Finally, I remove the last level of each factor to avoid singularities.
```{r echo= TRUE, warning = FALSE, message=FALSE}
dmy$ContractTwo.year <- NULL
dmy$InternetServiceNo <- NULL
dmy$PaymentMethodMailed.check <- NULL
dmy$tenure_year5.6.years <- NULL
```

Check the final data set.

```{r echo= FALSE, warning = FALSE, message=FALSE}
str(dmy)
```

Split the data into traning and test sets (75% vs 25%)

```{r echo= TRUE, warning = FALSE, message=FALSE}
set.seed(818)
assignment <- sample(0:1, size= nrow(dmy), prob = c(0.75,0.25), replace = TRUE)
train <- dmy[assignment == 0, ]
test <- dmy[assignment == 1, ]
```

Double check if the churn rates of two sets are close.

For the Training Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
train %>%
summarise(Total = n(), n_Churn = sum(Churn == 1), p_Churn = n_Churn/Total)
```
For the Test Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
test %>%
summarise(Total = n(), n_Churn = sum(Churn == 1), p_Churn = n_Churn/Total)
```
Now, the data is ready for training logistic regression models! 

### Train Models

I will first use all columns to build the model1. 
```{r echo= TRUE, warning = FALSE, message=FALSE}
model1 <- glm(Churn ~., family = "binomial", data = train)
summary(model1)
```

Notice there are 6 NAs in the model's summary for MultipleLinesYes, OnlineSecurityYes, OnlineBackupYes, 
DeviceProtectionYes, TechSupportYes, StreamingTVYes, StreamingMoviesYes. That's because I remove the "xxx.No Phone Service" 
or "xxx.No Internet Service" of them when processing dummy variables. Only two values of "xxx.yes" and "xxx.no" are left with obsolutely
multicollinearities between them. This problem will be address during the following variable selection.      

I use AIC to exclude variables based on their significance and create model2.
```{r echo= TRUE, warning = FALSE, message=FALSE}
model2 <- stepAIC(model1, trace = 0)
summary(model2)
```

Use VIF function to check multicollinearity
```{r echo= TRUE, warning = FALSE, message=FALSE}
vif(model2)
```

The VIFs for MonthlyCharges, InternetServiceDSL and InternetserviceFiber.optic are very high due to multicollinearity. 
Since TotalCharges has high correlation with MonthlyCharges and tenure (see the correlation plot above), I will remove 
the TotalCharges variable . The InternetserviceFiber.optic will also be removed from model3.

```{r echo= TRUE, warning = FALSE, message=FALSE}
model3 <- glm(formula = Churn ~  SeniorCitizen + Dependents + PhoneService + MultipleLinesNo + InternetServiceDSL + OnlineBackupNo +
DeviceProtectionNo + StreamingTVNo + StreamingMoviesNo + ContractMonth.to.month + ContractOne.year + 
PaperlessBilling + PaymentMethodElectronic.check + MonthlyCharges + tenure_year0.1.year + tenure_year1.2.years,
family = "binomial", data = train)
```

Then, check the model3 and its VIFs.

```{r echo= FALSE, warning = FALSE, message=FALSE}
summary(model3)
vif(model3)
```

Now all VIFs are fine below 5. but the p-values for StreamingTVNo and StreamingMoviesNo are still very high. 
So I remove these two variables and create model 4.

```{r echo= TRUE, warning = FALSE, message=FALSE}
model4 <- glm(formula = Churn ~  SeniorCitizen + Dependents + PhoneService + MultipleLinesNo + InternetServiceDSL + OnlineBackupNo +
DeviceProtectionNo + ContractMonth.to.month + ContractOne.year + 
PaperlessBilling + PaymentMethodElectronic.check + MonthlyCharges + tenure_year0.1.year + tenure_year1.2.years,
family = "binomial", data = train)
```

Check the model4 and its VIFs

```{r echo= FALSE, warning = FALSE, message=FALSE}
summary(model4)
vif(model4)
```
Model4 looks good! It is used as my final model to predict churn on train and test set. 

### Cross Validation (Confusion Matrix & ROC)

```{r echo= TRUE, warning = FALSE, message=FALSE}
model_logit <- model4
predict(model_logit, data = train, type = "response") -> train_prob
predict(model_logit, newdata = test, type = "response") -> test_prob
```

Set the threshold as 0.5 by default.

```{r echo= TRUE, warning = FALSE, message=FALSE}
train_pred <- factor(ifelse(train_prob >= 0.5, "Yes", "No"))
train_actual <- factor(ifelse(train$Churn == 1, "Yes", "No"))
test_pred <- factor(ifelse(test_prob >= 0.5, "Yes", "No"))
test_actual <- factor(ifelse(test$Churn == 1, "Yes", "No"))
```

For the Training Set:
```{r echo= TRUE, warning = FALSE, message=FALSE}
confusionMatrix(data = train_pred, reference = train_actual)
roc <- roc(train$Churn, train_prob, plot= TRUE, print.auc=TRUE)
```

For the Test Set: 
```{r echo= TRUE, warning = FALSE, message=FALSE}
confusionMatrix(data = test_pred, reference = test_actual)
roc <- roc(test$Churn, test_prob, plot= TRUE, print.auc=TRUE)
```

For the training set, the accuracy is 0.80 and the AUC is 0.85. For the test set, the accuracy is 0.79 and the AUC is 0.82.
It's a good model because the accuracy and AUC do not have big difference between the training and test sets. 
But the Specificities for two sets are as low as 0.46. 

In real case, we can adjust the threshold based on the cost of TN, FN, FP or TP to reduce cost or loss. But here, I just tend 
to find the optimal threshold (or cutoff) point that maximises the specificity (TN rate) and sensitivity (TP rate).

### Find the optimal cutoff and adjust the class of prediction

```{r echo= TRUE, warning = FALSE, message=FALSE}
pred <- prediction(train_prob, train_actual)
perf <- performance(pred, "spec", "sens")

cutoffs <- data.frame(cut=perf@alpha.values[[1]], specificity=perf@x.values[[1]], 
                      sensitivity= perf@y.values[[1]])
```
```{r echo= TRUE, warning = FALSE, message=FALSE}
opt_cutoff <- cutoffs[which.min(abs(cutoffs$specificity-cutoffs$sensitivity)),]
opt_cutoff
```
```{r echo= TRUE, warning = FALSE, message=FALSE}
ggplot(data = cutoffs) +
  geom_line(aes(x = cut, y = specificity, color ="red"), size = 1.5)+
  geom_line(aes(x = cut, y = sensitivity, color = "blue"), size = 1.5) +
  labs(x = "cutoff", y ="value") +
  scale_color_discrete(name = "", labels = c("Specificity", "Sensitivity"))+
  geom_vline(aes(xintercept = opt_cutoff$cut))+
  geom_text(aes(x= 0.55, y= 0.75),label="opt_cutoff = 0.3",hjust=1, size=4)
```

The optimal cutoff is 0.3. So I use it as the threshold to predict churn on training and test sets.

Prediction on training set with threshold = 0.3:
  ```{r echo= TRUE, warning = FALSE, message=FALSE}
train_pred_c <- factor(ifelse(train_prob >= 0.3, "Yes", "No"))
confusionMatrix(data = train_pred_c, reference = train_actual)
```

Prediction on test set with threshold = 0.3:
  ```{r echo= TRUE, warning = FALSE, message=FALSE}
predict(model_logit, newdata = test, type = "response") -> test_prob
test_pred_c <- factor(ifelse(test_prob >= 0.3, "Yes", "No"))
confusionMatrix(data = test_pred_c, reference = test_actual)
```

For the training set, the Accuracy is 0.76, and the Sensitivity and Specificity are both about 0.76.
For the test set, the Accuracy is 0.74, and the Sensitivity and Specificity are 0.74 and 0.73 respectively.
Overall, this model with adjusted cutoff works well. 


### Summary for Logistic Regression Model
The final Logistic Regression Model (with threshold = 0.5) has Accuracy of 0.79 and the AUC is 0.82. Based on the P values 
for variables, PhoneService, InternetServiceDSL, OnlineBackup, Contract, PaperleslsBilling, PaymentMethodElectronic.check, 
MonthlyCharges, tenure in 0-1 year and 1-2 years have more significant influence on predicting churn.


# Decision Tree
### Data Preparation
Decision tree models can handle categorical variables without one-hot encoding them, and one-hot encoding will degrade 
tree-model performance. Thus, I will re-prepare the data for decision tree and random forest models. I keep the "telco" data 
before I do logistic regression and change the charactor variables to factors. Here's the final dataset I use for training 
classification tree models.

```{r echo= TRUE, warning = FALSE, message=FALSE}
telcotree <- telco
telcotree$customerID <- NULL
telcotree %>%
mutate_if(is.character, as.factor) -> telcotree
str(telcotree)
```

Split the data into training and test sets.
```{r echo= TRUE, warning = FALSE, message=FALSE}
set.seed(818)
tree <- sample(0:1, size= nrow(telcotree), prob = c(0.75,0.25), replace = TRUE)
traintree <- telcotree[tree == 0, ]
testtree <- telcotree[tree == 1, ]
```

### Train Model1
First of all, I use all variables to build the model_tree1. 

```{r echo= TRUE, warning = FALSE, message=FALSE}
model_tree1 <- rpart(formula = Churn ~., data = traintree, 
method = "class", parms = list(split = "gini"))
```

### Cross Validation (Confusion Matrix and AUC) for modeltree1

```{r echo= TRUE, warning = FALSE, message=FALSE}
predict(model_tree1, data = traintree, type = "class") -> traintree_pred1
predict(model_tree1, data = traintree, type = "prob") -> traintree_prob1
predict(model_tree1, newdata= testtree, type = "class") -> testtree_pred1
predict(model_tree1, newdata = testtree, type = "prob") -> testtree_prob1
```
For the Training Set
```{r echo= TRUE, warning = FALSE, message=FALSE}
confusionMatrix(data = traintree_pred1, reference = traintree$Churn)
traintree_actual <- ifelse(traintree$Churn == "Yes", 1,0)
roc <- roc(traintree_actual, traintree_prob1[,2], plot= TRUE, print.auc=TRUE)
```

For the Test Set:
```{r echo= TRUE, warning = FALSE, message=FALSE}
confusionMatrix(data = testtree_pred1, reference = testtree$Churn)
testtree_actual <- ifelse(testtree$Churn == "Yes", 1,0)
roc <- roc(testtree_actual, testtree_prob1[,2], plot = TRUE, print.auc = TRUE)
```

For the training set, the Accuracy is 0.79 and the AUC is 0.800. For the test set, the Accuracy is 0.78 and the AUC is 0.78.

### Train Model2
Remember that Totalcharges, MonthlyCharges and tenure are highly correlated, which may effect the performance of the 
decision tree models. So I remove the TotalCharges column to train the second model.

```{r echo= TRUE, warning = FALSE, message=FALSE}
model_tree2 <- rpart(formula = Churn ~ gender + SeniorCitizen + Partner + Dependents + PhoneService + 
MultipleLines + InternetService + OnlineSecurity + TechSupport +
OnlineBackup + DeviceProtection + StreamingTV + StreamingMovies + 
Contract + PaperlessBilling + tenure_year +
PaymentMethod + MonthlyCharges, data = traintree, 
method = "class", parms = list(split = "gini"))
```

### Cross Validation for modeltree2
```{r echo= FALSE, warning = FALSE, message=FALSE}
predict(model_tree2, data = traintree, type = "class") -> traintree_pred2
predict(model_tree2, data = traintree, type = "prob") -> traintree_prob2
predict(model_tree2, newdata= testtree, type = "class") -> testtree_pred2
predict(model_tree2, newdata = testtree, type = "prob") -> testtree_prob2
```

For the Training Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
confusionMatrix(data = traintree_pred2, reference = traintree$Churn)
traintree_actual <- ifelse(traintree$Churn == "Yes", 1,0)
roc <- roc(traintree_actual, traintree_prob2[,2], plot= TRUE, print.auc=TRUE)
```

For the Test Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
testtree_actual <- ifelse(testtree$Churn == "Yes", 1,0)
confusionMatrix(data = testtree_pred2, reference = testtree$Churn)
roc <- roc(testtree_actual, testtree_prob2[,2], plot = TRUE, print.auc = TRUE)
```

For the training set, the Accuracy is 0.80 and the AUC is 0.80. For the test set, the Accuracy is 0.78 and the AUC is 0.78.
Compared to the performance of the first model, the performance of the second model is just a little bit better. So I will 
use model 2 as the final classification tree model. There is still a problem that the Specificity is too low. 
But since I don't have the real conditions about costs for this case, I don't do cutoff optimization here for tree models.


### Summary for Decision Tree Model
The final decision tree model has Accuracy of 0.78 and AUC of 0.78 for the test set. It does not perform as good as the logistic 
regression model.

# Random Forest

### Data Preparation
I use the same data prepared for Classification Tree models.

### Train Model
```{r echo= FALSE, warning = FALSE, message=FALSE}
set.seed(802)
modelrf1 <- randomForest(formula = Churn ~., data = traintree)
print(modelrf1)
```

### Cross Validation for modelrf1
```{r echo= FALSE, warning = FALSE, message=FALSE}
predict(modelrf1, traintree, type = "class") -> trainrf_pred
predict(modelrf1, traintree, type = "prob") -> trainrf_prob
predict(modelrf1, newdata = testtree, type = "class") -> testrf_pred
predict(modelrf1, newdata = testtree, type = "prob") -> testrf_prob
```

For the Training Set: 
```{r echo= FALSE, warning = FALSE, message=FALSE}
confusionMatrix(data = trainrf_pred, reference = traintree$Churn)
trainrf_actual <- ifelse(traintree$Churn == "Yes", 1,0)
roc <- roc(trainrf_actual, trainrf_prob[,2], plot= TRUE, print.auc=TRUE)
```

For the Test Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
confusionMatrix(data = testrf_pred, reference = testtree$Churn)
testrf_actual <- ifelse(testtree$Churn == "Yes", 1,0)
roc <- roc(testrf_actual, testrf_prob[,2], plot = TRUE, print.auc = TRUE)
```

For the training set, the Accuracy is 0.97 and the AUC is almost 1. For the test set, the Accuracy is 0.79 and the AUC is 0.82.

### Tunning 

#### Tunning mtry with tuneRF
```{r echo= TRUE, warning = FALSE, message=FALSE}
set.seed(818)
modelrf2 <- tuneRF(x = subset(traintree, select = -Churn), y = traintree$Churn, ntreeTry = 500, doBest = TRUE)
print(modelrf2)
```
When mtry = 2, OOB decreases from 20.11% to 19.67%

#### Grid Search based on OOB error

I first establish a list of possible values for mtry, nodesize and sampsize.
```{r echo= TRUE, warning = FALSE, message=FALSE}
mtry <- seq(2, ncol(traintree) * 0.8, 2)
nodesize <- seq(3, 8, 2)
sampsize <- nrow(traintree) * c(0.7, 0.8)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)
```

Then, I create a loop to find the combination with the optimal oob err. 
```{r echo= TRUE, warning = FALSE, message=FALSE}
oob_err <- c()
for (i in 1:nrow(hyper_grid)) {
model <- randomForest(formula = Churn ~ ., 
data = traintree,
mtry = hyper_grid$mtry[i],
nodesize = hyper_grid$nodesize[i],
sampsize = hyper_grid$sampsize[i])
oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])
```

The optimal hyperparameters are mtry = 2, nodesize = 7, sampsize = 3658.2

### Train model 2 with optimal hyperparameters.
```{r echo= TRUE, warning = FALSE, message=FALSE}
set.seed(802)
modelrf3 <- randomForest(formula = Churn ~., data = traintree, mtry = 2, nodesize = 7, sampsize = 3658.2)
print(modelrf3)
```

OOB of modelrf3 decreases a little bit to 19.79% with the optimal combination. The OOB of modelrf2 is 19.67%. 
So I will use modelrf2 as the final random forest model.

### Cross Validation for modelrf2
```{r echo= FALSE, warning = FALSE, message=FALSE}
predict(modelrf2, traintree, type = "class") -> trainrf_pred2
predict(modelrf2, traintree, type = "prob") -> trainrf_prob2
predict(modelrf2, newdata = testtree, type = "class") -> testrf_pred2
predict(modelrf2, newdata = testtree, type = "prob") -> testrf_prob2
```

For the Training Set: 
```{r echo= FALSE, warning = FALSE, message=FALSE}
confusionMatrix(data = trainrf_pred2, reference = traintree$Churn)
trainrf_actual <- ifelse(traintree$Churn == "Yes", 1,0)
roc <- roc(trainrf_actual, trainrf_prob2[,2], plot= TRUE, print.auc=TRUE)
```

For the Test Set:
```{r echo= FALSE, warning = FALSE, message=FALSE}
confusionMatrix(data = testrf_pred2, reference = testtree$Churn)
testrf_actual <- ifelse(testtree$Churn == "Yes", 1,0)
roc <- roc(testrf_actual, testrf_prob2[,2], plot = TRUE, print.auc = TRUE)
```

For the training set, the Accuracy is 0.88 and AUC is 0.95. For the test set, the Accuracy is 0.79 and the AUC is 0.82. 
Compared to the performance of the first model, which Accuracy = 0.97, AUC = 0.995 for the training set, and Accuracy = 0.79,
AUC = 0.82 for the test set. The second model works a little better.


### Variable Importance
```{r echo= TRUE, warning = FALSE, message=FALSE}
varImpPlot(modelrf2,type=2)
```

### Summary for Random Forest Model
The final random forest model has the Accuracy of 0.79 and AUC of 0.82 for the test set.   
According to the Variable Importance plot, TotalCharges, MonthlyCharges, Tenure_year and Contract are the top 4 most important 
variables to predict churn. The PhoneSerivce, Gender, SeniorCitizen, Dependents, Partner, MultipleLines, PaperlessBilling, StreamingTV,
Movies, DeviceProtection and OnlineBackup have very small effect on Churn.


# Comparison of ROC and AUC for Logistic Regression, Decision Tree and Random Forest models
```{r echo= TRUE, warning = FALSE, message=FALSE}
preds_list <- list(test_prob, testtree_prob2[,2], testrf_prob2[,2])
m <- length(preds_list)
actuals_list <- rep(list(testtree$Churn), m)

pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves for 3 Models")
legend(x = "bottomright",
legend = c("Logistic Regression", "Decision Tree", "Random Froest"),
fill = 1:m)
```

The logistic regression model and random forest model work better than the decision tree model. The Accuracies are 0.78 for 
Logistic Regression, 0.78 for Decision Tree and 0.79 for Random Forest, with 0.5 as threshold.

Regarding the variance importance, the logistic regression model and the random forest model have little differences.
They both have MonthlyCharges, tenure, Contract and PaymentMethod as important predictors and have gender, StreamingTV, Movies and 
Partner as unimportant predictors. However, in the logistic regression model, PaperlessBilling, PhoneService and OnlineBackup 
show significant influence on the churn, while in the randomforest model, they have very small predicting power. 
