df <- read.csv("Analysis.csv", header = T)

#Build a full model on the training data

fit <- lm(age ~ ., data = df)
summary(fit)

library(caret)

#Standard code
inTrain <- createDataPartition(df$age, p =0.8, list = F)

trainData <- df[inTrain,]
testData <- df[-inTrain,]

#Reduce parameters using stepwise regression
fit_step <- step(fit, k = log(nrow(trainData)))
summary(fit_step)

#BIC score for the full model and final model
BIC(fit)
BIC(fit_step)
#Get Bayes Factors using Wagenmaker (2007) formula
BF <- exp((BIC(fit)-BIC(fit_step))/2)
BF
#     exp(317821 - 317770.2)/2)
#Our best fitting model is BF time more likely to fit the data than the second best fitting model. 

#Evaluate how well the model is doing on unseen data
pred <- predict.lm(fit_step,testData)
pred
#Get R-squared for model performance on testdata
cor(pred,testData$age)^2

#check stepwise r squared
summary(fit_step)

#Add prediction to testData
testData$pred <- pred

#Draw the relationship between y-hat and y
graph <- plot(testData$age ~ pred)
linearmodel <- lm(age ~ pred, data = testData)
abline(linearmodel, col = "red")

#Finally, super important, look at diagnostic plots
plot(fit_step)
#check homoscedasticity and normality



#Logistic Regression Model 
Set.seed(5555)


df <- read.csv("Analysis.csv", header = T)  

df$duration=NULL

df$y <- as.factor(df$y)
df$y <- ifelse(df$y == 'yes',1,0)

library(caret)

#Standard code
inTrain <- createDataPartition(df$y, p = 0.8, list = F)
trainData <- df[inTrain,]
testData <- df[-inTrain,]

#Build a full model on the training data

fit <- glm(y ~ ., data = trainData, family = "binomial")
summary(fit)
#check the pseudo R using deviance
1-fit$deviance/fit$null.deviance

#Reduce parameters using stepwise regression
fit_step <- step(fit, k = log(nrow(trainData)))

#Quick BF
BIC(fit)
BIC(fit_step)
exp((BIC(fit)-BIC(fit_step))/2)

# the step wise model is 1.57 x 10^19 times more likely to fit the dataset than the full model
#check stepwise model for log odds
summary(fit_step)

#check the pseudo R using deviance
1-fit_step$deviance/fit_step$null.deviance

exp(fit_step$coefficients[2])

# when there is a married client there is a  20% decrease in the liklihood of y being yes
exp(fit_step$coefficients[4])
exp(fit_step$coefficients[5])
exp(fit_step$coefficients[6])
exp(fit_step$coefficients[7])
exp(fit_step$coefficients[8])
exp(fit_step$coefficients[9])
exp(fit_step$coefficients[10])
exp(fit_step$coefficients[11])
exp(fit_step$coefficients[12])
exp(fit_step$coefficients[13])
exp(fit_step$coefficients[15])
exp(fit_step$coefficients[16])
exp(fit_step$coefficients[17])
exp(fit_step$coefficients[18])
exp(fit_step$coefficients[19])
exp(fit_step$coefficients[20])
exp(fit_step$coefficients[22])

#Predict y
pred <- predict(fit_step, testData, type = "response")
head(pred)
# pred is probabilities for the actual y values.

#confusion matrix with default cut-off
pred <- ifelse(pred > 0.5, 1, 0)
print(confusionMatrix(as.factor(pred), as.factor(testData$y), positive = "1"))

#check roc plot
library(verification)
pred <- predict(fit_step, testData, type = "response")
roc.plot(testData$y == 1, pred)
#Roc plot show 0.1 as best threshold for
#maximising sensitivity.


#final confusion matrix
pred <- ifelse(pred > 0.1, 1, 0)
print(confusionMatrix(as.factor(pred),as.factor(testData$y), positive = "1"))
#sensitivity increased, specificity decreased

#Model performance
#precision 792/(792+2864) - 22%
#recall 792/(792+282) - 74%
#f1-score 2(792)/(2(792)+2864+282) - 33%

#Parametric Assumptions
#outliners
plot(fit_step)
#outliners confirmed (a limitation )

#multicollinearity
library(car)
vif(fit_step)
#no multicollinearity

