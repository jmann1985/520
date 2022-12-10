library (caret)
library (rpart)
library (rpart.plot)

set.seed(69)
trainIndex <- createDataPartition(MonkeyPox$MonkeyPox, p=0.8, list = FALSE)
Train <- MonkeyPox[ trainIndex, ]
Test <- MonkeyPox[ -trainIndex, ]
treemodel <- rpart(MonkeyPox~., data = Train)
treeplot <- rpart.plot(treemodel)
treemodel$variable.importance
Prediction <- predict(treemodel, Test, type = 'class')
Prediction
Confmatrix <- table(Test$MonkeyPox, Prediction)
Confmatrix
Acc <- (sum(diag(Confmatrix)) / sum(Confmatrix)*100)
Acc
