set.seed(69)
trainIndex <- createDataPartition(MonkeyPox$MonkeyPox, p=0.8, list = FALSE)
Train <- MonkeyPox[ trainIndex, ]
Test <- MonkeyPox[ -trainIndex, ]
svmGrid <- expand.grid(sigma = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), C = c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128))
trainctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
svm.model <- train(MonkeyPox~., data = Train, method = "svmRadial", trControl = trainctrl, tuneGrid = svmGrid)
svm.model$times
View(svm.model$results)
svm.predict <- predict(svm.model, Test)
confusionMatrix(svm.predict, as.factor(Test$MonkeyPox), mode= "prec_recall")
