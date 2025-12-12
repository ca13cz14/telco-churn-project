# 02_models.R
# Fit classification models and compare performance.

library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(pROC)

set.seed(123)

# Load preprocessed data --------------------------------------------------
obj <- readRDS("../data/prepared_telco.rds")

telco_train <- obj$telco_train
telco_test  <- obj$telco_test
x_train     <- obj$x_train
x_test      <- obj$x_test
y_train     <- obj$y_train
y_test      <- obj$y_test

# Helper for metrics ------------------------------------------------------

compute_metrics <- function(y_true, prob_positive, threshold = 0.5) {
  
  pred_class <- factor(ifelse(prob_positive >= threshold, "Yes", "No"),
                       levels = levels(y_true))
  
  cm <- confusionMatrix(pred_class, y_true, positive = "Yes")
  
  roc_obj <- roc(
    response  = y_true,
    predictor = prob_positive,
    levels    = rev(levels(y_true))
  )
  
  tibble(
    Accuracy    = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    AUC         = as.numeric(roc_obj$auc)
  )
}

# 1. Logistic regression (GLM) --------------------------------------------

glm_fit <- glm(churn ~ ., data = telco_train, family = binomial)

glm_prob_test <- predict(glm_fit, newdata = telco_test, type = "response")

metrics_glm <- compute_metrics(y_test, glm_prob_test)
print(metrics_glm)

# 2. Penalised logistic regression (Ridge & Lasso) ------------------------

y_train_num <- ifelse(y_train == "Yes", 1, 0)

cv_ridge <- cv.glmnet(
  x = x_train,
  y = y_train_num,
  family = "binomial",
  alpha = 0,
  nfolds = 5
)

cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train_num,
  family = "binomial",
  alpha = 1,
  nfolds = 5
)

ridge_prob_test <- predict(
  cv_ridge,
  newx = x_test,
  s = "lambda.min",
  type = "response"
) %>% as.vector()

lasso_prob_test <- predict(
  cv_lasso,
  newx = x_test,
  s = "lambda.min",
  type = "response"
) %>% as.vector()

metrics_ridge <- compute_metrics(y_test, ridge_prob_test)
metrics_lasso <- compute_metrics(y_test, lasso_prob_test)

print(metrics_ridge)
print(metrics_lasso)

# Optional: inspect non-zero lasso coefficients
coef_lasso <- coef(cv_lasso, s = "lambda.min")
nonzero_idx <- which(coef_lasso != 0)
cat("First few non-zero Lasso coefficients:\n")
print(rownames(coef_lasso)[nonzero_idx][1:15])

# 3. Random forest --------------------------------------------------------

control_rf <- trainControl(
  method = "cv",
  number = 3,            
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_grid <- expand.grid(mtry = c(5, 7)) 

set.seed(456)
rf_fit <- train(
  churn ~ .,
  data      = telco_train,
  method    = "rf",
  metric    = "ROC",
  trControl = control_rf,
  tuneGrid  = rf_grid,
  ntree     = 200          
)

print(rf_fit)

rf_prob_test <- predict(rf_fit, newdata = telco_test, type = "prob")[, "Yes"]

metrics_rf <- compute_metrics(y_test, rf_prob_test)
print(metrics_rf)

# 4. SVM (radial) ---------------------------------------------------------

svm_train_df <- data.frame(x_train, churn = y_train)

control_svm <- trainControl(
  method = "cv",
  number = 3,        
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

svm_grid <- expand.grid(
  sigma = 0.02,
  C     = 1
)

set.seed(789)
svm_fit <- train(
  churn ~ .,
  data      = svm_train_df,
  method    = "svmRadial",
  metric    = "ROC",
  trControl = control_svm,
  tuneGrid  = svm_grid
)

print(svm_fit)

svm_prob_test <- predict(
  svm_fit,
  newdata = data.frame(x_test),
  type    = "prob"
)[, "Yes"]

metrics_svm <- compute_metrics(y_test, svm_prob_test)
print(metrics_svm)

# 5. Neural network -----------------------------------------------------

if (requireNamespace("keras", quietly = TRUE)) {
  
  library(keras)
  
  x_train_scaled <- scale(x_train)
  x_test_scaled  <- scale(
    x_test,
    center = attr(x_train_scaled, "scaled:center"),
    scale  = attr(x_train_scaled, "scaled:scale")
  )
  
  y_train_num <- ifelse(y_train == "Yes", 1, 0)
  y_test_num  <- ifelse(y_test == "Yes", 1, 0)
  
  input_dim <- ncol(x_train_scaled)
  
  nn_model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = input_dim) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  nn_model %>% compile(
    loss      = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics   = c("accuracy")
  )
  
  history <- nn_model %>% fit(
    x = x_train_scaled,
    y = y_train_num,
    epochs = 40,
    batch_size = 64,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = 4,
        restore_best_weights = TRUE
      )
    )
  )
  
  nn_prob_test <- nn_model %>%
    predict(x_test_scaled) %>%
    as.vector()
  
  metrics_nn <- compute_metrics(y_test, nn_prob_test)
  print(metrics_nn)
  
} else {
  
  message("Package 'keras' not installed: neural network model not fitted.")
}

# Combine metrics into a single table --------------------------

all_metrics <- bind_rows(
  GLM          = metrics_glm,
  Ridge        = metrics_ridge,
  Lasso        = metrics_lasso,
  RandomForest = metrics_rf,
  SVM          = metrics_svm,
  .id = "Model"
)

print(all_metrics)
