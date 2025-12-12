# 01_eda_preprocessing.R
# Data loading, cleaning and train/test split.

library(tidyverse)
library(janitor)
library(caret)

set.seed(123)

# 1. Load data ------------------------------------------------------------
telco_raw <-
  readr::read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv") %>%
  clean_names()

# 2. Clean and preprocess -------------------------------------------------

telco <-
  telco_raw %>%
  mutate(
    total_charges = as.numeric(as.character(total_charges))
  ) %>%
  drop_na(total_charges) %>%
  select(-customer_id) %>%
  mutate(
    across(where(is.character), as.factor),
    churn = forcats::fct_relevel(churn, "No")
  )

# Quick sanity check
telco %>% glimpse()

# 3. Train???test split -----------------------------------------------------

train_idx <- createDataPartition(telco$churn,
                                 p = 0.8,
                                 list = FALSE)

telco_train <- telco[train_idx, ]
telco_test  <- telco[-train_idx, ]

# 4. Design matrices for models that need numeric input -------------------

x_train <- model.matrix(churn ~ ., data = telco_train)[, -1]
x_test  <- model.matrix(churn ~ ., data = telco_test)[, -1]

y_train <- telco_train$churn
y_test  <- telco_test$churn

# 5. Save prepared objects ------------------------------------------------
# Save into the SAME ../data/ folder
prepared <- list(
  telco_train = telco_train,
  telco_test  = telco_test,
  x_train     = x_train,
  x_test      = x_test,
  y_train     = y_train,
  y_test      = y_test
)

saveRDS(prepared, file = "../data/prepared_telco.rds")



