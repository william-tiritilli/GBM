

load("data\\all-data.RData")

str(dta)

# Select the train set
# We need to select only claims > 0 to predict severity
# The dataset is considerably reduced in size.
library(dplyr)
dta_trn <- dta %>% filter(train == TRUE)
dta_tst <- dta %>% filter(train == FALSE)

dta_trn_sev <- dta %>% filter(train == TRUE & clm.incurred > 0)
#dta_tst_sev <- dta %>% filter(train == FALSE & clm.incurred > 0) not ok 

dim(dta)
dim(dta_trn)
dim(dta_tst)
dim(dta_trn_sev)
head(dta_trn_sev)

# Remove the predictors that are not in use for the analysis
dta_trn_sev <- subset(dta_trn, select = -c(row.id, sev
))

dta_tst_sev <- subset(dta_tst, select = -c(row.id, sev
))

# Proportions of the number of claims in train data
dta_trn_sev$clm.count %>% table %>% prop.table %>% round(5)
# Proportions of the number of claims in test data
dta_tst_sev$clm.count %>% table %>% prop.table %>% round(5)


# Set a vector of features to be chosen for the training

features <- c('year' ,'hp', 'length', 'width', 'height', 'drv.age.gr1' , 'drv.age.gr2' , 'driver.gender' , 'marital.status' , 'yrs.lic' , 'fuel.type', 'prior.claims', 'ncd.level' , 'region.g1' )


library(gbm)
# for reproducibility
set.seed(1234)

# train GBM model
gbm.fit_sev <- gbm(
  formula = as.formula(paste('clm.incurred ~', paste(features, collapse = ' + '))),
  #formula = clm.incurred ~.,
  
  distribution = "gaussian", # gamma is not supported anymore
  data = dta_trn_sev,
  n.trees = 95,
  interaction.depth = 1, # a stump
  shrinkage = 0.01, # learning rate
  cv.folds = 5,
  bag.fraction = 1,
  n.minobsinnode = 0.01 * 1 * nrow(dta_trn_sev),
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

?gbm
# print results
print(gbm.fit_sev)

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit_sev$cv.error)

# get MSE and compute RMSE
sqrt(min(gbm.fit_sev$cv.error))

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit_sev,plot.it = TRUE, method = "cv")

# Prediction
# Generic prediction function
predict_model <- function(object, newdata) UseMethod('predict_model')


# Statistical performance
# Generic prediction function
predict_model <- function(object, newdata) UseMethod('predict_model')
# Prediction function for a regression tree
predict_model.rpart <- function(object, newdata) {
  predict(object, newdata, type = 'vector')
}
# Prediction function for a random forest
predict_model.rforest <- function(object, newdata) {
  predict(object, newdata, type= 'vector')
}
# Prediction function for a GBM
predict_model.gbm <- function(object, newdata) {
  predict(object, newdata, n.trees = object$n.trees, type = 'response')
}


oos_pred <- tibble::tibble(
  #tree_freq = tree_freq %>% predict_model(newdata = mtpl_tst),
  #rf_freq = rf_freq %>% predict_model(newdata = mtpl_tst),
  #gbm_freq = gbm.fit %>% predict_model(newdata = dta_tst),
  #tree_sev = tree_sev %>% predict_model(newdata = mtpl_tst),
  #rf_sev = rf_sev %>% predict_model(newdata = mtpl_tst),
  gbm_sev = gbm.fit_sev %>% predict_model(newdata = dta_tst)
)

head(oos_pred)

oos_pred2 <- tibble::tibble(
  #tree_freq = tree_freq %>% predict_model(newdata = mtpl_tst),
  #rf_freq = rf_freq %>% predict_model(newdata = mtpl_tst),
  #gbm_freq = gbm.fit %>% predict_model(newdata = dta_tst),
  #tree_sev = tree_sev %>% predict_model(newdata = mtpl_tst),
  #rf_sev = rf_sev %>% predict_model(newdata = mtpl_tst),
  gbm_sev2 = gbm.fit_sev %>% predict_model.gbm(newdata = dta_tst)
)

head(oos_pred2)



model_type.gbm <- function(x, ...) {
  return("regression")
}

predict_model.gbm <- function(x, newdata, ...) {
  pred <- predict(x, newdata, n.trees = x$n.trees)
  return(as.data.frame(pred))
}



# predict values for test data
pred <- gbm::predict.gbm(object = gbm.fit_sev, n.trees = gbm.fit_sev$n.trees, dta_tst)

# results - Root Mean Square Error
#caret::RMSE(pred, dta_tst$clm.incurred)


head(pred)
head(oos_pred)

# conclusion, with the count, it is not very friendly.





# Grid search

# create hyperparameter grid
hyper_grid <- expand.grid(
  trees = c(1000, 5000, 10000),
  shrinkage = c(.01, .1),
  interaction.depth = c(1, 3),
  n.minobsinnode = c(5, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(dta_trn_sev), nrow(dta_trn_sev))
random_dta_train_sev <- dta_trn_sev[random_index, ]


# grid search 
system.time(for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune_sev <- gbm(
    formula = as.formula(paste('clm.incurred ~', paste(features, collapse = ' + '))),
    distribution = "gaussian",
    data = random_dta_train_sev,
    n.trees = hyper_grid$trees[i],
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune_sev$valid.error)
  hyper_grid$min_Dev[i] <- sqrt(min(gbm.tune_sev$valid.error))
}
)

hyper_grid %>% 
  dplyr::arrange(min_Dev) %>%
  head(10)

# Model Optimal

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit_sev.final <- gbm(
  formula = as.formula(paste('clm.incurred ~', paste(features, collapse = ' + '))),
  distribution = "gaussian",
  data = dta_trn,
  n.trees = 39,
  interaction.depth = 3,
  shrinkage = 0.1,
  n.minobsinnode = 10,
  bag.fraction = 0.8, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
) 

# print results
print(gbm.fit_sev.final)


# Save model
saveRDS(gbm.fit_sev.final, file = "\\GBM Tuning\\gbm.fit_sev.final.rda")


# Variable Importance
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit_sev.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

# PArtial dependance plot
library(ggplot2)
gbm.fit_sev.final %>%
  pdp::partial(pred.var = "year", n.trees = gbm.fit_sev.final$n.trees, grid.resolution = 100) %>%
  autoplot(rug = TRUE, train = dta_trn) +
  scale_y_continuous(labels = scales::dollar)

# Prediction

# predict values for test data
pred <- predict(gbm.fit_sev.final, n.trees = gbm.fit_sev.final$n.trees, dta_tst)

head(pred)

# results
caret::RMSE(pred, dta_tst$clm.incurred)
