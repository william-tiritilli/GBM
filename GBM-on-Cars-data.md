Gradient Boosting Example on Car Data
================

# Introduction

Similar to bagging, boosting is a general technique to crete an ensemble
of any type of base learner. However, the two approaches are different.
Bagging is called a “Parralel approach” meaning that the trees won’t use
any information from each other. This is not the case in Boosting, where
the current tree uses information from all the past trees. It is called
a “Sequential approach”. By intuition, Boosting is superior to Bagging
because of this.

<!-- ![GB Search](C:\\Users\William.Tiritilli\\Documents\\Project P\\GBM\\images\\gradient-descent-fig-1.png "GB search") -->

``` r
# Define columnn class for dataset
colCls <- c("integer",         # row id
            "character",       # analysis year
            "numeric",         # exposure
            "character",       # new business / renewal business
            "numeric",         # driver age (continuous)
            "character",       # driver age (categorical)
            "character",       # driver gender
            "character",       # marital status
            "numeric",         # years licensed (continuous)
            "character",       # years licensed (categorical)
            "character",       # ncd level
            "character",       # region
            "character",       # body code
            "numeric",         # vehicle age (continuous)
            "character",       # vehicle age (categorical)
            "numeric",         # vehicle value
            "character",       # seats
            rep("numeric", 6), # ccm, hp, weight, length, width, height (all continuous)
            "character",       # fuel type
            rep("numeric", 3)  # prior claims, claim count, claim incurred (all continuous)
)
```

``` r
# Define the data path and filename
data.path <- "C:\\Users\\William.Tiritilli\\Documents\\Project P\\Frees\\Tome 2 - Chapter 1\\"
data.fn <- "sim-modeling-dataset2.csv"

# Read in the data with the appropriate column classes
dta <- read.csv(paste(data.path, data.fn, sep = "/"),
                colClasses = colCls)
str(dta)
```

    ## 'data.frame':    40760 obs. of  27 variables:
    ##  $ row.id        : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ year          : chr  "2010" "2010" "2010" "2010" ...
    ##  $ exposure      : num  1 1 1 0.08 1 0.08 1 1 0.08 1 ...
    ##  $ nb.rb         : chr  "RB" "NB" "RB" "RB" ...
    ##  $ driver.age    : num  63 33 68 68 68 68 53 68 68 65 ...
    ##  $ drv.age       : chr  "63" "33" "68" "68" ...
    ##  $ driver.gender : chr  "Male" "Male" "Male" "Male" ...
    ##  $ marital.status: chr  "Married" "Married" "Married" "Married" ...
    ##  $ yrs.licensed  : num  5 1 2 2 2 2 5 2 2 2 ...
    ##  $ yrs.lic       : chr  "5" "1" "2" "2" ...
    ##  $ ncd.level     : chr  "6" "5" "4" "4" ...
    ##  $ region        : chr  "3" "38" "33" "33" ...
    ##  $ body.code     : chr  "A" "B" "C" "C" ...
    ##  $ vehicle.age   : num  3 3 2 2 1 1 3 1 1 5 ...
    ##  $ veh.age       : chr  "3" "3" "2" "2" ...
    ##  $ vehicle.value : num  21.4 17.1 17.3 17.3 25 ...
    ##  $ seats         : chr  "5" "3" "5" "5" ...
    ##  $ ccm           : num  1248 2476 1948 1948 1461 ...
    ##  $ hp            : num  70 94 90 90 85 85 70 85 85 65 ...
    ##  $ weight        : num  1285 1670 1760 1760 1130 ...
    ##  $ length        : num  4.32 4.79 4.91 4.91 4.04 ...
    ##  $ width         : num  1.68 1.74 1.81 1.81 1.67 ...
    ##  $ height        : num  1.8 1.97 1.75 1.75 1.82 ...
    ##  $ fuel.type     : chr  "Diesel" "Diesel" "Diesel" "Diesel" ...
    ##  $ prior.claims  : num  0 0 0 0 0 0 4 0 0 0 ...
    ##  $ clm.count     : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ clm.incurred  : num  0 0 0 0 0 0 0 0 0 0 ...

``` r
# Define a train and test set
set.seed(54321) # reproducubility
# Create a stratified data partition
train_id <- caret::createDataPartition(
  y = dta$clm.count/dta$exposure,
  p = 0.8,
  groups = 100
)[[1]]


# Divide the data in training and test set
dta_trn <- dta[train_id,]
dta_tst <- dta[-train_id,]


library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
# Proportions of the number of claims in train data
dta_trn$clm.count %>% table %>% prop.table %>% round(5)
```

    ## .
    ##       0       1       2       3       4       5 
    ## 0.92257 0.07163 0.00537 0.00037 0.00003 0.00003

``` r
# Proportions of the number of claims in test data
dta_tst$clm.count %>% table %>% prop.table %>% round(5)
```

    ## .
    ##       0       1       2       3 
    ## 0.92098 0.07252 0.00613 0.00037

We focus on GBM to perform a stoachastic gradient boosting with decision
trees. In other terms, the goal is to optimize the loss function using a
gradient descent as illustrated below. For more information, please
refers to the excellent book “Hands On Machine Learning with R” from
Bradley Boehmke at <https://bradleyboehmke.github.io/HOML/gbm.html>

<img src="C:\\Users\William.Tiritilli\\Documents\\Project P\\GBM\\images\\gradient-descent-fig-1.png" alt="GB search" style="height: 300px; width:500px;"/>

``` r
# Transform variable into factors
dta_trn$fuel.type <-as.factor(dta_trn$fuel.type)
dta_trn$driver.gender <-as.factor(dta_trn$driver.gender)
dta_trn$yrs.lic  <-as.factor(dta_trn$yrs.lic )

# formula
response <- clm.count ~ 
  driver.age + vehicle.age + hp  + fuel.type + driver.gender + yrs.lic +
  offset(log(exposure))
```

First try using the GBM function

``` r
library(gbm)
```

    ## Warning: package 'gbm' was built under R version 4.1.1

    ## Loaded gbm 2.1.8

``` r
set.seed(76539) # reproducibility
fit <- gbm(formula = response,
           data = dta_trn,
           distribution = 'poisson',
           # var.monotone = c(0,0,1,0,0,0),
           n.trees = 200,
           interaction.depth = 3,
           n.minobsinnode = 1000,
           shrinkage = 0.1,
           bag.fraction = 0.75,
           cv.folds = 2 # needs to be >1 to use to track
                        # the CV error
)
# Track the improvement in the OOB error
oob_evo <- fit$oobag.improve
```

``` r
fit %>%
pretty.gbm.tree(i.tree = 1) %>%
print(digits = 4)
```

    ##   SplitVar SplitCodePred LeftNode RightNode MissingNode ErrorReduction Weight
    ## 0        5      0.000000        1         5           9          6.950  24457
    ## 1        1      3.500000        2         3           4          1.767  10445
    ## 2       -1     -0.006279       -1        -1          -1          0.000   4843
    ## 3       -1     -0.047174       -1        -1          -1          0.000   5602
    ## 4       -1     -0.028212       -1        -1          -1          0.000  10445
    ## 5        0     33.500000        6         7           8          3.103  14012
    ## 6       -1      0.042247       -1        -1          -1          0.000   2860
    ## 7       -1      0.008532       -1        -1          -1          0.000  11152
    ## 8       -1      0.015413       -1        -1          -1          0.000  14012
    ## 9       -1     -0.003218       -1        -1          -1          0.000  24457
    ##   Prediction
    ## 0  -0.003218
    ## 1  -0.028212
    ## 2  -0.006279
    ## 3  -0.047174
    ## 4  -0.028212
    ## 5   0.015413
    ## 6   0.042247
    ## 7   0.008532
    ## 8   0.015413
    ## 9  -0.003218

Single trees are not giving much information.

We want to tune a GBM by tracking the OOB error (Cross-validation error)

First, we set up a grid search

``` r
# Set up a search grid
tgrid <- expand.grid('depth' = c(1,3,5),
                     'ntrees' = NA,
                     'oob_err' = NA)
```

Then we iterate ovre the grid search

``` r
# Iterate over the search grid
for(i in seq_len(nrow(tgrid))){
  set.seed(76539) # reproducibility
  # Fit a GBM
  fit2 <- gbm(formula = response,
             data = dta_trn, distribution = 'poisson',
             #var.monotone = c(0,0,1,0,0,0,0,0,0),
             n.trees = 1000, shrinkage = 0.01,
             interaction.depth = tgrid$depth[i],
             n.minobsinnode = 1000,
             bag.fraction = 0.75, cv.folds = 0
  )
  # Retrieve the optimal number of trees
  opt <- which.max(cumsum(fit$oobag.improve))
  tgrid$ntrees[i] <- opt
  tgrid$oob_err[i] <- cumsum(fit$oobag.improve[1:opt])[opt]
}
```

``` r
library(magrittr)
# Order results on the OOB error
tgrid %<>% arrange(oob_err)
print(tgrid)
```

    ##   depth ntrees     oob_err
    ## 1     1     17 0.002451921
    ## 2     3     17 0.002451921
    ## 3     5     17 0.002451921

17 trees are enough.

We train a model with 17 trees.

``` r
# Fit the optimal GBM
set.seed(76539) # reproducibility
fit_gbm <- gbm(formula = response,
               data = dta_trn,
               distribution = 'poisson',
               #var.monotone = c(0,0,1,0,0,0,0,0,0),
               n.trees = tgrid$ntrees[1], #17 trees
               shrinkage = 0.01,
               interaction.depth = tgrid$depth[1],
               n.minobsinnode = 1000,
               bag.fraction = 0.75,
               cv.folds = 0
)

# Get the built-in feature importance
summary(fit_gbm)
```

![](GBM-on-Cars-data_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

    ##                         var  rel.inf
    ## yrs.lic             yrs.lic 89.80785
    ## driver.age       driver.age 10.19215
    ## vehicle.age     vehicle.age  0.00000
    ## hp                       hp  0.00000
    ## fuel.type         fuel.type  0.00000
    ## driver.gender driver.gender  0.00000

Let’s do the same work for the claim severity.

``` r
features <- c('driver.age', 'hp',
                'fuel.type ', 'driver.gender', 'body.code', 'yrs.licensed')
```

``` r
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 4.1.2

``` r
# Only retain the claims
dta_trn_claims <- dta_trn %>% dplyr::filter(clm.count > 0)
# Plot the density of all observations and those below 10 000 Euro
gridExtra::grid.arrange(
      ggplot(dta_trn_claims, aes(x = clm.incurred )) + 
      geom_density(adjust = 3, col = 'black', fill = 'gray') +
      labs(y = 'Density'),
      ggplot(dta_trn_claims, aes(x = clm.incurred )) + 
      geom_density(adjust = 3, col = 'black', fill = 'gray') +
      labs(y = 'Density') + xlim(0, 1e4),
      ncol = 2
        )
```

    ## Warning: Removed 1 rows containing non-finite values (stat_density).

![](GBM-on-Cars-data_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
dta_trn_claims$fuel.type <-as.factor(dta_trn_claims$fuel.type)
dta_trn_claims$driver.gender <-as.factor(dta_trn_claims$driver.gender)
dta_trn_claims$body.code <-as.factor(dta_trn_claims$body.code)
dta_trn_claims$yrs.licensed <-as.factor(dta_trn_claims$yrs.licensed)
```

``` r
set.seed(54321)
gbm_sev <- gbm(
  formula = as.formula(paste('clm.incurred ~', paste(features, collapse = ' + '))),
  data = dta_trn_claims,
  weights = clm.count,
  #distribution = 'gamma', # Gamma distribution not supported by the package?
  distribution = 'gaussian',

  n.trees = 500, # T in Table 3
  interaction.depth = 1, # d in Table 3
  shrinkage = 0.01, # lambda in Table 1
  bag.fraction = 0.75, # delta in Table 1
  n.minobsinnode = 0.01 * 0.75 * nrow(dta_trn_claims), # kappa * delta in Table 1
  verbose = FALSE
  )
```

``` r
summary(gbm_sev)
```

![](GBM-on-Cars-data_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

    ##                         var   rel.inf
    ## driver.age       driver.age 71.135071
    ## driver.gender driver.gender  8.451687
    ## yrs.licensed   yrs.licensed  8.360572
    ## body.code         body.code  6.244890
    ## hp                       hp  5.807780
    ## fuel.type         fuel.type  0.000000

First, we set up a grid search

``` r
# Iterate over the search grid
for(i in seq_len(nrow(tgrid))){
  set.seed(76539) # reproducibility
  # Fit a GBM
  fit_sev2 <- gbm(as.formula(paste('clm.incurred ~', paste(features, collapse = ' + '))),
             data = dta_trn_claims, distribution = 'gaussian',
             #var.monotone = c(0,0,1,0,0,0,0,0,0),
             n.trees = 1000, shrinkage = 0.01,
             interaction.depth = tgrid$depth[i],
             n.minobsinnode = 500,
             bag.fraction = 0.75, cv.folds = 0
  )
  # Retrieve the optimal number of trees
  opt <- which.max(cumsum(fit_sev2$oobag.improve))
  tgrid$ntrees[i] <- opt
  tgrid$oob_err[i] <- cumsum(fit_sev2$oobag.improve[1:opt])[opt]
}
```

``` r
# Order results on the OOB error
tgrid %<>% arrange(oob_err)
print(tgrid)
```

    ##   depth ntrees   oob_err
    ## 1     1      4 -4.366957
    ## 2     3      4 54.190625
    ## 3     5      4 54.190625

``` r
# Fit the optimal GBM
set.seed(76539) # reproducibility
fit_gbm_sev <- gbm(formula = response,
               data = dta_trn_claims,
               distribution = 'gaussian',
               #var.monotone = c(0,0,1,0,0,0,0,0,0),
               n.trees = tgrid$ntrees[1], #17 trees
               shrinkage = 0.01,
               interaction.depth = tgrid$depth[1],
               n.minobsinnode = 500,
               bag.fraction = 0.75,
               cv.folds = 0
)

# Get the built-in feature importance
summary(fit_gbm_sev)
```

![](GBM-on-Cars-data_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

    ##                         var  rel.inf
    ## vehicle.age     vehicle.age 75.47709
    ## yrs.lic             yrs.lic 24.52291
    ## driver.age       driver.age  0.00000
    ## hp                       hp  0.00000
    ## fuel.type         fuel.type  0.00000
    ## driver.gender driver.gender  0.00000
