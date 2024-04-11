---
layout: post
title: "Bayesian Optimization"
excerpt: "Test"
categories: [Data Science]
tags: [optimization, Bayesian optimization]
comments: true
image:
  feature: 
  credit: 
  creditlink: 
---

## Model Parameters vs. Hyperparameters
In machine learning we look for a suitable (mathematical) model that can be fitted to training data in order to make predictions about unseen data points. During the fitting the model learns the values of its parameters by optimizing a specific metric (e.g., root mean squared error). Examples of model parameters are the coefficients of linear model, weights in a neural network, or the feature and its split value in a decision tree.

Models can also have hyperparameters. These are not learned durig the training, but instead are configuration settings provided upfront. They control the behavior of the learning algorithm and can thus have a significant impact on model performance. Examples are the learning rate used in gradient descent, the number of hidden layers in a neural network, or the depth of a decision tree.

Hyperparameters are set before model training. They impact the model behavior.
{: .notice}

Tuning the model's hyperparameters can become quite complex due to their number, types (discrete, continuous, categorical, Boolean) and interdependence. Examples of one hyperparameter impacting the value of another are the learning rate and batch size in gradient descent, the number of clusters and distance metric in K-means clustering, or the window size and forecast horizon in time series forecasting.

## Hyperparameter tuning
The simplest way to approach this task is to create a grid where each point represents a combination of hyperparameter values and then test every configuration. This is the approach that I took to find the optimal number of estimators for random forest and the XGBoost regressors [here]({{ site.url }}/articles/2024-03/Predicting-house-prices). However, this ignored the fact that there are many more hyperparameters that can be tuned for these models. Take for example the XGBoost regressor from before.

```python
xgb.named_steps['xgb'].get_params()
```

    >>> {'objective': 'reg:squarederror',
    'base_score': None,
    'booster': None,
    'callbacks': None,
    'colsample_bylevel': None,
    'colsample_bynode': None,
    'colsample_bytree': None,
    'device': None,
    'early_stopping_rounds': None,
    'enable_categorical': False,
    'eval_metric': None,
    'feature_types': None,
    'gamma': None,
    'grow_policy': None,
    'importance_type': None,
    'interaction_constraints': None,
    'learning_rate': None,
    'max_bin': None,
    'max_cat_threshold': None,
    'max_cat_to_onehot': None,
    'max_delta_step': None,
    'max_depth': None,
    'max_leaves': None,
    'min_child_weight': None,
    'missing': nan,
    'monotone_constraints': None,
    'multi_strategy': None,
    'n_estimators': 50,
    'n_jobs': None,
    'num_parallel_tree': None,
    'random_state': 5,
    'reg_alpha': None,
    'reg_lambda': None,
    'sampling_method': None,
    'scale_pos_weight': None,
    'subsample': None,
    'tree_method': None,
    'validate_parameters': None,
    'verbosity': None}



