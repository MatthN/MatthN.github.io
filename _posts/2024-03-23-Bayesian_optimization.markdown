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
### The Limitation of Grid Search
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

Testing all points in a grid on your hyperparameter space quickly becomes undesirable or even infeasible.

### Random Search
A simple way to deal with the large possible number of possible configurations is to perform a **random search**. Instead of testing each possible option you pick $$n$$ random configurations and see which performs best.

Taking the house prices [example]({{ site.url }}/articles/2024-03/Predicting-house-prices) from before we can apply random search to find a good 'n_estimators' for the XGBoost regressor. Let's run this for 12 iterations since there were also 12 points in the grid used before.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(random_state=5))
])

param_dist = {
    'xgb__n_estimators': randint(25, 300)
}

xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                cv=5, scoring='neg_mean_squared_error',
                                n_iter=12, verbose=3, n_jobs=-1)
xgb_random.fit(X=X_train_oversampled, y=y_train_oversampled)

print(xgb_random.best_params_)
```
    >>> {'xgb__n_estimators': 72}

``` python
y_pred = xgb_random.predict(X=X_val)
print(root_mean_squared_error(y_true=y_val, y_pred=y_pred))
```
    >>> 22324.43301306317

The best option of 72 estimators is close to the previously found 50, and also the test scores are comparable.

Random search is great for its simplicity. It allows you to explore a vaster hyperparameter space and potentially find more optimal configurations than you otherwise would with grid search. On the flip side, since the search is not exhaustive the optimal configuration might not be found. The exploration of the search space could be uneven due to randomness. This can already be seen in the above example where sampled values were 38, 170, 72, 84, 287, 272 and 233. This strategy also treats each hyperparameter as independent. This can also lead to inefficient sampling in case some are strongly correlated. 

