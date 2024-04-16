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

If model training is expensive, testing all points in a grid on your hyperparameter space quickly becomes undesirable or even infeasible. For example, creating a grid for 3 hyperparameters with each 10 values to evaluate already gives in 1,000 possible combinations.

### Random Search
A simple way to deal with the large number of possible configurations is to perform a **random search**. Instead of testing each possible option you pick $$n$$ random configurations and see which performs best. Let's implement an example using `RandomizedSearchCV` from scikit-learn.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import randint

# import data and create training set
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# define regression model
xgb = XGBRegressor(random_state=7)

# define range from which n_estimator candidates can be sampled
param_dist = {
    'n_estimators': randint(1, 300)
}

# execute random search
xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                cv=5, scoring='neg_mean_squared_error',
                                n_iter=10, verbose=3, n_jobs=-1, random_state=7)
xgb_random.fit(X_train, y_train)

# print best result
print(xgb_random.best_params_)
```

    >>> Fitting 5 folds for each of 10 candidates, totalling 50 fits
    >>> {'n_estimators': 212}


This minimal example already required 50 model fits.

Random search is great for its simplicity. It allows you to explore a vaster hyperparameter space and potentially find more optimal configurations than you otherwise would with grid search. On the flip side, since the search is not exhaustive the optimal configuration might not be found. The exploration of the search space could be uneven due to randomness. The strategy also treats each hyperparameter as independent. This can lead to inefficient sampling in case some are strongly correlated.


### Bayesian Optimization
We have now established that there is a need for an algorithm that does not perform an exhaustive search, yet can still explore the hyperparameter space more efficiently than by sampling random points. This is where Bayesian optimization comes in. It is used to optimize black-box functions that are expensive to evaluate. The unknown objective function is modelled by a probabilistic surrogate model, often a Guassian process (see below). The model is updated after every iteration to incorporate the new information and determine the next sampling point. The algorithm balances exploration (sampling uncertain regions) and exploitation (sampling regions that likely contain the optimum) in order to find the globabl optimum of the objective function in a minimal number of iterations.

Gaussian processes define distributions over functions, where any finite set of points follows a multivariate Gaussian distribution. These distributions are characterized by a mean function and a covariance function, also known as a kernel, which encodes the smoothness and correlation properties of the functions.

Let's take the California housing example from above and start by defining our objective function.

```python
from sklearn.model_selection import cross_val_score

def objective(n_estimators):
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=7)
    mse_scores = -cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(mse_scores)
```

In order to fit the Gaussian process regressor we will first need to take some random samples to get started.

```python
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, RBF

np.random.seed(5)

X_samples = np.array([])
y_samples = np.array([])

# create 3 random samples
for x_sample in np.random.randint(bounds[:,0], bounds[:,1]+1, size=3):
    X_samples = np.append(X_samples, x_sample)
    y_samples = np.append(y_samples, objective(x_sample))

X_samples = X_samples.reshape(-1, 1)

# define the Gaussian process regressor and fit it to the samples
rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=50)
gpr.fit(X_samples, y_samples)

```

Setting 'n_restarts_optimizer' increases the chance of finding better hyperparameters for the model. When keeping this at the default (Which is 1) I sometimes noticed strange results. Now that the Gaussian process regressor (GPR) has been fitted we can plot the result.

```
import matplotlib.pyplot as plt

def plot_gpr(X_grid, X_samples, y_samples, gpr, next_sample=None):
    y_pred, sigma = gpr.predict(X_grid, return_std = True)
    y_pred_flat = y_pred.ravel()
    X_grid_flat = X_grid.ravel()
    plt.plot(X_grid, y_pred, 'b-', label='GPR Predictions')
    plt.fill_between(X_grid_flat, y_pred_flat - 1.96 * sigma, y_pred_flat + 1.96 * sigma, alpha=0.2, color='gray')
    plt.scatter(X_samples, y_samples, color='red', label='Evaluated Points')
    if next_sample:
        plt.axvline(x=next_sample, color='b', linestyle='--', label='Next Sample')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gaussian Process Regression')
    plt.legend()

X_grid = np.linspace(1, 300, 1000).reshape(-1, 1)
plot_gpr(X_grid=X_grid, X_samples=X_samples, y_samples=y_samples, gpr=gpr)
```

<img src="/img/posts/BayesOpt/gpr_iteration_00.png" width="50%">

The picture shows that where we have sampled the objective function the uncertainty is zero. Uncertainty increases with distance from the samples. In reality the samples are not entirely noise-free.

