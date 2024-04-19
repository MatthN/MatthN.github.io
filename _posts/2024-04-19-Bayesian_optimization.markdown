---
layout: post
title: "Unlocking Model Potential: Exploring Hyperparameter Tuning Strategies for Maximum Efficiency"
excerpt: "Optimizing a machine learning model often involves the relentless pursuit of the ideal configuration, a task that can seem never-ending. In this post, I aim to delve deeper into the realm of hyperparameter tuning, focusing specifically on the powerful technique of Bayesian optimization. I'll guide you through understanding how this algorithm operates, its distinct advantages, as well as its inherent limitations. Furthermore, I'll provide practical insights into implementing Bayesian optimization using popular libraries like scikit-optimize and Optuna. Join me on this journey as we uncover the intricacies of hyperparameter tuning and unlock the potential of your machine learning models."
categories: [Data Science]
tags: [hyperparameter tuning, Bayesian optimization, scikit-optimize, optuna, xgboost]
comments: true
image:
  feature: /posts/BayesOpt/Feature_image.jpg
  credit: 
  creditlink: 
---

Optimizing a machine learning model often involves the relentless pursuit of the ideal configuration, a task that can seem never-ending. In this post, I aim to delve deeper into the realm of hyperparameter tuning, focusing specifically on the powerful technique of Bayesian optimization. I'll guide you through understanding how this algorithm operates, its distinct advantages, as well as its inherent limitations. Furthermore, I'll provide practical insights into implementing Bayesian optimization using popular libraries like scikit-optimize and Optuna. Join me on this journey as we uncover the intricacies of hyperparameter tuning and unlock the potential of your machine learning models.

## Model Parameters vs. Hyperparameters
In machine learning we look for a suitable (mathematical) model that can be fitted to training data in order to make predictions about unseen data points. During the fitting the model learns the values of its parameters by optimizing a specific metric (e.g., root mean squared error). Examples of model parameters are the coefficients of linear model, weights in a neural network, or the feature and its split value in a decision tree.

Models can also have hyperparameters. These are not learned durig the training, but instead are configuration settings provided upfront. They control the behavior of the learning algorithm and can thus have a significant impact on model performance. Examples are the learning rate used in gradient descent, the number of hidden layers in a neural network, or the depth of a decision tree.

Hyperparameters are set before model training. They impact the model behavior.
{: .notice}

Tuning the model's hyperparameters can become quite complex due to their number, types (discrete, continuous, categorical, Boolean) and interdependence. Examples of one hyperparameter impacting the value of another are the learning rate and batch size in gradient descent, the number of clusters and distance metric in K-means clustering, or the window size and forecast horizon in time series forecasting.

## Strategies for Hyperparameter tuning
### Grid Search
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

If model training is expensive, testing all possible combinations in the hyperparameter space quickly becomes undesirable or even infeasible. For example, creating a grid for 3 hyperparameters with each 10 values to evaluate already gives 1,000 possible combinations. This is the **curse of dimensionality**. Adding hyperparameters exponentially increases the number of possible configurations.

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
We have now established that there is a need for an algorithm that does not perform an exhaustive search, yet can still explore the hyperparameter space more efficiently than by randomly sampling points. This is where Bayesian optimization comes in. It is used to optimize black-box functions that are expensive to evaluate. The unknown objective function is modelled by a probabilistic surrogate model, often a Guassian process (see below). The model is updated after every iteration to incorporate the new information and determine the next sampling point. The algorithm balances exploration (sampling uncertain regions) and exploitation (sampling regions that likely contain the optimum) in order to find the globab optimum of the objective function in a minimal number of iterations.

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

Setting `n_restarts_optimizer` increases the chance of finding better hyperparameters for the model. When keeping this at the default (Which is 1) I sometimes noticed strange results. Now that the Gaussian process regressor (GPR) has been fitted we can plot the result.

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

The picture shows that the uncertainty is zero at sampled locations and increases when moving away from them. In reality the samples are not entirely noise-free.

Now that we have a way to approximate our objective function, we can decide which position to sample next. This is achieved by the acquisition function which determines the utility of sampling each point in the search space. There are different options to go with. One commonly used is the expected improvement (EI). This method considers both the probability and magnitude of improvement. The model has a parameter $$\xi$$ that balances exploration and exploitation where increasing $$\xi$$ leads to more exploration. The next sampling point is determined by maximizing the EI.

```python
from skopt.acquisition import gaussian_ei

def plot_acquisition(X_grid, gpr, y_opt, xi=0.01):   
    acquisition_values = gaussian_ei(X_grid, gpr, xi=xi, y_opt=y_opt)
    next_sample = X_grid[np.argmax(acquisition_values)] # this is a shortcut that only works because the grid covers all values
    plt.plot(X_grid, acquisition_values, label='Expected Improvement (EI)', color='r')
    plt.axvline(x=next_sample, color='b', linestyle='--', label='Next Sample')
    plt.xlabel('x')
    plt.ylabel('Acquisition Value')
    plt.legend()

    return next_sample

y_opt = min(y_samples)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
next_sample = plot_acquisition(X_grid=X_grid, gpr=gpr, y_opt=y_opt, xi=0.00)
plt.title('xi = 0.00')
plt.subplot(1,2,2)
next_sample = plot_acquisition(X_grid=X_grid, gpr=gpr, y_opt=y_opt, xi=0.01)
plt.title('xi = 0.01')
```

<img src="/img/posts/BayesOpt/ei_iteration_00.png" width="100%">

The above picture shows the effect of $$\xi$$. With $$\xi=0$$ the expected improvement peaks at the minimum of the fitted GPR. This is not the case for $$\xi=0.01$$ where the peak is at the right tail because there is still a lot of uncertainty there.

We can now evaluate this next point, fit the GPR to the new data, recalculate the EI and determine the next most promising sampling position.

```python
def next_iteration(X_grid, X_samples, y_samples, next_sample, gpr, objective, xi=0.01, n_iterations=1):
    X_samples_new = np.append(X_samples, next_sample).reshape(-1, 1)
    y_samples_new = np.append(y_samples, objective(int(next_sample[0])))
    y_opt = min(y_samples)
    gpr.fit(X_samples_new, y_samples_new)

    plt.figure(figsize=(12, 6))

    plt.subplot(1,2,2)
    next_sample = plot_acquisition(X_grid=X_grid, gpr=gpr, y_opt=y_opt, xi=xi)

    plt.subplot(1,2,1)
    plot_gpr(X_grid, X_samples_new, y_samples_new, gpr, next_sample)

    if n_iterations == 1:
        return X_samples_new, y_samples_new, next_sample, gpr
    else:
        return next_iteration(X_grid, X_samples_new, y_samples_new, next_sample,
                              gpr, objective, xi=xi, n_iterations=n_iterations-1)

X_samples, y_samples, next_sample, gpr = next_iteration(X_grid=X_grid, X_samples=X_samples,
                                                        y_samples=y_samples, next_sample=next_sample,
                                                        gpr=gpr, objective=objective, xi=0.01,
                                                        n_iterations=7)
```

The result after 7 iterations (and 3 initial random points) is shown below.

<img src="/img/posts/BayesOpt/bo_iteration_07.png" width="100%">

This shows that our objective functions seems to reach a plateau, something you would expect with the `n_estimators` parameter. Once you have sufficient, adding more estimators does not improve the model performance.

The downside of Bayesian optimization is that it requires the information from the previous iterations. Therefore, it is very difficult to parallelize.

Let's look at some libraries that have Bayesian optimization methods.

#### scikit-optimize
The scikit-optimize package has an API to execute Bayesian optimization which is very intuitive. You define the search space and objective function, and then let `gp_minimize` do the work.

```python
from skopt.space import Integer
from skopt import gp_minimize
from skopt.utils import use_named_args

space = [
    Integer(1, 300, name='n_estimators')
]

@use_named_args(space)
def objective(n_estimators):
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=7)
    mse_scores = -cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(mse_scores)

result = gp_minimize(objective, space, n_calls=10, random_state=7, verbose=True,
                     base_estimator=gpr, acq_func='EI', xi=0.01, n_random_starts=3,
                     noise=1e-10)

gpr.fit(result.x_iters, result.func_vals)
plot_gpr(X_grid=X_grid, X_samples=result.x_iters, y_samples=result.func_vals, gpr=gpr)
```

<img src="/img/posts/BayesOpt/gp_minimize_iteration_07.png" width="50%">


The `noise` parameter allows you to specify the noise level of the observations (the part we ignored above).


#### optuna
Another library that offers Bayesian optimization is optuna. The API is similar to scikit-optimize. There is a possibility to store results that can be visualized afterwards in Optuna Dashboard, a VS code plugin.

```python
import optuna
from optuna.samplers import GPSampler

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 1, 300)
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=7)
    mse_scores = -cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(mse_scores)

study_name = "xgb_n_estimators"
storage = f"sqlite:///./{study_name}.db"
study = optuna.create_study(direction='minimize', sampler=GPSampler(n_startup_trials=3),
                            study_name=study_name, storage=storage)
study.optimize(objective, n_trials=10)
```

Optuna does not seem to have as many configuration options here. The acquisition function is the log expected improvement. There does not seem to be a way to set $$\xi$$. The kernel is a Matern kernel and also this does not seem configurable.

## Conclusion
As the hyperparameter space expands, Bayesian optimization emerges as a compelling choice over other methods like random search or grid search. Its ability to navigate the search space efficiently offers a clear advantage. However, it's essential to acknowledge that this efficiency comes at a cost – the complexity of parallelization increases significantly. Despite this challenge, there is a clear benefit of using Bayesian optimization in optimizing complex machine learning models. By understanding its nuances and leveraging its strengths, practitioners can unlock the full potential of their models and pave the way for groundbreaking advancements in the field of machine learning.


## Resources
- [Liu, P. (2023). Bayesian Optimization: Theory and Practice Using Python (1st ed.). Apress Berkeley, CA. https://doi.org/10.1007/978-1-4842-9063-7](https://doi.org/10.1007/978-1-4842-9063-7)
- [https://krasserm.github.io/2018/03/21/bayesian-optimization/](https://krasserm.github.io/2018/03/21/bayesian-optimization/)
- [https://medium.com/@gerbentempelman/comparing-hyperparameter-optimization-frameworks-in-python-a-conceptual-and-pragmatic-approach-24d9baa1cc69](https://medium.com/@gerbentempelman/comparing-hyperparameter-optimization-frameworks-in-python-a-conceptual-and-pragmatic-approach-24d9baa1cc69)

