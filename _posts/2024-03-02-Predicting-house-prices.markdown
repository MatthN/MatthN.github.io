---
layout: post
title: Predicting House Prices
excerpt: ""
categories: [Data Science, Regression]
tags: [scikit-learn, regression, Kaggle]
comments: true
image:
  feature: /posts/Predicting_house_prices/Feature_image.png
  credit: 
  creditlink: 
---

## Loading the Data and Initial Exploration
The data that we will be working with is available on Kaggle [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
We begin by loading the data and setting aside the label in a separate variable `y`.

```python
import pandas as pd

data_dir = r"path\to\your\data\\"
X = pd.read_csv(data_dir + "train.csv")

y = X['SalePrice']
X.drop(['SalePrice', 'Id'], axis=1, inplace=True)

```

We will start with some basic checks first to see what data types we have and how complete the data is.

```python
X.info()
```

    >>> <class 'pandas.core.frame.DataFrame'>
    >>> RangeIndex: 1460 entries, 0 to 1459
    >>> Data columns (total 79 columns):
    >>> #   Column         Non-Null Count  Dtype  
    >>> ---  ------         --------------  -----  
    >>> 0   MSSubClass     1460 non-null   int64  
    >>> 1   MSZoning       1460 non-null   object 
    >>> 2   LotFrontage    1201 non-null   float64
    >>> 3   LotArea        1460 non-null   int64  
    >>> 4   Street         1460 non-null   object 
    >>> ...  ...            ...            ...   
    >>> 74  MiscVal        1460 non-null   int64  
    >>> 75  MoSold         1460 non-null   int64  
    >>> 76  YrSold         1460 non-null   int64  
    >>> 77  SaleType       1460 non-null   object 
    >>> 78  SaleCondition  1460 non-null   object 
    >>> dtypes: float64(3), int64(33), object(43)
    >>> memory usage: 901.2+ KB


```python
mask = X_train.isna().sum()*100/X_train.shape[0] > 10
X_train.columns[mask]
```

    >>> Index(['LotFrontage', 'Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence',
    >>>        'MiscFeature'],
    >>>       dtype='object')

We can see that LotFrontage, Alley, MasVnrType, FireplaceQu, PoolQC, Fence and MiscFeature have more than 10% missing values. We will drop these features in our analysis.

```python
X.drop(X.columns[mask].tolist(), axis=1, inplace=True)
```

Let's also check if we have a label for all observations.

```python
sum(y.isna())
```
    >>> 0

We can now split our data into a training and validation set. For this we use `train_test_split` from scikit-learn. From here on we will continue with the training set and set aside the validation set.

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                test_size=0.25,
                                                random_state=5)
```

We can have a look at how the distributions of our features look, as well as of our label.

```python
y_train.hist(bins=50, xlabelsize=8, ylabelsize=8)
plt.xlabel('House price')
plt.ylabel('Count')
plt.title('Histogram of house prices')
plt.show()
```

<img src="/img/posts/Predicting_house_prices/histogram_features.png" width="100%" height="600">

```python
y_train.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
```

<img src="/img/posts/Predicting_house_prices/histogram_house_prices.png" width="25%" height="auto">


We can see that a lot of the distributions are skewed. Our house prices have a long tail towards more expensive houses. The same can be seen for the grond living area. The total basement square footage has a peak at 0, because not all houses have a basement. This is also the case for garage area, open porch square footage, wood deck square footage, and others. This may complicate prediction for simple methods such as linear regressions.

Finally, let's have a look at the correlation between features, and at the correlation of each feature with the label.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=False,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=.5,
            square=True)

plt.title('Correlation Matrix Heatmap')
plt.show()
```

<img src="/img/posts/Predicting_house_prices/heatmap_correlations.png" width="75%" height="auto">

We can see that the overall quality of the house has the highest positive correlation with its price. We can also see that GarageCars (the number of cars that fit in the garage) strongly correlates with the garage area, or that the year the garage was built correlates with the year the house was built. All these things make sense intuitively.


## Setting Up a Preprocessing Pipeline
All features in the dataset have a description in `data_description.txt`. Based on their description, I divided the features into
- **numerical.** Measurable quantities represented by numerical values (integers or floats).
- **categorical.** Features with a discrete number of possible values that do not have any particular order or hierarchy. We will apply one-hot encoding to these features.
- **ordinal.** Categorical features where there exists a meaningful order or ranking among the categories, although the distance between the categories is not necessarily known or meaningful. These categories can be encoded by integers. For some of these features that is already the case, for others this still needs to be done.

For each of the above we will define specific preprocessing steps. For the numerical features we first impute missing values using the `IterativeImputer`. Missing values are predicted based on the other features by Bayesian ridge regression (check [this](https://youtu.be/Z6HGJMUakmc?si=BCVO5VJchfKiRGM7) YouTube link for a good explanation on the subject). After that standardize the features using `StandardScaler`.

For categorical features we impute with `SimpleImputer` applying the *most_frequent* strategy. This will replace missing values with the most prevalent category in the training set for that feature. Each feature is then one-hot encoded with `OneHotEncoder`. To prevent that this pipeline fails on a category value it has not seen in the training data we pass the `handle_unknown='ignore'` argument.

For our ordinal features we have to make the distinction between those that are already encoded and those which are not. The ones that still require encoding can be handled with `OrdinalEncoder`. We provide it a list to specify the order of the values. Missing values in those features will receive encoded value -1. For those already encoded we use `SimpleImputer` to replace missing values with -1.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

numerical_pipe = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=10, random_state=5)),
    ('transform', StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_encode_pipe = Pipeline(steps=[
    ('encode', OrdinalEncoder(categories=ORDINAL_CATEGORICAL_ORDER,
                              handle_unknown='use_encoded_value',
                              unknown_value=-1))
])

ordinal_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipe, NUMERICAL),
    ('cat', categorical_pipe, CATEGORICAL),
    ('ord', ordinal_pipe, ORDINAL),
    ('ord_enc', ordinal_encode_pipe, ORDINAL_CATEGORICAL)
])
```

## Linear Regression
Earlier we saw that there is a strong correlation of the sales price with the overall quality of the house. Given this linear releationship we can do a first attempt with linear regression and see where that gets us. Given the large number of features in our data set we will make use of some regularization. In this case we can try Lasso regression which applies L1 regularization which tends to drive coefficients of unimportant features to 0. This can give us a first impression of which features are actually important.
We will use `LassoCV` which will determine the regularization penalty *alpha* automatically via cross-validation.

```python
from sklearn.linear_model import LassoCV

lm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LassoCV(cv=5, max_iter=10_000))
])
lm.fit(X=X_train, y=y_train)
```

Let's have a look at which features got non-zero coefficients.

```python
coef = lm.named_steps['lm'].coef_
feature_names = lm.named_steps['preprocessor'].get_feature_names_out()
features = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef
})
features.sort_values('coefficient', ascending=False)[abs(features['coefficient']) > 0]
```

  |      | feature                | coefficient |
  |------|------------------------|-------------|
  | 208  | ord__OverallQual       | 20159.02    |
  | 8    | num__GrLivArea         | 10053.32    |
  | 218  | ord__TotRmsAbvGrd      | 6242.16     |
  | 5    | num__1stFlrSF          | 5925.16     |
  | 227  | ord_enc__BsmtExposure  | 2166.85     |
  | 1    | num__BsmtFinSF1        | 1684.50     |
  | 4    | num__TotalBsmtSF       | 1235.24     |
  | 9    | num__GarageArea        | 933.87      |
  | 210  | ord__YearBuilt         | 443.36      |
  | 211  | ord__YearRemodAdd      | 436.62      |
  | 220  | ord__GarageYrBlt       | 6.25        |

As expected the overall quality got a large positive coefficient. Do note that even though GarageArea gets a non-zero coefficient, GarageCars does not. This is because when dealing with highly correlated features, Lasso tends to select one and shrink the coefficients of the others to zero.

We will use the RMSE evaluation metric, since this is the one that the Kaggle competition looks at.

```python
from sklearn.metrics import root_mean_squared_error
y_pred = lm.predict(X=X_val)
print(root_mean_squared_error(y_true=y_val, y_pred=y_pred))

min_val = min(min(y_val), min(y_pred))
max_val = max(max(y_val), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], linestyle='-', color='r')
plt.scatter(x=y_val, y=y_pred, alpha=0.25)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Lasso regression - actual vs. predicted')
plt.show()
```

    >>> 38632.28843759117

<img src="/img/posts/Predicting_house_prices/lasso_actual_vs_predicted.png" width="50%" height="auto">

Looking at the plot of the actual versus predicted price, it is obvious that we are making too low predictions for the most expensive houses. It seems that only a part of the entire range can be represented using a linear approach.


## Random Forest Regression
Given that our data does not seem to follow a linear trend across its entire range, we can try using a random forest regression model. This is a type of ensemble model that can handle non-linear releationships, and can capture interactions between features automatically. Even though it is very powerful, it is still quite easy to use and it can provide insight into the importance of each feature. First we will use the `GridSearchCV` method to search for the optimal value of *n_estimators*, the number of decision trees used. `GridSearchCV` will fit the model with every parameter value defined in our parameter grid and evaluate its performance by cross-validation. Notice that since we are using a pipeline we need to prepend our paramter name with our stepname and two underscores.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=5))
])

param_grid = {
    'rf__n_estimators': range(100, 1001, 100)
}

rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid,
                       cv=5, scoring='neg_mean_squared_error',
                       n_jobs=-1, verbose=2)
rf_grid.fit(X=X_train, y=y_train)
```

We can plot the cross validation scores for each parameter value.

```python
mean_test_scores = -rf_grid.cv_results_['mean_test_score']
n_estimators_values = [params['rf__n_estimators'] for params in rf_grid.cv_results_['params']]

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, mean_test_scores, marker='o', linestyle='-')
plt.title('CV Mean Test Score vs. Number of Trees (n_estimators)')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('CV Mean MSE')
plt.show()
```

<img src="/img/posts/Predicting_house_prices/random_forest_cv_parameters.png" width="50%" height="auto">

The best score was obtained by using 900 decision trees. Yet, we can see that our curve flattens off after 700 trees. Thus, we will use that value for our finetuned model.

```python
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=5, n_estimators=700))
])

rf.fit(X=X_train, y=y_train)
```

We can now evaluate the model the same way we did for the linear one above.

```python
y_pred = rf.predict(X=X_val)
print(root_mean_squared_error(y_true=y_val, y_pred=y_pred))

min_val = min(min(y_val), min(y_pred))
max_val = max(max(y_val), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], linestyle='-', color='r')
plt.scatter(x=y_val, y=y_pred, alpha=0.25)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random forest regression - actual vs. predicted')
plt.show()
```

    >>> 26025.936828384132

<img src="/img/posts/Predicting_house_prices/rf_actual_vs_predicted.png" width="50%" height="auto">

The random forest regressor is performing much better than the lasso model. Still we can see that the model has difficulties with the most expensive houses where the predicted values are too low. On the other end of the spectrum the cheapest houses seem to be consistently predicted too high. This seems like a good moment to dive a bit deeper into the data to understand this better.

## Deep Dive in the Data
For the deep dive I have imported the data in Power Bi. This allows for easy drill down.
Looking at the house price versus the ground living area shows that generally the price goes up as the house gets bigger (makes sense!).

<img src="/img/posts/Predicting_house_prices/powerbi_price_vs_grlivarea.png" width="50%" height="auto">

For houses with an overall quality of 10 there seem to be two clear exceptions to this rule which may pull that trend downards for that category. Zooming in on this we see that houses of the highest quality typically sell at a premium above the average price in the neighborhood.

<img src="/img/posts/Predicting_house_prices/powerbi_price_vs_neighborhood.png" width="100%" height="auto">

This is also the case in the Edwards neighborhood, but much less pronounced. The two outlying observations are both from this neighborhood. Meanwhile the two most expensive houses that get a much too low prediction are both from the NoRidge neighborhood where this effect is much more pronounced.

On the low side we see that there are only a few observations of overall quality 1 or 2. The low prevalence of these groups could explain the bad performance.

## Oversampling
To address the issues described above we will try to increase the representation of these groups in our training data by over-sampling them. We'll add some noise to the numerical variables too.

```python
import numpy as np

def add_additional_samples(X, y, OveralQual=[1,2,10], times=1,
                           noise_percentage=5, cols=NUMERICAL,
                           random_state=None):
    
    rng = np.random.RandomState(random_state)

    mask = X['OverallQual'].isin(OveralQual)
    X_sampled = X[mask].copy()
    y_sampled = y[mask].copy()

    for _ in range(times - 1):
        X_sampled = pd.concat([X_sampled, X[mask].copy()], axis=0, ignore_index=True)
        y_sampled = pd.concat([y_sampled, y[mask].copy()], axis=0, ignore_index=True)

    for col in cols:
        col_std = X_sampled[col].std()
        noise_std = col_std * noise_percentage / 100
        noise = rng.normal(0, noise_std, size=X_sampled[col].shape)

        if X_sampled[col].min() >= 0:
            adjusted_noise = np.where(X_sampled[col] + noise < 0, -X_sampled[col], noise)
        else:
            adjusted_noise = noise
        
        X_sampled[col] += adjusted_noise

    X_augmented = pd.concat([X, X_sampled], axis=0, ignore_index=True)
    y_augmented = pd.concat([y, y_sampled], axis=0, ignore_index=True)
    
    return X_augmented, y_augmented
    

X_train_oversampled, y_train_oversampled = add_additional_samples(X=X_train, y=y_train,
                                                                  times=3)
```

Now let's retrain our model and evaluate once more.

```python
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=5, n_estimators=700))
])

rf.fit(X=X_train_oversampled, y=y_train_oversampled)

y_pred = rf.predict(X=X_val)
print(root_mean_squared_error(y_true=y_val, y_pred=y_pred))

min_val = min(min(y_val), min(y_pred))
max_val = max(max(y_val), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], linestyle='-', color='r')
plt.scatter(x=y_val, y=y_pred, alpha=0.25)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random forest regression - actual vs. predicted')
plt.savefig('rf_sampled_actual_vs_predicted.png', dpi=300)
```

    >>> 24447.68667822811

<img src="/img/posts/Predicting_house_prices/rf_sampled_actual_vs_predicted.png" width="50%" height="auto">

We can see that this improves the prediction somewhat, but not a whole lot.


## Stacking





