---
layout: post
title: Predicting House Prices
excerpt: 
categories: [Data Science]
tags: [scikit-learn, regression, Kaggle]
comments: true
image:
  feature: 
  credit: 
  creditlink: 
---

## Loading the Data and Initial Exploration
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

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 79 columns):
    #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
    0   MSSubClass     1460 non-null   int64  
    1   MSZoning       1460 non-null   object 
    2   LotFrontage    1201 non-null   float64
    3   LotArea        1460 non-null   int64  
    4   Street         1460 non-null   object 
    ...  ...            ...            ...   
    74  MiscVal        1460 non-null   int64  
    75  MoSold         1460 non-null   int64  
    76  YrSold         1460 non-null   int64  
    77  SaleType       1460 non-null   object 
    78  SaleCondition  1460 non-null   object 
    dtypes: float64(3), int64(33), object(43)
    memory usage: 901.2+ KB


```python
mask = X_train.isna().sum()*100/X_train.shape[0] > 10
X_train.columns[mask]
```

    Index(['LotFrontage', 'Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence',
           'MiscFeature'],
          dtype='object')

We can see that LotFrontage, Alley, MasVnrType, FireplaceQu, PoolQC, Fence and MiscFeature have more than 10% missing values. We will drop these features in our analysis.

```python
X.drop(X.columns[mask].tolist(), axis=1, inplace=True)
```

Let's also check if we have a label for all observations.

```python
sum(y.isna())
```
    0

We can now split our data into a training and validation set. For this we use `train_test_split` from scikit-learn. From here on we will continue with the training set and set aside the validation set.

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                test_size=0.25,
                                                random_state=5)
```

We can have a look at how the distributions of our features look, as well as of our label.

```python
X_train.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
```

<img src="/img/posts/Predicting_house_prices/histogram_features.png" width="100%" height="600">

```python
y_train.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
```

<img src="/img/posts/Predicting_house_prices/histogram_label.png" width="25%" height="auto">


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




