---
layout: post
title: Overview Random Forest  
date: 2018-10-08 08:42:00
img: ml/concept/regression/Regression_print.png
categories: [ml-concept] 
tags: [machine learning, regression] # add tag
---

Your goal will be to use this data to predict the life expectancy in a given country based on features such as the country's GDP, fertility rate, and population.

As always, it is important to explore your data before building models. 
we have constructed a heatmap showing the correlation between the different features of the Gapminder dataset, which has been pre-loaded into a DataFrame as `df`

![heatmap](../assets/img/ml/concept/regression/heatmap.png)

Cells that are in green show positive correlation, while cells that are in red show negative correlation.

```python
### Import packages
import numpy as np
import pandas as pd
import seaborn as sns

# Read the CSV file into a DataFrame : df & check it
df = pd.read_csv("./data/gm_2008_region.csv")
df.head()

df.info()

df.describe()

# computes the pairwise correlation between columns
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
```
<br>

Since the target variable here is quantitative, this is a regression problem. To begin, you will fit a linear regression with just one feature: `'fertility'`
, which is the average number of children a woman in a given country gives birth to. In later exercises, you will use all the features to build regression models.
Before that, however, you need to import the data and get it into the form needed by scikit-learn. 
This involves creating feature and target variable arrays. Furthermore, since you are going to use only one feature to begin with, you need to do some reshaping using NumPy's `.reshape()` method.

$ a_{1} $






