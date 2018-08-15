---
layout: post
title: pandas-get_dummies
date: 2018-08-15 00:00:00
img: python/pandas/pandas.png
categories: [python-pandas] 
tags: [python, pandas, get_dummies] # add tag
---


# Pandas.get_dummies

[code link](http://nbviewer.jupyter.org/github/gaussian37/Python/blob/master/Data%20Analysis/pandas/pandas%20-%20get_dummies.ipynb)

[reference](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

Convert categorical variable into dummy/indicator variables.

In a strucrue data, You need to preprocess for training. For example, a feature has cateroty of 'Mon, Tue, ..., Sun'. In that case, those are not propoer to train, because they are not numerical value. Pandas's `get_dummies` function convert categorical variable into  0 / 1 variables. Let's learn it with example !


```python
import numpy as np
import pandas as pd

s = pd.Series(list('abca'))
```

Let me print the result. It's simple!


```python
print(s)
```

: 

    0    a
    1    b
    2    c
    3    a
    dtype: object
    

How about `pd.get_dummies` ? This function converts the categorical variable ('a', 'b', 'c') into 0 / 1 .


```python
print(pd.get_dummies(s))
```

:

       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    

It's the simple concept and useful to feed into machine learning model. Then, let's learn about options. <br>
Below example is about handling of `np.nan`. Without any option, **np.nan** will be ignored.


```python
s1 = ['a', 'b', np.nan]
pd.get_dummies(s1)
```

:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



If you want to consider it, `dummy_na = True` option is necessary.


```python
pd.get_dummies(s1, dummy_na=True)
```
:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



How about below example, feature "A" has ['a', 'b', ''c'] and "B" also has ['a', 'b', 'c'].


```python
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
df
```

:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>c</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



In this case, pandas automatically new feature name with existing one and categorical variable.


```python
pd.get_dummies(df)
```
:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>A_a</th>
      <th>A_b</th>
      <th>B_a</th>
      <th>B_b</th>
      <th>B_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



You can also designate new feature's prefix


```python
pd.get_dummies(df, prefix=['col1', 'col2'])
```
:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>col1_a</th>
      <th>col1_b</th>
      <th>col2_a</th>
      <th>col2_b</th>
      <th>col2_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>