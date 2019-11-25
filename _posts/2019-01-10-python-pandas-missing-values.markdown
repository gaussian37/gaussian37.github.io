---
layout: post
title: Numpy와 Pandas로 결측값 처리하기
date: 2019-01-11 00:00:00
img: python/pandas/pandas.png
categories: [python-basic] 
tags: [python, pandas, missing value, 결측값] # add tag
---

Structured data에서 결측값(Missing Value)를 제거하는 방법에 대하여 알아보겠습니다.

<br>

- 먼저 라이브러리를 입력합니다.

<br>

```python
# Load libraries
import numpy as np
import pandas as pd
```

<br>

## Numpy에서 결측값 제거하기

<br>

- numpy로 임의의 행렬을 만들어 보겠습니다.
- 이 때, `np.nan`이 결측값 입니다. 엑셀 시트에서 값이 아무것도 없는 셀에 속합니다.

<br>

```python
X = np.array([[1.1, 11.1], 
              [2.2, 22.2], 
              [3.3, 33.3], 
              [4.4, 44.4], 
              [np.nan, 55]])
```

- 결측값이 있는 행은 제거해 줍니다. 일반적으로 행을 기준으로 새로운 데이터가 추가 되므로 행을 제거합니다.

<br>

```python
>> X[~np.isnan(X).any(axis=1)]

array([[  1.1,  11.1],
       [  2.2,  22.2],
       [  3.3,  33.3],
       [  4.4,  44.4]])

```

<br>

## Pandas에서 결측값 제거하기

<br>

+ 행렬을 만들어 줍니다.

```python
X = np.array([[1, 2], 
              [6, 3], 
              [8, 4], 
              [9, 5], 
              [np.nan, 4]])
```

+ Numpy로 만든 행렬을 Pandas 타입으로 바꿔 줍니다.
+ `df.dropna()`를 통하여 결측값을 제거합니다.

```python
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])

# 결측값을 제거해 줍니다.
df.dropna()
```

<br>

## Pandas에서 결측값 인덱스 받기

<br>

- 임의의 DataFrame을 `NaN` 값을 포함하여 만든 후 NaN의 값이 있는 좌표만 따로 반환 받습니다.

<br>

```python
X = np.array([[1, 2], 
              [6, 3], 
              [8, 4], 
              [9, 5], 
              [np.nan, 4]])
              
df = pd.DataFrame(X)

print( np.asarray(df.isnull()).nonzero() )

>> (array([4], dtype=int64), array([0], dtype=int64))
```

<br>
