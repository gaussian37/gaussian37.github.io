---
layout: post
title: Numpy를 이용한 행렬 다루기 3탄
date: 2018-12-15 00:00:00
img: ml/la/matrix-handling-with-numpy/numpy-logo.png
categories: [ml-la] 
tags: [Linear algebra, Numpy] # add tag
---

이번 글에서는 Numpy를 이용할 때 가장 기본이 되는 기본중의 기본 연산에 대하여 한번 정리해 보려고 합니다.
한번 쭉쭉 읽으시면 도움이 되실거라 생각됩니다. 그럼 정리해 보겠습니다.

## Sparse 행렬 만들기

```python
import numpy as np
from scipy import sparse

# Dense matrix 형태를 먼저 만듭니다.
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

# Sparse Row (CSR) matrix로 변환합니다.
matrix_sparse = sparse.csr_matrix(matrix)

:

  (1, 1)        1
  (2, 0)        3

```

<br>

## Numpy 행렬 만들기

```python
import numpy as np

# np.array를 통하여 행렬 만들기
matrix = np.array([[1, 4],
                   [2, 5]])

```

<br>

## Dictionary 를 Matrix로 만들기

```python
from sklearn.feature_extraction import DictVectorizer


data_dict = [{'Red': 2, 'Blue': 4},
             {'Red': 4, 'Blue': 3},
             {'Red': 1, 'Yellow': 2},
             {'Red': 2, 'Yellow': 2}]
             
# DictVectorizer 객체 생성
dictvectorizer = DictVectorizer(sparse=False)

features = dictvectorizer.fit_transform(data_dict)

>> features

array([[ 4.,  2.,  0.],
       [ 3.,  4.,  0.],
       [ 0.,  1.,  2.],
       [ 0.,  2.,  2.]])

# feature의 열 이름 확인
>> dictvectorizer.get_feature_names()

['Blue', 'Red', 'Yellow']

```

<br>

## 행렬의 Trace 계산

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
>> matrix.diagonal().sum()

15

```

<br>

## 행렬의 Determinant 계산

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
>> np.linalg.det(matrix)

-9.5161973539299405e-16

```

<br>

## 행렬의 평균/분산/표준편차 계산

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 전체 행렬의 평균
np.mean(matrix)

# 전체 행렬의 분산
np.var(matrix)

# 전체 행렬의 표준 편차
np.std(matrix)
```

<br>

## 두 벡터의 dot product

```python
import numpy as np

vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

# 첫 번째 dot product
np.dot(vector_a, vector_b)

# 두 번째 dot product
vector_a @ vector_b

```

<br>

## vectorized function을 이용하여 원소에 연산 적용

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# 함수 생성
add_10 = lambda i: i + 10

# vectorized function 생성
vectorized_add_10 = np.vectorize(add_10)

# vectorized function 적용
>> vectorized_add_100(matrix)

array([[101, 102, 103],
       [104, 105, 106],
       [107, 108, 109]])
                   


```