---
layout: post
title: Numpy를 이용한 행렬 다루기 1탄
date: 2018-12-13 00:00:00
img: ml/la/matrix-handling-with-numpy/numpy-logo.png
categories: [ml-la] 
tags: [Linear algebra, Numpy] # add tag
---

이번 글에서는 Numpy를 이용할 때 가장 기본이 되는 기본중의 기본 연산에 대하여 한번 정리해 보려고 합니다.
한번 쭉쭉 읽으시면 도움이 되실거라 생각됩니다. 그럼 정리해 보겠습니다.

## Transpose 연산

```python
# Load library
import numpy as np

# 벡터 생성
vector = np.array([1, 2, 3, 4, 5, 6])

# 행렬 생성
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# Tranpose 
vector.T                   

# Transpose 
matrix.T
```

<br>

## Array 에서 element 선택

```python
# Load library
import numpy as np

# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# 벡터에서 원소를 선택하는 방법
vector[1]

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# 행렬에서 원소를 선택하는 방법
matrix[1,1]


# Create matrix
tensor = np.array([
                    [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
                    [[[3, 3], [3, 3]], [[4, 4], [4, 4]]]
                  ])
                  
                  
# 3차원 이상의 텐서에서 원소를 선택하는 방법
tensor[1,1,1]

```

<br>

## 행렬 shape 바꾸기 (reshape)

```python
# Load library
import numpy as np

# 4x3 행렬 생성
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
                   
# 행렬 reshape
matrix.reshape(2, 6)
```

<br>

## 역행렬 만들기

```python
import numpy as np

matrix = np.array([[1, 4],
                   [2, 5]])
                   
# 역행렬 생성 : linalg.inv
np.linalg.inv(matrix)

```

<br>

## 행렬의 대각 성분 구하기

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# 행렬의 대각 성분을 구합니다.
matrix.diagonal()

# 행렬의 Trace를 구합니다.
matrix.diagonal().sum()

```