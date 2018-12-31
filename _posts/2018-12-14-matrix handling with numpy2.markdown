---
layout: post
title: Numpy를 이용한 행렬 다루기 2탄
date: 2018-12-14 00:00:00
img: math/la/matrix-handling-with-numpy/numpy-logo.png
categories: [math-la] 
tags: [Linear algebra, Numpy] # add tag
---

이번 글에서는 Numpy를 이용할 때 가장 기본이 되는 기본중의 기본 연산에 대하여 한번 정리해 보려고 합니다.
한번 쭉쭉 읽으시면 도움이 되실거라 생각됩니다. 그럼 정리해 보겠습니다.

## 행렬 Flatten

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# 행렬 Flatten   
matrix.flatten()

```

<br>

## 행렬의 Rank 찾기

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# 행렬의 Rank 찾기      
np.linalg.matrix_rank(matrix)
# 결과 : 2
```

<br>

## 행렬의 최댓값과 최솟값 찾기

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 행렬 전체에서 최댓값 찾기     
np.max(matrix)
# 결과는 9

# 행렬 전체에서 최솟값 찾기
np.min(matrix)
# 결과는 1

# 열 기준으로 최댓값 찾기
np.max(matrix, axis=0)
# 결과는 array([7, 8, 9])

# 행 기준으로 최댓값 찾기
np.max(matrix, axis=1)
# 결과 : array([3, 6, 9])

```

<br>

## 행렬의 정보 확인

```python
import numpy as np

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
                   
# 행렬의 행과 열 확인       
matrix.shape
# (3, 4)

# 전체 사이즈 확인 (행 x 열)
matrix.size
# 12

# Dimension 확인
matrix.ndim
# 2

```

## 벡터 만등기

```python
import numpy as np

# 행 벡터 만들기
vector_row = np.array([1, 2, 3])

# 열 벡터 만들기
vector_column = np.array([[1],
                          [2],
                          [3]])
```