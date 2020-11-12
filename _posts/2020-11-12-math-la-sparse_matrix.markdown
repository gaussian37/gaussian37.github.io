---
layout: post
title: sparse matrix (희소 행렬)
date: 2020-11-12 00:00:00
img: math/la/sparse_matrix/0.png
categories: [math-la] 
tags: [Linear algebra, vector, projection, sparse matrix, 희소 행렬] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 참조 : https://en.wikipedia.org/wiki/Sparse_matrix#:~:text=The%20compressed%20sparse%20row%20(CSR,row%20indices%2C%20hence%20the%20name.
- 이번 글에서는 sparse matrix의 개념에 대하여 알아보도록 하겠습니다.

<br>

- `sparse matrix` 또는 `sparse array`는 행렬의 대부분의 요소가 0인 행렬입니다. 반대로 대부분의 요소가 0이 아니면 행렬은 `dense`하다고 표현합니다. 값이 0 인 갯수를 전체 값의 총 수로 나눈 값 (예 : m × n 행렬의 경우 m × n)을 행렬의 희소성 또는 `sparsity` 라고합니다.

<br>

- $$ \begin{pmatrix} 11 & 22 & 0 & 0 & 0 & 0 & 0 \\ 0 & 33 & 44 & 0 & 0 & 0 & 0 \\ 0 & 0 & 55 & 66 & 77 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 88 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 99 \end{pmatrix} $$

<br>

- 예를 들어 위 sparse matrix의 경우 오직 9개의 값만 0이 아니고 26개는 모두 0입니다. 따라서 sparsity = 76 % 가 되고 density = 24 %가 됩니다.

<br>

- sparse matrix는 대부분의 행렬의 값이 0이기 때문에 연산 시 필요없는 부분이 상당히 많아집니다. 따라서 행렬 연산에 수많은 0을 연산하는 것은 비효율적인 뿐 아니라 계산에서도 비효율적입니다.
- 따라서 sparse matrix를 좀 더 효율적으로 관리하기 위한 방법이 필요합니다. sparse matrix를 저장하기 위한 다양한 방법이 존재하는데 그 중 `CSR(Compressed Sparse Row)` 또는 `CRS(Compressed Row Storage)` 또는 `Yale format`으로 불리는 저장 방법에 대하여 다루어 보겠습니다.
- 이 방법은 **3개의 벡터를 이용하여 행렬을 표현**합니다. 각 벡터를 `V`, `COL_INDEX`, `ROW_INDEX` 라고 부르겠습니다.



<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

