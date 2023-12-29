---
layout: post
title: 행렬의 랭크 (rank)
date: 2021-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [rank, 행렬의 계수, 랭크] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 선형 대수학에서 중요한 개념 중 하나인 `rank` 또는 `행렬의 계수`로 불리는 개념에 대하여 살펴보도록 하겠습니다.

<br>

- `rank`는 행렬이 가지는 `independent`한 `column`의 수를 의미하며 이는 `column space`의 `dimension`에 해당합니다. 
- 예를 들어 `independent`한 `column` 2개로 `span` 하면 2D가 되고 3개로 `span` 하면 3D가 됩니다. 따라서 `independent`한 `column`이 $$ N $$ 개 이면 $$ N $$ `dimension`이 되도록 `span` 할 수 있습니다.
- 따라서 `rank`의 수가 `column space`의 `dimension`이 됩니다.

<br>

- 또한 `rank`의 개념을 이해하면 `independent`한 `column`의 수 = `independent`한 `row`의 수 임을 알 수 있습니다.

<br>

- $$ \text{rank}(A) = \text{rank}(A^{T}) $$

<br>

- 행렬 $$ A $$ 를 `transpose`를 하더라도 `independent`한 `column`의 수는 바뀌지 않습니다. 따라서 `rank`는 `column space`의 `dimension`임과 동시에 `row space`의 `dimension`이 됩니다.

<br>

- 아래 예제를 살펴보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \end{bmatrix} $$

<br>

- 행렬 $$ A $$ 의 `independent`한 `column`의 갯수는 1개 입니다. 따라서 $$ \text{rank}(A) = 1 $$ 이 됩니다.

<br>

- $$ B = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} $$

<br>

- 행렬 $$ B $$ 의 첫번째 두번째 `column`으로 세번쨰 `column`을 만들 수 있기 때문에 독립적인 `column`은 2개이며 $$ \text{rank}(B) = 2 $$ 가 됩니다.
- 따라서 `row space` 또는 2차원이 되는데, `row` 벡터의 element의 수는 3개 입니다. 이런 경우는 3차원 `row space` 안에서 2차원 만큼만 `span`할 수 있다는 것을 의미합니다.

<br>

- 행렬 $$ B $$ 와 같은 직사각형 행렬에서는 행/열 중 차원을 가지는 차원 만큼 최대 `rank`를 가질 수 있습니다. 행렬 $$ B $$ 는 2 X 3 행렬이므로 최대 `rank`는 2가 될 수 있습니다.
- 행렬 $$ B $$ 의 최대 `rank`는 2이고 실제 `rank` 또한 2인 경우 `full row rank` 라고 말합니다.

<br>

- 행렬 $$ A $$ 와 같은 경우 최대 `rank`는 2이지만 실제 `rank`는 1이었습니다. 이런 경우 `rank-deficient`라고 말합니다.

<br>

- 앞의 내용을 응용하여 만약 어떤 행렬의 크기가 3 X 2 이고 `rank`가 2인 경우 `full column rank` 라고 말할 수 있습니다.

<br>

- 정사각 행렬의 경우를 예시로 살펴 보겠습니다. 정사각 행렬의 크기가 3 X 3 이고 `rank`가 3인 경우 `full rank` 말합니다. 반면  정사각 행렬의 크기가 3 X 3 이고 `rank`가 2인 경우 `rank-deficient`라고 말하며 이는 직사각 행렬의 경우와 같습니다.


<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>
