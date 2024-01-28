---
layout: post
title: LU 분해와 Cholesky 분해
date: 2017-01-01 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, LU 분해, Cholesky 분해] # add tag
---

<br>

[선형대수학 관련 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 선형대수학에서 많이 사용되는 `LU 분해`와 `LU 분해`의 특수 형태인 `Cholesky 분해`에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [LU 분해](#lu-분해-1)
- ### [LDU 분해](#ldu-분해-1)
- ### [대칭행렬의 LDU 분해](#대칭행렬의-ldu-분해-1)
- ### [Cholesky 분해](#cholesky-분해-1)

<br>

## **LU 분해**

<br>

- 어떤 행렬 $$ A $$ 가 주어졌을 때, $$ A $$ 를 2개 이상의 행렬의 곱으로 나타내는 것을 행렬의 분해라고 합니다. 많이 쓰는 용어로는 `factorization` 또는 `decomposition` 이라고 합니다.
- 만약 어떤 행렬 $$ A $$ 를 다음과 같이 분해하였을 때, 이를 `LU 분해`라고 합니다.

<br>

- $$ A = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ j & 1 & 0 \\ k & l & 1 \end{bmatrix} \begin{bmatrix} m & n & o \\ 0 & p & q \\ 0 & 0 & r \end{bmatrix} = LU $$

<br>

- $$ L = \begin{bmatrix} 1 & 0 & 0 \\ j & 1 & 0 \\ k & l & 1 \end{bmatrix} $$

- $$ U = \begin{bmatrix} m & n & o \\ 0 & p & q \\ 0 & 0 & r \end{bmatrix} $$

<br>

- 위 식에서 $$ L $$ 은 `하삼각 행렬 (lower triangle matrix)`를 의미하며 대각선 원소는 모두 1이 됩니다. $$ U $$ 는 `상삼각 행렬 (upper triangle matrix)`라고 합니다.

<br>

- `LU 분해`는 가우스 소거법을 통해 도출할 수 있습니다. $$ 3 \times 3 $$ 행렬의 예시를 통해 가우스 소거법을 통한 `LU` 분해를 진행해 보도록 하겠습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 3 & 8 & 14 \\ 2 & 6 & 13 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 13 \\ 4 \end{bmatrix} $$

<br>

- 가우스 소거법을 적용해 보도록 하겠습니다. 위 행렬에서 ① 1 행을 3배를 한 뒤 2행에서 빼고 ② 1행의 2배를 해서 3행에서 빼주면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- `① 1 행을 3배를 한 뒤 2행에서 빼는 것`과 `② 1행의 2배를 해서 3행에서 빼는 것`을 행렬식으로 표현하면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} $$

<br>

- 따라서 앞에서 가우스 소거법을 적용한 것은 위 행렬식을 좌/우변에 곱하는 것과 같은 작업입니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 3 & 8 & 14 \\ 2 & 6 & 13 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 13 \\ 4 \end{bmatrix} $$

<br>

- $$ \Rightarrows \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- 이 때, 여기서 사용된 행렬은 이후에 `LU` 분해 시 사용됩니다.

<br>

- $$ \color{red}{\begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix}} $$

<br>

- 가우스 소거법을 계속 진행해 보도록 하겠습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$



<br>

## **LDU 분해**

<br>

<br>

## 대칭행렬의 LDU 분해

<br>


<br>

## **Cholesky 분해**

<br>

<br>




<br>

[선형대수학 관련 글 목차](https://gaussian37.github.io/math-la-table/)

<br>