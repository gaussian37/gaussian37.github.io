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

- 가우스 소거법을 적용해 보도록 하겠습니다. 위 행렬에서 ① 1 행을 3배를 한 뒤 2행에서 빼고 (2행 = 2행 - 3 x 1행) ② 1행의 2배를 해서 3행에서 빼주면 (3행 = 3행 - 2 x 1행) 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- 이 때, `① 1 행을 3배를 한 뒤 2행에서 빼는 것`과 `② 1행의 2배를 해서 3행에서 빼는 것`을 행렬식으로 표현하면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} $$

<br>

- 따라서 앞에서 가우스 소거법을 적용한 것은 위 행렬식을 좌/우변에 곱하는 것과 같은 작업입니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 3 & 8 & 14 \\ 2 & 6 & 13 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 13 \\ 4 \end{bmatrix} $$

<br>

- $$ \Rightarrow \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- 이 때, 여기서 사용된 행렬은 이후에 `LU` 분해 시 사용됩니다.

<br>

- $$ \color{red}{\begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix}} $$

<br>

- 가우스 소거법을 계속 진행해 보도록 하겠습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- 이번에는 3행에서 2행을 빼면 (3행 = 3행 - 2행) 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -6 \end{bmatrix} $$

<br>

- 이 결과는 다음 행렬을 곱한 것과 같습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} $$

<br>

- 따라서 다음과 같이 해석할 수 있습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \\ -2 \end{bmatrix} $$

<br>

- $$ \Rightarrow \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \\ -6 \end{bmatrix} $$

<br>

- 이 때, 가우스 소거법을 정의하는 행렬을 표시하면 다음과 같습니다.

<br>

- $$ \color{blue}{\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}} $$

<br>

- 가우스 소거법을 통해 정의된 빨간색과 파란색 행렬을 순서대로 곱하면 다음과 같습니다.

<br>

- $$ \color{blue}{\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}} \color{red}{\begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ -2 & 0 & 1 \end{bmatrix}} = \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 1 & -1 & 1 \end{bmatrix} $$

<br>

- 따라서 다음과 같이 위 행렬을 이용하면 다음과 같이 상삼각행렬을 구할 수 있습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 1 & -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 3 & 8 & 14 \\ 2 & 6 & 13 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} $$

<br>

- $$ \Rightarrow \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 1 & -1 & 1 \end{bmatrix} A = \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} $$

<br>

- 위 식에서 좌변의 행렬에 역행렬을 곱하여 우변으로 넘기면 다음과 같습니다.

<br>

- $$ A = \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 1 & -1 & 1 \end{bmatrix}^{-1} \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} $$

<br>

- 이 때, 하삼각 행렬의 역행렬은 다음과 같이 정형화 할 수 있습니다.

<br>
<center><img src="../assets/img/math/la/lu_and_cholesky_decomposition/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- $$ \begin{bmatrix} 1 & 0 & 0 \\ a & 1 & 0 \\ b & c & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ -a & 1 & 0 \\ ac-b & -c & 1 \end{bmatrix} $$

<br>

- 따라서 다음과 같이 정리할 수 있습니다.

<br>

- $$ \begin{align} A &= \begin{bmatrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 1 & -1 & 1 \end{bmatrix}^{-1} \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} \\ = \begin{bmatrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} \end{align} $$

<br>

- 분해된 행렬의 왼쪽은 `하삼각행렬`로 분해되고 오른쪽은 `상삼각행렬`로 분해됩니다. 그리고 `하삼각행렬`의 대각성분은 모두 1로 표현할 수 있습니다. 

<br>

- $$ A = LU = \begin{bmatrix} 1 & 0 & 0 \\ j & 1 & 0 \\ k & l & 1 \end{bmatrix} \begin{bmatrix} m & n & o \\ 0 & p & q \\ 0 & 0 & r \end{bmatrix} $$

<br>

- `LU 분해`는 지금까지 살펴본 바와 같이 **가우스 소거법을 행렬로 표현한 것**입니다. 따라서 다음과 같이 2 x 2 행렬과 3 x 3 행렬에 대하여 정형화 할 수 있습니다. 아래 내용은 가우스 소거법을 통하여 구한 것입니다.

<br>

- $$ A = LU $$

- $$ \text{ where } A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad L = \begin{bmatrix} 1 & 0 \\ \frac{c}{a} & 1 \end{bmatrix}, \quad U = \begin{bmatrix} a & b \\ 0 & d - \frac{bc}{a} \end{bmatrix} $$

<br>

- $$ A = LU $$ 

- $$ \text{ where } \\ A = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \\ \end{bmatrix}, $$

- $$ L = \begin{bmatrix} 1 & 0 & 0 \\ \frac{d}{a} & 1 & 0 \\ \frac{g}{a} & \frac{h - \frac{bg} {a}}{e - \frac{bd}{a}} & 1 \\ \end{bmatrix}, \quad U = \begin{bmatrix} a & b & c \\ 0 & e - \frac{bd}{a} & f - \frac{cd}{a} \\ 0 & 0 & i - \frac{cg}{a} - \frac{(h - \frac{bg}{a})(f - \frac{cd}{a})}{e - \frac{bd}{a}} \\ \end{bmatrix} $$

<br>

## **LDU 분해**

<br>

- `LDU 분해`는 `LU 분해`의 확장판이며 `D`는 `Diagonal Matrix`를 뜻하며 `LU 분해`의 `U`에 해당하는 `Upper Triangular Matrix`를 다음과 같이 분해하는 것입니다.

<br>

- $$ U = \begin{bmatrix} m & n & o \\ 0 & p & q \\ 0 & 0 & r \end{bmatrix} =\begin{bmatrix} m & 0 & 0 \\ 0 & p & 0 \\ 0 & 0 & r \end{bmatrix} \begin{bmatrix} 1 & n/m & o/m \\ 0 & 1 & q/p \\ 0 & 0 & 1 \end{bmatrix} $$

<br>

- 위 식을 보면 $$ U $$ 를 $$ DU' $$ 로 분해한 형태이고 $$ D $$ 의 대각 성분과 $$ U $$ 의 대각 성분이 같습니다. 분해된 식을 살펴보면 $$ U' $$ 와 같이 분해되는 것은 자명합니다.
- 따라서 `LDU 분해`를 적용하면 다음과 같습니다.

<br>

- $$ \begin{align} A &= \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ j & 1 & 0 \\ k & l & 1 \end{bmatrix} \begin{bmatrix} m & n & o \\ 0 & p & q \\ 0 & 0 & r \end{bmatrix} \\ &= \begin{bmatrix} 1 & 0 & 0 \\ j & 1 & 0 \\ k & l & 1 \end{bmatrix}   \begin{bmatrix} m & 0 & 0 \\ 0 & p & 0 \\ 0 & 0 & r \end{bmatrix} \begin{bmatrix} 1 & n/m & o/m \\ 0 & 1 & q/p \\ 0 & 0 & 1 \end{bmatrix} \\ &= LDU \end{align} $$

<br>

- 따라서 앞에서 구한 예제를 `LDU` 분해해 보도록 하겠습니다.

<br>

- $$ \begin{align} A = LU &= \begin{bmatrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix} \\ &= \begin{bmatrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix} \begin{bmatrix}1 & 2 & 4 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix} \end{align} $$

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