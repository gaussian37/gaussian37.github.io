---
layout: post
title: Skew Symmetric Matrix
date: 2020-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, vector, skew symmetric] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 행렬의 형태 중 하나인 `Skew Symmetric Matrix`에 대하여 알아보도록 하겠습니다.

<br>

- `Skew Symmetric Matrix`는 정사각행렬 $$ A $$ 에 대하여 전치 행렬이 $$ A^{T} $$ 라고 할 때, $$ A = -A^{T} $$ 를 만족하는 행렬을 의미합니다. 예를 들어 다음과 같습니다.

<br>

- $$ A = \begin{bmatrix} 0 & 2 & 4 \\ -2 & 0 & 3 \\ -4 & -3 & 0 \end{bmatrix} $$

- $$ A^{T} = \begin{bmatrix} 0 & 2 & 4 \\ -2 & 0 & 3 \\ -4 & -3 & 0 \end{bmatrix} $$

- $$ \therefore A = -A^{T} $$

<br>

- `Skew Symmetric Matrix` 는 다음과 같이 표현하기도 합니다.

<br>

- $$ A = [a_{\text{ij}}]_{text{n} \times text{n}} $$

<br>

- `Skew Symmetic Matrix`는 다음과 같은 성질을 가지며 아래 성질은 식을 전개할 때 종종 사용됩니다.

<br>

- ① $$ (A + B)^{T} = -(A + B) $$
- ② 실수 값으로 이루어진 `Real Skew Symmetric Matrix` $$ A $$ 의 모든 대각 성분은 0 입니다. ( $$ a_{ii} = -a_{ii} $$ )
- ③ `Real Skew Symmetric Matrix` $$ A $$ 의 `Eigenvalue` 중 실수값은 0을 가집니다. 즉, 0이 아닌 `Skew Symmetric Matrix`의 `Eigenvalue`는 실수가 아닙니다.
- ④ `Skew Symmetric Matrix`에 실수배를 해도 `Skew Symmetric Matrix` 성질은 유지 됩니다. $$ (kA)^{T} = -kA $$ ( $$ k \text{ is real number} $$ )
- ⑤ `Real Skew Symmetric Matrix` $$ A $$ 에 대하여 $$ I + A $$ 는 항상 `invertible` 합니다. ( $$ I \text{ is identity} $$ )
- ⑥ `Real Skew Symmetric Matrix` $$ A $$ 에 대하여 $$ A^{2} $$ 은 `Symmetric Negative Semi-Definite Matrix`를 만족합니다.

<br>

- `Skew Symmetric Matrix`와 관련하여 아래 2가지 정리 또한 많이 사용 됩니다.

<br>

- ⓐ `Real Skew Symmetric Matrix` $$ A $$ 에 대하여 $$ A + A^{T} $$ 는 `Symmetric Matrix`이며 $$ A - A^{T} $$ 는 `Skew Symmetric Matrix`입니다.

<br>

- 위 정리를 증명하면 다음과 같습니다.

<br>

- $$ \text{Let } P = A + A^{T} $$

- $$ P^{T} = (A + A^{T})^{T} = A^{T} + (A^{T})^{T} = A^{T} + A = A + A^{T} = P $$

- $$ \Rightarrow A + A^{T} \text{ is a symmetric matrix.} $$

<br>

- $$ \text{Let } Q = A - A^{T} $$

- $$ Q^{T} = (A + (-A)^{T})^{T} = A^{T} + (-A^{T})^{T} = A^{T} - (A^{T})^{T} = A^{T} - A = -(A - A^{T}) = -Q $$

- $$ \Rightarrow A - A^{T} \text{ is a skew-symmetric matrix.} $$

<br>

- ⓑ 임의의 정사각행렬 $$ A $$ 에 대하여 다음 성질을 만족합니다.

- $$ A + A^{T} \text{ is a symmetric matrix.} $$

- $$ A - A^{T} \text{ is a skew-symmetric matrix.} $$

<br>

- ⓒ 임의의 정사각행렬은 `symmetric matrix`와 `skew-symmetric matrix`의 합인 - $$ A = \frac{1}{2} (A + A^{T}) + \frac{1}{2} (A - A^{T}) $$ 으로 나타낼 수 있습니다. (④ 성질과 ⓑ 정리를 이용)

<br>


<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

