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

- $$ A = [a_{\text{ij}}]_{\text{n} \times \text{n}} $$

<br>

- `Skew Symmetic Matrix`는 다음과 같은 성질을 가지며 아래 성질은 식을 전개할 때 종종 사용됩니다.

<br>

- ① $$ (A + B)^{T} = -(A + B) $$
- ② 실수 값으로 이루어진 `Real Skew Symmetric Matrix` $$ A $$ 의 모든 대각 성분은 0 입니다. ( $$ a_{ii} = -a_{ii} $$ )
- ③ `Real Skew Symmetric Matrix` $$ A $$ 의 `Eigenvalue` 중 실수값은 오직 0입니다. 즉, 0이 아닌 `Skew Symmetric Matrix`의 `Eigenvalue`는 허수를 가집니다.
- ④ `Skew Symmetric Matrix`에 실수배를 해도 `Skew Symmetric Matrix` 성질은 유지 됩니다. $$ (kA)^{T} = -kA $$ ( $$ k \text{ is real number} $$ )
- ⑤ `Real Skew Symmetric Matrix` $$ A $$ 에 대하여 $$ I + A $$ 는 항상 `invertible` 합니다. ( $$ I \text{ is identity} $$ )
- ⑥ `Real Skew Symmetric Matrix` $$ A $$ 에 대하여 $$ A^{2} $$ 은 `Symmetric Negative Semi-Definite Matrix`를 만족합니다.

<br>

- 위 내용 중 ③의 `Eigenvalue`와 관련된 내용은 다음과 같이 확인할 수 있습니다.

<br>

- $$ Ax = \lambda x $$

- $$ \bar{x}^{T} A x = \lambda \bar{x}^{T} x = \lambda \Vert x \Vert^{2} $$

- $$ \bar{x} \text{ is cunjugate of eigenvector x} $$

- $$ \bar{x}^{T} A x = (Ax)^{T}\bar{x} = x^{T}A^{T}\bar{x} \ \ (\because \text{commutative of dot product.}) $$

- $$ \bar{x}^{T} A x = x^{T}A^{T}\bar{x} = -x^{T}A\bar{x} \ \ (\because A = -A^{T}) $$

<br>

- $$ Ax = \lambda x \to A\bar{x} = \bar{\lambda}\bar{x} \ \ (\bar{\lambda}, \bar{x} \text{ are conjugate.}) $$

- $$ A\bar{x} = \bar{\lambda}\bar{x} $$

- $$ \Rightarrow -x^{T}A\bar{x} = -x^{T}\bar{\lambda}\bar{x} = -\bar{\lambda} \Vert x \Vert^{2} $$

- $$ -\bar{\lambda} \Vert x \Vert^{2} = \lambda \Vert x \Vert^{2} (\because -x^{T}A\bar{x} = x^{T}A^{T}\bar{x} = \bar{x}^{T} A x) $$

- $$ (\lambda + \bar{\lambda}) \Vert x \Vert^{2} = 0 $$

- $$ \lambda + \bar{\lambda} = 0 (\because \Vert x \Vert^{2} \ge 0) $$

<br>

- 위 식에서 $$ \bar{\lambda} $$ 와 $$ \lambda $$ 는 `conjugate` 관계로 정의하였기 때문에 $$ \lambda $$ 2가지 경우의 값을 가집니다.

<br>

- 1) $$ \lambda = 0 $$

- 2) $$ \lambda \text{ is a purely imaginary number.} $$

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

- ⓒ 임의의 정사각행렬은 `symmetric matrix`와 `skew-symmetric matrix`의 합인 $$ A = \frac{1}{2} (A + A^{T}) + \frac{1}{2} (A - A^{T}) $$ 으로 나타낼 수 있습니다. (④ 성질과 ⓑ 정리를 이용)

<br>

#### **Determinant of Skew Symmetric Matrix**

<br>

- `Skew Symmetric`의 경우 행렬의 차원이 홀수 차원일 때와 짝수 차원일 때 계산 방법이 달라집니다.

<br>

- ⒜ 행렬의 차원이 홀수 차수인 경우: $$ \text{det}(A) = 0 $$

<br>

- 아래와 같이 $$ 3 \times 3 $$ 크기의 행렬이 있다고 가정해 보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} 0 & a & b \\ -a & 0 & m \\ -b & -m & 0 \end{bmatrix} $$

- $$ \begin{align} \text{det}(A) &= 0 \cdot (\text{cofactor of } a_{11}) + a \cdot (\text{cofactor of } a_{12}) + b \cdot (\text{cofactor of } a_{13}) \\ &= a \cdot (\text{cofactor of } a_{12}) + b \cdot (\text{cofactor of } a_{13}) \\ &= a \cdot ((-1)^{1 + 2}(0 - (-bm))) + b \cdot ((-1)^{1 + 3}(am)) \\ &= a \cdot (-1)^{3} \cdot (bm) + b \cdot (-1)^{4} \cdot (am) \\ &= -abm + abm = 0 \end{align} $$

- 위 예시와 같이 홀수 차수에 대해서는 항상 `determinant`가 0이 됩니다.
- 일반화를 위하여 다음과 같은 방식으로 적용해 볼 수 있습니다. 아래 식의 $$ n $$ 은 행렬의 차수입니다.

<br>

- $$ \text{det}(A) = \text{det}(-A^{T}) = (-1)^{n}\text{A^{T}} = (-1)^{2k-1} \text{det}(A^{T}) = -\text{det}(A) $$

- $$ \therefore \text{det}(A) = 0 $$

<br>

- ⒝ 행렬의 차원이 짝수 차수인 경우: $$ \text{det}(A) \ge 0 $$

<br>

- 앞에서 다룬 ③의 `Eigenvalue` 값 증명 과정에서 `conjugate`인 $$ \lambda = -\bar{\lambda} $$ 임을 확인하였습니다. 이 때, 2가지 경우가 있음을 확인하였습니다.

<br>

- $$ \lambda = 0, \quad \bar{\lambda} = 0 \ \ ( \lambda, \bar{\lambda} \text{ are real number.} ) $$

- $$ \lambda = bi, \quad \bar{\lambda} = -bi \ \ ( \lambda, \bar{\lambda} \text{ are pure imaginary number.} ) $$

<br>

- 먼저 $$ \lambda, \bar{\lambda} $$ 가 실수이면 0이어야 하므로 $$ \text{det}(A) = 0 $$ 이 되어 만족합니다. 모든 `Eigenvalue`의 곱이 `determinant`가 되기 때문입니다.

<br>

- 반면 $$ \lambda, \bar{\lambda} $$ 가 순허수이면 다음과 같이 나타낼 수 있습니다.

<br>

- $$ \text{det}(A) = \prod_{j=1}^{k} \lambda_{j}\bar{\lambda_{j}} = \prod_{j=1}^{k}(b_{j}i)(-b_{j}i) = \prod_{j=1}^{k}b_{j}^{2} \ge 0 $$

<br>

- 따라서 `Skew Symmetric Matrix`의 행렬의 차수가 짝수이면 $$ \text{det}(A) \ge 0 $$ 을 만족합니다.

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

