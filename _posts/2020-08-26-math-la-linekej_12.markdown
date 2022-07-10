---
layout: post
title: 행공간, 열공간, 영공간과 계수
date: 2020-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, 행공간, 열공간, 영공간, row space, column space, null space] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 `행공간 (row space)`, `열공간 (column space)`, `영공간 (null space)`에 대하여 알아보도록 하겠습니다.

<br>

## **행공간(Row Space), 열공간(Column Space), 영공간(Null Space)의 정의**

<br>

- 행공간 (Row Space)는 $$ m \times n $$ 행렬 $$ A $$ 의 행벡터 $$ A_{1}, A_{2}, ..., A_{m} $$ 에 의해 생성된 $$ \mathbb{R}^{n} $$ 의 부분공간 $$ \text{Row}(A) = \text{Span}(A_{1}, A_{2}, ..., A_{m}) $$ 을 의미합니다.
- 위 표기에서 $$ \mathbb{R}^{n} $$ 은 각각의 행벡터를 구성하는 성분은 $$ n $$ 개로 이루어져 있기 때문에 $$ \mathbb{R}^{n} $$ 의 부분공간으로 표기합니다.
- 행공간의 기저는 **사다리꼴 행렬에서 피봇이 존재하는 행을 선택**합니다.

<br>

- 열공간 (Column Space)는 $$ m \times n $$ 행렬 $$ A $$ 의 열벡터 $$ A^{(1)}, A^{(2)}, A^{(1)}, ... , A^{(n)} $$ 에 의해 생성된 $$ R^{m} $$ 의 부분공간 $$ \text{Col}(A) = \text{Span}(A^{(1)}, A^{(2)}, ... , A^{(n)}) $$ 을 의미합니다.
- 열공간의 기저는 행렬 $$ A $$ 에서 `피봇`이 존재하는 열을 선택합니다.

<br>

- 행공간과 열공간에서 선택되는 기저가 행공간에서는 사다리꼴 행렬에서 선택하고 열공간에서는 기존 행렬 $$ A $$ 에서 선택하는 지는 본 글에서 설명드리곘습니다.

<br>

- 영공간 (Null Space)는 $$ m \times n $$ 행렬 $$ A $$ 를 계수행렬로 가지는 제차연립방정식( $$ Av = 0 $$ )의 해집합으로 $$ \text{Null}(A) = \{v \in \mathbb{R}_{n} \vert Av = 0 \} $$ 으로 표기합니다.
- 영공간의 기저는 연립방정식을 풀어야 구할 수 있습니다.
- 영공간의 `해집합`이기 때문에 해가 유일하게 1개일 수도 있고 해가 없을 수도 있으며 무한히 많은 해를 가질 수도 있습니다.

<br>

- 공간 (Space)의 의미는 벡터 공간 (Vector Space)의 의미를 가지고 행공간과 열공간은 `Span`을 이용하여 표현하기 때문에 부분 공간이 됨을 만족합니다.
- 하지만 영공간의 경우 단순히 제차연립방정식의 해를 만족하는 집합인데 이 공간이 과연 벡터 공간의 성질을 만족하는 지는 별도 확인이 필요합니다. 이 내용을 먼저 확인해 보도록 하겠습니다.

<br>

- **(정리) $$ m \times n $$ 행렬 $$ A $$ 의 영공간은 $$ \mathbb{R}^{n} $$ 의 부분공간이다.**

<br>

- 아래 3가지 내용을 차례로 증명하면 위 정리 12를 증명할 수 있습니다. 아래 3가지 내용은 벡터 공간을 구성하기 위한 조건에 해당합니다.

<br>

- 1) $$ 0 \in \text{Null}(A) $$

- 2) $$ u, v \in \text{Null}(A) \Rightarrow u + v \in \text{Null}(A) $$

- 3) $$ u \in \text{Null}(A), c \in R \Rightarrow cu \in \text{Null}(A) $$

<br>

- 위 3가지 내용을 차례로 증면하면 아래와 같습니다.

<br>

- 1) $$ A0 = 0 $$ 이므로 $$ 0 \in \text{Null}(A) $$ 을 만족합니다.

- 2) $$ u, v \in \text{Null}(A) $$ 이면 $$ Au = 0 $$ 이고 $$ Av = 0 $$ 이므로 $$ A(u + v) = Au + Av = 0 + 0 = 0 $$ 입니다.

- 3) $$ u \in \text{Null}(A) $$ 와 $$ c \in \mathbb{R} $$ 에 대하여 $$ Au = 0 $$ 이므로 $$ A(cu) = c(Au) = c0 = 0 $$ 입니다. 따라서 $$ cu \in \text{Null}(A) $$ 입니다.

<br>

- 그러면 행공간, 열공간, 영공간에 대하여 좀 더 자세히 다루어 보도록 하겠습니다.
- 앞에서 정의한 행공간은 $$ \text{Row}(A) = \text{Span}(A_{1}, A_{2}, ..., A_{m}) $$ 형태를 따르고 만약 행벡터 $$ A_{1}, A_{2}, ..., A_{m} $$ 가 모두 일차 독립이라면 모두 기저가 되고 일차 종속인 행벡터가 있다면 `Row Echelon Form` 형태의 사다리꼴 행렬 ( $$ \mathbb{R} $$ )을 만들어 정리할 수 있습니다.
- 이 때, 사다리꼴 행렬에서 피봇이 존재하는 행을 선택하면 **기저에 해당하는 행공간을 구성할 수 있습니다.** 주의할 점은 행공간의 기저는 $$ A $$ 행렬 또는 $$ \mathbb{R} $$ 행렬에서 모두 선택할 수 있지만, `Row Echelon Form` 형태로 만들 시 행 간의 교환이 발생하면 $$ \mathbb{R} $$ 에서 구한 기저가 $$ A $$ 에서 구한 기저와 다를 수 있기 때문입니다.
- 기본행 연산을 이용하여 행렬의 변화가 발생할 때, 행공간은 변하지 않는 다는 성질을 이용하면 행공간은 사다리꼴 행렬에서 쉽게 구할 수 있습니다. 물론 원본 행렬 $$ A $$ 에서 행공간을 가져올 수 있으나 교환에 대한 추적을 정확히 해야 하는 불편함이 있습니다.

<br>

- 반면 열공간은 사다리꼴행렬 $$ mathbb{R} $$ 에서 얻은 기저를 직접적으로 사용하지 않고 사다리꼴을 통해 확인할 수 있는 열공간의 기저의 위치를 확인한 후 원본 행렬 $$ A $$ 에서 기저의 위치에 해당하는 열을 가져와서 사용합니다.
- 이와 같은 방법을 사용하는 이유는 기본행 연산을 통해 얻은 사다리꼴 행렬에서 열의 정보는 보존 되지 않기 때문입니다.

<br>

- 아래 예제를 살펴보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} 1 & -3 & 2 \\ -5 & 9 & 1 \end{bmatrix} $$ 일 때, 영공간을 구해보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} 1 & -3 & 2 \\ -5 & 9 & 1 \end{bmatrix} $$

- $$ \Rightarrow \begin{bmatrix} 1 & -3 & 2 \\ 0 & -6 & -9 \end{bmatrix} $$

<br>

- 위 사다라꼴 행렬 식에서 `pivot`이 생긴 열은 선행 변수라고 하며 `pivot`이 없는 열은 자유 변수라고 합니다.
- 1열과 2열은 각각 1, -6 이라는 피벗이 있기 때문에 선행 변수가 존재하며 3열은 피벗이 없기 때문에 자유 변수가 존재합니다.
- 따라서 위 사다리꼴 행렬 에서 첫번째 행이 행공간이 되고 원본 행렬에서 첫번째 열이 열공간이 됩니다.

<br>

- $$ R : \left\{ [1, -3, 2] \right\} $$

- $$ A : \left\{ [1, -5]^{T} \right\} $$

<br>

- 아래 쉬운 예제를 통하여 먼저 `영공간`에 대한 이해를 살펴보도록 하겠습니다.

<br>

- $$ \begin{bmatrix} 1 & -3 & 2 \\ 0 & -6 & -9 \end{bmatrix} \begin{bmatrix} x_{1} \\ x_{2} \\ x_{3} \end{bmatrix} = {bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} \tag{1} $$ 

<br>

- $$ x_{3} = a \tag{2} $$

- $$ -6x_{2} -9x_{3} = 0  \tag{3} $$

- $$ x_{2} = -\frac{3}{2}a \tag{4} $$

<br>

- $$ x_{1} - 3x_{2} - 2x_{3} = 0 \tag{5} $$

- $$ x_{1} = -\frac{9}{2}a + 2a = -\frac{5}{2}a \tag{6} $$

<br>

- $$ \text{Null}(A) = \left\{ \begin{bmatrix} -\frac{5}{2}a \\ -\frac{3}{2}a \\ a \end{bmatrix} \vert a \in \mathbb{R} \right\} \tag{7} $$

- $$ = \left\{ -\frac{a}{2} \begin{bmatrix} 5 \\ 3 \\ -2 \end{bmatrix} \vert a \in \mathbb{R} \right\} \tag{8} $$

<br>

- 영공간을 식 (8) 과 같이 정리할 수 있고 $$ Av = 0 $$ 의 해를 얻은 영공간은 독립이기 때문에 다음과 같은 `Span` 형태로 나타낼 수 있습니다.

<br>

- $$ \text{Span}\Biggl( \begin{bmatrix} 5 \\ 3 \\ -2 \end{bmatrix} \Biggr) \tag{9} $$

<br>

- 조금 더 복잡한 예제를 통하여 행공간, 열공간, 영공간을 구해보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} -3 & 6 & -1 & 1 & -7 \\ 1 & -2 & 2 & 3 & -1 \\ 2 & -4 & 5 & 8 & -4 \end{bmatrix} \tag{10} $$

<br>

- 위 행렬을 사다리꼴로 만들어 보도록 하곘습니다.

<br>

- $$ \Rightarrow \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ -3 & 6 & -1 & 1 & -7 \\ 2 & -4 & 5 & 8 & -4 \end{bmatrix} \tag{11} $$

- $$ \Rightarrow \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 5 & 10 & -10 \\ 0 & 0 & 1 & 2 & -2 \end{bmatrix} \tag{12} $$

- $$ \Rightarrow \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 5 & 10 & -10 \\ 0 & 0 & 1 & 2 & -2 \end{bmatrix} \tag{13} $$

- $$ \Rightarrow \begin{bmatrix} 1 & -2 & 2 & 3 & -1 \\ 0 & 0 & 1 & 2 & -2 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix} \tag{14} $$

<br>

- 따라서 `행공간`의 기저는 다음과 같습니다.

<br>

- $$ \text{Span}\Biggl( [1, -2, 2, 3, -1], [0, 0, 1, 2, -2] \Biggr) \tag{15} $$

<br>

- `열공간`의 기저는 다음과 같습니다.

<br>

- $$ \text{Span}\Biggl( \begin{bmatrix} -3 \\ 1 \\ 2 \end{bmatrix}, \begin{bmatrix} -1 \\ 2 \\ 5 \end{bmatrix} \Biggr) \tag{16} $$

<br>

- `영공간`의 기저는 다음과 같습니다.

<br>

- $$ x_{2} = a, x_{4} = b, x_{5} = c \tag{17} $$

- $$ x_{3} + 2x_{4} - 2x_{5} = 0 \tag{18} $$

- $$ x_{1} - 2x_{2} + 2x_{3} + 3x_{4} - x_{5} = 0 \tag{19} $$

- 식 (18), 식 (19)를 이용하면 영공간의 기저는 다음과 같이 구할 수 있습니다.

- $$ \text{Span}\Biggl( \begin{bmatrix} 2 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ -2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ 2 \\ 0 \\ 1 \end{bmatrix} \Biggr) \tag{20} $$

<br>

## **계수(Rank)의 정의**

<br>

- `계수 (Rank)`는 $$ m \times n $$ 행렬 $$ A $$ 에 대하여 열공간의 차원을 행렬 $$ A $$ 의 계수라고 합니다. 즉, $$ \text{rank}(A) = \text{dim}(\text{Col}(A)) $$ 가 성립하며 다음 성질을 가집니다.

<br>

- ① $$ \text{rank}(A) $$ 는 사다리꼴 행렬에서 `pivot`의 갯수와 같다.
- ② $$ \text{rank}(A) + \text{dim}(\text{Null}(A)) = n $$
- ③ $$ \text{rank}(A) = \text{dim}(\text{Col}(A)) = \text{dim}(\text{Row}(A)) $$

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>