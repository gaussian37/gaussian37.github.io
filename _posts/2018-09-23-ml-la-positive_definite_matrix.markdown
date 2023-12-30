---
layout: post
title: 양의 정부호 행렬 (Positive Definite Matrix)
date: 2018-09-23 03:49:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 양의 정부호 행렬, positive definite matrix] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 본 글에서 다룰 내용이 길어질 수 있으므로 양의 정부호/준정부호 행렬의 정의와 성질을 먼저 상단부에 정리해 놓도록 하겠습니다.

<br>

#### **양의 정부호/준정부호 행렬의 정의**

<br>

- `Positive Definite Matrix` : 대칭행렬 $$ A (A = A^{T}) $$ 가 모든 $$ n $$ 차원 벡터 $$ x \ne 0 $$ 에 대하여 $$ x^{T} A x \gt 0 $$ 이면, $$ A $$ 를 `PDM` 이라고 합니다.
- `Positive Semi-Definite Matrix` : `PDM`의 조건에서 $$ x^{t} A x \ge 0 $$ 이면 $$ A $$ 를 `PSDM`이라고 합니다.

<br>

#### **양의 정부호/준정부호 행렬의 성질**

<br>

- ① 임의의 행렬 (앞에서 가정한 대칭행렬과 상관 없음) $$ A $$ 에 대하여 $$ A^{T}A, AA^{T} $$ 는 `PSDM`입니다.
- ② $$ n \times n $$ 대칭 행렬 $$ A $$ 에 대하여, $$ A $$ 가 `PDM`일 **필요충분 조건**은 $$ A $$ 의 모든 `고유값`이 양수인 경우입니다.


<br>

---

<br>

- 이번 글에서는 `Positivie Definite Matrix (양의 정부호 행렬)`와 `Positivie Semi-Definite Matrix (양의 준정부호 행렬)`의 정의와 그 성질에 대하여 알아보겠습니다. 용어는 `PDM`과 `PSDM`으로 줄여서 사용하겠습니다.
- `PDM`과 `PSDM`의 정의는 다음과 같습니다.

<br>

- `Positive Definite Matrix` : 대칭행렬 $$ A (A = A^{T}) $$ 가 모든 $$ n $$ 차원 벡터 $$ x \ne 0 $$ 에 대하여 $$ x^{T} A x \gt 0 $$ 이면, $$ A $$ 를 `PDM` 이라고 합니다.
- `Positive Semi-Definite Matrix` : `PDM`의 조건에서 $$ x^{t} A x \ge 0 $$ 이면 $$ A $$ 를 `PSDM`이라고 합니다.

<br>

- 두가지 정의를 살펴보면 임의의 벡터 $$ x $$ 를 대칭행렬 $$ A $$ 를 이용하여 선형변환 하고 ($$ x^{T}A $$) 다시 $$ x $$ 를 곱한 뒤 결과의 `부호`를 살펴보는 과정입니다. 이 과정의 의미는 글의 내용을 살펴보면서 차근 차근 설명해 보도록 하겠습니다.
- 지금부터 `PDM`과 `PSDM`의 성질 등을 하나씩 살펴보도록 하겠습니다.

<br>

- ① 임의의 행렬 (앞에서 가정한 대칭행렬과 상관 없음) $$ A $$ 에 대하여 $$ A^{T}A, AA^{T} $$ 는 `PSDM`입니다.
- ① 내용의 증명은 다음과 같습니다.

<br>

- $$ x^{T}A^{T}A x = (Ax)^{T}(Ax) = b^{T}b \ge 0 $$

- $$ x^{T}AA^{T} x = (A^{T}x)^{T}(A^{T}x) = c^{T}c \ge 0 $$

<br>

- 위 2가지 내용을 모두 살펴보면 연산 결과 같은 벡터 ($$ b, c $$)의 내적이 되고 같은 벡터의 내적은 0 이상의 값을 가지므로 ① 내용을 만족할 수 있습니다.

<br>

- ② $$ n \times n $$ 대칭 행렬 $$ A $$ 에 대하여, $$ A $$ 가 `PDM`일 **필요충분 조건**은 $$ A $$ 의 모든 `고유값`이 양수인 경우입니다.
- ② 내용의 증명은 다음과 같습니다. 먼저 충분 조건 (→) 부터 살펴보도록 하겠습니다.

<br>

- (→) 행렬 $$ A $$ 가 `PDM`이고 $$ Av = \lambda v $$ 라고 가정하겠습니다. 이 때, 다음 식을 만족합니다.

<br>

- $$ v^{t} A v = \lambda v^{t} v > 0 $$

<br>

- 이 때, $$ v $$ 는 0이 아니고 (`PDM`을 만족해야 함) 같은 벡터를 이용한 내적의 결과는 항상 0 이상의 양수이므로 $$ v^{t} v > 0 $$ 을 만족합니다. 따라서 $$ \lambda > 0 $$ 을 만족해야 합니다.

<br>

- (←) 고유값, 고유벡터를 이용한 식 $$ Av_{1} = \lambda_{1}v_{1} $$, ~ , $$ Av_{n} = \lambda_{n}v_{n} $$ 에서 $$  \lambda_{i} > 0, \ i=1, ... , n $$ 이고 $$ v_{1}, v_{2}, ... , v_{n} $$ 이 `orthonormal basis`라고 가정하겠습니다. 이와 같이 가정하면 임의의 벡터 $$ v $$ 는 `orthonormal basis`인 고유벡터와 고유값을 이용하여 다음과 같이 표현할 수 있습니다.

<br>

- $$ v = \lambda_{1}v_{1} + \lambda_{2}v_{2} + \cdots + \lambda_{n}v_{n} $$

- $$ Av = A(\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n}) $$

<br>

- $$ \begin{align} v^{T}Av &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}A(\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}Av_{1} + \cdots \lambda_{n}Av_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}\lambda_{1}v_{1} + \cdots \lambda_{n}\lambda_{n}v_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}^{2}v_{1} + \cdots + \lambda_{n}^{2}v_{n}) \\ &= (v_{1}^{3} + v_{2}^{3} + \cdots + v_{n}^{3}) > 0 \ (\because v_{i}^{T}v_{j} = 1 \text{, if} \ i = j \text{ and } v_{i}^{T}v_{j} = 0, \text{ if } i \ne j) \end{align} $$

<br>

- 따라서 위 식의 전개와 같이 $$ v^{T}Av > 0 $$ 임을 확인할 수 있었습니다.
- 최종적으로 대칭행렬 $$ A $$ 에 대하여 $$ A $$ 가 `PDM`인 것과 모든 고유값이 0보다 크다는 것은 필요충분조건임을 확인할 수 있었습니다.

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>