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
- ② $$ n \times n $$ 대칭 행렬 $$ A $$ 에 대하여, $$ A $$ 가 `PDM`일 **필요충분 조건**은 $$ A $$ 의 모든 `eigenvalue`가 양수인 경우입니다.
- ③ $$ n \times n $$ `대칭 행렬` $$ A $$ 에 대하여 다음 네가지 명제는 동치입니다. 즉, 하나를 만족하면 나머지 3개도 모두 만족한다는 뜻입니다.
    - ③-1 : 행렬 $$ A $$ 가 `PDM` 입니다.
    - ③-2 : $$ A $$ 의 모든 `eigenvalue`가 양의 실수 입니다.
    - ③-3 : $$ A = U^{T}U $$ 를 만족하는 `regular matrix` (역행렬이 존재하는 행렬) $$ U $$ 가 존재합니다.
    - ③-4 : $$ A $$ 의 모든 `sub-determinant`가 양의 실수입니다.
- ④ $$ n \times n $$ 크기의 행렬 $$ A, B $$ 가 각각 `PDM`이면 다음을 만족합니다.
    - ④-1 : $$ A^{T}, sA + tB $$ 모두 `PDM`을 만족합니다. 단, $$ s, t \gt 0 $$
    - ④-2 : $$ A^{-1} $$ 또한 `PDM`을 만족합니다.
    - ④-3 : $$ U $$ 가 `regular matrix`이면 $$ U^{T} A U $$ 또한 `PDM`을 만족합니다.

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

- ② $$ n \times n $$ 대칭 행렬 $$ A $$ 에 대하여, $$ A $$ 가 `PDM`일 **필요충분 조건**은 $$ A $$ 의 모든 `eigenvalue`가 양수인 경우입니다.
- ② 내용의 증명은 다음과 같습니다. 먼저 충분 조건 (→) 부터 살펴보도록 하겠습니다.

<br>

- (→) 행렬 $$ A $$ 가 `PDM`이고 $$ Av = \lambda v $$ 라고 가정하겠습니다. 이 때, 다음 식을 만족합니다.

<br>

- $$ v^{t} A v = \lambda v^{t} v > 0 $$

<br>

- 이 때, $$ v $$ 는 0이 아니고 (`PDM`을 만족해야 함) 같은 벡터를 이용한 내적의 결과는 항상 0 이상의 양수이므로 $$ v^{t} v > 0 $$ 을 만족합니다. 따라서 $$ \lambda > 0 $$ 을 만족해야 합니다.

<br>

- (←) eigenvalue, `eigenvector`를 이용한 식 $$ Av_{1} = \lambda_{1}v_{1} $$, ~ , $$ Av_{n} = \lambda_{n}v_{n} $$ 에서 $$  \lambda_{i} > 0, \ i=1, ... , n $$ 이고 $$ v_{1}, v_{2}, ... , v_{n} $$ 이 `orthonormal basis`라고 가정하겠습니다. 이와 같이 가정하면 임의의 벡터 $$ v $$ 는 `orthonormal basis`인 `eigenvector`와 `eigenvalue`를 이용하여 다음과 같이 표현할 수 있습니다.

<br>

- $$ v = \lambda_{1}v_{1} + \lambda_{2}v_{2} + \cdots + \lambda_{n}v_{n} $$

- $$ Av = A(\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n}) $$

<br>

- $$ \begin{align} v^{T}Av &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}A(\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}Av_{1} + \cdots \lambda_{n}Av_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}\lambda_{1}v_{1} + \cdots \lambda_{n}\lambda_{n}v_{n}) \\ &= (\lambda_{1}v_{1} + \cdots + \lambda_{n}v_{n})^{T}(\lambda_{1}^{2}v_{1} + \cdots + \lambda_{n}^{2}v_{n}) \\ &= (\lambda_{1}^{3} + \lambda_{2}^{3} + \cdots + \lambda_{n}^{3}) > 0 \ (\because v_{i}^{T}v_{j} = 1 \text{, if} \ i = j \text{ and } v_{i}^{T}v_{j} = 0, \text{ if } i \ne j) \end{align} $$

<br>

- 따라서 위 식의 전개와 같이 $$ v^{T}Av > 0 $$ 임을 확인할 수 있었습니다.
- 최종적으로 대칭행렬 $$ A $$ 에 대하여 $$ A $$ 가 `PDM`인 것과 모든 `eigenvalue`가 0보다 크다는 것은 필요충분조건임을 확인할 수 있었습니다.

<br>

- ③ $$ n \times n $$ `대칭 행렬` $$ A $$ 에 대하여 다음 네가지 명제는 동치입니다. 즉, 하나를 만족하면 나머지 3개도 모두 만족한다는 뜻입니다.
    - ③-1 : 행렬 $$ A $$ 가 `PDM` 입니다.
    - ③-2 : $$ A $$ 의 모든 `eigenvalue`가 양의 실수 입니다.
    - ③-3 : $$ A = U^{T}U $$ 를 만족하는 `regular matrix` (역행렬이 존재하는 행렬) $$ U $$ 가 존재합니다.
    - ③-4 : $$ A $$ 의 모든 `sub-determinant`가 양의 실수입니다.

<br>

- 먼저 앞에서 ③-1과 ③-2는 필요 충분조건인 것을 확인하였습니다. **③-2 → ③-3** 인 것을 먼저 확인하고 **③-3 → ③-1** 인 것을 확인하여 ③-3 또한 동치임을 보이도록 하겠습니다.

<br>

- 대칭 행렬은 `orthogonal matrix` $$ Q $$ 와 `diagonal matrix` $$ D $$ 로 다음과 같이 표현가능합니다. 아래 내용을 참조해 주시기 바랍니다.
    - [행렬의 대각화](https://gaussian37.github.io/math-la-diagonalization/)

<br>

- $$ A = Q^{T} D Q \quad (\text{Q : Orthogonal Matrix, D : Diagonal Matrix}) $$ 

- $$ D = \begin{bmatrix} \lambda_{1} & 0 & \cdots & 0 \\ 0 & \lambda_{2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n} \end{bmatrix} $$

<br>

- 위 식과 같이 `diagonal matrix` $$ D $$ 의 대각성분이 `eigenvalue`이므로 모두 양수입니다. (③-2의 조건) 따라서 `diagonal matrix` $$ D $$ 의 대각 성분이 모두 양수이므로 $$ D = C^{2} $$ 으로 표현할 수 있습니다. 추가적으로 식을 전개해 보면 다음과 같습니다.

<br>

- $$ \begin{align} A = Q^{T}D Q &= Q^{T} CC Q \\ &= Q^{T}C^{T}C Q \quad (\because C = C^{T}) \\ &= (CQ)^{T}(CQ)  \\ &= U^{T}U \quad (CQ = U) \end{align} $$

- $$ \therefore \ A = U^{T}U $$

<br>

- 따라서 **③-2 → ③-3** 인 것을 확인하였습니다. 이번에는 **③-3 → ③-1** 임을 확인해 보겠습니다.

<br>

- $$ x^{T} A x = x^{T}U^{T}U x = (Ux)^{T}Ux \gt 0 $$

- $$ U \text{ : regular matrix, } \quad \therefore x \ne 0 \to Ux \ne 0 $$

<br>

- 마지막으로 **③-4**의 내용을 살펴보도록 하겠습니다.

<br>

- 먼저 $$ 3 \times 3 $$ 크기의 행렬 케이스를 살펴보도록 하겠습니다.

<br>

- 대칭 행렬 $$ A = \begin{bmatrix} a_{11} & b & c \\ b & a_{22} & d \\ c & d & a_{33} \end{bmatrix} $$ 가 `PDM`이라고 가정하겠습니다. 그러면 **③-2**에 의하여 $$ A $$ 의 모든 $$ \lambda_{1}, \lambda_{2}, \lambda_{3} $$ 이 양의 실수가 됩니다. 이 때, `orthogonal matrix` $$ U $$ 가 존재하여 다음을 만족합니다.

<br>

- $$ A = U^{T} \begin{bmatrix} \lambda_{1} & \ & \ \\ \ & \lambda_{2} & \ \\ \ & \ & \lambda_{3} \end{bmatrix} U $$

- $$ \text{det}(A) = \text{det}(U^{T})\text{det}\left(\begin{bmatrix} \lambda_{1} & \ & \ \\ \ & \lambda_{2} & \ \\ \ & \ & \lambda_{3} \end{bmatrix}\right) \text{det}(U) = \lambda_{1}\lambda_{2}\lambda_{3} > 0 $$

- $$ \text{det}(U^{T})\text{det}(U) = 1 \quad (\because U \text{ and } U^{T} \text{ are inversly related.}) $$

<br>

- 즉, $$ A $$ 가 `PDM`이면 $$ \text{det}(A) > 0 $$ 를 만족함을 알 수 있습니다.

<br>

- 행렬 $$ A $$ 의 `principal submatrices`는 다음과 같습니다.

<br>

- $$ \begin{bmatrix} a_{11} \end{bmatrix}, \begin{bmatrix} a_{22} \end{bmatrix}, \begin{bmatrix} a_{33} \end{bmatrix} $$

- $$ \begin{bmatrix} a_{11} & b \\ b & a_{22} \end{bmatrix} $$

- $$ \begin{bmatrix} a_{22} & d \\ d & a_{33} \end{bmatrix} $$

<br>

- 행렬 $$ A $$ 가 `PDM`이라는 가정으로 인하여 위에서 나열한 모든 `principal submatrices`는 `PDM`을 만족합니다. 그 이유는 다음 예시를 통해 확인할 수 있습니다.

<br>

- $$ \begin{bmatrix} x & 0 & 0 \end{bmatrix} \begin{bmatrix} a_{11} & b & c \\ b & a_{22} & d \\ c & d & a_{33} \end{bmatrix} \begin{bmatrix} x \\ 0 \\ 0 \end{bmatrix} > 0 \quad (\text{Assumed } A \text{ is P.D.M}) $$

- $$ x (a_{11}) x > 0 \quad \text{satisfied P.D.M} $$ 

<br>

- 위 식과 같이 $$ x $$ 를 $$ \begin{bmatrix} x \\ 0 \\ 0 \end{bmatrix} $$ 와 같이 정의하면 (또는 유사하게 정의하면) $$ \begin{bmatrix} a_{ii} \end{bmatrix} $$ 와 같은 형태의 `principal submatrices`를 만들 수 있습니다.
- 유사한 방식으로 2개의 행과 열을 선택하는 방식은 다음과 같이 만들 수 있습니다.

<br>

- $$ \begin{bmatrix} x & y & 0 \end{bmatrix} \begin{bmatrix} a_{11} & b & c \\ b & a_{22} & d \\ c & d & a_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 0 \end{bmatrix} > 0 $$

- $$ \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} a_{11} & b \\ b & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} > 0 \quad \text{satisfied P.D.M} $$ 

<br>

- 이와 같은 방식으로 $$ 2 x 2 $$ 크기의 `principal submatrices`를 만들 수 있습니다.

<br>

- 여기서 확인해야 할 점은 모든 `principal submatrices`의 `determinant`가 양수인 지 확인하는 것입니다.
- ③-1과 ③-2에 따라 `PDM`을 만족하는 행렬의 `eigenvalue`는 모두 양의 실수이므로 `principal submatrices`의 `eigenvalue` 또한 양의 실수가 됩니다.

<br>

- $$ A = U^{T} \begin{bmatrix} \lambda_{1} & \ & \ \\ \ & \lambda_{2} & \ \\ \ & \ & \lambda_{3} \end{bmatrix} U $$

- $$ \text{det}(A) = \text{det}(U^{T})\text{det}\left(\begin{bmatrix} \lambda_{1} & \ & \ \\ \ & \lambda_{2} & \ \\ \ & \ & \lambda_{3} \end{bmatrix}\right) \text{det}(U) = \lambda_{1}\lambda_{2}\lambda_{3} > 0 $$

<br>

- 앞에서 살펴본 위 식에서 전개한 바와 동일하게 `principal submatrices`에 동일하게 적용하면 `eigenvalue`가 모두 양의 실수 이기 때문에 `principal submatrices`의 `determinant`인 `sub-determinant`가 모두 양수인 것을 확인할 수 있습니다.

<br>

- 만약 `sub-determinant`가 모두 양수이면 행렬 $$ A $$ 를 구성하는 모든 `eigenvalue`가 양의 실수임을 만족합니다. 앞의 예제를 살펴보면 확인 가능합니다.
- 즉, `sub-determinant`가 모두 양수이면 `eigenvalue`가 양의 실수이고 따라서 행렬 $$ A $$ 가 `PDM`이 됩니다.

<br>

- 지금까지 살펴본 내용으로 `PDM`이면 `eigenvalue`가 양의 실수임을 계속 확인하였습니다.
- `eigenvector`는 `basis`의 역할을 하는 반면 `eigenvalue`는 `eigenvector`의 스케일 및 방향을 결정하는 역할을 합니다.
- **`PDM`에서는 `eigenvalue`가 모두 양수이기 때문에 `eigenvector`의 크기만 바뀔 뿐 방향이 바뀌지 않습니다.** `PDM`인 행렬 $$ A $$ 의 기하학적인 의미는 이와 같이 `basis`의 방향을 바꾸지 않는 행렬로 이해하면 좀 더 쉽게 이해하실 수 있습니다.

<br>

- 다음으로 `PDM`의 추가적인 성질에 대하여 알아보도록 하겠습니다.

<br>

- ④ $$ n \times n $$ 크기의 행렬 $$ A, B $$ 가 각각 `PDM`이면 다음을 만족합니다.
    - ④-1 : $$ A^{T}, sA + tB $$ 모두 `PDM`을 만족합니다. 단, $$ s, t \gt 0 $$
    - ④-2 : $$ A^{-1} $$ 또한 `PDM`을 만족합니다.
    - ④-3 : $$ U $$ 가 `regular matrix`이면 $$ U^{T} A U $$ 또한 `PDM`을 만족합니다.

<br>

- 먼저 **④-1** 부터 살펴보도록 하겠습니다. $$ x^{T}A^{T} x $$ 와 $$ (x^{T}A^{T}x)^{T} $$ 모두 실수 스칼라 값만을 가지므로 다음과 같이 식을 적용할 수 있습니다.

<br>

- $$ x^{T}A^{T}x = (x^{T}A^{T}x)^{T} $$

- $$ \therefore x^{T}A^{T}x = (x^{T}A^{T}x)^{T} = x^{T}Ax \gt 0 $$

<br>

- 따라서 $$ A $$ 가 `PDM`이면 $$ A^{T} $$ 또한 `PDM`을 만족합니다.

<br>

- $$ x^{T}(sA + tB)x = x^{T}(sAx + tBx) = sx^{T}Ax + tx^{T}Bx > 0 $$

<br>

- 따라서 $$ A, B $$ 가 `PDM`이면 $$ sA + tB \ (s, t \gt 0) $$ 또한 `PDM`임을 만족합니다.

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>