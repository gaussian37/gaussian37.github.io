---
layout: post
title: 특이값 분해(SVD)
date: 2016-12-01 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, SVD, singular vector decomposition] # add tag
---

<br>

- 참조 : https://darkpgmr.tistory.com/106
- 참조 : https://angeloyeo.github.io/2019/08/01/SVD.html
- 참조 : https://dowhati1.tistory.com/7
- 참조 : https://bskyvision.com/

<br>

- 아래 내용은 사전 지식 이므로 가능한 먼저 읽으시길 추천 드립니다.
- `고유값 분해 (EVD, Eigen Value Decomposition)` : [https://gaussian37.github.io/math-la-evd/](https://gaussian37.github.io/math-la-evd/)

<br>

## **목차**

<br>

- ### [SVD 간단 정의](#svd-간단-정의-1)
- ### [SVD 계산 방법](#svd-계산-방법-1)
- ### [SVD 간단 예제](#svd-간단-예제-1)
- ### [SVD 의미 해석](#svd의-의미-해석-1)
- ### [SVD 관련 성질](#svd-관련-성질-1)
- ### [SVD with Python](#svd-with-python-1)
- ### [SVD의 활용](#svd의-활용-1)

<br>

## **SVD 간단 정의**

<br>

- `SVD (Singular Value Decomposition)`, 특이값 분해는 `고유값 분해`와 같이 행렬을 `대각화`하는 방법 중 하나입니다. `고유값 분해`는 정방 행렬에만 사용가능하고 정방 행렬 중 일부 행렬에 대해서만 적용 가능한 반면, 특이값 분해는 `직사각형 행렬일 때에도 사용 가능`하므로 활용도가 높습니다.
- 즉, `고유값 분해`에서는 행렬 $$ A $$ 가 `대칭 행렬` & `정사각행렬`이면 $$ A = PDP^{T} ( P : \text{orthogonal matrix}, D : \text{diagonal matrix} )$$ 로 분해할 수 있으나 $$ A $$ 가 이 조건을 만족하지 못하는 경우에도 $$ \text{orthogonal matrix} \text{diagonal matrix} \text{orthogonal matrix} $$ 형태로 분해하고자 하는 것이 `SVD`의 목적입니다.

<br>

- `m x n` 크기의 행렬 $$ A $$ 를 `특이값 분해`하면 다음과 같이 분해됩니다.

<br>

- $$ A = U \Sigma V^{t} $$

- $$ A : m \times n \text{    (rectangular matrix)} $$

- $$ U : m \times m \text{    (orthogonal matrix)} $$

- $$ \Sigma : m \times n \text{    (diagonal matrix)} $$

- $$ V : n \times n \text{    (orthogonal matrix)} $$

<br>

- 여기서 $$ U, V $$ 는 각각 서로 다른 `직교 행렬`이며 `특이 벡터`로 구성된 행렬입니다. $$ \Sigma $$ 는 `특이값` $$ \sigma_{1}, \sigma_{1}, \cdots \sigma_{r} $$ 들을 대각요소로 갖고 있는 대각 행렬로서 `특이값 행렬`이라고 불립니다. $$ \sigma_{r} $$ 의 $$ r $$ 은 대각행렬의 `rank`를 의미합니다.
- 행렬 $$ A $$ 의 크기가 $$ m \time n $$ 이고 $$ m \ge n $$ 이라면 $$ r $$ 과의 관계는 다음과 같습니다.

<br>

- $$ m \ge n \ge r $$

<br>

- 따라서 $$ \Sigma $$ 는 $$ \text{rank}(A) = r $$ 만큼의 대각 성분을 가지고 나머지 대각 성분은 0을 가지는 대각 행렬이 됩니다.

<br>

- $$ \Sigma = \begin{bmatrix} \sigma_{1} &  &  & \\  & \sigma_{2} &  & \\ & & \ddots & \\ & & & \sigma_{r} \\ & & & & \sigma_{r+1} \\ & & & & &\ddots \\ & & & & && \sigma_{n}
 \end{bmatrix} $$ 

- $$ \sigma_{r+1}, \cdots \sigma_{n} = 0 $$

- `특이값 행렬`은 (m x n) 크기의 직사각행렬이므로 m과 n의 크기에 따라 다음과 같은 형태를 가질 수 있습니다.

<br>
<center><img src="../assets/img/math/la/svd/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `직교 행렬(orthogonal matrix)`와 `대각 행렬(diagonal matrix)`에 대한 성질을 살펴보면 다음과 같습니다. $$ U $$ 를 직교 행렬이라고 하겠습니다.

<br>

- $$ UU^{T} = U^{T}U = I $$

- $$ U^{-1} = U^{T} $$

<br>

- 벡터가 서로 직교한다고 하면 내적이 0이 됩니다. $$ u $$ 와 $$ v $$ 벡터가 `직교`한다면 다음 수식을 만족합니다.

<br>

- $$ u^{T}v = 0 $$

<br>

- 만약 $$ n $$ 차원 공간에서 행 방향과 열 방향에 대하여 서로 직교하는 $$ n $$ 개의 `단위 벡터 (unit vector)`를 이용하여 $$ n $$ 차원의 벡터를 만들면 `직교 행렬`이 됩니다.

<br>

- $$ U = (u_{1}, u_{2}, ... , u_{n}) $$

<br>

- 각 벡터가 `단위 벡터`이기 때문에 서로 다른 행 또는 열 끼리의 벡터의 내적 연산은 0이고 같은 행 또는 열 끼리의 벡터의 내적 연산은 1이므로 앞에서 언급한 조건이 만족합니다.

<br>

- $$ UU^{T} = U^{T}U = I $$

- $$ U^{-1} = U^{T} $$

<br>

- `SVD`를 구하는 근본적인 목적은 `선형 연립방정식`의 **해를 찾거나 근사화 해를 찾기 위함**입니다.
- 일반적으로 $$ Ax = b $$ 의 식에서 $$ A $$ 는 m x n 크기의 행렬이고 $$ x, b $$ 는 열벡터일 때, 이 식을 만족하는 열벡터 $$ x $$ 를 찾는 것이 선형 대수학의 근본적인 질문입니다.
- 이와 같은 문제를 풀 때, 크게 3가지 경우의 수가 발생합니다. `① 해가 1개 존재하는 경우`, `② 해가 여러개 존재하는 경우`, `③ 해가 존재하지 않아 근사화 하는 경우` ( $$ \text{min} \Vert Ax - b \Vert $$ ) 입니다.
- `SVD`를 이용하면 3가지 경우의 수에 대하여 모두 동일한 방법으로 접근할 수 있습니다. 따라서 `SVD`를 통하여 `선형 연립방정식`의 **해를 찾거나 근사화 해를 찾기 위한 일반화 방법**을 얻을 수 있습니다.

<br>

## **SVD 계산 방법**

<br>

- 어떤 행렬 $$ A $$ 를 특이값 분해를 하면 $$ U, \Sigma, V $$ 로 분해가 됩니다. 그러면 어떤 방법으로 분해할 수 있을까요? 먼저 간단하게 분해 방법에 대하여 서술해보겠습니다.
- 행렬 $$ A $$ 의 `특이값`들은 $$ AA^{T} $$ 또는 $$ A^{T}A $$ 의 **0이 아닌 고유값들에 루트를 적용**한 것입니다. 이 때, $$ AA^{T} $$ 와 $$ A^{T}A $$ 는 `동일한 고유값`들을 가집니다. (이러한 이유는 아래 `SVD 관련 성질` 부근을 참조하시면 됩니다.)
- 여기서 $$ U $$ 는 $$ AA^{T} $$ 의 고유벡터 행렬이고 $$ V $$ 는 $$ A^{T}A $$ 의 고유벡터 행렬입니다. 앞으로는 이 벡터들을 `특이 벡터 (Singular vector)`라고 하며 $$ U $$ 의 열벡터를 `left singular vectors`, $$ V $$ 의 열벡터를 `right singular vectors`라고 부르겠습니다.
- [고유값 분해](https://gaussian37.github.io/math-la-evd/)에서 다룬 바와 같이 `대칭 행렬 (symmetric matrix)`은 항상 고유값 분해가 가능하며 `직교 행렬 (orthogonal matrix)`로 대각화 할 수 있습니다. $$ AA^{T} $$ 와 $$ A^{T}A $$ 는 모두 `대칭 행렬`이므로 고유값 분해가 가능하여 항상 $$ U $$ , $$ V $$ 를 구할 수 있습니다.
- 그리고 $$ U $$ 와 $$ V $$ 는 `정규 직교 벡터`들을 열벡터로 갖는 `직교 행렬`인데 처음 $$ r $$ 개의 열벡터는 0이 아닌 고유값들에 해당하는 고유벡터들로 채우면 되고 (고유값과 고유벡터의 짝이 맞아야 합니다.) 나머지는 그것들에 직교인 `정규 직교 벡터`를 자유롭게 찾아서 채워넣으면 됩니다. (이 부분은 아래 예제를 참조하시면 됩니다.)

<br>

- 이와 같은 방법을 사용하면 `SVD`를 할 수 있습니다. 여기서 $$ A^{T}A $$ 를 사용하는 이유는 $$ A^{T}A $$ 가 `대칭 행렬`이 되기 때문입니다.

<br>

- $$ (AA^{T})^{T} = (A^{T})^{T}A^{T} = AA^{T} $$

<br>

- `대칭 행렬`은 `고유값 분해`가 가능함을 이용하는 것이 `SVD`의 조건이고 따라서 $$ AA^{T} $$ 와 $$ A^{T}A $$ 를 모두 이용합니다.

<br>

## **SVD 간단 예제**

<br>

- 아래 행렬 $$ A $$를 특이값 분해 해보도록 하겠습니다.

<br>

- $$ A = \begin{bmatrix} -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} $$

<br>

- 먼저 행렬 $$ A $$의 특이값들을 찾기 위해 $$ AA^{T} $$와 $$ A^{T}A $$의 고유값을 구합니다. 먼저 $$ AA^{T} $$의 고유값부터 구해보도록 하겠습니다.

<br>

- $$ AA^{T} = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} $$

<br>

- 행렬 $$ AA^{T} $$의 고유값이 $$ \lambda_{1}, \lambda_{2} $$라고 하면 $$ AA^{T} $$의 고유값들의 합 $$ \lambda_{1} + \lambda_{2} $$은 $$ AA^{T} $$의 대각요소들의 합과 같고, 고유값들의 곱 $$ \lambda_{1}\lambda_{2} $$은 행렬식의 값과 같으므로 아래와 같습니다.

<br>

- $$ \lambda_{1} + \lambda_{2} = 4, \lambda_{1}\lambda_{2} = 3 $$

<br>

- 따라서 $$ AA^{T} $$ 의 고유값들은 3, 1이 됩니다. 이것들에 **루트를 씌운 것**이 행렬 $$ A $$ 의 `특이값 (Singular Value)`이 됩니다. 따라서 $$ \sqrt{3}, 1 $$이 특이값이 됩니다. 특이값 행렬 $$ \Sigma $$는 특이값들을 대각요소로 갖고 있는 (m x n) 크기의 행렬로 이 문제에서는 (2 x 3) 행렬이 됩니다.

<br>

- $$ \Sigma = \begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} $$

<br>

- `특이값 행렬`의 `고유값`의 작성 순서는 **큰 값을 기준으로 내림차순 순서**로 작성하겠습니다. 이렇게 작성해야 하는 것은 아니나 고유값이 큰 값 순서대로 활용성이 커지기 때문에 이와 같은 방법을 흔히 사용합니다.
- 이번에는 $$ A^{T}A $$의 고유값을 구해보도록 하겠습니다. 앞에서 설명한 바와 같이 동일하게 3, 1이 나올 것입니다.

<br>

- $$ A^{T}A = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix} $$

- $$ \vert A^{T}A - \lambda I \vert = \begin{vmatrix} 1-\lambda & -1 & 0 \\ -1 & 2-\lambda & -1 \\ 0 & -1 & 1-\lambda \end{vmatrix} $$

- $$ (1-\lambda)^{2}(2-\lambda)-2(1-\lambda) = 0 $$

- $$ (1-\lambda)( (1-\lambda)(2-\lambda)-2 ) = 0 $$

- $$ (1-\lambda)(\lambda^{2} - 3\lambda) = 0 $$

- $$ \lambda(\lambda-1)(\lambda-3) = 0 $$

<br>

- 3, 1, 0이 $$ A^{T}A $$의 고유값으로 계산됩니다. 이 중에서 0이 아닌 고유값들은 3과 1이므로 $$ AA^{T} $$의 고유값들과 동일함을 확인할 수 있습니다. 특이값행렬을 구했으므로 이번에는 특이벡터행렬인 $$ U $$와 $$ V $$를 구해보도록 하곘습니다.
- 먼저 $$ U $$는 $$ AA^{T} $$의 고유벡터들을 열로 가진 행렬이므로 $$ AA^{T} $$의 고유벡터를 구해야 합니다.

<br>

- $$ AA^{T}x = \lambda x $$

- $$ (AA^{T} - \lambda I)x = 0 $$

<br>

- 위 식에서 $$ \lambda_{1} = 3, \lambda{2} = 1 $$ 을 각각 대입하여 고유벡터 $$ x_{1}, x_{2} $$ 를 구해보도록 하겠습니다. 먼저 $$ \lambda_{1} = 3 $$ 을 대입하여 구하면 다음과 같습니다.

<br>

- $$ (AA^{T} - \lambda_{1}I)x_{1} = \begin{bmatrix} 2 - 3 & -1 \\ -1 & 2 - 3 \end{bmatrix} x_{1} =  \begin{bmatrix} -1 & -1 \\ -1 & -1 \end{bmatrix} x_{1} = 0 $$

<br>

- 위 조건을 만족시키는 `정규직교`인 고유벡터를 구하면 다음과 같습니다.

<br>

- $$ x_{1} = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix} $$

<br>

- 그리고 앞의 방식과 동일하게 $$ AA^{T} $$ 의 고유값 $$ \lambda_{2} = 1 $$ 일 때의 고유벡터 $$ x_{2} $$ 를 찾으면 다음과 같습니다.

<br>

- $$ (AA^{T} - \lambda_{1}I)x_{2} = \begin{bmatrix} 2 - 1 & -1 \\ -1 & 2 - 1 \end{bmatrix} x_{2} =  \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} x_{2} = 0 $$

<br>

- 위 식을 만족시키는 `정규직교`인 고유벡터를 구하면 다음과 같습니다.

<br>

- $$ x_{2} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} $$

<br>

- 따라서 왼쪽 특이행렬 $$ U $$ 는 다음과 같습니다. $$ x_{1}, x_{2} $$ 의 순서는 고유값에 대응되며 앞에서 고유값이 큰 값을 기준으로 내림차순으로 사용하기로 하였습니다.

<br>

- $$ U = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 & 1 \\ 1 & 1 \end{bmatrix} $$

<br>

- 이와 동일한 방법으로 $$ A^{T}A $$의 고유벡터들로 이루어진 오른쪽 특이벡터행렬 $$ V $$를 구할 수 있습니다.

<br>

- $$ x_{1} = \frac{1}{\sqrt{6}}  \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} $$

- $$ x_{2} = \frac{1}{\sqrt{2}}  \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix} $$

<br>

- 고유값이 0에 해당하는 고유벡터는 다른 고유벡터와 직교인 임의의 `정규직교벡터`를 사용하면 됩니다. 일반적으로 다음과 같은 정규직교벡터를 사용하면 됩니다.

<br>

- $$ x_{3} = \frac{1}{\sqrt{3}}  \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} $$

<br>

- 따라서 오른쪽 특이벡터행렬 $$ V $$ 는 다음과 같습니다.

<br>

- $$ V = \begin{bmatrix} \frac{1}{\sqrt{6}} & \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{3}} \\ \frac{-2}{\sqrt{6}} & 0 & \frac{1}{\sqrt{3}} \\ \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{3}} \end{bmatrix} $$

<br>

- 이와 같이 $$ \Sigma, U, V $$를 모두 구했으므로 $$ A $$는 아래와 같이 특이값 분해가 됩니다.

<br>

- $$ \begin{align} A &= \begin{bmatrix} -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} = U \Sigma V^{T} \\ &= \frac{1}{\sqrt{2}} \begin{bmatrix} -1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{6}} & \frac{-2}{\sqrt{6}} & \frac{1}{\sqrt{6}} \\ \frac{-1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} \end{bmatrix} \end{align} $$

<br>

## **SVD 의미 해석**

<br>

- `SVD`는 임의의 행렬 $$ A $$ 를 $$ A = U \Sigma V^{T} $$ 로 분해하는 것을 확인하였습니다. 즉, 분해된 각 성분은 역할이 있는데 그 역할에 대하여 간략하게 살펴보겠습니다.
- 임의의 행렬 $$ A $$ 는 어떤 벡터 $$ x $$ 를 $$ x' $$ 로 변환할 때 사용됩니다. 즉, $$ x' = Ax $$ 가 되므로 $$ A $$ 는 선형 변환의 역할로 사용됩니다.
- 행렬 $$ A = U \Sigma V^{T} $$ 에서 $$ U, V $$ 는 `직교 행렬`이고 $$ \Sigma $$ 는 `대각 행렬` 입니다. `직교 행렬`은 `회전 변환`의 역할을 하고 `대각 행렬`dms `스케일 변환`을 하게 됩니다.
- 따라서 $$ x' = Ax = U \Sigma V^{T} x $$ 는 벡터 $$ x $$ 를 $$ V^{T} $$ 만큼 `회전 변환`을 한 후 $$ \Sigma $$ 만큼 `스케일 변환`을 한 다음에 다시 $$ U $$ 만큼 `회전 변환`을 적용하여 $$ x \to x' $$ 로 변환합니다. 그림으로 나타내면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/la/svd/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 `SVD`를 통해 얻은 `특이값`은 행렬의 `스케일 변환`에 사용됨을 알 수 있습니다.
- `EVD (고유값 분해)`에서는 `고유값`을 얻을 수 있고 이 `고유값`은 선형 변환에 의해 변환되지 않는 `고유 벡터`에 대한 스케일 값인 반면에 `SVD`에서 얻은 `특이값`은 선형 변환 자체의 스케일 값인 것을 알 수 있습니다. 즉, 선형 변환 $$ A $$ 에 의한 기하학적 변환은 특이값들에 의해서만 결정되는 것을 확인할 수 있습니다.

<br>

## **SVD 관련 성질**

<br>

- `SVD`를 다양한 관점에서 보면 `SVD`에 관련된 다양한 성질이 있음을 확인할 수 있습니다. `SVD`에 관한 다양한 성질을 나열해 보도록 하겠습니다. 아래 나열된 순서의 기준은 우선순위와는 무관합니다.

<br>

- `SVD`를 전개할 때 사용하는 $$ AA^{T} $$ 와 $$ A^{T}A $$ 의 `고유값`은 ① `모두 0 이상`이며 ② 0이 아닌 `고유값들은 서로 동일`하다는 것입니다. `특이값`이 `고유값`에 루트를 적용한 것이므로 `고유값`이 0보다 커야 하고 $$ A = U \Sigma V^{T} $$ 에서 하나의 행렬 $$ \Sigma $$ 를 사용하려면$$ AA^{T} $$ 와 $$ A^{T}A $$ 의 고유값은 같아야 합니다.
- 먼저 **첫번째 조건**인 $$ AA^{T} $$ 와 $$ A^{T}A $$ 의 `고유값`은 모두 0 이상임을 확인해 보도록 하겠습니다. 아래와 같이 고유값과 고유벡터의 정의를 이용하여 전개해 보겠습니다.

<br>

- $$ A^{T}Av = \lambda v ( v \ne 0) $$

- $$ v^{T}A^{T}Av = \lambda v^{T}v $$

- $$ (Av)^{T}Av = \lambda v^{T}v $$

- $$ \Vert Av \Vert^{2} = \lambda \Vert v \Vert^{2} $$

<br>

- 위 식에서 좌변과 우변이 제곱으로 모두 양수이어야 하기 때문에 $$ \lambda \ge 0 $$ 이 되어야 합니다.

<br>

- **두번째 조건**인 $$ AA^{T} $$ 와 $$ A^{T}A $$ 의 `고유값이 서로 동일`하다는 것을 확인해 보도록 하겠습니다.

<br>

- $$ (A^{T}A)v = \lambda v $$

- $$ A(A^{T}A)v = \lambda Av  $$

- $$ AA^{T}(Av) = \lambda (Av) $$

<br>

- 위 식에서 $$ Av \ne 0 $$ 을 만족해야 한다면 마지막 식에서 $$ AA^{T} $$ 의 `고유값`이 $$ \lambda $$ 가 되며 첫번째 식에서 $$ A^{T}A $$ 의 고유값이 $$ \lambda $$ 임을 전제로 시작하였기 때문에  $$ AA^{T} $$ 와 $$ A^{T}A $$ 의 고유값은 $$ \lambda $$ 로 동일해 집니다.

<br>

- 정리하면 $$ AA^{T} $$ 와 $$ A^{T}A $$ 의 `공통의 고유값`은 $$ \lambda_{1} \ge \lambda_{2} \ge \cdots \lambda_{r} \ge 0 $$ ($$ \text{rank}(A) = \text{rank}(AA^{T}) = \text{rank}(A^{T}A) = r $$) 이 되고 루트를 적용한 값 $$ \sqrt{\lambda_{1}} \ge \sqrt{\lambda_{2}} \ge \cdots \sqrt{\lambda_{r}} \ge 0 $$ (기호를 바꿔 쓰면) $$ \sigma_{1} \ge \sigma_{2} \ge \cdots \sigma_{r} \ge 0 $$ 이 `공통의 특이값`이 됩니다.

<br>

## **SVD with Python**

<br>




<br>

## **SVD의 활용**

<br>




<br>
