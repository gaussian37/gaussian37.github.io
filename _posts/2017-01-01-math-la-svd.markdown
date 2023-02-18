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

- 특이값 분해는 고유값 분해와 같이 행렬을 `대각화`하는 방법 중 하나입니다. 고유값 분해는 정방 행렬에만 사용가능한 반면, 특이값 분해는 `직사각형 행렬일 때에도 사용 가능`하므로 활용도가 높습니다. m x n 크기의 행렬 $$ A $$를 특이값 분해하였을 때 다음과 같이 분해됩니다.

<br>

- $$ A = U \Sigma V^{t} $$

- $$ A : m \times n \text{    (rectangular matrix)} $$

- $$ U : m \times m \text{    (orthogonal matrix)} $$

- $$ \Sigma : m \times n \text{    (diagonal matrix)} $$

- $$ V : n \times n \text{    (orthogonal matrix)} $$

<br>

- 여기서 $$ U, V $$는 각각 서로 다른 `직교행렬`인 `특이 벡터 행렬`이고 $$ \Sigma $$는 특이값 $$ \sigma_{1}, \sigma_{1}, \cdots \sigma_{r} $$들을 대각요소로 갖고 있는 대각 행렬로서 `특이값 행렬`이라고 불립니다. $$ \sigma_{r} $$의 $$ r $$은 대각행렬의 `rank`를 의미합니다.
- 특이값 행렬은 (m x n) 크기의 직사각행렬이므로 m과 n의 크기에 따라 다음과 같은 형태를 가질 수 있습니다.

<br>
<center><img src="../assets/img/math/la/svd/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 직교 행렬(orthogonal matrix)와 대각 행렬(diagonal matrix)에 대한 성질을 살펴보면 다음과 같습니다. $$ U $$를 직교 행렬이라고 하겠습니다.

<br>

- $$ UU^{T} = U^{T}U = I $$

- $$ U^{-1} = U^{T} $$

<br>





<br>

## **SVD 계산 방법**

<br>

- 어떤 행렬 $$ A $$를 특이값 분해를 하면 $$ U, \Sigma, V $$로 분해가 됩니다. 그러면 어떤 방법으로 분해할 수 있을까요? 먼저 간단하게 분해 방법에 대하여 서술해보겠습니다.
- 행렬 $$ A $$의 특이값들은 $$ AA^{T} $$ 또는 $$ A^{T}A $$ 의 0이 아닌 고유값들에 루트를 적용한 것입니다.
- 이 때, $$ AA^{T} $$ 와 $$ A^{T}A $$ 는 `동일한 고유값`들을 가집니다.
- 여기서 $$ U $$는 $$ AA^{T} $$의 고유벡터 행렬이고 $$ V $$는 $$ A^{T}A $$의 고유벡터 행렬입니다. 
- 그리고 $$ U $$와 $$ V $$는 `정규직교벡터`들을 열벡터로 갖는 `직교 행렬`인데 처음 r개의 열벡터는 0이 아닌 고유값들에 해당하는 고유벡터들로 채우면 되고 나머지는 그것들에 직교인 정규직교벡터를 자유롭게 찾아서 채워넣으면 됩니다.



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

- 따라서 $$ AA^{T} $$의 고유값들은 3, 1이 됩니다. 이것들에 루트를 씌운 것이 행렬 $$ A $$의 특이값이 됩니다. 따라서 $$ \sqrt{3}, 1 $$이 특이값이 됩니다. 특이값 행렬 $$ \Sigma $$는 특이값들을 대각요소로 갖고 있는 (m x n) 크기의 행렬로 이 문제에서는 (2 x 3) 행렬이 됩니다.

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

- $$ \begin{align} A &= \begin{bmatrix} -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix} = U \Sigma V^{T} \\ &= \frac{1}{\sqrt{2}} = \begin{bmatrix} -1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{6}} & \frac{-2}{\sqrt{6}} & \frac{1}{\sqrt{6}} \\ \frac{-1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} \end{bmatrix} \end{align} $$

<br>

## **SVD 의미 해석**

<br>




<br>

## **SVD 관련 성질**

<br>



<br>

## **SVD with Python**

<br>




<br>

## **SVD의 활용**

<br>




<br>
