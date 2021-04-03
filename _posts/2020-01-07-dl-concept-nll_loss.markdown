---
layout: post
title: Softmax와 Negative Log Likelihood Loss
date: 2020-01-07 00:00:00
img: dl/concept/nll_loss/0.png
categories: [dl-concept]
tags: [deep learning, loss function, softmax, negatics log likelihood] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

## **목차**

<br>

- ### Softmax
- ### Negative Log-Likelihood (NLL)
- ### Negative Log Likelihood에 대한 Softmax 함수의 미분
- ### Pytorch에서 사용 방법

<br>

- 이번 글에서는 `softmax`와 `Negative Log-Likelihood`를 사용하는 `NLL Loss`에 대하여 설명해 보겠습니다.

<br>

## **Softmax**

<br>

- `softmax`는 보통 뉴럴 네트워크의 출력 부분 layer에 위치하고 있으며 사용 목적은 multi-class classification과 같은 문제에서 최총 출력을 할 때에 출력값을 0과 1사이의 확률값과 같이 나타내기 위해 사용됩니다. 

<br>

- $$ S(f_{y_{i}}) = \frac{e^{f_{ y_{i} } }}{\sum_{j}e^{f_{j}}} $$

<br>

- 위 식과 같이 모든 출력되는 모든 아웃풋을 `exp()`를 이용하여 양수화 하고 normalization 작업을 거치면 모든 값이 0과 1 사이의 값이 될 뿐 아니라 총 합이 1이 되어 확률 처럼 다룰 수 있게 됩니다.
- 아래와 같이 3개의 이미지 고양이, 말, 개를 이용하여 한번 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/dl/concept/nll_loss/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 고양이 이미지 입력을 넣었을 때, 출력은 분홍색, 말은 파란색, 개는 초록색에 해당합니다. 가운데 행렬이 출력값에 해당하며 행렬의 첫번째 열은 고양이, 두번째 열은 개, 세번째 열은 말에 해당하며 가장 큰 값을 가지는 클래스가 선택됩니다.
- 첫번째 분홍색을 살펴보면 고양이가 열의 값이 5로 가장 크므로 고양이가 선택됩니다. 파란색을 살펴보면 세번째 열인 말이 가장 큰 값인 8을 가지므로 말이 선택됩니다.
- 초록색을 살펴보면 첫번째 열과 두번째 열의 값이 모두 4로 같으므로 어떤 것을 선택해야 할 지 애매한 상황에 놓이게 됩니다.

<br>

- 위와 같은 방법으로 가장 값이 큰 값을 선택해도 되지만 확률적으로 표현하면 좀 더 직관적으로 살펴볼 수 있습니다. 이 때, `softmax` 계산을 적용하여 가장 오른쪽 행렬과 같이 나타낼 수 있습니다. softmax 연산은 대소 크기를 바꾸지 않으므로 앞에서 선택된 것과 동일한 열이 선택되며 초록색을 살펴 보더라도 첫번째 열과 두번째 열의 값이 0.49로 같음을 확인할 수 있습니다.
- 이와 같은 방법을 통하여 `softmax`의 출력은 뉴럴 네트워크의 `confidence`와 같은 역할을 할 수 있습니다.

<br>

## **Negative Log-Likelihood (NLL)**

<br>

- 앞에서 다룬 `softmax`와 더불어서 `NLL`을 이용하여 딥러닝의 `Loss`를 만들 수 있습니다. 적용하는 방법은 간단합니다. 어떤 입력에 대한 `sofrmax` 출력 중 **가장 큰 값**을 다음 식에 적용하면 됩니다. $$ y $$는 softamx 출력값 중 가장 큰 값입니다.

<br>

- $$ L(\mathbf{y}) = -\log(\mathbf{y}) $$

<br>

- 위 식에서 **log 함수**와 **음의 부호**를 사용함으로써 softmax의 출력이 낮은 값은 $$ -log(y) $$ 값이 크도록 만들 수 있고 softmax의 출력이 높은 값은 $$ -log(y) $$의 값이 0에 가깝도록 만듭니다.

<br>
<center><img src="../assets/img/dl/concept/nll_loss/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 정답 클래스의 softmax 출력이 높은 값을 가진다면 출력이 정확한 것이므로 $$ -\log(\mathbf{y}) $$가 매우 작아지게 됩니다. 마치 Loss가 작아지는 것과 같습니다.
- 정답 클래스의 softmax 출력이 낮다면 $$ -\log(\mathbf{y}) $$는 큰 값을 가집니다. 즉, 출력이 정확하지 못한 부분에 대해서는 높은 Loss를 가지는 것 처럼 나타낼 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/nll_loss/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 따라서 위 그래프와 같이 softmax 출력에 대한 $$ -\log(\mathbf{y}) $$는 Loss 처럼 나타낼 수 있습니다.

<br>

## **Negative Log Likelihood에 대한 Softmax 함수의 미분**

<br>

- Loss function을 적용하기 위해서는 미분이 가능해야 하므로 **Negative Log Likelihood에 대한 Softmax 함수의 미분**을 해보도록 하겠습니다. 먼저 $$ f $$를 딥 러닝 네트워크의 출력인 클래스 별 최종 score를 저장한 벡터로 정의합니다. 
- 이 때, $$ f_{k} $$는 전체 클래스의 갯수가 $$ j $$인 벡터에서 $$ k $$번째 클래스의 score로 정의 됩니다.
- 먼저 softmax 함수는 다음과 같이 정의 할 수 있습니다.

<br>

- $$ p_k = \dfrac{e^{f_k}}{\sum_{j} e^{f_j}} $$

<br>

- 다음으로 softmax 함수의 출력을 받는 NLL을 다음과 같이 정의 할 수 있습니다.

<br>

- $$ L_i = -log(p_{y_{i}}) $$

<br>

- $$ \dfrac{\partial L_i}{\partial f_k} $$

<br>









<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
