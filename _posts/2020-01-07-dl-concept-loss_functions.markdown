---
layout: post
title: Loss function 모음
date: 2020-01-07 00:00:00
img: dl/concept/loss_functions/0.png
categories: [dl-concept]
tags: [deep learning, loss function, softmax, negatics log likelihood] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 이 글에서는 다양한 Loss function들에 대하여 정리하려고 합니다. 기본적으로 Framework에서 제공하는 Loss function 뿐 아니라 최근 논문에서 제안한 Loss function들도 포함합니다.
- Pytorch를 이용한 Loss function의 구현 전체는 아래 링크에서 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/dl-pytorch-loss_functions/](https://gaussian37.github.io/dl-pytorch-loss_functions/)

<br>

## **목차**

<br>

- ### Softmax와 Negative Log-Likelihood Loss

<br>

- 이번 글에서는 `softmax`와 `Negative Log-Likelihood`를 사용하는 `NLL Loss`에 대하여 설명해 보겠습니다.

<br>

#### **softmax**

<br>

- `softmax`는 보통 뉴럴 네트워크의 출력 부분 layer에 위치하고 있으며 사용 목적은 multi-class classification과 같은 문제에서 최총 출력을 할 때에 출력값을 0과 1사이의 확률값과 같이 나타내기 위해 사용됩니다. 

<br>

- $$ S(f_{y_{i}}) = \frac{e^{f_{ y_{i} } }}{\sum_{j}e^{f_{j}}} $$

<br>

- 위 식과 같이 모든 출력되는 모든 아웃풋을 `exp()`를 이용하여 양수화 하고 normalization 작업을 거치면 모든 값이 0과 1 사이의 값이 될 뿐 아니라 총 합이 1이 되어 확률 처럼 다룰 수 있게 됩니다.
- 아래와 같이 3개의 이미지 고양이, 말, 개를 이용하여 한번 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/dl/concept/loss_functions/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 고양이 이미지 입력을 넣었을 때, 출력은 분홍색, 말은 파란색, 개는 초록색에 해당합니다. 가운데 행렬이 출력값에 해당하며 행렬의 첫번째 열은 고양이, 두번째 열은 개, 세번째 열은 말에 해당하며 가장 큰 값을 가지는 클래스가 선택됩니다.
- 첫번째 분홍색을 살펴보면 고양이가 열의 값이 5로 가장 크므로 고양이가 선택됩니다. 파란색을 살펴보면 세번째 열인 말이 가장 큰 값인 8을 가지므로 말이 선택됩니다.
- 초록색을 살펴보면 첫번째 열과 두번째 열의 값이 모두 4로 같으므로 어떤 것을 선택해야 할 지 애매한 상황에 놓이게 됩니다.

<br>

- 위와 같은 방법으로 가장 값이 큰 값을 선택해도 되지만 확률적으로 표현하면 좀 더 직관적으로 살펴볼 수 있습니다. 이 때, `softmax` 계산을 적용하여 가장 오른쪽 행렬과 같이 나타낼 수 있습니다. softmax 연산은 대소 크기를 바꾸지 않으므로 앞에서 선택된 것과 동일한 열이 선택되며 초록색을 살펴 보더라도 첫번째 열과 두번째 열의 값이 0.49로 같음을 확인할 수 있습니다.
- 이와 같은 방법을 통하여 `softmax`의 출력은 뉴럴 네트워크의 `confidence`와 같은 역할을 할 수 있습니다.

<br>

#### **Negative Log-Likelihood (NLL)**

<br>

- 앞에서 다룬 `softmax`와 더불어서 `NLL`을 이용하여 딥러닝의 `Loss`를 만들 수 있습니다. 적용하는 방법은 간단합니다. `sofrmax` 출력에 다음 식을 적용하면 됩니다.

<br>

- $$ L(\mathbf{y}) = -\log(\mathbf{y}) $$

<br>

- 위 식에서 **log 함수**와 **음의 부호**를 사용함으로써 

<br>
<center><img src="../assets/img/dl/concept/loss_functions/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>


<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
