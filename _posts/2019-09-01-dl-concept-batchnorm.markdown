---
layout: post
title: 배치 정규화(Batch Normalization)
date: 2019-09-01 00:00:00
img: dl/concept/batchnorm/batchnorm.png
categories: [dl-concept] 
tags: [배치 정규화, 배치 노멀라이제이션, batch normalization] # add tag
---

<br>

- 참조
- https://youtu.be/TDx8iZHwFtM?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS

- 이번 글에서는 딥러닝 개념에서 중요한 것 중 하나인 `batch normalization`에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### Batch
- ### Internal Covariant Shift
- ### Batch Normalization
- ### Internal Covariant Shift 더 알아보기]
- ### Batch Normalization의 효과
- ### Pytorch에서의 사용 방법

<br>

## **Batch**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 배치 정규화를 설명하기에 앞서서 gradient descent 방법에 대하여 한번 생각해 보도록 하겠습니다.
- 먼저 위와 같이 일반적인 gradient descent에서는 gradient를 한번 업데이트 하기 위하여 모든 학습 데이터를 사용합니다.
- 즉, 학습 데이터 전부를 넣어서 gradient를 다 구하고 그 모든 gradient를 평균해서 한번에 모델 업데이트를 합니다. 
- 이런 방식으로 하면 대용량의 데이터를 한번에 처리하지 못하기 때문에 데이터를 `batch` 단위로 나눠서 학습을 하는 방법을 사용하는 것이 일반적입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 그래서 사용하는 것이 stochastic gradient descent 방법입니다.
- SGD에서는 gradient를 한번 업데이트 하기 위하여 `일부의 데이터`만을 사용합니다. 즉, `batch` size 만큼만 사용하는 것이지요.
- 위 gradient descent의 식을 보면 $$ \Sum $$에 $$ j = Bi $$가 있는데 $$ B $$가 `batch`의 크기가 됩니다. 
- 한 번 업데이트 하는 데 $$ B $$개의 데이터를 사용하였기 때문에 평균을 낼 때에도 $$ B $$로 나누어 주고 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 용어를 살펴보면 학습 데이터 전체를 한번 학습하는 것을 `Epoch` 라고 하고 Gradient를 구하는 단위를 `Batch` 라고 합니다. 

<br>

## **Internal Covariant Shift**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- `Batch` 단위로 학습을 하게 되면 발생하는 문제점이 있는데 이것이 논문에서 다룬 **Internal Covariant Shift** 입니다.
- 먼저 Internal Covariant Shift 의미를 알아보면 위 그림과 같이 학습 과정에서 계층 별로 입력의 데이터 분포가 달라지는 현상을 말합니다.
- 각 계층에서 입력으로 feature를 받게 되고 그 feature는 convolution이나 위와 같이 fully connected 연산을 거친 뒤 activation function을 적용하게 됩니다.
- 그러면 연산 전/후에 데이터 간 분포가 달라질 수가 있습니다. 
- 이와 유사하게 Batch 단위로 학습을 하게 되면 Batch 단위간에 데이터 분포의 차이가 발생할 수 있습니다.
- 즉, Batch 간의 데이터가 상이하다고 말할 수 있는데 위에서 말한 Internal Covariant Shift 문제입니다.
- 이 문제를 개선하기 위한 개념이 **Batch Normalization** 개념이 적용됩니다.

<br>

## **Batch Normalization**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- batch normalization은 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도 **각 배치별로 평균과 분산을 이용해 정규화**하는 것을 뜻합니다. 
- 위 그림을 보면 batch 단위나 layer에 따라서 입력 값의 분포가 모두 다르지만 정규화를 통하여 분포를 zero mean gaussian 형태로 만듭니다. 
- 그러면 평균은 0, 표준 편차는 1로 데이터의 분포를 조정할 수 있습니다.

<br>

$$ BN(X) = \gamma \Bigl(  \frac{X - \mu_{batch} }{\sigma_{batch} } \Bigr) + \beta $$

<br>

- 여기서 $$ \gamma $$는 스케일링 역할을 하고 $$ \beta $$는 bias입니다. 물론 backprpagation을 통하여 

