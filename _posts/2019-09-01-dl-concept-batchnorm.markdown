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
- ### 
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
<center><img src="../assets/img/dl/concept/batchnorm/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

