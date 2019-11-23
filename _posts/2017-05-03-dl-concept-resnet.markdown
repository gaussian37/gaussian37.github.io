---
layout: post
title: 딥 러닝 관련 글 목차
date: 2017-05-03 00:00:00
img: dl/concept/resnet/resnet.png
categories: [dl-concept] 
tags: [python, deep learning, resnet, residual network] # add tag
---

<br>

- 이번 글에서는 `Residual Network`에 대하여 알아보겠습니다.  현재는 가장 기본이 되는 네트워크 중 하나인데, 처음에 나왔을 때에는 상당히 큰 성능 개선의 역할을 이루어 낸 중요한 뉴럴 네트워크입니다. 그러면 `resnet`에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

- ### ResNet 관련 배경 
- ### ResNet의 구조
- ### Skip Connection
- ### pytorch 코드

<br>

## **ResNet 관련 배경**

<br>
<center><img src="../assets/img/dl/concept/resnet/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `ResNet`은 Kaimimg He의 논문에서 소개 되었는데 classification 대회에서 기존의 20계층 정도의 네트워크 수준을 152 계층 까지 늘이는 성과를 거두었고 위의 그래프와 같이 에러율 또한 3.57%로 인간의 에러율 수준 (약 5%)을 넘어서게 된 시점이 되겠습니다.
- 여기서 2014년도의 VGG, GoogLeNet 같은 경우에는 레이어의 수가 20 내외 였는데, ResNet의 경우에는 152개로 7배 이상 레이어를 쌓는 결과를 보였습니다. 즉, ResNet 이전에는 **레이어를 계속 쌓는 데 문제가 있었기 때문에** 무한정 레이어를 쌓기 어려웠습니다. 하지만 ResNet에서 그 문제를 개선하였기 때문에 더 깊은 레이어로 성능을 낼 수 있었습니다.

<br>

## **ResNet의 구조**

- 전체적인 ResNet의 구조를 먼저 살펴보겠습니다.

<br>
<center><img src="../assets/img/dl/concept/resnet/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 ResNet은 레이어 사이 사이에 연결된 구조가 보이는 데 이것을 `skip connection` 이라고 합니다.
- 이것이 아주 중요한 역할을 하는데, 위 그림과 같이 **great gradient highway** 즉, gradient를 전달하기 위한 좋은 통로가 됩니다. 

<br>

## **Skip Connection**

<br>

<br>
<center><img src="../assets/img/dl/concept/resnet/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 왼쪽이 일반적인 구조의 (convolutional) neural network 입니다. 입력이 들어오면 layer를 거쳐서 (e.g. convolution filter 와의 연산) activation이 적용되고 이러한 작업이 연속적으로 이루어 지는 것입니다.
- 반면 오른쪽의 Residual 구조에서는 **입력을 출력과 더해주는 형태**를 가지게 됩니다.
- 즉, 위의 형태는 처음 제안되었던 skip connection의 구조로 **feature를 추출하기 전 후를 더하는 특징**을 가지고 있습니다.
