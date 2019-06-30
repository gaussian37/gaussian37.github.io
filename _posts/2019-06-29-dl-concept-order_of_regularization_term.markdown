---
layout: post
title: BatchNormalization, Dropout, Pooling 적용 순서
date: 2019-06-30 00:00:00
img: dl/concept/deep-neural-network.jpg
categories: [dl-concept] 
tags: [python, deep learning, batchnormalization, dropout, order, 순서] # add tag
---

- 딥러닝 프레임워크를 이용하여 네트워크를 구성할 때 다양한 정규화 기법들을 사용하곤 합니다.
- 배치 정규화(batch normaliation), 드랍 아웃을 사용할 때, 어떤 순서로 사용가능한 지 알아보겠습니다.
- 출처 : [stackoverflow](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout#answer-40295999)
- 아래 내용은 배치 정규화와 드랍아웃의 **저자**들의 의견만 정리한 것입니다. 

<br>

- 먼저 배치 정규화의 경우 저자에 따르면 Convoluter Layer 또는 Fully Connected Layer 등의 layer 뒤에 적용되는 것이 맞다고 봅니다.
- 그리고 ReLU와 같은 Activation function을 적용하기 전에 적용하는 것을 추천하고 있습니다.
- 왜냐하면 배치 정규화의 목적이 **네트워크 연산 결과가 원하는 방향의 분포대로 나오는 것이기 때문에** 핵심 연산인 Convolution 연산 뒤에 바로 적용하여 정규화 하는 것이 핵심입니다.
    - 즉, Activation function이 적용되어 분포가 달라지기 전에 적용하는 것이 올바릅니다.

<br>

- 반면 dropout의 경우 저자에 따르면, activation function을 적용한 뒤에 적용하는 것으로 제안되었습니다.

<br>

- 마지막으로 Convolution 연산과 함께 사용되는 pooling 연산은 정규화 기법 적용이 끝난 뒤에 적용하는 것을 추천합니다.
- 이것은 제 경험으로 말씀드리는 것으로 정확하지 않을 수 있지만, dropout 이후에 적용하니 성능이 잘 나왔었습니다.(물론 task 마다 다르겠지요?)

<br>

- 정리하면 **`Convolution - Batch Normalization - Activation - Dropout - Pooling`** 순서로 네트워크를 구성하면 됩니다. 
- 아래 그림은 CS231n의 Batchn Normalization 관련 강의 자료 중 일부입니다.

<center><img src="../assets/img/dl/concept/order_of_regularization/1.PNG" alt="Drawing" style="width: 600px;"/></center>
