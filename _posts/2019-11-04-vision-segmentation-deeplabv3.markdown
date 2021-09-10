---
layout: post
title: DeepLab v3 (Rethinking Atrous Convolution for Semantic Image Segmentation)
date: 2019-11-04 00:00:00
img: vision/segmentation/deeplabv3/0.png
categories: [vision-segmentation] 
tags: [segmentation, deeplab v3+, deeplab, deeplab v3] # add tag
---

<br>

- 참조 : https://arxiv.org/abs/1706.05587
- 참조 : https://medium.com/free-code-camp/diving-into-deep-convolutional-semantic-segmentation-networks-and-deeplab-v3-4f094fa387df
- 참조 : https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42

<br>

- 이번 글에서는 Semantic Segmentation 내용과 DeepLab v3 내용에 대하여 간략하게 알아보도록 하겠습니다.
- DeepLab v3의 핵심은 `ASPP (Atrous Spatial Pyramid Pooling)`이며 이 개념의 도입으로 DeepLab v2 대비 성능 향상이 되었고 이전에 사용한 추가적인 Post Processing을 제거함으로써 End-to-End 학습을 구축하였습니다.

<br>

## **목차**

<br>

- ### Semantic Segmentation
- ### Model Architecture
- ### ResNets
- ### Atrous Convolutions
- ### Atrous Spatial Pyramid Pooling
- ### Implementation Details
- ### Results

<br>

## **Semantic Segmentation**

<br>
<center><img src="../assets/img/vision/segmentation/deeplabv3/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 기본적인 classification 문제를 다루는 CNN 구조를 나타냅니다. 기본적인 convolution, activation function, pooling, fc layer 등을 가지는 것을 알 수 있으며 입력 이미지를 받았을 때, 이러한 operation들을 이용하여 feature vector를 만들어내고 이 값을 이용하여 classification을 하게 됩니다. (이미지가 어떤 클래스에 해당하는 지 출력합니다.)

<br>
<center><img src="../assets/img/vision/segmentation/deeplabv3/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- image classification 문제와는 다르게 semantic segmentation에서는 모든 픽셀에 대하여 픽셀 단위 별로 classification을 하고자 합니다. 즉, semantic segmentation에서는 이미지 내의 모든 픽셀에 대한 의미를 이해해야 합니다.
- 일반적으로 사용하는 image classification 모델을 이용하여 단순히 semantic segmentation을 하면 잘 동작하지는 않습니다. 크게 2가지 이유가 있습니다.
- ① image classification 모델은 `input feature`의 `spatial dimension`을 줄이는 데 집중되어 있습니다. 결과적으로 이러한 목적으로 만들어진 layer는 semantic segmentation을 할 만큼 디테일한 정보를 가지고 있기 어려워 집니다. 첫번째 그림의 예시를 살펴보면 `feature learning`이란 부분에서 feature의 크기가 계속 작아지는 것을 살펴볼 수 있습니다.
- ② fully connected layer는 고정된 크기의 layer만 가질 수 있으며 계산 과정 중에 공간적인 정보를 잃어버리는 성질이 있습니다. 

<br>
<center><img src="../assets/img/vision/segmentation/deeplabv3/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위 그림과 같은 예시에서는 pooling 또는 FC layer 대신에 연속적인 convolution을 거치도록 만들었습니다. 특히 convoltuion의 `stride=1` 과 `padding=same` 이라는 조건을 사용하였는데 이와 같은 조건의 효과는 **convolution이 input의 sptial dimension을 보존**하도록 합니다.
- 따라서 위 구조에서는 단순히 convolution layer을 쌓기만 하였고 그 결과 segmentation 모델을 만들 수 있습니다.
- 이 모델은 `(W, H, C)`와 같은 형태의 출력을 가지게 되며 `W`, `H`는 입력 이미지의 사이즈와 같으며 `C`는 각 픽셀 별 구분하고자 하는 클래스의 수와 같아집니다. 마지막 `C`의 갯수 만큼 확률 분포를 가지게 되며 `argmax` 연산을 통하여 가장 큰 확률 값을 가지는 클래스를 선택하였을 때, `(W, H, 1)`의 크기를 가지는 segmentation 결과를 가지게 됩니다.
- 이 결과를 이용하여 `Cross Entropy`와 같은 Loss function을 통해 실제 ground-truth 이미지와의 차이를 학습을 하게 됩니다.

<br>

- 위 과정은 전체적인 semantic segmentation 과정을 잘 설명합니다. 하지만 **효율성에 문제가 있습니다.**
- 위 구조와 같이 단순히 `stride=1`, `padding=same`을 가지는 convolution layer를 계속 쌓게 되면 공간 정보를 계속 유지할 수 있다는 장점은 있지만, 연산량이 많이 증가한다는 단점이 있어 메모리 낭비가 심하며 한 개의 입력을 처리하는 데 처리 시간도 많이 필요합니다.

<br>
<center><img src="../assets/img/vision/segmentation/deeplabv3/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이러한 문제를 해결하기 위하여 가장 많이 사용되는 방법이 위 그림과 같이 `downsampling`와 `upsampling` layer를 사용하여 연산량을 줄이는 방법입니다.
- 먼저 `downsampling` 방법을 살펴보면 `convolution layer`를 `stride` 또는 `pooling` 연산과 함께 사용하는 방법이 있습니다. `downsampling`의 목적은 feature map의 spatial dimension을 줄여서 효율성을 높이는 것입니다. 위 그림의 `Encoder`영역을 보면 이와 같은 역할을 볼 수 있습니다. 연산의 효율성을 높이는 대신에 feature 일부를 소실한 것을 확인할 수 있습니다.
- 이와 같은 `Encoder` 구조는 `classification`에서 fully connected layer를 적용하기 이전의 구조와 같습니다. 즉 `feature extraction`을 하는 역할을 한다고 볼 수 있습니다.