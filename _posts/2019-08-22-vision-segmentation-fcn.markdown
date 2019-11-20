---
layout: post
title: Fully Convolutional Networks for Semantic Segmentation
date: 2019-08-21 00:00:00
img: vision/segmentation/fcn/fcn.jpg
categories: [vision-segmentation] 
tags: [vision, segmentation, fcn] # add tag
---

- 이번 글에서 다루어 볼 논문은 `FCN`으로 유명한 **Fully Convolutional Networks for Semantic Segmentation** 입니다.

<br>

## **목차**

- ### FCN의 배경
- ### FCN 구조 설명 - upsampling
- ### FCN 구조 설명 - downsampling
- ### FCN 구조 설명 - skip connection

<br>

## **FCN의 배경**

<br>

- 이미지 처리를 위한 딥러닝 네트워크의 시작은 `CNN` 기반의 알렉스넷이 대회에서 성능을 거두면서 부터 시작되었습니다.
- '12년도에 `알렉스넷`을 시작으로 다양한 딥러닝 기반의 `classification`을 위한 네트워크가 고안되기 시작하였고
- '14년도에 `VGG`와 `GoogLeNet` 그리고 '15년도에 대망의 `ResNet` 이 나오면서 `classification` 성능에 비약적인 발전이 있어왔습니다.
- 이렇게 `classification`에서 좋은 성능을 보인 CNN 기반의 딥러닝 네트워크를 `Localization` 이나 `Segmentation`에도 적용시켜서 성능 향상을 해보려는 시도가 있었고 그 결과 확인하게 된것들이 `FCN`에 반영됩니다.

<br>

## **FCN 구조 설명 - upsampling**

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 `classification`을 할 때, 마지막 FC layer를 softmax로 출력을 하게 되면 확률 값이 나타나게 되고 가장 큰 확률값에 해당하는 클래스가 이미지에 해당하는 것으로 판단하게 됩니다.
- 이와 같은 classification 작업에서는 물체의 **위치 정보는 없고** 단지 물체가 어떤 물체인지에 대한 확률 값만 가지게 됩니다. 즉, 물체의 위치 정보를 잃어버리게 되는 것이지요.
- 하지만 입력 값을 보면 이미지 내의 어떤 물체에 대한 **공간 정보를 가지고 있었고** 하위 레벨로 내려가면서도 그 공간 정보는 계속 가지고 있었습니다. 
- 이런 공간 정보를 중간에 잃어버리게 되는 데 그 시점이 바로 `fully connected layer`가 되게 됩니다.
    - 왜냐하면 `fully connected layer`에서는 모든 노드들이 연결되어 버리기 때문입니다. (모든 노드들이 서로 곱해져서 더해지는 형태가 되지요)
- 따라서 classifier 용도로 사용한 `fully connected layer`를 사용하면 안되겠다고 생각하게 됩니다.

<br>

- 그래서 `fully connected layer`를 대신하여 [NIN(Network In Network, 1x1 Network)](https://gaussian37.github.io/dl-dlai-network_in_network/)를 사용하게 됩니다.
    - `NIN`은 현재 효율적인 네트워크 설계를 위해 많이 사용되었고(**차원 축소 및 연산량 감소**) 위에서 언급한 `Inception`에서도 사용되었습니다.
    - `NIN`은 이름 그대로 네트워크 에서 Multi layer perceptron의 역할을 수행하고 있습니다. (위 그림 참조)






- CNN으로 end-to-end, pixels-to-pixels로 학습한 Semantic Segmentation 모델이 좋은 성능을 보였습니다.
- 여기서 이 논문의 핵심은 **FCN(Fully Convolutional Network)** 가 임의의 사이즈의 이미지를 입력으로 받아서 그것에 상응하는 사이즈의 아웃풋을 만들어 내는 것에 있습니다.
  - 첨언하면 기존의 CNN 기반의 작업에서는 고정 입력, 고정 출력이 전제였는데 그 점을 개선한 것입니다.
- 논문에서 다루는 내용은 **FCN**에 대한 정의와 그것에 대한 상세화를 하고 **FCN**이 어떻게 공간상에 prediction 하는 작업을 하는 지 설명합니다. 그리고 기존에 사용되었던 딥러닝 모델과 연결해 보려고 합니다.
- 여기서는 현대 classification network를 FCN에 접목시켜 보았습니다. 그리고 transfer learning을 사용하였는데 기존에 학습된 representation을 segmentation 작업에 fine tuning 작업을 거쳤습니다.
- 그리고 **skip architecture** 구조에 대하여 설명을 하였는데, 이 구조는 깊은 layer와 얕은 layer를 결합하여 정확도와 디테일한 segmentation 작업에 도움을 줍니다.
- 결과적으로 이 논문은 그 당시에 segmentation 작업에 좋은 성능을 내었었습니다.

<br>

## **Introduction**

<br>

