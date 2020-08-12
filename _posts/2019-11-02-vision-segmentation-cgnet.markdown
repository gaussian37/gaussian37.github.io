---
layout: post
title: CGNet, A Light-weight Context Guided Network for Semantic Segmentation
date: 2019-11-05 00:00:00
img: vision/segmentation/cgnet/0.png
categories: [vision-segmentation] 
tags: [segmentation, cgnet] # add tag
---

<br>

- 논문 : https://arxiv.org/abs/1811.08201
- 코드 : https://github.com/wutianyiRosun/CGNet
- Cityscape Benchmarks 성능 : ① IoU class : 64.8 %, ② Runtime : 20 ms
    - 링크 : https://www.cityscapes-dataset.com/method-details/?submissionID=2095


<br>

- 이번 글에서는 CGNet, A Light-weight Context Guided Network for Semantic Segmentation 에 대하여 알아보도록 하겠습니다.
- Network의 이름에도 포함이 되어 있듯이 `Light-weight` 이므로 weight의 수가 작은 Realtime 용도의 Segmentation 모델입니다.

<br>

## **목차**

<br>

- ### Abstract
- ### Introduction
- ### Related Work
- ### Proposed Approach
- ### Experiments
- ### Conclusion
- ### Pytorch code

<br>

## **Abstract**

<br>

- 세그멘테이션(semantic segmentation)을 모바일 디바이스 환경에 적용하려는 시도가 많이 증가하고 있습니다.
- 성능이 좋은 세그멘테이션 모델들은 많은 파라미터와 연산량으로 인해 모바일 디바이스에는 적합하지 않기 때문에 모바일 디바이스에는 경량화 모델이 필요합니다.
- 경량화 세그멘테이션 모델에 대한 연구들의 일부 문제점은 classification에서 사용된 방법들을 사용하고  세그멘테이션에서 고려해야 할 특성들을 무시한 상태로 구조가 만들어 졌다는 것에 있습니다.
- 이 논문에서는 이러한 문제점들을 개선하기 위하여 `Context Guided Network (CGNet)`을 소개합니다. 이 모델 또한 가볍고 계산에 효율적인 세그멘테이션 모델입니다.
- CGNet에서 사용된 `CG block`은 **local feature**와 local feature를 둘러싼 **surrounding context**를 학습합니다. 그리고 더 나아가 **global context**와 관련된 feature 또한 이용하여 성능을 향상시킵니다. CGNet은 `CG block`을 기반으로 네트워크의 모든 단계에서 상황에 맞는 정보를 이해하고 세그멘테이션 정확도를 높이기 위해 설계됩니다. (local feature는 convolutional filter가 연산되는 영역입니다.)
- CGNet은 또한 **파라미터 수를 줄이고 메모리 공간을 절약**하도록 정교하게 설계되었습니다. 동등한 수의 매개 변수 하에서 제안된 CGNet은 기존 세그먼테이션 네트워크보다 훨씬 뛰어납니다.
- Cityscape 및 CamVid 데이터 세트에 대한 광범위한 실험은 제안된 접근 방식의 효과를 검증합니다.
- 특히 post-processing 및 multi-scale testing 없이 제안 된 CGNet은 0.5M 미만의 매개 변수로 64.8 %의 Cityscape에서 평균 IoU를 달성합니다.

<br>

## **Introduction**

<br>
<center><img src="../assets/img/vision/segmentation/cgnet/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 최근 자율주행 및 로봇 시스템에 대한 관심이 높아지면서 모바일 장치에 세그멘테이션 모델을 배치해야 한다는 요구가 거세지고 있습니다. 하지만 작은 메모리을 사용하면서 높은 정확도를 모두 갖춘 모델을 설계하는 것은 중요하고 어려운 일입니다.
- 위 그림은 Cityscape 데이터 셋에서 여러 가지 모델의 정확도와 매개변수 수를 보여줍니다. 그래프의  파란색 점은 정확도가 높은 모델을 나타내고 빨간색 점은 메모리 사용량이 작은 모델을 나타냅니다. `CGNet`은 메모리 설치 공간이 작은 방법에 비해 파라미터 수가 적으면서도 정확도가 높아 왼쪽 상단에 위치합니다.
- 위 그래프의 파란색 점에 해당하는 모델들은 모바일 디바이스에서 사용하기 적합하지 않습니다.
- 반면 빨간색 점들은 이미지 분류의 설계 원리를 따를 뿐 세그멘테이션의 고유한 속성은 무시하기 때문에 세그멘테이션 정확도가 낮습니다.
- 따라서 CGNet은 정확성을 높이기 위해 **세그멘테이션의 내재적 특성을 활용**하는 방법으로 설계됩니다.

<br>

- 세그멘테이션은 픽셀 수준 분류와 개체 위치 지정을 모두 포함합니다. 따라서 공간 의존성(spatial dependency)과 상황별 정보(contextual information)는 정확성을 향상시키는 중요한 역할을 합니다.
- ① `CG 블록`은 local feature와 주변 context가 결합된 joint feature를 학습합니다. 따라서 CG블록은 local feature와 local feature 주변의 context가 공간 상 공유하는 특징들을 잘 학습하게 됩니다.
- ② `CG 블록`은 global context를 사용하여 ①에서 만든 joint feature를 개선합니다. global context는 유용한 구성요소를 강조하고 쓸모 없는 구성요소를 억제하기 위해 채널별로 joint feature의 가중치를 재조정하는 데 적용됩니다. global context에 대한 상세 내용은 뒤에서 알아보겠습니다.
- ③ `CG 블록`은 CGNet의 모든 단계에서 활용됩니다. 따라서, CCNet은 (깊은 레이어) semantic level 과 (얕은 레이어) spatial level 모두에서 context 정보를 캡처합니다. 이는 기존 이미지 분류 방법에 비해 세그멘테이션에 더 적합합니다.

<br>

- 기존 세그멘테이션 프레임워크는 두 가지 유형으로 나눌 수 있습니다.

<br>

- 앞에서 다룬 CGNet의 성과에 대하여 정리하면 다음과 같습니다.
- ① local feature와 local feature의 주변 context feature를 합친 joint feautre를 학습하고 global context로 joint feature를 더욱 향상시키는 CG 블록을 제안하여 세그멘테이션 성능을 높였습니다.
- ② CG 블록을 적용하여 모든 단계에서 context 정보를 효과적이고 효율적으로 캡처하는 CGNet을 설계하였습니다. 특히, CCNet의 backbone은 세그멘테이션 정확도를 높이기 위해 맞춤 제작되었습니다.
- ③ 파라미터 수와 메모리 사용량을 줄이기 위해 CCNet의 아키텍처를 정교하게 설계하였습니다. 동일한 수의 매개 변수에서 제안된 CGNet은 기존 세그멘테이션 네트워크(예: ENet 및 ESPNet)의 성능을 크게 능가합니다.

<br>

## **Related Work**

<br>

- Related Work에서는 CGNet과 관련된 작은 세그멘테이션 모델(small semantic segmentation model), 상황별 정보(contextual information) 모델 그리고 어텐션 모델에 대하여 간략하게 다루어 보겠습니다.

<br>

#### **Small semantic segmentation models**

<br>

- 작은 세그멘테이션 모델을 사용하려면 정확성과 모델 매개변수 또는 메모리 공간 간에 적절한 trade-off가 필요합니다.
- `ENet`은 FCN과 같은 기존 세그멘테이션 모델의 마지막 단계를 제거하는 방법을 제안하고 임베디드 장치에서 세그멘테이션이 가능하다는 것을 보여주었습니다.
- 반면 그러나 `ICNet`은 compressed-PSPNet 기반 이미지 캐스케이드 네트워크를 제안하여 의미 분할 속도를 높였습니다.
- 최근의 `ESPNet`에서는 리소스 제약 하에서 고해상도 이미지를 세그멘테이션할 수 있는 빠르고 효율적인 콘볼루션 네트워크를 도입했습니다.
- 하지만 `ENet`, `ICNet`, `ESPNet`과 같은 모델 대부분은 영상 분류의 설계 원리를 따르기 때문에 픽셀 별 세그멘테이션 정확도가 떨어집니다.

<br>

#### **Contextual information models**

<br>

- 최근 연구에서는 상황별 정보가 고품질 세그멘테이션 결과를 예측하는 데 도움이 된다는 것을 보여 주었습니다.
- 한 가지 방법은 필터의 receptive field를 확대하거나 또는 상황에 맞는 정보를 캡처하도록 특정 모듈을 구성하는 것입니다.
- 예를 들어 `dilation 8`은 Class likelihood map 이후에 multiple dilated convolutional layers을 사용하여 exercise multi-scale context를 합칩니다. (aggregation)
- 또는 `SAC`(scale-adaptive convolution)는 가변적인 크기의 receptive field를 적용합니다.
- `DeepLab v3`는 **ASPP, Atrous Spatial Pyramid Pooling**을 도입합니다. ASPP를 이용하여 상황별 정보를 다양한 크기(스케일)로 얻을 수 있습니다.

<br>

#### **Attention models**

<br>

## **Proposed Approach**

<br>

- 지금 부터 `CG block`에 대하여 다루어 보고 CG block과 유사한 다른 구조의 block과 비교를 해보겠습니다.

<br>

#### **Context Guided Block**

<br>
<center><img src="../assets/img/vision/segmentation/cgnet/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 위 그림을 설명하고 Context Guided Block에 대하여 다루어 보도록 하겠습니다.
- 먼저 (a)를 보면 그림의 조그마한 노란색 영역만 보았을 때, 그 영역에 해당하는 클래스가 무엇인지 판단하기가 어렵습니다.
- 반면 (b)와 같이 노란색 영역 주위에 빨간색 영역을 포함하여 같이 본다면 인식하기 좋아집니다. 여기서 빨간색 영역을 `surrounding context` 라고 합니다.
- 마지막으로 (c) 그림을 보면 전체 이미지를 포함하는 보라색의 사각형이 있습니다. 전체 영역을 이용하여 노란색 영역의 클래스가 무엇인 지 판단한다면 더 높은 정확도로 판단할 수 있습니다. 보라색 사각형을 `global context` 라고 하겠습니다.
- CG 블록의 형태는 (d)와 같습니다. 블록의 구성 요소 중 $$ f_{loc}(*) $$ 이 그림의 노란색 영역에 해당하는 **local feature**입니다. 그리고 $$ f_{sur}(*) $$은 빨간색 영역에 해당하는 **surrounding context extractor** 입니다. $$ f_{joi}(*) $$는 $$ f_{loc}(*) $$와 $$ f_{sur}(*) $$을 합친 joint feature 입니다. 마지막으로 $$ f_{glo}(*) $$는 **global context extractor** 입니다. 그림의 마지막에 있는 ⊙ 기호는 **element-wise multiplication**을 뜻합니다.

<br>

- 

#### **Context Guided Network**

<br>

#### **Comparision with Similar Works**



