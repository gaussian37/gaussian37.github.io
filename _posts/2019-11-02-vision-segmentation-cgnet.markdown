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

- semantic segmentation을 모바일 디바이스 환경에 적용하려는 시도가 많이 증가하고 있습니다.
- 성능이 좋은 semantic segmentation 모델들은 많은 파라미터와 연산량으로 인해 모바일 디바이스에는 적합하지 않기 때문에 모바일 디바이스에는 경량화 모델이 필요합니다.
- 경량화 segmentation model에 대한 연구들의 일부 문제점은 classification에서 사용된 방법들을 사용하고 semantic segmentation에서 고려해야 할 특성들을 무시한 상태로 구조가 만들어 졌다는 것에 있습니다.
- 이 논문에서는 이러한 문제점들을 개선하기 위하여 `Context Guided Network (CGNet)`을 소개합니다. 이 모델 또한 가볍고 계산에 효율적인 segmentation 모델입니다.
- CGNet에서 사용된 `CG block`은 local feature와 local feature를 둘러싼 surrounding context를 학습합니다. 그리고 더 나아가 global context와 관련된 feature 또한 이용하여 성능을 향상시킵니다. CGNet은 CG block을 기반으로 네트워크의 모든 단계에서 상황에 맞는 정보를 이해하고 segmentation 정확도를 높이기 위해 설계됩니다.
- CGNet은 또한 파라미터 수를 줄이고 메모리 공간을 절약하도록 정교하게 설계되었습니다. 동등한 수의 매개 변수 하에서 제안 된 CGNet은 기존 세그먼테이션 네트워크보다 훨씬 뛰어납니다.
- Cityscape 및 CamVid 데이터 세트에 대한 광범위한 실험은 제안 된 접근 방식의 효과를 검증합니다.
- 특히 post-processing 및 multi-scale testing 없이 제안 된 CGNet은 0.5M 미만의 매개 변수로 64.8 %의 Cityscape에서 평균 IoU를 달성합니다.

<br>

## **Introduction**

<br>



<br>

## **Related Work**

<br>

- Related Work에서는 CGNet과 관련된 small semantic segmentation model, contextual information model 그리고 attention model에 대하여 간략하게 다루어 보겠습니다.

<br>

#### **Small semantic segmentation models**

<br>

<br>

#### **Contextual information models**

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

- 먼저 위 그림을 먼저 설명하고 Context Guided Block에 대하여 다루어 보도록 하겠습니다.
- 먼저 (a)를 보면 그림의 조그마한 노란색 영역만 보았을 때, 그 영역에 해당하는 클래스가 무엇인지 판단하기가 어렵습니다.
- 반면 (b)와 같이 노란색 영역 주위에 빨간색 영역을 포함하여 같이 본다면 인식하기 좋아집니다. 여기서 빨간색 영역을 `surrounding context` 라고 합니다.
- 마지막으로 (c) 그림을 보면 전체 이미지를 포함하는 보라색의 사각형이 있습니다. 전체 영역을 이용하여 노란색 영역의 클래스가 무엇인 지 판단한다면 더 높은 정확도로 판단할 수 있습니다. 보라색 사각형을 `global context` 라고 하겠습니다.
- CG Block의 형태는 (d)와 같습니다. Block의 구성 요소 중 $$ f_{loc}() $$ 이 그림의 노란색 영역에 해당하는 **local feature**입니다. $$ f_{sur}() $$은 빨간색 영역에 해당하는 **surrounding context extractor** 입니다. $$ f_{joi} $$는 $$ f_{loc} $$와 $$ f_{sur} $$을 합친 feature 입니다. 마지막으로 $$ f_{glo} $$는 **global context extractor** 입니다. 그림의 마지막에 있는 ⊙ 기호는 **element-wise multiplication**을 뜻합니다.

<br>

- 

#### **Context Guided Network**

<br>

#### **Comparision with Similar Works**



