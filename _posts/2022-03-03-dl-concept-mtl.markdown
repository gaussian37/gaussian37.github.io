---
layout: post
title: Multi Task Deep Learning (멀티 태스크 러닝) 개념 및 컴퓨터 비전 태스크 실습
date: 2022-03-03 00:00:00
img: dl/concept/mtl/0.png
categories: [dl-concept]
tags: [deep learning, multi task, 멀티 태스크] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 이번 글에서는 딥러닝을 이용한 멀티 태스트 러닝 내용과 컴퓨터 비전의 대표적인 태스크인 sementic segmentation과 depth estimation을 멀티 태스크 러닝으로 풀어보는 과정을 설명해 보도록 하겠습니다. 
- 본 글에서 다룰 sementic segmentation과 depth estimation의 멀티 태스크 논문은 [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf) 입니다.
- 물론 가장 대표적인 컴퓨터 비전의 멀티 태스크인 object detection은 bounding box의 클래스를 classification 함과 동시에 box의 위치를 regression 합니다. 따라서 object detection을 이해하시면 멀티 태스킹의 원리에 도움이 될 것입니다.

<br>

## **목차**

<br>

- ### [멀티 태스크 러닝의 소개](#)
- ### [멀티 태스크의 구현 및 실습](#)
- ### [멀티 태스크 최적화 방법론](#)
- ### [멀티 태스크 모델 (Sementic Segmentation과 Depth Estimation) 논문 리뷰](#)
- ### [멀티 태스크 모델 모델링 및 학습](#)
- ### [멀티 태스크 모델 출력 결과 확인](#)

<br>

## **멀티 태스크 러닝의 소개**

<br>

- 멀티 태스크 러닝은 하나의 모델을 이용하여 다양한 태스크를 처리하기 위해 사용합니다. 컴퓨터 비전에서 다루는 대표적인 태스크는 object detection, sementic segmentation, depth estimation. optical flow 등이 있습니다.
- 이와 같은 태스크를 모두 개별적으로 동작하기 보다 한 개의 모델을 이용하여 멀티 태스크를 처리하면 ① `one forward propagation`, ② `one backpropagation`, ③ `lower parameter`와 같은 장점을 얻을 수 있습니다. 
- 뿐만 아니라 여러 모델을 사용할 때 보다 한 개의 모델을 이용하여 멀티 태스크를 하는 경우 메모리 사용량도 더 적어서 실시간으로 동작할 때 효율적일 수 있습니다.
- 또한 멀티 태스크를 하는 태스크가 서로 연관되어 있으면 태스크를 같이 학습하는 데 전체적인 성능을 향상할 수도 있습니다. 이와 관련된 논문도 있으며 이는 태스크를 잘 조합해야 의미가 있습니다.

<br>
<center><img src="../assets/img/dl/concept/mtl/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 멀티 태스크 러닝은 일반적으로 위 구조와 같이 `shared network`를 통하여 공통의 feature를 추출하고 그 이후에 분기별로 나뉘어져 각 태스크를 해결할 수 있도록 추가 학습되는 구조를 가집니다.
- 앞에서 설명한 다양한 장점이 있는 반면에 멀티 태스크는 상대적으로 잘 학습하기가 어려울 수 있습니다. 
    - 먼저 ① **태스크 별 데이터셋의 크기**가 다릅니다. 따라서 데이터 불균형이 발생할 수 있기 때문에 학습하는 데 문제가 발생할 수 있습니다. 
    - ② **태스크 별 학습 난이도**가 다를 수 있습니다. 따라서 태스크 별 다른 Learning Rate를 사용하거나 필요한 Epoch 수가 다를 수 있는데 학습할 때 이 점들을 모두 고려하여 반영하기가 까다롭습니다.
    - ③ Loss function을 설계 시, 각 태스크가 잘 학습될 수 있도록 Loss function을 설계해야 합니다. 간단하게 $$ L = \alpha_{1} L_{1} + \alpha_{2} L_{2} + \alpha_{3} L_{3} $$ 과 같이 스케일만 조절하더라도 그 스케일 값을 결정해야 합니다.
    - ④ Fine Tuning을 할 경우에 어떻게 멀티 태스크를 반영해야 할 지 검토해야 합니다.




<br>


## **멀티 태스크의 구현 및 실습**

<br>

- 

<br>


## **멀티 태스크 최적화 방법론**

<br>

- 다음은 멀티 태스크 러닝의 학습을 잘 하기 위한 대표적인 학습 방법론입니다. 멀티 태스크 학습을 최적화 할 때 사용하면 도움이 됩니다.
- [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics]()

<br>


## **멀티 태스크 모델 (Sementic Segmentation과 Depth Estimation) 논문 리뷰**

<br>

<br>


## **멀티 태스크 모델 모델링 및 학습**

<br>

<br>


## **멀티 태스크 모델 출력 결과 확인**

<br>

<br>



<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
