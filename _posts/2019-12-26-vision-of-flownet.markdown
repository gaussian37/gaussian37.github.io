---
layout: post
title: FlowNet 알아보기
date: 2019-12-26 00:00:00
img: vision/of/flownet/0.png
categories: [vision-of] 
tags: [vision, optical flow, flownet] # add tag
---

<br>

- 논문 : https://arxiv.org/abs/1504.06852
- 참조 : https://github.com/ClementPinard/FlowNetPytorch
- 참조 : https://medium.com/swlh/what-is-optical-flow-and-why-does-it-matter-in-deep-learning-b3278bb205b5
- 참조 : https://towardsdatascience.com/a-brief-review-of-flownet-dca6bd574de0

<br>

## **목차**

<br>

- ### **Contribution**
- ### **Abstract**
- ### **1. Introduction**
- ### **2. Related Work**
- ### **3. Network Architectures**
- ### **4. Training Data**
- ### **5. Experiments**
- ### **6. Concolusion**
- ### **Code Review**

<br>

## **Contribution**

<br>

- 이번 글에서는 **딥러닝을 이용한 옵티컬 플로우를 추정**하는 논문인 `FlowNet`을 알아보도록 하겠습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/k_wkDLJ8lJE" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

- 위 영상을 통해 FlowNet의 데모 영상을 확인할 수 있습니다. FlowNet의 목적에 맞게 옵티컬 플로우를 잘 추정하는 것을 볼 수 있습니다.
- FlowNet의 논문을 읽으면서 느낀 `contribution`은 다음과 같습니다.
- ① **Optical Flow를 위한 최초의 딥러닝 모델**의 의미가 있다고 생각합니다. 초기 모델인 만큼 아이디어와 네트워크 아키텍쳐도 간단합니다.
- ② 현실적으로 만들기 어려운 학습 데이터를 **저자가 직접 합성 데이터를 만들어서 학습을 진행**한 점입니다. 이 학습 데이터는 이후에 다른 옵티컬 플로우 논문에서도 학습을 위해 사용됩니다.
- ③ **합성 영상을 이용하여 학습을 하여도 현실 영상에 적용이 가능한 수준**임을 확인한 것입니다.

<br>

- 정리하면 `FlowNet`은 옵티컬 플로우 문제를 딥러닝을 이용하여 해결해 보려는 시도였고 학습을 하기 위해 합성 영상 데이터셋을 만들어서 학습을 하였는데 이 결과가 현실 영상에서도 적용 가능했다는 점입니다.
- 옵티컬 플로우는 보통 옵티컬 플로우 자체가 목적이라기 보다는 픽셀 별 `모션 벡터`를 이용하여 어플리케이션에 적용하기 위해 사용됩니다. 즉, `FlowNet`을 통하여 딥러닝 방식으로 옵티컬 플로우를 추정하게 되면 `End-To-End` 방식의 어플리케이션을 적용할 수 있다는 의의도 있습니다.
- 현재 시점으로는 `FlowNet`을 발전시킨 `FlowNet 2.0`이 많이 사용되고 있습니다. 개선된 버전의 핵심 내용은 FlowNet을 따르므로 이 글을 잘 이해하고 그 다음 FlowNet 2.0 글을 읽으시길 추천드립니다.

<br>

## **Abstract**

<br>

<br>
<center><img src="../assets/img/vision/of/flownet/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- FlowNet은 옵티컬 플로우 추정을 supervised learning 방식으로 학습을 하는 알고리즘을 뜻합니다.
- 네트워크의 아키텍쳐에 따라서 `generic architecture`에 해당하는 `FlowNetS` 모델이 있고 반면에 서로다른 이미지의 featue vector의 correlation을 이용하는 방법인 `FlowNetC`이 있습니다.
- 앞에서 언급하였듯이 학습을 하기에 충분한 데이터가 없으므로 `Flying Chairs`라는 인공적으로 만든 합성 데이터를 이용하여 학습을 진행하였습니다.

<br>

## **1. Introduction**

<br>

<br>
<center><img src="../assets/img/vision/of/flownet/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/of/flownet/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>



<br>

## **2. Related Work**

<br>

<br>

## **3. Network Architectures**

<br>

<br>

## **4. Training Data**

<br>

<br>

## **5. Experiments**

<br>

<br>

## **6. Concolusion**

<br>

<br>

## **Code Review**

<br>

<br>






<br>
