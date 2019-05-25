---
layout: post
title: 11. Detection and Segmentation
date: 2018-01-11 01:00:00
img: vision/cs231n/cs231n.jpg
categories: [vision-cs231n] 
tags: [cs231n, detection, segmentation] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/nDPWywWRIRo" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

<img src="../assets/img/vision/cs231n/11/1.png" alt="Drawing" style="width: 800px;"/>

+ 이 때까지의 수업에서는 주로 Image Classification 문제를 다루었습니다.
+ 입력 이미지가 들어오면 Deep ConvNet을 통과하고 네트워크를 통과하면 Feature Vector가 나옵니다.
    + AlexNet이나 VGG의 경우에는 4096차원의 Feature Vector가 생성되었습니다.
+ 그리고 최종 Fully Connected Layer에서는 1000개의 클래스 스코어를 나타냅니다.
    + 이 예제에서 1000개의 클래스는 ImageNet의 클래스를 의미합니다.
+ 즉, 전체 구조는 입력 이미지가 들어오면 전체 이미지가 속하는 카테고리의 출력입니다.
    + 위 과정은 가장 기본적인 Image Classification이고 Deep Learning으로 더 흥미로운 작업이 가능합니다.
+ 이번 강의에는 Deep Learning의 다양한 Task들에 대하여 알아보도록 하겠습니다.
     
<img src="../assets/img/vision/cs231n/11/2.png" alt="Drawing" style="width: 800px;"/>

+ 이번 강의에서 배울 내용은 크게 4가지 입니다.
    + Semantic Segmentation
    + Classification + Localization
    + Object Dectection
    + Instance Segmentation

<br>

## Semantic Segmentation

<img src="../assets/img/vision/cs231n/11/3.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation 문제에서는 입력은 이미지이고 출력으로 이미지의 모든 픽셀에 카테고리를 정합니다.
    + 예를 들어 위 슬라이드의 예제를 보면 입력은 고양이 입니다. 
    + 출력은 모든 픽셀에 대하여 그 픽셀이 고양이, 잔디, 하늘, 나무, 배경인지를 결정합니다.
+ Semantic Segmentation에서도 Classification 처럼 카테고리가 있습니다.
+ 하지만 차이점은 Classification처럼 이미지 전체에 카테고리 하나가 아니라 모든 픽셀에 카테고리가 매겨집니다.


<img src="../assets/img/vision/cs231n/11/4.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation은 개별 객체를 구별하지 않습니다.
+ 위 슬라이드의 Semantic Segmentation 결과를 보면 픽셀의 카테고리만 구분해 줍니다.
    + 즉 오른쪽 슬라이드의 결과를 보면 소가 2마리가 있는데 2마리 각각을 구분할 수는 없습니다.
+ 이것은 Semantic Segmentation의 단점이고 나중에 배울 Instance Segmentation에서 이 문제를 해결할 수 있습니다.

<img src="../assets/img/vision/cs231n/11/5.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation 문제에 접근해 볼 수 있는 방법 중 하나는 Classification을 통한 접근 방법입니다.
+ Semantic Segmentation을 위해서 `Sliding Window`를 적용해 볼 수 있습니다.   
+ 먼저 입력 이미지를 아주 작은 단위로 쪼갭니다.
