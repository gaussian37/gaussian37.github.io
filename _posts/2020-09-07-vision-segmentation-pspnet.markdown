---
layout: post
title: PSPNet(Pyramid Scene Parsing Network)
date: 2020-10-05 00:00:00
img: vision/segmentation/pspnet/0.png
categories: [vision-segmentation] 
tags: [vision, deep learning, segmentation, PSPNet, Pyramid, Scene, Parsing, Network] # add tag
---

<br>

[Segmentaion 관련 글 목록](https://gaussian37.github.io/vision-segmentation-table/)

<br>

- 참조 : https://youtu.be/siwbdHhQPXE?list=WL
- 참조 : https://arxiv.org/pdf/1612.01105.pdf

<br>

## **목차**

<br>

- ### [PSPNet의 소개](#PSPNet의-소개-1)
- ### [Pyramid Pooling Module](#Pyramid-Pooling-Module-1)
- ### [실험 결과](#실험-결과-1)
- ### [Pytorch Code](#Pytorch-Code-1)

<br>

## **PSPNet의 소개**

<br>


- 이 글에서 알아볼 `Pyramid Scene Parsing Network`, 줄여서 `PSPNet`은 CVPR 2017에서 발표된 Semantic Segmentation 모델입니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프와 같이 성능 지표를 살펴보면 PSPNet은 높은 정확도 성능을 가지지만 1.5 FPS로 실시간으로 사용할 수 없는 느린 수행 속도를 가집니다.
- 대표적인 데이터셋인 CityScape에서 80.2% 의 정확도를 얻은 만큼 좋은 성능을 가지므로 PSPNet의 주요 아이디어를 통해 배울점이 있어 보입니다.
- PSPNet의 특징을 살펴보기 위하여 아래 그림과 같이 가장 기본적인 Segmentation 모델인 [FCN](https://gaussian37.github.io/vision-segmentation-fcn/)과 비교하여 살펴보도록 하곘습니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같은 이미지가 있을 때, 노란색 박스의 영역은 보트로 분류가 되어야 합니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 하지만 FCN에서는 노란색 박스 영역을 자동차로 분류하였습니다.
- 노란색 영역에서 보트의 일부가 가려졌기 때문에 주변 상황을 고려하지 않고 노란색 박스 영역 내부만 살펴보면 자동차로 분류할 가능성도 있어보입니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 왜냐하면 FCN의 경우 가장 기본적인 Convolution / Transposed Convolution을 사용하므로 `Local context information`만 고려하여 세그멘테이션 하기 때문입니다. Local context information에 해당하는 대표적인 정보로는 **모양, 형상, 재질의 특성**등이 있습니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 PSPNet에서는 노란색 박스의 영역을 보트로 정확하게 분류할 수 있습니다.

<br>
<center><img src="../assets/img/vision/segmentation/pspnet/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- FCN의 경우와는 다르게 PSPNet에서는 `Global context information`을 얻을 수 있기 때문입니다.
- 즉, 어떤 픽셀값의 클래스를 분류하기 위해 단순히 그 근처의 local 정보들만 이용하지 않고 좀 더 넓은 영역(Global)을 고려합니다.
- 

<br>

[Segmentaion 관련 글 목록](https://gaussian37.github.io/vision-segmentation-table/)

<br>
