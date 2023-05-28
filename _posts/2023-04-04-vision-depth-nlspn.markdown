---
layout: post
title: NLSPN, Non-Local Spatial Propagation Network for Depth Completion
date: 2023-04-04 00:00:00
img: vision/depth/nlspn/0.png
categories: [vision-depth]
tags: [depth completion, depth estimation, nlspn] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 논문 : https://arxiv.org/abs/2007.10042
- 사전 지식 : [deformable convolution](https://gaussian37.github.io/dl-concept-deformable_convolution/)

<br>

- 이번 글에서는 대표적인 `Depth Completion` 논문인 `NLSPN`을 다루어 보도록 하겠습니다. 저자의 아래 영상 또한 이해하는 데 도움이 많이 되었습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/gQlwsauWKRk" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [1. Introduction](#1-introduction-1)
- ### [2. Related Work](#2-related-work-1)
- ### [3. Non-Local Spatial Propagation](#3-non-local-spatial-propagation-1)
- ### [4. Confidence-Incorporated Affinity Learning](#4-confidence-incorporated-affinity-learning-1)
- ### [5. Depth Completion Network](#5-depth-completion-network-1)
- ### [6. Experimental Results](#6-experimental-results-1)
- ### [7. Conclusion](#7-conclusion-1)
- ### [Pytorch 코드](#pytorch-코드-1)

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/depth/nlspn/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 본 논문은 성능 측면에서 강건한 `depth completion` 방법론을 제안합니다. `depth completion`은 `RGB 이미지`와 `sparse depthmap`을 입력으로 받아서 `dense depthmap`을 예측하는 문제입니다.
- `NLSPN`이 예측하는 값들은 `Non-Local Neighbors and Affinities`, `Initial Depth Prediction`, `Initional Depth Confidence` 이며 이 개념들에 대하여 본 글에서 알아볼 예정입니다. 이 3가지 값을 이용하여 `NLSPN`은 depth를 주변 픽셀로 전파하는 `propagation` 작업 과정에서 depth propagation에 무관한 `local neighbor`를 피하고 `non-local neighbor`이지만 depth propagation에 관련되어 있는 영역에 집중할 수 있도록 합니다. 이 부분이 `NLSPN`의 핵심입니다. 추가적으로 `learnable affinity normalization` 개념 또한 도입하여 이전 논문들의 방법에 비해 좀 더 강건한 `depth completion`을 해냅니다.

<br>

## **1. Introduction**

<br>
<center><img src="../assets/img/vision/depth/nlspn/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `depth completion`은 `RGB 이미지`와 `sparse depthmap`이 있을 때, `sparse depthmap`을 `dense depthmap`으로 생성하는 태스크를 의미합니다. 이와 같은 태스크를 수행하는 이유는 `lidar`의 하드웨어 한계로 `lidar`의 해상도가 이미지의 해상도 만큼 높지 않기 때문에 `sparse depthmap`의 형태로 데이터를 취득할 수 밖에 없습니다.
- 이러한 문제를 개선하기 위하여 `RGB 이미지`와 `sparse depthmap`이 주어졌을 때, `dense depthmap`을 예측하는 문제를 `depth completion`으로 정의 하였고 이번 논문에서는 이 태스크에서 의미 있는 결과를 도출한 방법을 살펴볼 예정입니다.

<br>
<center><img src="../assets/img/vision/depth/nlspn/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이전에 사용하였던 `depth completion` 방법은 단순히 `sparse depthmap`을 이용하는 방법이었으나 이와 같은 방법은 `blurry`나 `mixed-depth`와 같은 `artifact` 문제가 발생하였습니다. 따라서 최근에 사용하는 방법과 같이 `RGB 이미지`를 이용하여 `dense depth`를 어떻게 생성해야 할 지 가이드를 주는 방법이 주류로 사용되고 있습니다.

<br>
<center><img src="../assets/img/vision/depth/nlspn/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 `RGB` 또는 `RGB-D` 이미지를 이용하여 딥러닝 모델을 이용하여 `depth completion`을 하는 방법을 사용하여 성능 개선을 하였지만 이 방법의 경우 depth가 변화가 심한 경계 부분에서 실제 depth와 같이 급격하게 변하지 않고 `blur`해지는 문제가 발생하게 됩니다.
- 이 문제는 어떤 픽셀이 그 픽셀의 주변 픽셀과 얼만큼 `affinity`가 있는 지 학습하는 방식을 통해 개선하는 시도들이 있었고 `affinity` 학습과 반복(`iteration`)을 통한 `depth prediction refinement`를 통해 앞에서 제기한 `blur` 문제를 포함하여 전체적인 성능 개선을 진행해 왔습니다.
- 또한 `RGB` 또는 `RGB-D` 이미지 기반의 `depth completion`에서는 `mixed-depth` 문제가 발생하였습니다. 그 이유는 기존의 방식이 `convolution` 기반의 `fixed-local neighborhood`를 참조하는 `depth propagation` 방식이었는데 `convolution` 연산은 직사각형 형태의 `fixed-local neighborhood`만 참조하여 `depth`를 주변 픽셀로 전파할 수 밖에 없기 때문에 원하지 않는 `depth`가 섞일 수 있기 때문입니다.

<br>
<center><img src="../assets/img/vision/depth/nlspn/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이번 글에서 다룰 `NLSPN (Non-Local Spatial Propagation Network)`은 각 픽셀에서 `depth propagation`을 위하여 참조할 `non-local neighbor`를 어떻게 예측하여 선정할 지에 대한 방법과 선정된 픽셀에 대하여 `sparially-varing affinities`를 통해 `depth` 정보를 어떻게 모아서 `propagation` 할 지에 대한 방법론을 제시합니다.
- `depth propagation` 방식을 `fixed local` → `non local` 방식으로 변경함에 따라 `mixed-depth` 문제에 강건해 질 수 있었으며 이 때 사용되는 `affinity`의 `learnable affinity normalization` 방법과 `initial dense depth`의 `confidence`를 사용하는 방법을 통해 추가적인 성능 개선을 할 수 있었습니다. 이 방법에 대해서는 본론에서 알아보도록 하겠습니다.

<br>

## **2. Related Work**

<br>

<br>


## **3. Non-Local Spatial Propagation**

<br>

<br>


## **4. Confidence-Incorporated Affinity Learning**

<br>

<br>


## **5. Depth Completion Network**

<br>

<br>


## **6. Experimental Results**

<br>

<br>


## **7. Conclusion**

<br>

<br>


## **Pytorch 코드**

<br>

<br>



<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>
