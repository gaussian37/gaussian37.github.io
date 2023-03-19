---
layout: post
title: Self-Supervised Surround-View Depth Estimation with Volumetric Feature Fusion
date: 2023-03-18 00:00:00
img: vision/fusion/volumetric_feature_fusion/0.png
categories: [vision-fusion]
tags: [depth estimation, supervised depth estimation, volumetric feature fusion] # add tag
---

<br>

[멀티 카메라/뷰 퓨전 관련 글 목록](https://gaussian37.github.io/vision-fusion-table/)

<br>

- 논문 : https://openreview.net/forum?id=0PfIQs-ttQQ
- 발표 자료 : https://nips.cc/virtual/2022/poster/54283

<br>

- 이번 글에서는 `NIPS 2022`에 발표된 `Self-Supervised Surround-View Depth Estimation with Volumetric Feature Fusion` 논문에 대한 내용 리뷰를 진행하겠습니다.
- 논문에서 주목하고자 하는 부분은 멀티 카메라를 사용하였을 때 카메라 간 겹치는 영역이 발생하는데 그 영역에 대하여 어떻게 Fusion을 잘 할 지에 대한 방법과 그 효과를 보여줍니다. 이 방법을 논문에서는 `Volumetric Feature Fusion`이라고 명하였습니다.

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [Introduction](#introduction-1)
- ### [Related Work](#related-work-1)
- ### [Surround-View Depth Estimation via Volumetric Feature Fusion](#surround-view-depth-estimation-via-volumetric-feature-fusion-1)
- ### [Experiments](#experiments-1)
- ### [Conclusions](#conclusions-1)
- ### [Supplementary](#supplementary-1)
- ### [42dot dataset](#42dot-dataset-1)

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>


<br>


## **Introduction**

<br>

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/7.gif" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>


## **Related Work**

<br>

<br>


## **Surround-View Depth Estimation via Volumetric Feature Fusion**

<br>

<br>


## **Experiments**

<br>

<br>


## **Conclusions**

<br>

## **Supplementary**

<br>

<br>


## **42dot dataset**

<br>

- 아래 링크에서 42dot 데이터셋을 받아볼 수 있습니다.
    - 링크 : https://www.42dot.ai/akit/dataset/mcmot
- 본 논문에서는 `front`, `front-left`, `front-right`에 대한 시각화 사례가 있었는데, 42dot 데이터셋을 보면 `front`, `front-left`, `front-right`을 사용한 것을 볼 수 있습니다. 링크의 설명은 보면 `front`는 60도 화각의 카메라이며 `front-left`, `front-right`는 120도 화각의 카메라임을 알 수 있습니다.
- 아래는 `volumetric feature fusion`을 한 영역으로 추정되는 영역을 표시하였습니다.

<br>
<center><img src="../assets/img/vision/fusion/volumetric_feature_fusion/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>

[멀티 카메라/뷰 퓨전 관련 글 목록](https://gaussian37.github.io/vision-fusion-table/)

<br>
