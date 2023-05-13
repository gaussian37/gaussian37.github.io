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

<br>

- 이번 글에서는 대표적인 Depth Completion 논문인 `NLSPN`을 다루어 보도록 하겠습니다. 저자의 아래 영상 또한 이해하는 데 도움이 많이 되었습니다.

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
- `NLSPN`가 예측하는 값들은 `Non-Local Neighbors and Affinities`, `Initial Depth Prediction`, `Initional Depth Confidence` 이며 이 개념들에 대하여 본 글에서 알아볼 예정입니다. 이 3가지 값을 이용하여 `NLSPN`은 depth를 주변 픽셀로 전파하는 `propagation` 작업 과정에서 depth propagation에 무관한 `local neighbor`를 피하고 `non-local neighbor`이지만 depth propagation에 관련되어 있는 영역에 집중할 수 있도록 합니다. 이 부분이 `NLSPN`의 핵심입니다. 추가적으로 `learnable affinity normalization` 개념 또한 도입하여 이전 논문들의 방법에 비해 좀 더 강건한 `depth completion`을 해냅니다.

<br>


## **1. Introduction**

<br>

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
