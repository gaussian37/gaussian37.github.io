---
layout: post
title: CompletionFormer, Depth Completion with Convolutions and Vision Transformers
date: 2023-04-25 00:00:00
img: vision/depth/completionformer/0.png
categories: [vision-depth]
tags: [depth completion, depth estimation, completionformer] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 논문 : https://arxiv.org/abs/2304.13030
- 사전 지식 : [NLSPN](https://gaussian37.github.io/vision-depth-nlspn/)

<br>

- 이번 글에서 다룰 논문은 CVPR 2023 Paper로 등재된 논문이며 `Transformer`를 이용하여 `Depth Completion`을 적용한 뒤 `NLSPN`을 이용하여 `refinement`를 적용한 논문으로 요약할 수 있습니다.
- `NLSPN` 논문만 이해하면 본 논문에서는 이해하기 어려운 부분은 없으니 쉽게 읽을 수 있을 것으로 생각됩니다. 본 글을 읽기 전에 사전 지식으로 `NLSPN`을 꼭 읽기를 추천드립니다.

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [1. Introduction](#1-introduction-1)
- ### [2. Related Work](#2-related-work-1)
- ### [3. Method](#3-method-1)
- ### [4. Experiments](#4-experiments-1)
- ### [5. Conclusion and Limitations](#5-conclusion-and-limitations-1)
- ### [Pytorch 코드](#pytorch-코드-1)

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/depth/completionformer/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `sparse depthmap`과 `RGB image`가 주어졌을 때, `sparse depthmap`의 값을 전체 이미지의 픽셀로 확장하는 것이 `depth completion`의 역할입니다.
- 지금까지 많이 시도되어 왔던 CNN 기반의 `depth completion`은 `locality`에 집중하는 Convlution Layer의 특성 상 픽셀 간의 거리가 멀었을 때, `depth completion`이 정확하게 되지 않는 문제가 발생하였습니다. 최근에 급속도로 발전 중이며 많이 사용되는 `transformer` 구조에서는 `global receptive field`를 가지기 때문에 이러한 문제를 개선할 수 있음을 논문에서 보여줍니다.
- 본 논문에서는 단순히 `CNN` 이나 `Transformer`만을 사용하지 않고 `CNN`의 성질을 통해 `locality`의 `depth completion` 성능을 확보하고 `Transformer`의 성질을 통해 `globality`의 `depth completion`의 성능을 확보하고자 `JCAT`, `Joint Convolutional Attention and Transformer block` 구조를 제안합니다.

<br>

## **1. Introduction**

<br>



<br>

## **2. Related Work**

<br>

<br>

## **3. Method**

<br>

<br>

## **4. Experiments**

<br>

<br>

## **5. Conclusion and Limitations**

<br>

<br>

## **Pytorch 코드**

<br>

<br>




<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>
