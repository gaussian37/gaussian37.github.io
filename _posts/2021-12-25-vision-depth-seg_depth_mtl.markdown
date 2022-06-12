---
layout: post
title: Semantic Segmentation과 Depth Estimation의 Multi Task Learning
date: 2021-12-25 00:00:00
img: vision/depth/seg_depth_mtl/0.png
categories: [vision-segmentation]
tags: [deep learning, segmentation, loss] # add tag
---

<br>

[depth estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 이번 글에서는 Semantic Segmentation과 Depth Estimation을 멀티 태스크 러닝을 통하여 한번에 학습하고 출력하는 전체 과정에 대하여 다루어 보도록 하겠습니다.
- 이와 관련된 내용으로 ICRA 2019에 발표된 [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf) 논문을 기준으로 설명해 보도록 하겠습니다.
- 이 논문의 아이디어나 학습 방법 등에 초점을 두지 않고 Hard Parameter 방식의 멀티 태스크 러닝이 어떤 메카니즘으로 동작하는 지 살펴보면 되겠습니다.

<br>




<br>

[depth estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>
