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

<br>

- 이번 글에서는 CGNet, A Light-weight Context Guided Network for Semantic Segmentation 에 대하여 알아보도록 하겠습니다.
- Network의 이름에도 포함이 되어 있듯이 `Light-weight` 이므로 weight의 수가 작은 Realtime 용도의 Segmentation 모델입니다.

<br>

## **목차**

<br>

- ### Abstract
- ### 1. Introduction
- ### 2. Related Work
- ### 3. Proposed Approach
    - #### 3.1. Context Guided Block
    - #### 3.2. Context Guided Network
    - #### 3.3. Comparison with Similar Works
- ### 4. Experiments
    - #### 4.1. Experiments Settings
    - #### 4.2. Ablation Studies
    - #### 4.3. Comparison with state-of-the-arts
- ### 5. Conclusion

<br>
<center><img src="../assets/img/vision/segmentation/cgnet/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>


