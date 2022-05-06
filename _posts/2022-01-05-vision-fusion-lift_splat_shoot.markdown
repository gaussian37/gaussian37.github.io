---
layout: post
title: Lift, Splat, Shoot (Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D)
date: 2022-01-05 00:00:00
img: vision/fusion/lift_splat_shoot/0.png
categories: [vision-fusion]
tags: [camera fusion, multi camera, nvidia, lift, splat, shoot] # add tag
---

<br>

[멀티 카메라 퓨전 관련 글 목록](https://gaussian37.github.io/vision-fusion-table/)

<br>

- 논문 : https://arxiv.org/abs/2008.05711
- 논문 : https://nv-tlabs.github.io/lift-splat-shoot/
- 참조 : https://towardsdatascience.com/monocular-birds-eye-view-semantic-segmentation-for-autonomous-driving-ee2f771afb59

<br>

- 이번 글에서는 NVIDIA의 멀티 카메라 기반의 BEV 세그멘테이션 논문인 `Lift, Splat, Shoot`을 자세하게 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 0. Abstract
- ### 1. Introduction
- ### 2. Related Work
- ### 3. Method
- ### 4. Implementation
- ### 5. Experiments and Results
- ### 6. Conclusion

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>

[멀티 카메라 퓨전 관련 글 목록](https://gaussian37.github.io/vision-fusion-table/)

<br>