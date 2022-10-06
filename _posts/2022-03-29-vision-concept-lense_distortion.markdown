---
layout: post
title: 카메라 모델과 렌즈 왜곡 (lense distortion)
date: 2022-03-29 00:00:00
img: vision/concept/lense_distortion/0.png
categories: [vision-concept] 
tags: [lense distortion, 카메라 모델, 렌즈 왜곡] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 참조 : https://kr.mathworks.com/help/vision/ug/camera-calibration.html
- 참조 : http://jinyongjeong.github.io/2020/06/19/SLAM-Opencv-Camera-model-%EC%A0%95%EB%A6%AC/
- 참조 : http://jinyongjeong.github.io/2020/06/15/Camera_and_distortion_model/
- 참조 : https://docs.nvidia.com/vpi/algo_ldc.html
- 참조 : http://www.gisdeveloper.co.kr/?p=6868

<br>

- 이번 글에서는 카메라 모델의 특성, 카메라 렌즈 왜곡 모델 그리고 렌즈 왜곡을 제거하는 방법 등에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 화각에 따른 카메라의 종류
- ### 표준 카메라 렌즈 모델 
- ### 표준 카메라 렌즈 왜곡 보정 방법
- ### 표준 카메라의 렌즈 왜곡 보정 실습
    - ### 왜곡 영상 Points → 핀홀 모델 영상 Points
    - ### undistort를 이용한 왜곡 영상 → 핀홀 모델 영상
    - ### remap을 이용한 왜곡 영상 → 핀홀 모델 영상
- ### 어안 카메라 렌즈 모델
- ### 어안 카메라 렌즈 왜곡 보정 방법
- ### 어안 카메라의 렌즈 왜곡 보정 실습
    - ### 왜곡 영상 Points → 핀홀 모델 영상 Points
    - ### remap을 이용한 왜곡 영상 → 핀홀 모델 영상

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>