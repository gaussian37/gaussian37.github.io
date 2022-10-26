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
- 참조 : http://www.close-range.com/docs/Decentering_Distortion_of_Lenses_Brown_1966_may_444-462.pdf

<br>

- 이번 글에서는 카메라 모델의 특성, 카메라 렌즈 왜곡 모델 그리고 렌즈 왜곡을 제거하는 방법 등에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [화각에 따른 카메라의 종류](#)
- ### [표준 카메라 렌즈 왜곡 모델 ](#)
- ### [표준 카메라 렌즈 왜곡 보정 방법](#)
- ### [표준 카메라의 렌즈 왜곡 보정 실습](#)
    - ### [왜곡 영상 Points → 핀홀 모델 영상 Points](#)
    - ### [undistort를 이용한 왜곡 영상 → 핀홀 모델 영상](#)
    - ### [remap을 이용한 왜곡 영상 → 핀홀 모델 영상](#)
    - ### [Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상](#)
- ### [어안 카메라 렌즈 왜곡 모델 ](#)
- ### [어안 카메라 렌즈 왜곡 보정 방법](#)
- ### [어안 카메라의 렌즈 왜곡 보정 실습](#)
    - [### 왜곡 영상 Points → 핀홀 모델 영상 Points](#)
    - [### remap을 이용한 왜곡 영상 → 핀홀 모델 영상](#)
    - [### Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상](#)
- ### [렌즈 왜곡 모델링의 이론적 이해](#)

<br>

## **화각에 따른 카메라의 종류**

<br>

- 카메라에서 가장 중요한 부분 중 하나가 렌즈입니다. 핀홀 모델 카메라는 이론적으로 빛의 직진성을 이용하여 만든 이상적이면서 간단한 카메라 모델이지만 빛의 유입량이 적어 정상적인 이미지를 만들어낼 수 없습니다.
- 따라서 렌즈를 이용하여 빛의 양이 많이 유입될 수 있도록 (사람의 수정체와 같습니다.) 카메라에서 사용하며 **렌즈의 형태에 따라 카메라가 빛을 유입할 수 있는 영역이 달라지기 때문에** 아래 그림과 같이 **렌즈에 따른 화각이 결정됩니다.**

<br>
<center><img src="../assets/img/vision/concept/lense_distortion/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 화각과 초점 거리 (focal length)는 설명을 위한 예시이며 절대적인 기준은 아닙니다.
- 이번 글에서는 `표준 렌즈`를 사용하는 `표준 카메라`와 `어안 렌즈`를 사용하는 `어안 카메라`에서 렌즈로 인한 물체가 휘어져 보이는 현상이 발생하였을 때, 컴퓨터 비전에서 어떻게 처리하는 지 그 방식과 원리에 대하여 살펴보도록 하겠습니다.

## **표준 카메라 렌즈 왜곡 모델 **

<br>

<br>

## **표준 카메라 렌즈 왜곡 보정 방법**

<br>

<br>

## **표준 카메라의 렌즈 왜곡 보정 실습**

<br>

<br>

### **왜곡 영상 Points → 핀홀 모델 영상 Points**

<br>

<br>

### **undistort를 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

### **remap을 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

### **Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

## **어안 카메라 렌즈 왜곡 모델 **

<br>

<br>

## **어안 카메라 렌즈 왜곡 보정 방법**

<br>

<br>

## **어안 카메라의 렌즈 왜곡 보정 실습**

<br>

<br>

### **왜곡 영상 Points → 핀홀 모델 영상 Points**

<br>

<br>

### **remap을 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

### **Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

## **렌즈 왜곡 모델링의 이론적 이해**

<br>

<br>


<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>