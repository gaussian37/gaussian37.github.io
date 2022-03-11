---
layout: post
title: 카메라 캘리브레이션의 이해와 Python 실습
date: 2022-01-28 00:00:00
img: vision/concept/calibration/0.png
categories: [vision-concept] 
tags: [vision, concept, calibaration, 캘리브레이션] # add tag
---

<br>

- 참조 : https://ms-neerajkrishna.medium.com/
- 참조 : https://darkpgmr.tistory.com/32
- 참조 : https://www.mathworks.com/help/vision/ug/camera-calibration.html

<br>

- 이번 글에서는 컴퓨터 비전을 위한 카메라 내용 및 카메라 캘리브레이션 관련 내용과 파이썬을 이용하여 실습을 해보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 이미지 형성과 핀홀 모델 카메라
- ### Camera Extrinsic Matrix with Example in Python
- ### Camera Intrinsic Matrix with Example in Python
- ### Find the Minimum Stretching Direction of Positive Definite Matrices
- ### Camera Calibration with Example in Python

<br>

## **이미지 형성과 핀홀 모델 카메라**

<br>

- 이미지 형성의 기본 아이디어는 `object`에서 `medium`으로 반사되는 광선(Rays)을 포착하는 것에서 부터 시작합니다.
- 가장 단순한 방법은 `object` 앞에 `medium`을 놓고 반사되어 들어오는 광선을 캡쳐하면 됩니다. 하지만 단순히 이러한 방식으로 하면 필름 전체에 회색만 보일 수 있습니다. 왜냐하면 object의 다른 지점에서 나오는 광선이 필름에서 서로 겹쳐서 엉망이 되기 때문입니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 살펴보면 

<br>

## **Camera Extrinsic Matrix with Example in Python**

<br>


<br>

## **Camera Intrinsic Matrix with Example in Python**

<br>


<br>

## **Find the Minimum Stretching Direction of Positive Definite Matrices**

<br>


<br>

## **Camera Calibration with Example in Python**

<br>


<br>


