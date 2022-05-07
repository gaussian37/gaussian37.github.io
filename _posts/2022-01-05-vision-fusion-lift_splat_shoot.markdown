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
- ### Pytorch 코드 분석

<br>

## **0. Abstract**

<br>

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 논문의 목적은 위 그림의 왼쪽과 같이 멀티 뷰 카메라 데이터를 입력 받아서 위 그림의 오른쪽과 같이 BEV(Bird Eye View) 좌표계에 직접적으로 인퍼런스 출력하는 것입니다.
- 출력을 살펴보면 파란색은 차, 주황색은 주행 가능 영역 그리고 초록색은 차선을 의미합니다.
- 위 그림의 왼쪽에서 각 카메라 뷰 별 파란색, 주황색, 초록색의 점들이 찍혀 있습니다. 이 점들은 BEV의 출력에 해당하는 결과를 다시 입력 이미지로 가져와서 projection 한 것입니다.

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- Abstract에 나온 바와 같이 자율주행 자동차의 인식 관련 주요 기능은 멀티 센서의 정보를 퓨전하여 얻은 representation으로 부터 semantic representation을 추출하고 단일 BEV 좌표계에 정보를 표현하여 motion planning 까지 하는 것입니다.
- 이러한 기능을 구현하기 위하여 본 논문에서는 BEV 좌표계에 직접 인식 결과를 출력하는 end-to-end 구조를 소개하였습니다. 이 구조의 입력은 임의의 갯수의 카메라 영상을 입력받습니다.
- 이 논문의 제목과 같이 `lift`, `splat`, `shoot`은 각 단계별 task를 의미합니다. 
- 먼저 `lift`는 각 카메라의 이미지를 각 `frustum`으로 변경하는 역할을 하는 단계를 의미합니다. `frustum`은 피라미드 모양의 위면이 잘린 입체형태를 의미하며 `lift`를 구현하기 위한 feature의 구조가 이와 같이 생겨서 frustum으로 표현하였습니다.
- 그 다음으로 `splat`은 모든 `frustum` feature를 rasterized BEV grid로 표현하는 것을 의미합니다. [rasterized](https://www.computerhope.com/jargon/r/rasterize.htm)란 뜻은 discrete pixel 형태의 이미지를 나타내었다는 뜻으로 흔히 알고 있는 픽셀 형태로 BEV grid를 표현했다고 이해하시면 됩니다.
- 차량에 장착된 카메라의 영상 전체를 한번에 학습함으로써 **본 논문의 모델은 이미지를 어떻게 represent 하는 지 학습하고 모든 카메라 이미지를 통해 얻은 출력을 퓨전하여 결합된 단일 BEV scene에 표현하는 방법을 학습**합니다. 이 과정 속에서 차량에 장착된 카메라에서 발생하는 **캘리브레이션 에러에 강건**해지는 장점을 얻을 수 있었습니다. 이러한 학습 방법을 통하여 기존의 object segmentation 이나 map segmentation 보다 좋은 성능을 얻을 수 있었습니다.
- 그리고 BEV 에 representation 하는 방식을 통하여 end-to-end 방식의 motion planning을 할 수 있으며 이 방식을 `shooting`이라고 표현합니다.
- `shooting`은 template trajectory를 BEV cost map output에 적용하는 과정을 의미하며 자세한 내용은 본문 내용에서 살펴보겠습니다.
- 최종적으로 성능 검증은 lidar 데이터를 통하여 확인하였습니다.

<br>

## **1. Introduction**

<br>

- 



<br>

## **2. Related Work**

<br>



<br>

## **3. Method**

<br>



<br>

## **4. Implementation**

<br>



<br>

## **5. Experiments and Results**

<br>



<br>

## **6. Conclusion**

<br>



<br>

## **Pytorch 코드 분석**

<br>






<br>

[멀티 카메라 퓨전 관련 글 목록](https://gaussian37.github.io/vision-fusion-table/)

<br>