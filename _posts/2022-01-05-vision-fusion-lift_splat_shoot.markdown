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

- 이번 글에서는 NVIDIA의 멀티 카메라 기반의 BEV (Bird Eye View) 세그멘테이션 논문인 `Lift, Splat, Shoot`을 자세하게 다루어 보도록 하겠습니다.

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
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 컴퓨터 비전 알고리즘은 좌표계가 무관한 classification과 같은 task가 있고 detection, semantic segmentation과 같이 입력 좌표계와 동일한 좌표계에서 문제를 해결하는 task도 있습니다.
- 자율주행 관련된 task 에서는 여러 개의 센서로부터 입력을 받아서 자차 (ego car)를 중심으로 만든 새로운 좌표계를 기준으로 예측값을 출력합니다.
- 위 Fig. 2. 와 같이 기존의 semantic segmentation에서는 입력 이미지와 동일한 해상도의 출력 이미지를 만들어내는 FIg. 2. 의 오른쪽 이미지는 planning을 할 때 BEV 환경에서 동작하는 예시를 나타냅니다.
- 본 논문은 멀티 뷰 이미지를 입력으로 받아서 각 frame 별 출력으로 만든 BEV에서 인식을 하고 end-to-end 방식으로 planning 까지 다룹니다.

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 한 개의 이미지에서 멀티 뷰 이미지로 확장하기 위해서 여러 개의 카메라들을 각 카메라의 intrinsic과 extrinsic 정보를 이용하여 각 카메라의 좌표계 기준을 기준 좌표계 기준으로 변환해야 합니다. 이 때, 카메라의 intrinsic과 extrinsic을 이용합니다.
- 이와 같은 멀티 뷰 이미지로의 개념 확장은 아래 3가지 특성을 만족을 기대합니다.
- ① `Translation equivariance` : 각 이미지 내의 픽셀 좌표계에서 물체가 이동하면, 동일한 크기만큼 출력 좌표계에서도 물체가 이동되어야 합니다.
- ② `Permutation invariance` : 최종 출력은 카메라 입력의 순서에 의존적이지 않아야 합니다.
- ③ `Ego-frame isometry equivariance` : 멀티 뷰 이미지에서 동일한 물체를 인식하면 자차 기준의 ego-frame에서 같은 물체로 인식되어야 합니다. 즉, 차량에 장착된 카메라가 회전 및 이동이 발생하면 그 변경양에 맞춰서 보정이 되어서 같은 물체를 다른 물체 또는 다른 위치에서 인식하지 않도록 해야 합니다. (멀티 뷰 전체를 고려한 Frame을 ego-frame 이라는 용어로 사용하였습니다.)

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 가장 간단한 방법은 post process 방식을 이용하여 인식 결과를 처리하는 것입니다. 하지만 이러한 접근 방식은 ego-frame 에서 서로 다른 센서를 통해 얻은 예측을 구분하는 데 방해가 될 수 있습니다.
- 그리고 post process 방식을 시용하면 모델이 data-driven 방식으로 학습을 할 수 없습니다. 가장 좋은 방식은 멀티 뷰 카메라 간 정보를 퓨전하는 것인데 data-driven 방식으로 학습하면서 자동적으로 개선되는 방식이 좋습니다. 

<br>
<center><img src="../assets/img/vision/fusion/lift_splat_shoot/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 본 논문에서는 `Lift-Splat` 이라는 모델의 학습을 통하여 앞에서 언급한 멀티 뷰 이미지의 3가지 요소인 `Translation equivariance`, `Permutation invariance`, `Ego-frame isometry equivariance` 를 만족하도록 하려고 합니다.

<br>

- 이후에 다룰 내용 중 `3.Method` 에서는 `lift-splat`모델의 내용을 다룰 예정입니다. 
- 먼저 `lifts`에서는 2D 이미지에서 얻은 정보를 frustum 모양의 contextual feature 포인트 클라우드로 생성하여 3D로 나타내는데 이 과정을 `lift` 라고 합니다. 즉, 2D 이미지를 3D feature로 lift (들어 올린다) 한다고 생각하시면 됩니다.
- 그 다음 `splat`에서는 모든 frustum feature들을 `reference plane`에 펼칩니다. 말 그대로 `splat` 하는 것입니다. 이 과정을 통해 그 이후에 진행되는 motion planning에 도움이 됩니다.
- 마지막으로 `shooting`이라는 과정은 reference plnae 상에서 trajectory를 제안하는 방법입니다. 이 방법을 통하여 해석 가능한 end-to-end 방식의 motion planning을 접근합니다.

<br>

- `4. Implementation`에서는 멀티 뷰 카메라에서 어떻게 `lift-splat` 모델이 학습하는 지 상세히 다루어 보겠습니다. `5. Experiments and Results`에서는 실험적 증거를 통하여 `lift-splat` 모델이 여러 카메라의 정보를 효과적으로 퓨전한 지를 소개하겠습니다.

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