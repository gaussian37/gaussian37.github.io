---
layout: post
title: Object Detection 관련 글 목차   
date: 2000-01-01 00:00:00
img: vision/detection/detection.png
categories: [vision-detection] 
tags: [vision, deep learning, detection] # add tag
---

<br>

- 블로그에서 Object Detection 관련 내용을 정리한 리스트 입니다. 크게 One-Stage Detector, Two-Stage Detector, 그 이외로 분류하였습니다. One-Stage Detector, Two-Stage Detector 관련 정의는 이 글의 아래 부분에서 확인하실 수 있습니다.

<br>

## **One-Stage Detector**

<br>

- [CenterNet](https://gaussian37.github.io/vision-detection-centernet/)


<br>

## **Two-Stage Detector**

<br>




<br>

## **Detector 알고리즘 이외 내용**

<br>

- [IoU(Intersection over Union)](https://gaussian37.github.io/math-algorithm-iou/)
- [GIoU(Generalized Intersection over Union)](https://gaussian37.github.io/vision-detection-giou/)
- [Monocular 3D Object Detection 에서의 Orientation Estimation (방향 추정)](https://gaussian37.github.io/vision-detection-orientation_estimation_monocular_3d_od/)

<br>

#### **One-Stage Detector와 Two-Stage Detector란?**

<br>

- 참조 : https://jdselectron.tistory.com/m/101
- Object Detection은 이미지에 존재하는 여러 object들을 Bounding box를 통해 `Localization`을 하고, `Classification`하는 것을 말합니다. Localization은 위치를 찾는 것이고 Classification은 어떤 물체인지 분류를 하는 것이므로 즉, **Object Detection 이라고 하는 것은 어떤 물체라고 생각되는 위치를 찾고 그 물체가 무엇인지 판단하는 것**이라고 할 수 있습니다.

<br>
<center><img src="../assets/img/vision/detection/one_two_stage/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Object Detection에서는 `One-Stage`와 `Two-Stage`의 Detector가 있습니다. One-Stage Detector는 Localization(Region Proposal)과 Classification을 동시에 작업 하는 Detector를 뜻하고 Two-Stage Detector는 Localization → Classification 순서로 순차적으로 작업하는 Detector를 뜻합니다.
- 일반적으로 One-Stage Detector는 Two-Stage Detector에 비해 수행 속도는 빠르지만 Detector 정확도는 떨어집니다. 

<br>

- 먼저 One-Stage Detector는 Localization(Region Proposal)과 Classification이 동시에 일어납니다. 즉 딥러닝 네트워크를 통하여 얻은 feature에서 위치와 물체 정보까지 한번에 뽑아 내는 방식입니다.

<br>
<center><img src="../assets/img/vision/detection/one_two_stage/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이미지 내 모든 위치를 잠재적인 후보 object들로 고려하여 ROI를 배경 또는 타겟 object들로 분류하도록 합니다.
- 이 방식의 장점은 빠른 속도로 실시간으로 detection이 필요한 작업에 사용할 수 있습니다. 하지만 Two-Stage Detector 보다는 인식 성능에서 열세입니다.
- 대표적으로 다음과 같은 모델들이 있습니다.

<br>
<center><img src="../assets/img/vision/detection/one_two_stage/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 Two-Stage Detector는 Localization(Regional Proposal)과 Classification이 순차적으로 이루어집니다.
- 첫번째 단계인 Region Proposal에서 proposal 셋을 생성하고, 두번째 단계 Classification에서는 생성된 proposal에 대한 feature vector를 딥러닝 네트워크를 이용하여 encoding한 후 object에 대한 클래스를 예측합니다.
- 이 방법은 정확도가 높지만 일반적으로 인퍼런스 하는 데 속도가 느리다는 단점이 있습니다.

<br>

- 먼저 Regional Proposal이란 무엇인지 간단하게 알아보도록 하겠습니다. **regional proposal은 object의 위치를 찾는 localization문제**입니다.
- Region Proposal을 하기 위한 가장 간단하고 원시적인 방법은 sliding window 방식입니다. 이 방법은 이미지에서 모든 영역을 다양한 크기의 window(different scale & ratio)로 탐색하는 것입니다.
- 하지만 전체 영역을 다양한 사이즈의 window로 모두 탐색하는 것은 매우 비효율적이므로 object가 있을 만한 영역을 찾는 Seletive Search 방법이 처음에 사용되었습니다. Selective Search는 비슷한 질감, 색, 강도를 갖는 인접 픽셀로 구성된 다양한 크기의 window를 생성하는 방법입니다.

<br>
<center><img src="../assets/img/vision/detection/one_two_stage/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음은 대표적인 Two-Stage Detector의 예시입니다.

<br>
<center><img src="../assets/img/vision/detection/one_two_stage/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>