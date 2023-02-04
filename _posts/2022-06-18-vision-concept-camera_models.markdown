---
layout: post
title: Double Sphere 카메라 모델 및 다양한 카메라 모델의 종류 (Pinhole, UCM, EUCM, Kannala-Brandt Camera Model 등)
date: 2022-06-18 00:00:00
img: vision/concept/camera_models/0.png
categories: [vision-concept] 
tags: [camera model, 카메라 모델, 핀홀, UCM, EUCM, Double Sphere, Kannala-Brandt Camera Model] # add tag
---

<br>

- 논문 : https://arxiv.org/abs/1807.08957
- 참조 : https://vision.in.tum.de/research/vslam/double-sphere

<br>

- 본 글은 `The Double Sphere Camera Model`이라는 논문 내용을 기준으로 작성하였습니다. 논문을 작성한 랩실이 유명한 `Daniel Cremers`의 랩실에서 작성한 논문인 것에서 관심이 가기도 하였습니다.
- 이번 글에서는 최종적으로 `Double Sphere`라는 카메라 모델을 소개하고자 하며 `Double Sphere` 카메라 모델은 `UCM` 및 `EUCM` 카메라 모델 등의 `Generic Camera Model`을 발전시킨 카메라 모델입니다.

<br>
<center><img src="../assets/img/vision/concept/camera_models/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- `Double Sphere` 카메라 모델은 위 그림과 같이 2개의 `Sphere`를 기준으로 모델링 되었기 때문에 `Double Sphere`라는 이름으로 지어졌으며 이는 단일 `Spehere`를 가지는 `UCM, EUCM`과의 차이점입니다.
- `Double Sphere` 카메라 모델의 파라미터는 총 6개로 적당한 수준이며 fisheye 카메라와 같은 넓은 화각에 대해서도 카메라 캘리브레이션 성능이 좋으며 `projection` 및 `unprojection`이 모두 `closed-form` 형태로 구성되어 있습니다. 그리고 `projection` 및 `unprojection`의 계산 시 삼각 함수와 같이 계산 비용이 큰 함수도 없기 때문에 계산 복잡도도 낮은 이점이 있습니다.

<br>

## **목차**

<br>

- ### Abstract
- ### Introduction
- ### Related Work
- ### Pinhole Camera Model
- ### Unified Camera Model
- ### Extended Unified Camera Model
- ### Kannala-Brandt Camera Model
- ### Field-of-View Camera Model
- ### Double Sphere Camera Model
- ### Calibration
- ### Evaluation
- ### Conclusion

<br>


