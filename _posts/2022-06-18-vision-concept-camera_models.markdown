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

- ### [Abstract](#)
- ### [Introduction](#)
- ### [Related Work](#)
- ### [Pinhole Camera Model](#)
- ### [Unified Camera Model](#)
- ### [Extended Unified Camera Model](#)
- ### [Kannala-Brandt Camera Model](#)
- ### [Field-of-View Camera Model](#)
- ### [Double Sphere Camera Model](#)
- ### [Calibration](#)
- ### [Evaluation](#)
- ### [Conclusion](#)

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/concept/camera_models/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 비전 기반의 다양한 어플리케이션을 개발하기 위하여 넓은 영역의 FOV (Field Of View)를 가지는 카메라가 많이 사용되고 있습니다. 대표적인 카메라가 `fisheye` 카메라로 저렴한 가격과 넓은 시야의 장점으로 많이 사용되고 있으며 본 논문에서도 `fisheye` 카메라를 이용하였습니다. 추가적으로 다양한 광범위한 FOV를 가지는 카메라를 이용하여 실험을 진행하였습니다.
- 본 논문에서 제안하는 `Double Sphere` 카메라 모델은 `projection` 및 `unprojection` 각각에 대하여 `closed-form` 형태의 식을 가지며 계산 복잡도 낮은 장점을 가집니다. 
- `Double Sphere` 카메라 모델의 유효성을 다양한 카메라 모델과 다양한 카메라 데이터셋에서 검증하였으며 `reprojection error`, `projection, unprojection, Jacobian computational time`에 대하여 우수한 성능을 지님을 보여줍니다.

<br>

## **Introduction**

<br>

<br>

## **Related Work**

<br>

<br>

## **Pinhole Camera Model**

<br>

<br>

## **Unified Camera Model**

<br>

<br>

## **Extended Unified Camera Model**

<br>

<br>

## **Kannala-Brandt Camera Model**

<br>

<br>

## **Field-of-View Camera Model**

<br>

<br>

## **Double Sphere Camera Model**

<br>

<br>

## **Calibration**

<br>

<br>

## **Evaluation**

<br>

<br>

## **Conclusion**

<br>

<br>
<center><img src="../assets/img/vision/concept/camera_models/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 본 논문에서는 16개의 카메라 데이터셋과 6개의 카메라 모델을 통하여 실험을 진행하였고 `Double Sphere` 카메라 모델의 타당함을 보여주었습니다.
- 정확도 성능 측면에서는 8개의 파라미터를 사용하는 `Kannala-Brandt` 카메라 모델 성능이 가장 좋았으며 `Double Sphre` 모델이 근소한 차이로 그 뒤를 따랐습니다. 다만 `Kannala-Brandt` 카메라 모델은 계산량이 많아 다른 카메라 모델에 비해 5 ~ 10배 정도 느린 단점이 있었습니다.
- `Kannala-Brandt`는 `closed-form`형태의 `unprojection` 식이 없다는 단점과 계산량이 크다는 단점이 있는 반면 그 다음으로 성능이 좋은 `Double Sphere` 카메라 모델은 `closed-form` 형태의 `unprojection` 식이 있으며 계산량도 적다는 장점이 있습니다.
- 따라서 본 논문에서는 넓은 화각의 굴곡이 큰 카메라 렌즈에서 `Double Sphere` 카메라 모델을 사용하는 것의 유효성을 확인할 수 있었습니다.

<br>
