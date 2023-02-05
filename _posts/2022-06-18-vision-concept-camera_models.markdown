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

- ### [Abstract](#abstract-1)
- ### [Introduction](#introduction-1)
- ### [Related Work](#related-work-1)
- ### [Pinhole Camera Model](#pinhole-camera-model-1)
- ### [Unified Camera Model](#unified-camera-model-1)
- ### [Extended Unified Camera Model](#extended-unified-camera-model-1)
- ### [Kannala-Brandt Camera Model](#kannala-brandt-camera-model-1)
- ### [Field-of-View Camera Model](#field-of-view-camera-model-1)
- ### [Double Sphere Camera Model](#double-sphere-camera-model-1)
- ### [Calibration](#calibration-1)
- ### [Evaluation](#evaluation-1)
- ### [Conclusion](#conclusion-1)
- ### [Appendix](#appendix-1)

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/concept/camera_models/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 비전 기반의 다양한 어플리케이션을 개발하기 위하여 넓은 영역의 FOV (Field Of View)를 가지는 카메라가 많이 사용되고 있습니다. 본 논문에서 사용하는 카메라 렌즈는 `fisheye` 카메라 렌즈로 저렴한 가격과 넓은 시야의 장점으로 많이 사용되고 있습니다. 따라서 본 논문에서는 `fisheye` 카메라 렌즈의 다양한 카메라를 이용하여 실험을 진행하였습니다.
- 논문에서 제안하는 `Double Sphere` 카메라 모델은 `projection` 및 `unprojection` 각각에 대하여 `closed-form` 형태의 식을 가지며 계산 복잡도 낮은 장점을 가집니다. 
- `Double Sphere` 카메라 모델의 유효성을 다양한 카메라 모델과 다양한 카메라 데이터셋에서 검증하였으며 `reprojection error`, `projection, unprojection, Jacobian computational time`에 대하여 우수한 성능을 지님을 보여줍니다.

<br>

## **Introduction**

<br>

<br>
<center><img src="../assets/img/vision/concept/camera_models/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `Visual Odometry` 또는 `SLAM` 등의 어플리케이션이 중요해 지고 있으며 이러한 어플리케이션의 성능을 향상시키기 위하여 하드웨어와 소프트웨어의 개선이 필요해지고 있습니다. `large field-of-view` 카메라는 이와 같은 어플리케이션에서 몇가지 개선점을 제공합니다.
- ① `large FOV` 카메라는 더 많은 texture 영역을 이미지 내에 담을 수 있고 이러한 특성은 비전 기반의 motion estimation에 필수적입니다.
- ② `large FOV` 카메라는 같은 해상도의 `small FOV` 카메라에 비하여 카메라의 움직임을 더 작은 픽셀의 움직임으로 매핑하여 표현할 수 있습니다. 즉, 카메라의 움직임을 더 촘촘하게 표현할 수 있다는 뜻입니다. 이와 같은 장점은 픽셀의 움직임을 예측하는 `optical flow`에 큰 도움이 됩니다.
- 이러한 장점들이 `motion estimation`에 큰 도움을 주므로 `large FOV`의 대표적인 카메라 렌즈인 `fisheye lense`가 실생활에 많이 새용되고 있습니다. 따라서 본 논문에서는 `large FOV` 카메라 렌즈 중 `fisheye lense`를 기준으로 실험 및 기술이 될 예정입니다.

<br>
<center><img src="../assets/img/vision/concept/camera_models/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞으로 다룰 내용은 다양한 카메라 모델을 `fisheye` 카메라 렌즈를 사용하는 다양한 카메라에 대하여 성능을 검토하였으며 카메라 모델 별 `projection`, `unprojection` 함수 등을 제공합니다.
- 기존에 사용하던 다양한 카메라 모델에 대한 설명과 함께 논문에서 제시하는 `Double Sphere` 카메라 모델에 대한 설명을 추가적으로 이어가겠습니다. 앞에서 언급한 바와 같이 `Double Sphere` 카메라 모델은 ① 화각이 넓은 `fisheye` 렌즈와 같은 환경에서 projection이 잘 되도록 모델링 되어 있고 ② `projection`, `unprojection` 시 삼각함수와 같은 계산 비용이 큰 계산을 필요로 하지 않습니다. 또한 ③ `unprojection` 함수 식이 근사식으로 구하는 것이 아니라 `closed-form` 형태로 지정되어 있다는 장점이 있습니다.
- 각 카메라 모델에 대한 소개를 한 후에 비전 기반의 motion estimation과 연관된 지표인 `reprojection error`와 계산 효율성과 관련된 projection, unprojection `computational time`과 Jacobbian 연산 시간을 통해 비교해 보도록 하겠습니다.

<br>

## **Related Work**

<br>

- 앞으로 사용할 기호에 대하여 먼저 정리하도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/camera_models/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- ① $$ \text{scalars}, u, v, ... $$ : 소문자로 나타낸 값을 스칼라 값으로 표현합니다.
- ② $$ \text{matrices} \mathbf{R} $$ : 대문자로 나타낸 값을 행렬로 표현합니다.
- ③ $$ \text{vectors}, \mathbf{u}, \mathbf{v} $$ : 굵은 글씨체의 소문자로 나타낸 값을 벡터로 표현합니다.
- ④ $$ \Theta$$ : 3D 포인트가 projection 된 2D 이미지 공간을 의미합니다.
    - 따라서 $$ \mathbf{u} = [u, v]^{T} \in \Theta \subset \mathbb{R}^{2} $$ 표현은 이미지 좌표계의 각 픽셀 $$ \mathbf{u} $$ 는  $$ \Theta$$ 에 속함을 의미합니다.
- ⑤ $$ \Omega$$ : 3D 포인트 집합을 의미합니다.
    - 따라서 $$ \mathbf{x} = [x, y, z]^{T} \in \Omega \subset \mathbb{R}^{3} $$ 표현은 3D 포인트 $$ \mathbf{x} $$ 가 $$ \Omega $$ 에 속함을 의미합니다.
- ⑥ $$ x, y, z $$ : 카메라 좌표계는 `principal axis` 축이 $$ z $$ 축이 되며 `image plane`의 width 방향이 $$ x $$, height 방향이 $$ y $$ 인 것과 동일하게 적용됩니다.
- ⑦ $$ \text{SE(3)} $$ : 논문에서 소개하는 캘리브레이션 패턴의 3D 좌표계에서 카메라 좌표계로 변환하는 `Transformation Matrix`는 `Special Euclidean Group`임을 뜻합니다. `SE(3)`에 대한 상세 내용은 글 가장 아랫부분의 `Appendix`부분에 따로 정리하였습니다.
- ⑧ $$ \pi $$ : 3D 포인트를 `2D 이미지 공간`에 projection 하는 함수를 의미합니다. 따라서 다음과 같이 기호로 표현합니다. $$ \pi : \Omega \to \Theta $$
- ⑨ $$ \pi^{-1} $$ :` 2D 이미지 공간`의 픽셀 값을 focal length (vector of unit lenght)가 1인 normalized(standardized) image plane으로 변환하는 함수를 의미합니다. 2D 이미지 공간으로 projection 되었기 때문에 $$ z $$ 값이 사라져서 `unprojection` 결과는 z 값을 곱하기 직전 상태인 `normalized image plane`까지만 변환할 수 있습니다. `normalized image plane`을 논문에서는 $$ \mathbb{S}^{2} $$ 로 표현합니다. 따라서 다음과 같이 기호로 표현할 수 있습니다.$$ \pi^{-1} : \Theta \to \mathbb{S}^{2} $$

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

## **Appendix**

<br>

#### **SO(3)와 SE(3)**

<br>

- 참조 : https://alida.tistory.com/9#org3029dbb
- `Lie Group` 종류 중 `Special Orthogonal 3 Group` 줄여서 `SO(3) Group`과 `Special Euclidaen 3 Group`  줄여서 `SE(3) Group`이 있습니다. 3D 공간 상에서 Transformation Matrix를 다룰 때 `SO(3) Group`은 3차원 회전 행렬을 다루고 `SE(3) Group`은 3차원 강체 변환 (Rigid Transformation) 행렬을 다룹니다.
- 여기서는 본 논문의 이해를 돕기 위하여 `SO(3) Group`과 `SE(3) Group`의 의미와 간단한 성질만 살펴보겠습니다. 상세 내용은 위 참조 링크를 확인해 보시기 바랍니다.

<br>

