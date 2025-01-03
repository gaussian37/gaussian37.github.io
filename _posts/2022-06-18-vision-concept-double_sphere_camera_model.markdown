---
layout: post
title: Double Sphere 카메라 모델 및 다양한 카메라 모델의 종류 (Pinhole, UCM, EUCM, Kannala-Brandt Camera Model 등)
date: 2022-06-18 00:00:00
img: vision/concept/double_sphere_camera_model/0.png
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
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/1.gif" alt="Drawing" style="width: 400px;"/></center>
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
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 비전 기반의 다양한 어플리케이션을 개발하기 위하여 넓은 영역의 FOV (Field Of View)를 가지는 카메라가 많이 사용되고 있습니다. 본 논문에서 사용하는 카메라 렌즈는 `fisheye` 카메라 렌즈로 저렴한 가격과 넓은 시야의 장점으로 많이 사용되고 있습니다. 따라서 본 논문에서는 `fisheye` 카메라 렌즈의 다양한 카메라를 이용하여 실험을 진행하였습니다.
- 논문에서 제안하는 `Double Sphere` 카메라 모델은 `projection` 및 `unprojection` 각각에 대하여 `closed-form` 형태의 식을 가지며 계산 복잡도 낮은 장점을 가집니다. 
- `Double Sphere` 카메라 모델의 유효성을 다양한 카메라 모델과 다양한 카메라 데이터셋에서 검증하였으며 `reprojection error`, `projection, unprojection, Jacobian computational time`에 대하여 우수한 성능을 지님을 보여줍니다.

<br>

## **Introduction**

<br>

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `Visual Odometry` 또는 `SLAM` 등의 어플리케이션이 중요해 지고 있으며 이러한 어플리케이션의 성능을 향상시키기 위하여 하드웨어와 소프트웨어의 개선이 필요해지고 있습니다. `large field-of-view` 카메라는 이와 같은 어플리케이션에서 몇가지 개선점을 제공합니다.
- ① `large FOV` 카메라는 더 많은 texture 영역을 이미지 내에 담을 수 있고 이러한 특성은 비전 기반의 motion estimation에 필수적입니다.
- ② `large FOV` 카메라는 같은 해상도의 `small FOV` 카메라에 비하여 카메라의 움직임을 더 작은 픽셀의 움직임으로 매핑하여 표현할 수 있습니다. 즉, 카메라의 움직임을 더 촘촘하게 표현할 수 있다는 뜻입니다. 이와 같은 장점은 픽셀의 움직임을 예측하는 `optical flow`에 큰 도움이 됩니다.
- 이러한 장점들이 `motion estimation`에 큰 도움을 주므로 `large FOV`의 대표적인 카메라 렌즈인 `fisheye lense`가 실생활에 많이 새용되고 있습니다. 따라서 본 논문에서는 `large FOV` 카메라 렌즈 중 `fisheye lense`를 기준으로 실험 및 기술이 될 예정입니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞으로 다룰 내용은 다양한 카메라 모델을 `fisheye` 카메라 렌즈를 사용하는 다양한 카메라에 대하여 성능을 검토하였으며 카메라 모델 별 `projection`, `unprojection` 함수 등을 제공합니다.
- 기존에 사용하던 다양한 카메라 모델에 대한 설명과 함께 논문에서 제시하는 `Double Sphere` 카메라 모델에 대한 설명을 추가적으로 이어가겠습니다. 앞에서 언급한 바와 같이 `Double Sphere` 카메라 모델은 ① 화각이 넓은 `fisheye` 렌즈와 같은 환경에서 projection이 잘 되도록 모델링 되어 있고 ② `projection`, `unprojection` 시 삼각함수와 같은 계산 비용이 큰 계산을 필요로 하지 않습니다. 또한 ③ `unprojection` 함수 식이 근사식으로 구하는 것이 아니라 `closed-form` 형태로 지정되어 있다는 장점이 있습니다.
- 각 카메라 모델에 대한 소개를 한 후에 비전 기반의 motion estimation과 연관된 지표인 `reprojection error`와 계산 효율성과 관련된 projection, unprojection `computational time`과 Jacobbian 연산 시간을 통해 비교해 보도록 하겠습니다.

<br>

## **Related Work**

<br>

- 앞으로 사용할 기호에 대하여 먼저 정리하도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/6.png" alt="Drawing" style="width: 400px;"/></center>
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
- ⑨ $$ \pi^{-1} $$ :` 2D 이미지 공간`의 픽셀 값을 focal length (vector of unit lenght)가 1인 normalized(standardized) image plane으로 변환하는 함수를 의미합니다. 2D 이미지 공간으로 projection 되었기 때문에 $$ z $$ 값이 사라져서 `unprojection` 결과는 `normalized image plane` 까지 변환할 수 있습니다. 본 논문에서는 `normalized image plane`에서 더 나아가 `unit sphere`에 대응하는 것 까지 확장합니다. `sphere`는 3차원 공간에서 다른 점으로부터 같은 거리에 있는 모든 점의 집합을 찾아 생성된 `2차원 표면`입니다. 수식으로 표현하면 다음과 같습니다.

<br>

- $$ d(x, c) = \sqrt{\sum_{i=1}^{3} (x_{i} - c_{i})^{2}} $$

<br>

- `unit sphere`는 구의 모든 점이 중심에서 하나의 거리에 있도록 하는 구입니다. 휘어져 있기 때문에 3차원 공간에 내장되어 표현되는 경우가 많습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/15.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 2차원 표면을 `sphere`라고 하며 3차원으로 확장하면 `ball` 이라고 합니다. 
- 이러한 이유로 논문에서는 `sphere`를 $$ \mathbb{S}^{2} $$ 로 표현합니다. 따라서 다음과 같이 기호로 표현할 수 있습니다. $$ \pi^{-1} : \Theta \to \mathbb{S}^{2} $$

<br>

## **Pinhole Camera Model**

<br>

- 먼저 소개하는 카메라 모델은 가장 기본적인 `Pinhole` 카메라 모델 입니다. `Pinhole` 카메라 모델은 카메라 렌즈의 왜곡 (distortion)이 없다고 가정한 카메라 모델이며 앞으로 다룰 `distortion` 모델들은 `Pinhole` 카메라 모델에 렌즈 왜곡을 반영한 모델입니다.
- 먼저 가장 기본이 되는 `Pinhole` 카메라 모델의 `projection`, `unprojection`에 대하여 다루어 보도록 하겠습니다. `Pinhole` 카메라 모델의 파라미터는 4개이며 다음과 같습니다.

<br>

- $$ i = [f_{x}, f_{y}, c_{x}, c_{y}]^{T} $$

<br>

- 위 4가지 파라미터의 상세 내용은 아래 글을 참조하시면 됩니다.
    - 링크 : [https://gaussian37.github.io/vision-concept-calibration/](https://gaussian37.github.io/vision-concept-calibration/)

<br>

- `Pinhole` 카메라 모델에서는 3D 포인트를 2D 이미지 좌표계로 변형하려면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/14.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 카메라 렌즈의 왜곡이 없기 때문에 단순히 $$ z $$ 로 나누기만 하면 `undistorted normalized image plane`으로 사영됩니다.
- `projection` 시 유효한 값들은 $$ z > 0 $$ 인 범위의 값이며 기호로 나타내면 다음과 같습니다.

<br>

- $$ \Omega = \{\mathbf{x} \in \mathbb{R}^{3} \vert z > 0 \} $$

<br>

- 위 조건으로 인하여 `FOV (field-of-view)`는 180도 이하만 유효합니다. 즉 카메라 원점 기준으로 뒤의 값은 투영될 수 없습니다.
- 본 글에서는 `Pinhole` 카메라 모델에 `lense distortion`을 반영하기 위한 모델이 추가되어 120도 이상의 화각을 가지는 카메라 렌즈를 어떻게 사용할 지 다룰 예정입니다.

<br>

- `Pinhole` 카메라 모델의 `unprojection` 식은 다음 식과 같습니다. $$ \pi^{-1} : \Theta \to \mathbb{S}^{2} $$ 이기 때문에 최종 변환 지점은 `unit sphere`입니다. 즉, 일반적으로 사용하는 `normalized image plane`이 아니므로 식이 조금 다릅니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/16.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 일반적으로 위 식의 $$ [m_{x}, m_{y}, 1]^{T} $$ 가 `2D image plane`에서 `normalized image plane`으로 변환하는 식에 해당합니다.
- 카메라 중점과 `depth`가 1이 되는 지점을 `normalized image plane`이라고 부릅니다. 반면 카메라 중점과 `distance`가 1이 되는 지점을 `unit sphere`라고 부르며 이번 글에서는 `unit sphere`를 위주로 다룹니다.
- 위 식에서는 $$ 1 / \sqrt{m_{x}^{2} + m_{y}^{2} + 1} $$ 이 추가로 곱해집니다. 이 값은 $$ \sqrt{(m_{x} - 0)^{2} + (m_{y} - 0)^{2} + (1 - 0)^{2}} $$ 인 `distance` 값을 나누어 준 값으로 `unit sphere` 형태로 만들어 주기 위해 곱해집니다. 즉, `normalized image plane`에서의 모든 `distance`를 각 픽셀 별로 나누어 줌으로써 모든 거리가 동일한 `unit sphere`로 변환됩니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/17.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `unit sphere`는 위 그림처럼 `distance`가 1이 되도록 만든 `sphere` 입니다.
- 이 글에서 표현하는 모든 `projection`은 `3D Point → unit sphere → normalized image plane → image plane`이고 `unprojection`은 그 역순입니다.
- 따라서 본 논문의 `unprojection`의 결과는 `image plane` → `unit sphere`로의 변환이며 `unit sphere`의 좌표에서 `depth` 방향의 값을 곱하여 scale을 조정하면 실제 `3D Point`의 $$ X, Y, Z $$ 좌표값이 됩니다. 예를 들어 `unit sphere` 상에서 어떤 점은 $$ (X, Y, Z) $$ 값을 가지고 `depth` 방향으로 $$ d $$ 값을 가지면 depth 값을 곱하여 $$ (X*d, Y*d, Z*d) $$ 가 됩니다. 왜냐하면 `3D Point` 에서 `unit sphere` 표면 까지는 직진하기 때문입니다.
- 반면 `unit sphere` → `normalized image plane` 까지는 `unit sphere` 표면에서 한번 굴절되며 이 굴절되는 정도는 카메라 모델에 따라서 바뀌게 됩니다. 따라서 다음의 형태를 가집니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/18.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/19.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림은 카메라 원점으로 부터 `distance`가 1인 지점 까지의 `unit sphere`가 `depth`가 1인 지점인 `normalized image plane`으로 변환되어야 하는 것을 나타낸 것입니다. 개념적으로 나타내면 위 그림과 같이 곡면이 평면으로 펴지게 됩니다.

<br>

- `Pinhole` 모델에서는 `unit sphere`에서 `normalized image plane` 까지 굴절이 없이 직진하기 때문에 `projection` 시 별다른 수식이 없었습니다.
- 반면 앞으로 다룰 다른 카메라 모델에서는 `unit sphere`에서 `normalized image plane` 까지 다양한 방식의 굴절을 반영하기 때문에 `Pinhole` 모델보다 다소 복잡한 식을 가지게 되며 이 내용을 다루어 보도록 하겠습니다.

<br>

## **Unified Camera Model**

<br>

- `Unified Camera Model`에 대한 상세 내용은 아래 링크에서 확인할 수 있습니다.
    - 참조 : [https://gaussian37.github.io/vision-concept-unified_camera_model/](https://gaussian37.github.io/vision-concept-unified_camera_model/)

<br>

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/20.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 

<br>

## **Extended Unified Camera Model**

<br>

<br>

## **Kannala-Brandt Camera Model**

<br>

- `Kannala-Brandt Camera Model`에 대한 상세 내용은 아래 링크에서 확인할 수 있습니다.
    - 참조 : [https://gaussian37.github.io/vision-concept-generic_camera_model/](https://gaussian37.github.io/vision-concept-generic_camera_model/)

<br>

<br>

## **Field-of-View Camera Model**

<br>

<br>

## **Double Sphere Camera Model**

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 논문에서 제시하는 `Double Sphere` 카메라 모델에 대하여 설명해 보도록 하곘습니다. 글 서두에 말씀드린 바와 같이 `Double Sphere` 카메라 모델은 화각이 넓은 `fisheye lense`와 같은 환경에서도 잘 동작하며 `unprojection`을 위한 `closed-form` 형태의 식이 존재하고 그 계산량도 적다는 것을 확인하였었습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/8.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `Double Sphere` 카메라 모델은 위 그림과 같이 이름 그대로 3D 포인트가 2개의 단위 구에 투영되는 형태를 가집니다.
- 3D 포인트가 먼저 초록색 구에 투영되면서 꺽이게 되고 그 다음 빨간색 구에 투영되면서 한번 더 꺽이면서 렌즈 굴절이 반영됩니다. 빨간색 구의 중심은 $$ \xi $$ 만큼 이동되어 있고 이 값이 구의 반영을 결정하는 파라미터가 됩니다.
- 가장 아래 부분의 검은색 가로 실선인 `normalized image plane`과 빨간색 구의 중심 간의 거리가 $$ \frac{\alpha}{1 - \alpha} $$ 로 결정됩니다. 따라서 $$ \xi $$ 와 $$ \frac{\alpha}{1 - \alpha} $$ 가 같이 결합되어 변수가 동작하도록 되어 있습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/9.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 Double Sphere 모델에서 구에 의해 꺽이는 양을 표현하면 위 그림과 같습니다. 다만 Double Sphere 카메라 모델 에서는 앞의 다른 카메라 모델과 같이 실제 꺽이는 양을 모델링 하지 않고 $$ \alpha, \xi $$ 값을 이용하여 normalized image plane에 어떻게 투영되는 지 모델링 하는 방법을 사용하였습니다.
- 즉, `UCM`과 비교하였을 때, $$ \frac{\alpha}{1 - \alpha} $$ 만큼 `shift`가 시작되는 시점이 $$ \xi $$ 만큼 이동되는 것이고 그 첫번째 구와 두번째 구의 복합적인 꺽임 양은 $$ \xi, \alpha $$의 결합으로 이루어짐을 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/10.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림을 기준으로 $$ \xi = 0, \alpha = 0 $$ 이 되면 핀홀 모델이 됩니다. 즉, 빨간색 구의 중심이 초록색 구의 중심으로 이동되면서 구의 중심이 하나가 되고 구로 인해 꺽이는 양도 사라져서 normalized image plane이 초록색 및 빨간색 구의 중심선에 생기게 되는 경우입니다.
- 논문에서는 이와 같은 방식의 카메라 모델을 사용하였을 때 캘리브레이션 패턴에서의 꼭지점이 에러가 적은 형태로 잘 projection 되는 것을 확인할 수 있음을 보여줍니다.
- 그러면 `Double Sphere` 카메라 모델에 관련된 수식 내용을 상세히 살펴보도록 하겠습니다.

<br>

- `Double Sphere`는 6개의 파라미터를 가집니다. 

<br>

- $$ i = [f_{x}, f_{y}, c_{x}, c_{y}, \xi, \alpha]^{T} $$

<br>

- 위 파라미터 $$ i $$ 를 이용하여 3D 포인트를 2D로 `projection` 하는 함수 $$ \pi $$ 를 정의하면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/11.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 살펴본 `UCM` 계열과 같이 분모의 $$ \alpha d_{2} + (1-\alpha)(\xi d_{1} + z) $$ 를 통하여 단번에 `undistorted normalized plane` 으로 접근합니다.

<br>

- 다음은 `UCM`과 마찬가지로 $$ \alpha $$ 값의 크기에 따라 `undistorted normalized plane`의 위치가 바뀌므로 $$ \alpha $$ 와 다른 파라미터 값에 따라서 3D 포인트가 2D 이미지의 FOV에 유효한 지 확인하는 함수 식입니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/12.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음으로 `unprojection`하는 `closed-form` 형태의 식이며 `unprojection` 결과 2D 이미지 좌표계에서 `distorted normalized plane`으로 변환하는 역할을 합니다.
- 만약 `z`값을 복원할 수 있으면 $$ z \pi^{-1}(u, i) = \mathbf{x} $$ 로 3D 포인트를 구할 수 있습니다. `unprojection` 식은 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/13.png" alt="Drawing" style="width: 600px;"/></center>
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
<center><img src="../assets/img/vision/concept/double_sphere_camera_model/3.png" alt="Drawing" style="width: 400px;"/></center>
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

- `SO(3) Group`은 3차원 Rotation Matrix와 이에 닫혀 있는 연산들로 구성된 군을 의미하며 3차원 물체의 회전을 표현하는 데 사용됩니다.

<br>

- $$ \begin{equation} \begin{aligned} SO(3)= \{\mathbf{R} \in \mathbb{R}^{3 \times 3} \ \vert \ \mathbf{R} \mathbf{R}^{T} = \mathbf{I}, \text{det}(\mathbf{R})=1 \} \end{aligned} \end{equation} $$

<br>

- `SO(3) Group`의 속성은 다음과 같습니다.
- `Associativity` : $$ \mathbf{R}_{1} \cdot \mathbf{R}_{2}) \cdot \mathbf{R}_{3} = \mathbf{R}_{1} \cdot (\mathbf{R}_{2} \cdot \mathbf{R}_{3}) $$ 즉, 결합 법칙이 성립합니다.
- `Identity element` : $$ \mathbf{R} \cdot \mathbf{I} = \mathbf{I} \cdot \mathbf{R}  = \mathbf{R} $$ 를 만족하는 3 X 3 항등 행렬 $$ \mathbf{I} $$ 가 존재합니다.
- `Inverse` : $$ \mathbf{R}^{-1} \cdot \mathbf{R} = \mathbf{R}\cdot\mathbf{R}^{-1} = \mathbf{I} $$ 을 만족하는 역행렬이 존재하며 $$ \mathbf{R}^{-1} = \mathbf{R}^{\intercal} $$ 가 됩니다. 따라서 $$ \mathbf{R}\cdot \mathbf{R}^{\intercal} = \mathbf{I} $$ 를 만족합니다.
- `Composition` : `SO(3) Group`의 합성은 다음과 같이 행렬의 곱셈 연산으로 표현할 수 있습니다. $$ \begin{equation} \begin{aligned} & \mathbf{R}_{1} \cdot \mathbf{R}_{2} = \mathbf{R}_{3} \in SO(3) \end{aligned} \end{equation} $$
- `Non-commutative` : $$ \mathbf{R}_{1}\cdot\mathbf{R}_{2} \neq \mathbf{R}_{2}\cdot\mathbf{R}_{1} $$ 교환 법칙이 성립하지 않습니다.
- `Determinant` : $$ \text{det}(\mathbf{R})=1 $$ 을 만족합니다.
- `Rotation` : $$ \mathbb{P}^{2} $$ 공간 상의 점 또는 벡터 $$ \mathbf{x} = \begin{bmatrix} x&y&z \end{bmatrix}^{\intercal} \in \mathbb{P}^{2} $$ 를 다른 방향의 점 또는 벡터 $$ \mathbf{x}' $$ 로 회전시킬 수 있습니다. $$ \begin{equation} \begin{aligned} \mathbf{x}' = \mathbf{R}\cdot \mathbf{x} \end{aligned} \end{equation} $$

<br>

- 논문에서 사용한 `SE(3) Group`은 3차원 공간 상에서 `Rigid Body Transformation`과 관련된 행렬과 이에 닫혀 있는 연산들로 구성된 Group을 의미합니다. 즉, `Rotation`과 `Translation`이 모두 반영되어 있습니다.

<br>

- $$ \begin{equation} \begin{aligned} SE(3) = \left \{ \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix} \in \mathbb{R}^{4\times4} \ | \ \mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^{3}  \right \} \end{aligned} \end{equation} $$

<br>

- `SE(3) Group`의 속성은 다음과 같습니다.
- `Associativity` : $$ \mathbf{T}_{1} \cdot \mathbf{T}_{2}) \cdot \mathbf{T}_{3} = \mathbf{T}_{1} \cdot (\mathbf{T}_{2} \cdot \mathbf{T}_{3}) $$ 와 같이 결합 법칙이 성립합니다.
- `Identity element` : $$ \mathbf{T} \cdot \mathbf{I} = \mathbf{I} \cdot \mathbf{T}  = \mathbf{T} $$ 를 만족하는 4 x 4 항등 행렬 $$ \mathbf{I} $$ 가 존재합니다.
- `Inverse` : $$ \mathbf{T}^{-1} \cdot \mathbf{T} = \mathbf{T}\cdot\mathbf{T}^{-1} = \mathbf{I} $$ 을 만족하는 역행렬이 다음과 같습니다.

<br>

- $$ \begin{equation} \begin{aligned} \mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^{T} & -\mathbf{R}^{T}\mathbf{t} \\ \mathbf{0} & 1\end{bmatrix} \end{aligned} \end{equation} $$ 

<br>

- `Composition` : `SE(3) Group`의 Composition은 아래와 같이 행렬의 곱셈 연산으로 이루어집니다.

<br>

- $$ \begin{equation} \begin{aligned} \mathbf{T}_{1} \cdot \mathbf{T}_{2} & = \begin{bmatrix} \mathbf{R}_{1} & \mathbf{t}_{1} \\ \mathbf{0} & 1 \end{bmatrix} \cdot \begin{bmatrix} \mathbf{R}_{2} & \mathbf{t}_{2} \\ \mathbf{0} & 1 \end{bmatrix} \\ & = \begin{bmatrix} \mathbf{R}_{1}\mathbf{R}_{2} & \mathbf{R}_{1}\mathbf{t}_{2} + \mathbf{t}_{1} \\ \mathbf{0} & 1 \end{bmatrix} \in SE(3) \end{aligned} \end{equation} $$

<br>

- `Non-commutative` : $$ \mathbf{T}_{1}\cdot\mathbf{T}_{2} \neq \mathbf{T}_{2}\cdot\mathbf{T}_{1} $$ 교환 법칙이 성립하지 않습니다.
- `Transformation` : $$ \mathbb{P}^{3} $$ 공간 상의 점 또는 벡터 $$ \mathbf{X} = \begin{bmatrix} X&Y&Z&W \end{bmatrix}^{\intercal} \in \mathbb{P}^{3} $$ 를 다른 방향과 위치를 가지는 점 또는 벡터 $$ \mathbf{X}' $$ 로 변환할 수 있습니다.

<br>

- $$ \begin{equation} \begin{aligned} \mathbf{X}' = \mathbf{T}\cdot \mathbf{X} & = \begin{bmatrix} \mathbf{R}&\mathbf{t}\\\mathbf{0}&1 \end{bmatrix} \cdot \mathbf{X} \\ & = \begin{bmatrix} \mathbf{R}(X \ Y \ Z)^{T} + W \cdot \mathbf{t} \\ W \end{bmatrix} \end{aligned} \end{equation} $$

<br>