---
layout: post
title: IPM(Inverse Perspective Mapping)을 이용한 Bird Eye View 변환
date: 2022-01-29 00:00:00
img: vision/concept/ipm/0.png
categories: [vision-concept] 
tags: [IPM, Bird Eye View, BEV, Top-Down, Top View] # add tag
---

<br>

- 참조 : https://towardsdatascience.com/a-hands-on-application-of-homography-ipm-18d9e47c152f
- 참조 : https://github.com/darylclimb/cvml_project
- 참조 : https://csyhhu.github.io/2015/07/09/IPM/
- 참조 : https://kr.mathworks.com/help/driving/ref/birdseyeview.imagetovehicle.html

<br>

- 사전 지식 : [Homogeneous Coordinate (동차좌표계)](https://gaussian37.github.io/vision-concept-homogeneous_coordinate/)
- 사전 지식 : [카메라 캘리브레이션](https://gaussian37.github.io/vision-concept-calibration/)
- 사전 지식 : [포인트 클라우드와 뎁스 맵의 변환 관계 정리](https://gaussian37.github.io/vision-depth-pcd_depthmap/)

<br>

- 이번 글에서 다룰 내용은 위 그림과 같이 흔히 보는 왼쪽 그림과 같은 `perspective view` 형태의 이미지를 `Bird Eye View` 형태로 변환하는 방법에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [IPM의 사용 배경](#ipm의-사용-배경-1)
- ### [IPM을 위한 배경 설명](#ipm을-위한-배경-설명-1)
- ### [IPM 적용 방법](#ipm-적용-방법-1)
- ### [Python 코드](#python-코드-1)

<br>

## **IPM의 사용 배경**

<br>

- `IPM`은 `Inverse Perspective Mapping`의 줄임말이며 그 역할은 2D 이미지를 `BEV(Bird Eye View)` 방식의 3D로 변환하는 것입니다. `BEV`로 나타내기 때문에 최종 표현되는 결과는 2D 이며 따라서 2D 이미지로 표현 가능합니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림에서 좌측이 `2D 이미지`이고 우측이 `BEV` 입니다. 우측의 `BEV`의 하단 부분에서 상단 부분으로 갈수록 RGB 영역이 점점 커지는 이유는 실제 카메라가 촬영하는 화각에 의해 반영되는 것입니다. 빛이 카메라 렌즈로 입사할 때, 실제 이미지로 투영되는 영역이 위에 보이는 화각 영역 정도라고 생각하시면 됩니다.

<br>

- `IPM`의 이름을 살펴보면 `Inverse`, `Perspective Mapping` 2가지 의미로 나뉘게 됩니다. 
- 먼저 `Perspective Mapping`은 `Perspective Projection`을 의미합니다. `Perspective Projection`은 3D Scene을 2D image plane으로 투영하는 작업 방식 중에 하나이며 `Perspective Projection`을 하면 일반적으로 보는 사진과 같이 **원근감이 있도록** 2D 이미지가 생성 됩니다. 같은 크기의 물체임에도 불구하고 가까우면 크게 보이고 멀리 있으면 작게 보이는 것이 `Perspective Projection`에 나타나는 왜곡의 형태 (`Perspective Distortion`) 입니다. 
- `Inverse`는 `Perspective Projection`의 과정을 역으로 진행하면서 앞에서 설명한 `원근감 (Perspective)`에 의한 왜곡을 제거하는 것을 의미합니다. 따라서 `IPM`의 과정은 `Perspective Distortion`을 제거하여 전체 3D Scene에 대하여 카메라와의 거리에 상관 없이 일관성 있게 표현하는 것을 목표로 하며 표현 방식은 `BEV` 형식으로 표현하게 됩니다.
- 여기서 `BEV`로 표현하는 이유는 `BEV`로 보았을 때, 카메라와의 원근에 상관 없이 동일한 크기로 표현할 수 있다는 점을 이용하는 것과 3D 상의 실제 물체의 `Depth`는 알 수 없으므로 모든 물체는 높이가 없이 지면(ground)에 붙어 있는 것을 가정해야 하기 때문입니다. 2D 이미지의 정보를 3D 로 변환하기 위해서는 `Depth`를 통하여 물체의 높이 정보를 알 수 있는데 ([포인트 클라우드와 뎁스 맵의 변환 관계 정리](https://gaussian37.github.io/vision-depth-pcd_depthmap/) 참조) 이 정보를 알 수 없으니 모든 물체의 높이를 무시할 수 있는 방법인 `BEV`를 선택한 것입니다. 하늘에서 땅을 정면으로 바라보았을 때, 높이는 알 수 없는 형태의 시점으로 밖에 볼 수 없는데 그 점을 이용한 것입니다.

<br>

- 정리하면 `IPM, Inverse Perspective Mapping`은 2D 이미지를 3D 공간으로 변환하는데, `Perspective Distortion`을 제거하기 위함과 `Depth`값의 부재로 인하여 `BEV` 형태로 변환하는 과정을 의미합니다.
- 이와 같은 가정을 전제 조건으로 두고 `IPM`을 진행하기 때문에 아래 그림과 같이 한계 상황들이 이미 발생합니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 전제 조건인 **모든 물체의 높이는 0이다** 라는 가정 즉, 모든 물체는 높이가 없이 지면에 붙어 있다는 가정으로 인하여 실제 지면에 있는 정보들 (도로, 차선 풀 등)은  `Perspective Distortion` 없이 정상적으로 보이지만 자동차와 같이 높이가 있는 물체는 이상하게 보입니다. 즉, 전제 조건에 위배하기 때문에 정상적으로 나타낼 수 없습니다.
- 또한 도로 자체가 오르막길이거나 내리막길이면 발생하면 **모든 물체의 높이는 0이다**라는 가정에 위배되기 때문에 이상한 형상이 발생합니다. 아래 영상의 39초 정도에 그 현상을 살펴볼 수 있습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/ZI6NrQ3ZK2w" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 이와 같은 단점에도 불구하고 이상적인 평지 (flat ground) 환경에서는 지면의 정보를 인식하는 데에는 `Perspective Distortion`이 없다는 장점이 있기 때문에 상황에 따라서 사용하기도 합니다. 
- 아래는 `KITTI` 데이터 셋의 이미지를 `IPM`을 통하여 `BEV`로 생성한 데모이며 `KITTI` 데이터셋 에서는 지면의 높낮이가 적은 평지이기 때문에 이상적으로 잘 보이는 것을 알 수 있습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/QhFS6tj4_mo" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 그러면 지금부터는 실제 `IPM`을 적용하는 방법에 대하여 살펴보도록 하겠습니다.

<br>

## **IPM을 위한 배경 설명**

<br>

- 지금부터 살펴볼 방법은 front view 이미지를 `BEV` 이미지로 변경하는 방법입니다. `IPM`은 이러한 처리를 전방 카메라의 `원근 효과(perspective effect)`를 제거하고 top-view 기반의 2D 도메인 상에 다시 매핑하는 방법을 사용합니다. 즉, `BEV` 이미지는 원근 효과를 보정하여 거리와 평행선을 유지하는 성질을 가지고 있습니다.

<br>

- 다음 내용은 `homography` 기반의 `IPM`의 절차를 나타냅니다.
- ① 만들고자 하는 `BEV` 이미지는 `perspective view` 이미지의 도로를 평평한 2차원 평면으로 변환하여 생성합니다. 따라서 ( $$ X, Y, Z = 0 $$ ) 과 같이 모델링 하여 높이 정보가 없는 2차원 평면으로 간주합니다. 
- ② `projection matrix` $$ P $$ 를 `extrinsic`, `intrinsic` 파라미터를 통해 구성합니다. 보통 이 값은 `calibration`을 통해 얻을 수 있습니다.
- ③ $$ P $$ 행렬을 기존 이미지에 적용하여 이미지 변환을 합니다. 이 작업을 `perspective projection` 이라고 말하기도 합니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 설명한 ② 과정의 `extrinsic`, `intrinsic`에 관한 설명은 [카메라 캘리브레이션](https://gaussian37.github.io/vision-concept-calibration/) 내용을 참조해 주시기 바랍니다. `calibration`을 알고자 하는 목적은 실제 장착되어 있는 카메라와 도로 간의 위치 관계를 알기 위해서입니다.
- 위 그림과 같이 어떤 좌표계를 사용하는 지에 따라서 같은 물체를 서로 다른 위치로 표현할 수 있습니다. 위 그림에서는 `Road`, `Vehicle`, `Camera` 3개의 다른 좌표계가 있고 `Camera`와 `Vehicle` 간의 관계를 알아야 하기 때문에 두 좌표계를 변환할 수 있는 정보를 `calibration`을 통해 얻고 이 값을 통하여 `Vehicle` 좌표계의 3D Scene을 2D 이미지로 `perspective projection` 하게 됩니다. (참고로 `Road` 좌표계는 일반적으로 사용하진 않습니다.)

<br>

#### **Perspective projection**

<br>

- `Perspective projection`은 3D 공간을 2D 평면에 매핑하는 것입니다. 매핑 과정 중, 3D 공간 상의 평행한 2개의 선이 2D 평면에 표현될 때, 평행한 선이 어떤 특정점에서 만나게 되는 현상이 나타납니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 선을 보면 3D 공간 상에서는 평행한 선이지만 2D 공간에서는 선이 만나는 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 왼쪽 그림은 차량의 윗부분에 위치하는 카메라를 통해 도로를 본 관점 (view) 이고 오른쪽 그림은 BEV로 본 주변 환경 입니다. `viewpoint` 가 perspective view로 보는 지 또는 bird eye view로 보는 지에 따라서 관측되는 장면이 달라집니다.
- 두 view의 차이점 중 중요한 점은 왼쪽과 같이 `projective transformation`을 적용하게 되면 평행선이 더 이상 보존되지 않는 다는 점입니다. 반면 BEV의 경우 평행선이 그대로 유지되는 것을 볼 수 있습니다. 앞에서 설명한 `Perspective Distortion`이 제거된 것입니다.

<br>

#### **Camera Projective Geometry**

<br>

<br>
<center><img src="../assets/img/vision/concept/ipm/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>


<br>
<center><img src="../assets/img/vision/concept/ipm/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>


- 카메라 모델은 3D scene 에서 2D 이미지로 `perspective projection` 을 하며 2D 이미지를 만들 때, `calibration`을 통해 구한 `intrinsic`과 `extrinsic` 값에 따라 원하는 2D 이미지가 만들어 집니다.
- `extrinsic` $$ [R \vert t] $$ 는 `world`와 `camera` 간의 상태 위치 및 방향을 나타내어 `world coordinate`를 `camera coordinate`로 변환하는 역할을 합니다.
- `intrinsic` $$ K $$ 는 `camera coordinate` 기준으로 3D scene이 어떻게 2D 이미지 상의 `image coordinate`로 변환되는 지와 연관되어 있으며 $$ K $$ 는 카메라의 구성 성분인 `focal length`와 `camera center` 성분 값을 가집니다.  (가장 간단한 핀홀 카메라 케이스에 해당하며 `pixel skew`, `lens distortion` 등은 생략하였습니다.)

<br>
<center><img src="../assets/img/vision/concept/ipm/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- BEV의 원점은 image plane 2의 왼쪽 상단 모서리로 설정됩니다. 위 그림의 아이디어는 image plane 2에 있는 전체 scene $$ (X, Y, Z) $$ 을 카메라 image plane 1에 투영하는 것입니다.

<br>

- 앞에서 언급 하였듯이, `IPM` 을 적용하기 위해서는 **도로가 지면에 평평하다고 가정**했습니다. 따라서 도로에 있는 모든 점에 대해 `Z = 0` 입니다.
- 따라서 만약 도로가 평지가 아니라면 `Z = 0`이라는 전제 조건을 만족하지 못하기 때문에 `IPM`의 결과가 왜곡이 되어 보일 것입니다. 예를 들어 도로가 휘어지거나, 차선이 직선이 아닌 형태로 나타나는 경우가 발생하는데 도로가 지면에 평평하지 않기 때문에 발생할 수 있습니다.

<br>

## **IPM 적용 방법**

<br>

- 지금까지는 `IPM` 적용을 위한 배경 지식을 살펴보았고 실제로 `IPM`을 적용하기 위한 프로세스를 살펴보겠습니다.

<br>

- ① `calibraition` 정보 읽기 
- ② `BEV` 이미지와 `world` 좌표간의 관계 정하기
- ③ `BEV` 이미지와 `Image` 좌표간의 LUT (Look Up Table) 구하기
- ④ `backward` 방식으로 `IPM` 처리하여 `BEV` 이미지 생성하기

<br>

#### **① `calibraition` 정보 읽기 **

<br>

<br>

#### **② `BEV` 이미지와 `world` 좌표간의 관계 정하기**

<br>

<br>

#### **③ `BEV` 이미지와 `Image` 좌표간의 LUT (Look Up Table) 구하기**

<br>

<br>

#### **④ `backward` 방식으로 `IPM` 처리하여 `BEV` 이미지 생성하기**

<br>

<br>

## **Python 코드**

<br>