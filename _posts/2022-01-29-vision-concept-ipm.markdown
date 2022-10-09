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
- 참조 : https://csyhhu.github.io/2015/07/09/IPM/

<br>

- 이번 글을 읽기 전에 아래 2가지 개념에 대한 이해도가 있으면 이해하기가 더욱 쉽습니다.
    - [Homogeneous Coordinate (동차좌표계)](https://gaussian37.github.io/vision-concept-homogeneous_coordinate/)
    - [카메라 캘리브레이션](https://gaussian37.github.io/vision-concept-calibration/)

<br>
<center><img src="../assets/img/vision/concept/ipm/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이번 글에서 다룰 내용은 위 그림과 같이 흔히 보는 왼쪽 그림과 같은 perspective view 형태의 이미지를 Bird Eye View 형태로 변환하는 방법에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### IPM의 사용 배경
- ### IPM 적용 방법
- ### Python 코드

<br>

## **IPM의 사용 배경**

<br>

- 

<br>


## **IPM 적용 방법**

<br>

- 지금부터 살펴볼 방법은 front view 이미지를 `BEV` 이미지로 변경하는 방법입니다. `IPM`은 이러한 처리를 전방 카메라의 `원근 효과(perspective effect)`를 제거하고 top-view 기반의 2D 도메인 상에 다시 매핑하는 방법을 사용합니다. 즉, BEV 이미지는 원근 효과를 보정하여 거리와 평행선을 유지하는 성질을 가지고 있습니다.

<br>

- 다음 내용은 `homography` 기반의 `IPM`의 절차를 나타냅니다.
- ① perspective view 이미지의 도로를 평평한 2차원 평면으로 가정합니다. 따라서 ( $$ X, Y, Z = 0 $$ ) 과 같이 모델링 하여 높이 정보가 없는 2차원 평면으로 간주합니다. 
- ② `projection matrix` $$ P $$ 를 `extrinsic`, `intrinsic` 파라미터를 통해 구성합니다. 보통 이 값은 `calibration`을 통해 얻을 수 있습니다.
- ③ $$ P $$ 행렬을 기존 이미지에 적용하여 이미지 변환을 합니다. 이 작업을 `perspective projection` 이라고 말하기도 합니다.

<br>
<center><img src="../assets/img/vision/concept/ipm/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 설명한 ② 과정의 `extrinsic`, `intrinsic`에 관한 설명은 [카메라 캘리브레이션](https://gaussian37.github.io/vision-concept-calibration/) 내용을 참조해 주시기 바랍니다. `calibration`을 알고자 하는 목적은 실제 장착되어 있는 카메라와 도로 간의 위치 관계를 알기 위해서입니다.
- 위 그림과 같이 어떤 좌표계를 사용하는 지에 따라서 같은 물체를 서로 다른 위치로 표현할 수 있습니다. 위 그림에서는 `Road`, `Vehicle`, `Camera` 3개의 다른 좌표계가 있고 카메라와 도로 간의 관계를 알아야 하기 때문에 두 좌표계를 변환할 수 있는 정보를 `calibration`을 통해 얻고 실제 `perspective projection` 할 때 사용합니다.

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
- 두 view의 차이점 중 중요한 점은 왼쪽과 같이 `projective transformation`을 적용하게 되면 평행선이 더 이상 보존되지 않는 다는 점입니다. 반면 BEV의 경우 평행선이 그대로 유지되는 것을 볼 수 있습니다.

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
- `intrinsic` $$ K $$ 는 `camera coordinate` 기준으로 3D scene 이 어떻게 2D 이미지 상의 `image coordinate`로 변환되는 지와 연관되어 있으며 $$ K $$ 는 카메라의 구성 성분인 `focal length`와 `camera center` 성분 값을 가집니다.  (가장 간단한 핀홀 카메라 케이스에 해당하며 `pixel skew`, `lens distortion` 등은 생략하였습니다.)

<br>
<center><img src="../assets/img/vision/concept/ipm/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- BEV의 원점은 image plane 2의 왼쪽 상단 모서리로 설정됩니다. 위 그림의 아이디어는 image plane 2에 있는 전체 scene $$ (X, Y, Z) $$ 을 카메라 image plane 1에 투영하는 것입니다.

<br>

- 앞에서 언급 하였듯이, `IPM` 을 적용하기 위해서는 **도로가 지면에 평평하다고 가정**했습니다. 따라서 도로에 있는 모든 점에 대해 `Z = 0` 입니다. 이러한 문제 접근 방법은 `planar homography`를 사용하는 문제로 변환합니다. (이러한 방법은 OpenCV를 통해 간단하게 이미지 워핑을 수행할 수 있습니다.)

<br>

- 아래 살펴볼 예정인 코드 중 `ipm_from_parameters` 라는 함수가 있습니다. 이 함수 부분이 전체 코드의 핵심이 되는데 처리 순서를 살펴보겠습니다.
- ① `Defining the plane on the ground` : 먼저 `BEV` 평면에서 보려는 도로 영역을 잘라냅니다. 이 영역에 대해 픽셀 해상도, 픽셀당 절대 거리(스케일) 및 포즈(위치 및 방향)를 정의합니다.
- ② ` Deriving and Applying Perspective projection` : 픽셀 좌표에 카메라 투영 모델을 사용하여 영역의 모든 3D 점(X, Y, Z=0)에 대한 `perspective projection`을 적용합니다.
- ③ `Resampling pixels` : front view 이미지에서 해당 픽셀을 다시 샘플링하고 image plane 2에 다시 매핑합니다. 매핑하였을 때, 발생한 hole과 aliasing 을 방지하기 위하 일부 형태의 `interpolation`이 필요합니다. 살펴볼 코드에서는 `bilinear interpolation`을 사용합니다.

<br>

- 위 3가지 내용을 스텝 별로 상세하게 알아보도록 하겠습니다.

<br>

#### **Defining the plane on the ground**

<br>
<center><img src="../assets/img/vision/concept/ipm/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 차량이 이동하더라도 인식 가능한 영역이 일정하게 유지되기를 원하기 때문에 2D 평면은 차량의 상태에 맞춰서 정의되어야 합니다.
- 먼저 평면의 원점은 왼쪽 상단 모서리로 정의되며 인식 가능한 영역은 위 그림에서 밝게 보이는 부분입니다. (위 그림에서 어두어진 부분은 인식 가능 영역이 아님) 이것은 카메라의 `FOV (Field Of View)`에 따라 영역이 달라지게 됩니다.
- 카메라의 `FOV` 에 따라 보이지 않는 영역이 발생하기 때문에 `IPM` 이후 이미지에 관찰할 수 없는 픽셀(검은색)이 생기게 됩니다. 이런 점들을 고려하여 `plane`의 속성값들을 정해야 합니다. 예를 들면 다음과 같습니다.
    - 픽셀 사이즈 : 500 x 500
    - 해상도 : 픽셀 당 0.1 m
    - 카메라는 평면의 y축 중간점에 위치하고 정렬됨
    - ...

<br>

#### **Deriving and Applying Perspective projection**

<br>


```python
{
    "baseline": 0.21409619719999115,
    "roll": 0.0,
    "pitch": 0.03842560000000292,
    "yaw": -0.009726800000000934,
    "x": 1.7,
    "y": 0.026239999999999368,
    "z": 1.212400000000026,
    "fx": 2263.54773399985,
    "fy": 2250.3728170599807,
    "u0": 1079.0175620000632,
    "v0": 515.0066006000195
}
```

<br>

```python
def load_camera_params(file):
    """
    Get the intrinsic and extrinsic parameters
    Returns:
        Camera extrinsic and intrinsic matrices
    """
    with open(file, 'rt') as handle:
        p = json.load(handle)

    fx, fy = p['fx'], p['fy']
    u0, v0 = p['u0'], p['v0']

    pitch, roll, yaw = p['pitch'], p['roll'], p['yaw']
    x, y, z = p['x'], p['y'], p['z']

    # Intrinsic
    K = np.array([[fx, 0, u0, 0],
                  [0, fy, v0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Extrinsic
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x, -y, -z))

    # Rotate to camera coordinates
    R = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])

    RT = R @ R_veh2cam @ T_veh2cam
    return RT, K
```



- 


<br>

## **Python 코드**

<br>

<br>




