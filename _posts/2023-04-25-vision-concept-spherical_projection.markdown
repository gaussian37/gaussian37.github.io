---
layout: post
title: 이미지의 구면 투영법 (Spherical Projection) 및 원통 투영볍 (cylindrical projection)
date: 2023-04-25 00:00:00
img: vision/concept/spherical_projection/0.png
categories: [vision-concept] 
tags: [구면 좌표계, 원통 좌표계, 구면 투영법, 원통 투영법, spherical, cylindrical] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 사전 지식 : [직교 좌표계, 원통 좌표계 및 구면 좌표계](https://gaussian37.github.io/math-calculus-cylindrical_spherical_coordinate_system/)
- 사전 지식 : [카메라 모델 및 카메라 캘리브레이션의 이해와 Python 실습](https://gaussian37.github.io/vision-concept-calibration/)
- 사전 지식 : [카메라 모델과 렌즈 왜곡 (lens distortion)](https://gaussian37.github.io/vision-concept-lens_distortion/)

<br>

- 이번 글에서는 `구면 좌표계`를 이용하여 이미지를 `구면 투영법`에 적용하는 방법에 대하여 알아보도록 하겠습니다.
- 앞으로 다루는 내용은 [직교 좌표계, 원통 좌표계 및 구면 좌표계](https://gaussian37.github.io/math-calculus-cylindrical_spherical_coordinate_system/), [카메라 모델 및 카메라 캘리브레이션의 이해와 Python 실습](https://gaussian37.github.io/vision-concept-calibration/), [카메라 모델과 렌즈 왜곡 (lens distortion)](https://gaussian37.github.io/vision-concept-lens_distortion/) 에서 다룬 내용을 기반으로 설명할 예정입니다. 따라서 생략된 용어에 대한 설명은 각 링크를 통해 참조해 주시면 됩니다.

<br> 

## **목차**

<br>

- ### [구면 투영법 사용 이유](#구면-투영법-사용-이유-1)
- ### [카메라 기준의 구면 투영법](#카메라-기준의-구면-투영법-1)
- ### [회전을 고려한 카메라 기준의 구면 투영법](#회전을-고려한-카메라-기준의-구면-투영법-1)
- ### [회전을 고려한 World 기준의 구면 투영법](#회전을-고려한-world-기준의-구면-투영법-1)
- ### [회전을 고려한 World 기준의 구면 파노라마 투영법](#회전을-고려한-world-기준의-구면-파노라마-투영법-1)
- ### [원통 투영법 적용 방법](#원통-투영법-적용-방법-1)

<br>

## **구면 투영법 사용 이유**

<br>

- 카메라를 통하여 이미지 데이터를 취득하였을 때, 일반적으로 사용할 수 있는 2가지 방법은 `원본 이미지`를 사용하는 것과 `원근 투영법(Perspective Projection)`을 사용하는 것입니다. 이번 글에서 소개하고자 하는 방법은 `구면 투영법(Spherical Projection)`입니다. 각각의 투영법에 대한 정의와 장단점 및 특성등을 정의해 보면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 구면 투영 이미지는 오른쪽 그림과 같은 형태로 투영되며 세계 지도와 같이 구(Sphere) 형태에서 `azimuth`, `elevation`의 각도를 격자 단위로 나누고 각 격자를 이미지의 픽셀로 나타낸 것 표현 방식을 의미합니다.

<br>

### **각 투영법의 정의**

<br>

- ① `원본 이미지`
    - ⓐ `정의`: 카메라를 통하여 취득한 원본 이미지를 뜻합니다. 원본 이미지에는 카메라 렌즈에 의하여 발생된 왜곡이 영상에 그대로 반영되어 있습니다.
    - ⓑ `각 픽셀의 의미` : 각 픽셀은 렌즈를 통과하는 `ray`의 비선형 매핑에 해당합니다. 정확한 `ray`의 방향을 확인하기 위해서는 `카메라 캘리브레이션`이 필요합니다.
    - ⓒ `장점` : 원본 이미지이므로 영상의 `artifact`가 존재하지 않으며 `fisheye 카메라`와 같은 경우에는 넓은 화각 영역을 커버할 수 있는 장점이 있습니다.
    - ⓓ `단점` : 카메라 렌즈에 의해 발생한 왜곡으로 인하여 직선이 곡선으로 나타나는 현상이 발생합니다.
- ② `원근 투영법 (Perspective Projection)`
    - ⓐ `정의`: 핀홀 카메라 모델을 의미합니다. 핀홀 카메라 모델에서는 직선은 직선 그대로 모양을 가지는 특성이 있습니다.
    - ⓑ `각 픽셀의 의미` : 각 픽셀은 카메라 핀홀에 의해 투사된 `ray`에 의해 대응되며 원본 이미지와는 다르게 `ray`의 선형 매핑에 해당합니다.
    - ⓒ `장점` : 실제 직선이 이미지 상에서 그대로 직선 형태로 나타나며 픽셀 매핑 시 선형식을 통한 매핑이 가능하다는 단순함이 있습니다.
    - ⓓ `단점` : 원본 이미지를 `원근 투영법` 반영 시 이미지의 `FOV`가 제한적으로 표현됩니다. 특히 넓은 화각을 위한 카메라의 경우 `FOV`의 손실이 크게 발생할 수 있습니다.
- ③ `구면 투영법 (Spherical Projection)`
    - ⓐ `정의`: 수평 화각은 최대 360도, 수직 화각은 최대 180도의 구(sphere)에 매핑이 되는 투영법입니다.
    - ⓑ `각 픽셀의 의미` : 각 픽셀은 구에서 각 위치를 나타내는 방법과 동일합니다. 따라서 각 픽셀은 `azimuth angle`과 `elevation angle`을 의미합니다.
    - ⓒ `장점` : 실제 3D 공간이 구 형태로 되어 있으므로 실제 3D 환경을 표현하기에 용이 합니다. 따라서 VR 등에서도 구면 투영법을 통한 영상 투영을 사용하기도 합니다.
    - ⓓ `단점` : 구의 양쪽 극단에서 왜곡이 발생할 수 있습니다.
- ④ `원통 투영법 (Cylindrical Projection)`
    - ⓐ `정의`: 수직 원통에 이미지가 투영된 다음에 원통의 옆면이 펴진 형태의 투영법입니다.
    - ⓑ `각 픽셀의 의미` : 각 픽셀은 `azimuth angle`과 원통의 높이에 대응됩니다. 원본 이미지나 원근 투영법에서는 각 픽셀이 `ray`에 대응된 반면 원통 투영법에는 각 픽셀이 실제 원통을 구성하는 `azimuth angle`과 높이에 대응된다는 차이가 있습니다.
    - ⓒ `장점` : 최대 360도 까지의 수평 화각을 커버할 수 있도록 설계할 수 있습니다. 원통 기둥을 생각해 보면 이 점을 이해할 수 있을 것입니다. 원통을 모델링하여 표현하기 때문에 수직 방향으로는 왜곡이 보정이 되는 장점도 존재합니다. 따라서 수직 방향의 직선은 직선 형태로 나타내어 집니다.
    - ⓓ `단점` : 수직 화각을 표현하는 데 제한이 생기고 원통의 수직 방향으로 양쪽 끝지점에서 왜곡이 생기거나 불균일하게 샘플링 됩니다.

<br>

- 이번 글에서는 `구면 투영법`을 적용하는 방법에 대하여 다룰 것이고 위에 설명한 `원통 투영법`은 구면 투영법 코드에서 어떤 부분만 변경하면 되는 지 마지막에 간략하게 언급할 예정입니다.
- `구면 투영법`은 ① `카메라 좌표계` 기준에서 투영하는 방법과 ② `World 좌표계` 기준에서 투영하는 방법이 존재합니다.
- `카메라 좌표계` 기준의 `구면 투영법`은 기본적으로 카메라가 바라보는 방향과 3차원 좌표계의 좌표축이 동일하다는 관점에서 `구면 투영법`을 진행합니다. 반면 `World 좌표계` 기준의 `구면 투영법`은 `World 좌표계`의 좌표축을 기준으로 `구면 투영법`을 진행하기 때문에 카메라와 `World 좌표계` 간의 `Extrinsic` 관계를 고려해야 합니다. 이 방법을 이용하면 한 개의 `World 좌표계` 기준으로 여러개의 카메라를 `구면 투영`할 수 있고 각 카메라의 기준이 동일한 `World 좌표계`이기 때문에 파노라마 뷰를 생성할 수 있습니다. 따라서 `카메라 좌표계` 기준의 `구면 투영법`은 카메라의 `Intrinsic`만 사용하는 반면 `World 좌표계` 기준의 `구면 투영법`은 카메라의 `Intrinsic`, `Extrinsic`을 사용합니다.

<br>

- 먼저 아래 그림들을 통하여 `카메라 좌표계` 기준과 `World 좌표계` 기준의 차이를 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 `카메라 좌표계` 기준의 그림을 나타냅니다. 따라서 카메라 좌표계에서 사용하는 `RDF(Right-Down-Forward)`축과 카메라가 바라보는 방향이 일치하는 것을 알 수 있습니다.

<br>

- 반면 `World 좌표계` 기준의 구면 투영법을 적용할 때에는 3차원 좌표축이 카메라 좌표축과는 별개로 존재합니다. 먼저 현재 사용되는 카메라가 사전에 계산된 `Rotation(Extrinsic)`을 통하여 좌표축과 어떤 관계에 있는 지 계산한 뒤 구면 투영을 한다는 점에서 차이가 있습니다. 따라서 `World 좌표계` 기준의 구면 투영에서는 카메라의 `Extrinsic` 중 `Rotation`과 카메라 `Intrinsic`을 필요로 합니다. 

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 그림에는 3개의 카메라가 존재합니다. 각 카메라는 서로 다른 방향을 바라보고 있습니다. 반면 실선 화살표로 표시된 좌표축은 동일한 방향을 가리킵니다. 이 실선 화살표는 `World 기준`의 좌표축을 의미합니다. 반면 점선 화살표는 `각 카메라 좌표계`를 나타냅니다.
- `World 기준`의 구면 투영법을 사용하는 이유는 다수의 카메라 영상을 **동일한 좌표축 기준으로 영상을 생성**할 수 있다는 장점이 있기 때문입니다. 이와 같은 사용법은 **멀티 카메라 환경에서 이미지를 생성할 때 도움이 됩니다.** 각 카메라의 영상을 구면 투영할 때, 하나의 좌표축 기준으로 생성할 수 있으므로 실제 카메라가 장착된 각도를 고려하여 영상을 투영할 수 있고 더 나아가 360도 환경의 파노라마 이미지를 고려할 수 있기 때문입니다.

<br>

- 위 내용을 고려하여 본 글에서는 다음 순서로 구면 투영하는 방법을 살펴볼 예정입니다.
- ① `카메라 기준의 구면 투영법`: 이미지를 단순히 구면 좌표계로 투영하는 방법을 의미합니다. 따라서 카메라 중심축과 구면좌표계의 중심축의 방향이 일치합니다.
- ② `회전을 고려한 카메라 기준의 구면 투영법`: 구면 좌표계 중심축의 `roll`, `pitch`, `yaw` 방향 회전을 고려하여 이미지를 구면좌표계로 투영하는 방법을 의미합니다. 이 방법을 이용하면 임의의 방향으로 카메라가 회전하였을 때 영상을 만들어낼 수 있습니다.
- ③ `회전을 고려한 World 기준의 구면 투영법`: 구면 좌표계의 `roll`, `pitch`, `yaw`의 기준이 카메라 중심축이 아닌 외부의 `World 좌표`가 기준이 된다는 점에서 ②와 차이점이 있습니다.
- ④ `회전을 고려한 World 기준의 구면 파노라마 투영법`: ③의 개념을 확장하여 복수개의 카메라를 하나의 구면 좌표 공간으로 투영하는 방법을 다룹니다.

<br>

## **카메라 기준의 구면 투영법**

<br>

- 사전 지식 : [직교 좌표계, 원통 좌표계 및 구면 좌표계](https://gaussian37.github.io/math-calculus-cylindrical_spherical_coordinate_system/)

<br>

- 카메라 기준의 구면 투영법은 이미지를 단순히 구면 좌표계로 투영하는 방법을 의미합니다. 따라서 `카메라 중심축`과 `구면좌표계의 중심축` 방향이 일치합니다. 아래 구면 좌표계를 참조하시면 됩니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 구면 좌표계는 `RDF(Right-Down-Forward)` 좌표축으로 정의되었으며 카메라 좌표계와 축의 방향을 일치시키기 위함입니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림의 왼쪽은  카메라 중심축 기준으로 표현한 이미지이고 오른쪽은 구면 좌표계 존재하는 구면을 표현한 것입니다.
- 구면 투영을 위하여 필요한 정보는 **왼쪽의 오른쪽 이미지의 $$ (u, v) $$ 좌표와 오른쪽 구면에 존재하는 $$ (\phi, \theta) $$ 좌표를 대응 시키는 방법**입니다.
- 구면 투영을 거치면 원본 이미지의 $$ (u, v) $$ 좌표가 구면 투영 이미지의 $$ (\phi, \theta) $$ 좌표에 대응되기 때문에 아래 그림과 같은 관계를 가집니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, 구면 투영법의 핵심은 원본 이미지의 임의의 픽셀 좌표 $$ (u_{n}, v_{m}) $$ 과 구면 투영 이미지의 $$ (\phi_{n}, \theta_{m}) $$ 의 대응 방법입니다.
- 위 예시에서는 `카메라 중심축`과 `구면좌표계의 중심축` 방향이 일치하기 때문에 구면 투영 이미지의 중점에서 $$ \phi = 0, \theta = 0 $$ 임을 알 수 있습니다. 구면 투영 이미지의 오른쪽 방향으로 $$ \phi $$ 가 증가하고 아래쪽 방향으로 $$ \theta $$ 가 증가합니다.
- 오른쪽 그림에서 `hfov(horizontal fov)`는 구면 투영 이미지의 수평 화각입니다. 원점을 중심으로 좌우 대칭으로 화각을 설정할 때, 최대 몇 화각까지 볼 지 결정합니다. 예를들어 `hfov`가 120도이면 구면 좌표계 중심 기중 좌/우 각각 60도씩 수평 화각을 가집니다. 이와 같은 논리로 `vfov(vertical fov)` 또한 존재합니다.
- 가로축인 `hfov`를 구성하는 픽셀이 `W`개이면 가로축으로 1픽셀 증가 (우측으로 한 칸)할 때 마다 `hfov / W` 만큼 화각이 증가합니다. 같은 논리로 세로축인 `vfov`를 구성하는 픽셀이 `H`개이면 세로축으로 1픽셀 증가(아래쪽으로 한 칸)할 때 마다 `vfov / H` 만큼 화각이 증가합니다.

<br>

- 지금부터 살펴볼 내용은 **구면 투영 이미지에서 표현해야 할 모든 $$ (\phi_{n}, \theta_{m}) $$ 위치에 대한 색상 정보를 원본 이미지의 어떤 픽셀 좌표 $$ (u_{n}, v_{m}) $$ 에서 가져와야 할 지 찾는 과정**입니다. 이 과정을 통해 `LUT(Look Up Table)`를 만들고 `LUT`를 통해 원본 이미지를 구면 투영 이미지로 쉽게 변환하는 과정을 코드로 살펴보려고 합니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 따라서 `LUT`는 위 그림과 같이 모든 $$ (\phi_{n}, \theta_{m}) $$ 픽셀에 대하여 대응되는 원본 이미지의 좌표  $$ (u_{n}, v_{m}) $$ 의 값을 저장해야 합니다.

<br>

- ... 작성중 ...

<br>

- 데이터셋 링크: https://drive.google.com/drive/folders/15cnXNjEaztZl0CBT25oCaJ9-8qyfRYAw?usp=drive_link
- 데이터 관련 설명: [링크](https://gaussian37.github.io/vision-concept-ipm/#custom-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-ipm-%EC%A0%81%EC%9A%A9-%EC%98%88%EC%8B%9C-1)

<br>

## **회전을 고려한 카메라 기준의 구면 투영법**

<br>

<br>

## **회전을 고려한 World 기준의 구면 투영법**

<br>

<br>

## **회전을 고려한 World 기준의 구면 파노라마 투영법**

<br>

<br>


```python
def get_camera_cylindrical_spherical_lut(
    K, D, conversion_mode, target_width, target_height, hfov_deg, vfov_deg, roll_degree, pitch_degree, yaw_degree):
    '''
    - K : (3, 3) intrinsic matrix
    - D : (5, ) distortion coefficient
    - conversion_mode: "cylindrical", "spherical"
    - target_width, target_height: output image size
    - hfov_deg: 0 ~ 360
    - vfov_deg: 0 ~ 180
    - roll_degree: 0 ~ 360
    - pitch_degree: 0 ~ 360
    - yaw_degree: 0 ~ 360
    '''

    fx = K[0][0]
    skew = K[0][1]
    cx = K[0][2]
    
    fy = K[1][1]        
    cy = K[1][2]
    
    k0, k1, k2, k3, k4 = D[0], D[1], D[2], D[3], D[4]

    # 원통/구면 투영 시 생성할 azimuth/elevetion 각도 범위
    # 원통/구면 투영 시, azimuth 사용
    # 구면 투영 시, elevation 사용
    hfov=np.deg2rad(hfov_deg)
    vfov=np.deg2rad(vfov_deg)
    
    x_angle = pitch_degree
    y_angle = yaw_degree
    z_angle = roll_degree
    
    # X 축 (Pitch) 회전 행렬 (좌표축 회전) 
    Rx_PASSIVE = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(x_angle)), -np.sin(np.radians(x_angle))],
        [0, np.sin(np.radians(x_angle)), np.cos(np.radians(x_angle))]])
    
    # Y 축 (Yaw) 회전 행렬 (좌표축 회전)
    Ry_PASSIVE = np.array([
        [np.cos(np.radians(y_angle)), 0, np.sin(np.radians(y_angle))],
        [0, 1, 0],
        [-np.sin(np.radians(y_angle)), 0, np.cos(np.radians(y_angle))]])
    
    # Z 축 (Roll) 회전 행렬 (좌표축 회전)
    Rz_PASSIVE = np.array([
        [np.cos(np.radians(z_angle)), -np.sin(np.radians(z_angle)), 0],
        [np.sin(np.radians(z_angle)), np.cos(np.radians(z_angle)), 0],
        [0, 0, 1]])
    
    # X, Y, Z 축 전체 회전을 반영한 회전 행렬 (좌표축 회전)
    # SRC: 어떤 회전이 반영되지 않은 카메라 좌표축
    # TARGET: Roll/Pitch/Yaw 회전이 반영된 카메라 좌표축    
    # new_R_RDF_SRC_RDF_TARGET_PASSIVE: SRC → TARGET의 좌표축 회전
    new_R_RDF_SRC_RDF_TARGET_PASSIVE = Ry_PASSIVE @ Rx_PASSIVE @ Rz_PASSIVE
    # new_R_RDF_SRC_RDF_TARGET_ACTIVE: SRC → TARGET의 좌표 회전
    new_R_RDF_SRC_RDF_TARGET_ACTIVE = new_R_RDF_SRC_RDF_TARGET_PASSIVE.T
    ##############################################################################################################
    
    # 원통/구면 투영 시, normalized → image 로 적용하기 위한 intrinsic 행렬렬
    new_K = np.array([
        [target_width/hfov,       0,                     target_width/2],
        [0,                       target_height/vfov,    target_height/2],
        [0,                       0,                     1]], dtype=np.float32)
    new_K_inv = np.linalg.inv(new_K)
    
    # Create pixel grid and compute a ray for every pixel
    # xv : (target_height, target_width), yv : (target_height, target_width)
    xv, yv = np.meshgrid(range(target_width), range(target_height), indexing='xy')
    
    # p.shape : (3, #target_height, #target_width)
    p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates    
    # p.shape : (#target_height, #target_width, 3, 1)    
    p = p.transpose(1, 2, 0)[:, :, :, np.newaxis] # [u, v, 1]
    '''
    p.shape : (H, W, 3, 1)
    p[:, : 0, :] : 0, 1, 2, ..., W-1
    p[:, : 1, :] : 0, 1, 2, ..., H-1    
    p[:, : 2, :] : 1, 1, 1, ..., 1
    '''
    # p_norm.shape : (#target_height, #target_width, 3, 1)
    p_norm = new_K_inv @ p  # r is in normalized coordinate
    
    '''
    p_norm[:, :, 0, :]. phi (azimuthal angle. horizontal) : -hfov/2 ~ hov/2
    p_norm[:, :, 1, :]. theta (elevation angla. vertical) : -vfov/2 ~ vfov/2
    p_norm[:, :, 2, :]. 1.    
    '''

    # x, y, z : cartesian coordinate in camera coordinate system (RDF, Right-Down-Front)
    # hemisphere
    if conversion_mode == "cylindrical":
        # azimuthal angle
        phi = p_norm[:, :, 0, :]
        
        x = np.sin(phi)
        y = p_norm[:, :, 1, :]
        z = np.cos(phi)
        
    elif conversion_mode == "spherical":
        # azimuthal angle
        phi = p_norm[:, :, 0, :]
        # elevation angle
        theta = p_norm[:, :, 1, :] 
        
        x =np.cos(theta)*np.sin(phi) # -1 ~ 1
        y =np.sin(theta) # -1 ~ 1
        z =np.cos(theta)*np.cos(phi) # 0 ~ 1
    else:
        print("wrong conversion_mode: ", conversion_mode)
        exit()
    
    RDF_cartesian = np.zeros(p_norm.shape).astype(np.float32)
    RDF_cartesian[:,:,0,:]=x
    RDF_cartesian[:,:,1,:]=y
    RDF_cartesian[:,:,2,:]=z    
    
    # RDF_rotated_cartesian = Rz @ Ry @ Rx @ RDF_cartesian
    # SRC → TARGET의 좌표 회전울 통하여 생성된 좌표들을 회전함
    RDF_rotated_cartesian = new_R_RDF_SRC_RDF_TARGET_ACTIVE @ RDF_cartesian
            
    # compute incidence angle
    x_un = RDF_rotated_cartesian[:, :, [0], :]
    y_un = RDF_rotated_cartesian[:, :, [1], :]
    z_un = RDF_rotated_cartesian[:, :, [2], :]
    # theta = np.arccos(RDF_rotated_cartesian[:, :, [2], :] / np.linalg.norm(RDF_rotated_cartesian, axis=2, keepdims=True))
    theta = np.arccos(z_un / np.sqrt(x_un**2 + y_un**2 + z_un**2))
    
    mask = theta > np.pi/2
    mask = mask.squeeze(-1).squeeze(-1)
    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    r_dn = k0*theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
    
    r_un = np.sqrt(x_un**2 + y_un**2)
    
    x_dn = r_dn * x_un / (r_un + 1e-6) # horizontal
    y_dn = r_dn * y_un / (r_un + 1e-6) # vertical    
    
    map_x_origin2new = fx*x_dn[:, :, 0, 0] + cx + skew*y_dn[:, :, 0, 0]
    map_y_origin2new = fy*y_dn[:, :, 0, 0] + cy
    
    DEFAULT_OUT_VALUE = -100
    map_x_origin2new[mask] = DEFAULT_OUT_VALUE
    map_y_origin2new[mask] = DEFAULT_OUT_VALUE
    
    map_x_origin2new = map_x_origin2new.astype(np.float32)
    map_y_origin2new = map_y_origin2new.astype(np.float32)
    return map_x_origin2new, map_y_origin2new
```

<br>

```python
image = cv2.cvtColor(cv2.imread("ELP-USB16MP01-BL180-2048x1536_EXTRINSIC.png", -1), cv2.COLOR_BGR2RGB)
calib = json.load(open("ELP-USB16MP01-BL180-2048x1536_calibration.json", "r"))

origin_height, origin_width, _ = image.shape
target_height, target_width  = origin_height, origin_width

intrinsic = np.array(calib['ELP-USB16MP01-BL180-2048x1536']['Intrinsic']['K']).reshape(3, 3)
intrinsic[0, :] *= (target_width/origin_width)
intrinsic[1, :] *= (target_height/origin_height)
distortion = np.array(calib['ELP-USB16MP01-BL180-2048x1536']['Intrinsic']['D'])

map_x, map_y = get_camera_cylindrical_spherical_lut(intrinsic, distortion, "cylindrical", target_width, target_height, hfov_deg=180, vfov_deg=180, roll_degree=0, pitch_degree=0, yaw_degree=0)
new_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
plt.imshow(new_image)
```

<br>


<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/YkfjGxAVY2w" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/LoP7H3K_wt4" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/If-p9DcBjAM" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

## **카메라 기준의 구면 투영법**

<br>



<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/oOhKlkkEL4c" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/2b9ennd6F_4" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/sMnwPiBMOAs" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>


<br>

## **원통 투영법 적용 방법**

<br>


<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>