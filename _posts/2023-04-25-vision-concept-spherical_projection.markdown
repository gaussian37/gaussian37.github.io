---
layout: post
title: 이미지의 구면 좌표계 투영법 (Spherical Projection)
date: 2023-04-25 00:00:00
img: vision/concept/spherical_projection/0.png
categories: [vision-concept] 
tags: [구면 좌표계, 구면 투영법, spherical] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 참조 : https://plaut.github.io/fisheye_tutorial/
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
- ### [카메라 기준 구면 투영법](#카메라-기준-구면-투영법-1)
- ### [회전을 고려한 카메라 기준 구면 투영법](#회전을-고려한-카메라-기준-구면-투영법-1)
- ### [회전을 고려한 World 기준 구면 투영법](#회전을-고려한-world-기준-구면-투영법-1)
- ### [회전을 고려한 World 기준 구면 투영법의 World-to-Image, Image-to-World](#회전을-고려한-world-기준-구면-투영법의-world-to-image-image-to-world-1)
- ### [회전을 고려한 World 기준 구면 투영 이미지의 Topview 이미지 생성](#회전을-고려한-world-기준-구면-투영-이미지의-topview-이미지-생성-1)
- ### [회전을 고려한 World 기준 구면 파노라마 투영법](#회전을-고려한-world-기준-구면-파노라마-투영법-1)

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

- 이번 글에서는 `구면 투영법`을 적용하는 방법을 단계적으로 살펴보고 그 성질에 대하여 살펴보도록 하겠습니다.
- `구면 투영법`은 ① `카메라 좌표계` 기준에서 투영하는 방법과 ② `World 좌표계` 기준에서 투영하는 방법이 존재합니다.
- `카메라 좌표계` 기준 `구면 투영법`은 기본적으로 카메라가 바라보는 방향과 3차원 좌표계의 좌표축이 동일하다는 관점에서 `구면 투영법`을 진행합니다. 반면 `World 좌표계` 기준 `구면 투영법`은 `World 좌표계`의 좌표축을 기준으로 `구면 투영법`을 진행하기 때문에 카메라와 `World 좌표계` 간의 `Extrinsic` 중 `Rotation`의 관계를 고려해야 합니다. 즉, 이 방법을 이용하면 한 개의 `World 좌표계` 기준으로 여러개의 카메라를 `구면 투영`할 수 있고 각 카메라의 기준이 동일한 `World 좌표계`이기 때문에 파노라마 뷰를 생성할 수 있습니다. 따라서 `카메라 좌표계` 기준 `구면 투영법`은 카메라의 `Intrinsic`만 사용하는 반면 `World 좌표계` 기준 `구면 투영법`은 카메라의 `Intrinsic`, 과 `Extrinsic`의 `Rotation`을 사용합니다.

<br>

- 먼저 아래 그림들을 통하여 `카메라 좌표계` 기준과 `World 좌표계` 기준의 차이를 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 `카메라 좌표계` 기준의 그림을 나타냅니다. 카메라 좌표계에서 사용하는 `RDF(Right-Down-Forward)`축과 카메라가 바라보는 방향이 일치하는 것을 알 수 있습니다. 일반적으로 카메라 좌표계에서 $$ X $$ 축은 `Right`, $$ Y $$ 축은 `Down` 그리고 $$ Z $$ 축은 `Forward`를 나타내기 때문입니다.

<br>

- 반면 `World 좌표계` 기준 구면 투영법을 적용할 때에는 3차원 좌표축이 카메라 좌표축과는 별개로 존재합니다. 먼저 현재 사용되는 카메라가 사전에 계산된 `Rotation(Extrinsic)`을 통하여 좌표축과 어떤 관계에 있는 지 계산한 뒤 구면 투영을 한다는 점에서 차이가 있습니다. 따라서 `World 좌표계` 기준 구면 투영에서는 카메라의 `Extrinsic` 중 `Rotation`과 카메라 `Intrinsic`을 필요로 합니다. 

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 그림에는 3개의 카메라가 존재합니다. 각 카메라는 서로 다른 방향을 바라보고 있습니다. 반면 실선 화살표로 표시된 좌표축은 동일한 방향을 가리킵니다. 이 실선 화살표는 `World 기준`의 좌표축을 의미합니다. 반면 점선 화살표는 `각 카메라 좌표계`를 나타냅니다.
- `World 기준`의 구면 투영법을 사용하는 이유는 다수의 카메라 영상을 **동일한 좌표축 기준으로 영상을 생성**할 수 있다는 장점이 있기 때문입니다. 이와 같은 사용법은 **멀티 카메라 환경에서 이미지를 사용하거나 생성할 때 도움이 됩니다.** 각 카메라의 영상을 구면 투영할 때, 하나의 좌표축 기준으로 생성할 수 있으므로 실제 카메라가 장착된 각도를 고려하여 영상을 투영할 수 있고 더 나아가 360도 환경의 파노라마 이미지를 고려할 수 있기 때문입니다.

<br>

- 위 내용을 고려하여 본 글에서는 다음 순서로 구면 투영하는 방법을 살펴볼 예정입니다.
- ① `카메라 기준 구면 투영법`: 이미지를 단순히 구면 좌표계로 투영하는 방법을 의미합니다. 이 과정에서 구면 좌표계로 영상을 투영하는 원리를 이해할 수 있 습니다. 이 단계에서 카메라 중심축과 구면 좌표계 중심축의 방향이 일치합니다.
- ② `회전을 고려한 카메라 기준 구면 투영법`: 카메라가 바라보는 방향이 `roll`, `pitch`, `yaw` 각 방향으로 회전이 발생한 상황을 고려합니다. 이와 같이 카메라 자세에 회전이 발생하였을 때 구면 좌표계로 투영하는 방법을 이해해 보려고 합니다. 이 방법을 이용하면 임의의 방향으로 카메라가 회전하였을 때 영상을 만들어낼 수 있습니다.
- ③ `회전을 고려한 World 기준 구면 투영법`: 구면 좌표계의 `roll`, `pitch`, `yaw`의 기준이 카메라 중심축이 아닌 외부의 `World 좌표`가 기준이 된다는 점에서 ②와 차이점이 있습니다.
- ④ 앞에서 다룬 내용을 이용하여 `회전을 고려한 World 기준 구면 투영법`에서 `Image-to-World`와 `World-to-Image`에 대하여 다루어 보고 멀티 카메라 이미지를 이용하여 `Topivew` 생성을 통해 `World 좌표계`와 `구면 좌표계` 간의 관계를 명확히 이해해보도록 하겠습니다.
- ⑤ `회전을 고려한 World 기준 구면 파노라마 투영법`: ③의 개념을 확장하여 복수개의 카메라를 하나의 구면 좌표 공간으로 투영하는 방법을 다루어 보겠습니다.

<br>

## **카메라 기준 구면 투영법**

<br>

- 본 글을 이해하기 위해서는 아래 사전 지식의 `직교 좌표계`와 `구면 좌표계`의 관계에 대한 명확한 이해가 필요합니다. 아래 글을 먼저 읽고 본문 내용을 읽어 보시는 것을 추천드립니다.
- 사전 지식 : [직교 좌표계, 원통 좌표계 및 구면 좌표계](https://gaussian37.github.io/math-calculus-cylindrical_spherical_coordinate_system/)

<br>

- 본 글에서 사용할 샘플 데이터의 링크 및 설명은 아래를 참조하시면 됩니다.
    - 데이터셋 링크: https://drive.google.com/drive/folders/15cnXNjEaztZl0CBT25oCaJ9-8qyfRYAw?usp=drive_link
    - 데이터 관련 설명: [링크](https://gaussian37.github.io/vision-concept-ipm/#custom-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-ipm-%EC%A0%81%EC%9A%A9-%EC%98%88%EC%8B%9C-1)

<br>

- 카메라 기준 구면 투영법은 이미지를 단순히 구면 좌표계로 투영하는 방법을 의미합니다. 따라서 `카메라 중심축`과 `구면 좌표계의 중심축` 방향이 일치합니다. 아래 구면 좌표계를 참조하시면 됩니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 구면 좌표계는 `RDF(Right-Down-Forward)` 좌표축으로 정의되었으며 카메라 좌표계와 축의 방향을 일치시키기 위함입니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림의 왼쪽은 카메라 중심축 기준으로 표현한 직교 좌표계 상의 이미지이고 오른쪽은 구면 좌표계 상에 존재하는 구면을 표현한 것입니다.
- 구면 투영을 위하여 필요한 정보는 **왼쪽 이미지의 $$ (u, v) $$ 좌표와 오른쪽 구면에 존재하는 $$ (\phi, \theta) $$ 좌표를 대응 시키는 방법**입니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 실제 의미는 구면을 다루는 것이지만 이미지 데이터나 구면 좌표로 투영된 이미지를 다룰 때에는 행렬로 나타내어야 하기 때문에 평면 이미지를 다루게 됩니다. 따라서 앞으로 생성할 이미지는 비록 직사각형의 이미지이지만 의미론적으로는 구면을 생각해 주시면 됩니다.
- 위 그림을 살펴보겠습니다. 왼쪽 그림의 이미지 각 픽셀은 구면 좌표계에서 `azimuth`와 `elevation`을 뜻하는 $$ (\phi_{i}, \theta_{j}) $$ 입니다. 따라서 의미적으로는 오른쪽 그림에서의 각 분할된 면과 대응될 수 있습니다. 위 그림에서 주황색으로 표시된 픽셀 및 작은 면적이 의미적으로 대응된다는 것을 이해하면 됩니다.
- 따라서 구면 투영을 거쳐 원본 이미지의 $$ (u, v) $$ 좌표가 구면 투영 이미지의 $$ (\phi, \theta) $$ 좌표에 대응되면 아래 그림과 같은 관계를 가집니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, **구면 투영법의 핵심**은 **원본 이미지의 임의의 픽셀 좌표 $$ (u_{n}, v_{m}) $$ 과 구면 투영 이미지의 $$ (\phi_{i}, \theta_{j}) $$ 의 1:1 매핑 방법**입니다.
- 위 예시에서는 `카메라 중심축`과 `구면 좌표계 중심축` 방향이 일치하기 때문에 구면 투영 이미지의 중점에서 $$ \phi = 0, \theta = 0 $$ 임을 알 수 있습니다. 구면 투영 이미지의 오른쪽 방향으로 $$ \phi $$ (`azimuth`)가 증가하고 아래쪽 방향으로 $$ \theta $$ (`elevation`)이 증가합니다.
- 오른쪽 그림에서 `hfov(horizontal fov)`는 구면 투영 이미지의 수평 화각입니다. 원점을 중심으로 좌우 대칭으로 화각을 설정할 때, 최대 몇 화각까지 볼 지 결정합니다. 예를들어 `hfov`가 120도이면 구면 좌표계 중심 기준 좌/우 각각 60도씩 수평 화각을 가집니다. 이와 같은 논리로 `vfov(vertical fov)` 또한 존재합니다. (좌/우 또는 상/하를 비대칭으로 설계할 수도 있지만 본 글에서는 대칭 화각으로 설계할 예정입니다.)
- 가로축인 `hfov`를 구성하는 픽셀이 `W`개이면 가로축으로 1픽셀 증가 (우측으로 한 칸)할 때 마다 `hfov / W` 만큼 화각이 증가합니다. 같은 논리로 세로축인 `vfov`를 구성하는 픽셀이 `H`개이면 세로축으로 1픽셀 증가(아래쪽으로 한 칸)할 때 마다 `vfov / H` 만큼 화각이 증가합니다.

<br>

- 지금부터 살펴볼 내용은 **구면 투영 이미지에서 표현해야 할 모든 $$ (\phi_{i}, \theta_{j}) $$ 위치에 대한 색상 정보를 원본 이미지의 어떤 픽셀 좌표 $$ (u_{n}, v_{m}) $$ 에서 가져와야 할 지 찾는 과정**입니다. 이 과정을 통해 `LUT(Look Up Table)`를 만들고 `LUT`를 통해 원본 이미지를 구면 투영 이미지로 쉽게 변환하는 과정을 코드로 살펴보려고 합니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 `LUT`는 위 그림과 같이 모든 $$ (\phi_{i}, \theta_{j}) $$ 픽셀에 대하여 대응되는 원본 이미지의 좌표  $$ (u_{n}, v_{m}) $$ 의 값을 저장해야 합니다.
- 즉, $$ (\phi_{i}, \theta_{j}) $$ 와 일대일 대응이 되는 $$ (u_{n}, v_{m}) $$ 을 찾아야 하므로 다음과 같은 순서로 접근을 해야 합니다. 전체적인 순서는 `backward mapping`으로 최종 생성하고자 하는 구면 좌표계의 정보인 $$ (\phi_{i}, \theta_{j}) $$ 에서 부터 원본 이미지의 정보인 $$ (u_{n}, v_{m}) $$ 로 접근하는 과정을 차례대로 나타냅니다.
    - ① `구면 투영 이미지`: 최종적으로 생성하고자 하는 구면으로 정의된 이미지 공간 입니다.
    - ② `normalized 구면 투영 이미지`: 구면 투영 이미지의 `normalized` 공간을 의미합니다. 원본 이미지에 접근 하기 위한 중간 과정입니다.
    - ③ `normalized 이미지`: 원본 이미지의 `normalized` 공간을 의미합니다.
    - ④ `이미지`: 원본 이미지를 의미하며 구면 투영 이미지에서 사용할 RGB 값을 가져올 때 사용 됩니다.
- ① `구면 투영 이미지`에서 최종 생성해야 하는 이미지의 공간을 정의해 놓고 이 이미지의 $$ (\phi_{i}, \theta_{j}) $$ 와 대응되는 ④ `이미지`의 $$ (u_{n}, v_{m}) $$ 를 매핑 시키는 작업을 해야 합니다. ②, ③ 은 중간 과정으로 거쳐야 하는 공간입니다. 
- `normalize` 공간에 대한 정의는 아래 글에서 확인해 보시기 바랍니다.
    - 사전 지식 : [카메라 모델 및 카메라 캘리브레이션의 이해와 Python 실습](https://gaussian37.github.io/vision-concept-calibration/)
    - 사전 지식 : [카메라 모델과 렌즈 왜곡 (lens distortion)](https://gaussian37.github.io/vision-concept-lens_distortion/)

<br>

- `LUT`를 만들기 위한 시작은 가상의 이미지 공간 생성입니다. 위 그림과 같은 `LUT` 사이즈의 이미지를 생성한다고 가정해 보겠습니다.
- 생성할 `구면 투영 이미지`의 가로 사이즈를 `target_width`, 세로 사이즈를 `target_height`로 정의하고 생성할 이미지의 라디안 단위의 가로 화각을 `hfov`, 세로 화각을 `vfov`라고 정의한다면 생성할 이미지의 각 픽셀 한 칸의 의미는 다음과 같습니다.

- $$ \Delta \phi_{\text{pixel}} = \frac{\text{target width}}{\text{hfov}} $$

- $$ \Delta \theta_{\text{pixel}} = \frac{\text{target height}}{\text{vfov}} $$

<br>

- ② `normalized 구면 투영 이미지` →  ① `구면 투영 이미지` 로 변환하기 위한 `intrinsic` 행렬은 다음과 같이 정의할 수 있습니다.

<br>

- $$ \text{new K} = \begin{bmatrix} f_{x} & \text{skew} & c_{x} \\ 0 & f_{y} & c_{y} \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} \frac{\text{target width}}{\text{hfov}} & 0 & \frac{\text{target width}}{2} \\ 0 & \frac{\text{target height}}{\text{vfov}} & \frac{\text{target height}}{2} \\ 0 & 0 & 1 \end{bmatrix} $$

<br>

- 각 픽셀 당 $$ \Delta \phi_{\text{pixel}}, \quad \Delta \theta_{\text{pixel}} $$ 의 비율 만큼 증가하므로 `intrinsic`에서 사용되는 $$ f_{x} = \Delta \phi_{\text{pixel}}, \quad f_{y} = \Delta \theta_{\text{pixel}} $$ 에 대응됩니다. 그리고 `principal point`은 $$ c_{x} = \frac{\text{target width}}{2}, \quad c_{y} = \frac{\text{target height}}{2} $$ 에 대응됩니다. 즉, $$ c_{x} = \frac{\text{target width}}{2}, \quad c_{y} = \frac{\text{target height}}{2} $$ 를 통해 생성하고자 하는 **가상 이미지의 중심**에 해당하는 $$ \phi, \theta $$ 가 정해지면 좌/우, 상/하 화각이 대칭이 되도록 배치합니다.
- 새롭게 생성되는 `구면 투영 이미지`의 `intrinsic`인 $$ \text{new K} $$ 는 이와 같은 방법으로 생성됩니다.

<br>

- 가상의 `구면 투영 이미지`의 픽셀 좌표는 다음과 같이 생성합니다.

<br>

```python
# xv : (target_height, target_width), yv : (target_height, target_width)
xv, yv = np.meshgrid(range(target_width), range(target_height), indexing='xy')

# p.shape : (3, #target_height, #target_width)
p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates
# p.shape : (#target_height, #target_width, 3, 1)
p = p.transpose(1, 2, 0)[:, :, :, np.newaxis] # [u, v, 1]
'''
- p[:, : 0, :] : 0, 1, 2, ..., W-1
- p[:, : 1, :] : 0, 1, 2, ..., H-1    
- p[:, : 2, :] : 1, 1, 1, ..., 1
'''
```

<br>

- 위 과정을 통하여 ① `구면 투영 이미지`의 모든 좌표들을 생성합니다.

<br>

- 다음 과정을 통하여 ① `구면 투영 이미지` → ② `normalized 구면 투영 이미지`로 변경합니다.

<br>

```python
# hfov_deg: 0 ~ 360
# vfov_deg: 0 ~ 180
# 구면 투영 시 생성할 azimuth/elevation 각도 범위
# hfov: azimuth
# vfov: elevation
hfov=np.deg2rad(hfov_deg)
vfov=np.deg2rad(vfov_deg)
    
# 구면 투영 시, normalized → image 로 적용하기 위한 intrinsic 행렬
new_K = np.array([
    [target_width/hfov,       0,                     target_width/2],
    [0,                       target_height/vfov,    target_height/2],
    [0,                       0,                     1]
], dtype=np.float32)

new_K_inv = np.linalg.inv(new_K)

# p_norm.shape : (#target_height, #target_width, 3, 1)
p_norm = new_K_inv @ p  # r is in normalized coordinate

'''
p_norm[:, :, 0, :]. phi (azimuthal angle. horizontal) : -hfov/2 ~ hov/2
p_norm[:, :, 1, :]. theta (elevation angla. vertical) : -vfov/2 ~ vfov/2
p_norm[:, :, 2, :]. 1.    
'''
```

<br>

- 다음으로 ② `normalized 구면 투영 이미지` → ③ `normalized 이미지` 으로 변경합니다. $$ \phi, \theta $$ 를 이용하여 $$ x, y, z $$ 의 직교 좌표계로 변경하는 과정에 해당합니다.
- 다음 과정은 아래 링크의 내용을 사전에 이해해야 합니다.
    - 사전 지식 : [직교 좌표계, 원통 좌표계 및 구면 좌표계](https://gaussian37.github.io/math-calculus-cylindrical_spherical_coordinate_system/)

<br>

```python
# azimuthal angle
phi = p_norm[:, :, 0]
# elevation angle
theta = p_norm[:, :, 1]   

RDF_cartesian = np.zeros(p_norm.shape).astype(np.float32)

# x, y, z : cartesian coordinate in camera coordinate system (RDF, Right-Down-Front)
# hemisphere
x =np.cos(theta)*np.sin(phi) # -1 ~ 1
y =np.sin(theta) # -1 ~ 1
z =np.cos(theta)*np.cos(phi) # 0 ~ 1

RDF_cartesian[:,:,0,:]=x
RDF_cartesian[:,:,1,:]=y
RDF_cartesian[:,:,2,:]=z    

# x_un, y_un, z_un: (target_height, target_width)
x_un = RDF_cartesian[:, :, 0, 0]
y_un = RDF_cartesian[:, :, 1, 0]
z_un = RDF_cartesian[:, :, 2, 0]
```

<br>

- 마지막으로 ③ `normalized 이미지` → ④ `이미지`로 변경하는 과정입니다. 이 과정을 통하여 ① 에서 정의한 `구면 투영 이미지`의 좌표를 원본 이미지와 대응시킬 수 있으므로 `LUT`를 생성할 수 있습니다. 여기서 사용하는 `LUT`는 `구면 투영 이미지`에서 원본 이미지의 색상 정보를 접근하기 위한 `backward` 매핑을 의미합니다.
- 아래 코드에서 카메라 모델을 이용한 렌즈 왜곡을 반영한 부분은 아래 링크를 참조하시기 바랍니다.
    - 링크: https://gaussian37.github.io/vision-concept-lens_distortion/

<br>

```python
theta = np.arccos(z_un / np.sqrt(x_un**2 + y_un**2 + z_un**2))
mask = theta > np.pi/2

# project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
r_dn = D[0]*theta + D[1]*theta**3 + D[2]*theta**5 + D[3]*theta**7 + D[4]*theta**9

r_un = np.sqrt(x_un**2 + y_un**2)

x_dn = r_dn * x_un / (r_un + 1e-6) # horizontal
y_dn = r_dn * y_un / (r_un + 1e-6) # vertical    

map_x_origin2new = K[0][0]*x_dn + K[0][1]*y_dn + K[0][2]
map_y_origin2new = K[1][1]*y_dn + K[1][2]

map_x_origin2new[mask] = DEFAULT_OUT_VALUE
map_y_origin2new[mask] = DEFAULT_OUT_VALUE

map_x_origin2new = map_x_origin2new.astype(np.float32)
map_y_origin2new = map_y_origin2new.astype(np.float32)
```

<br>

- 위 코드를 정리하여 표현하면 아래와 같습니다.
    - 링크: https://colab.research.google.com/drive/118sQlforFfkE45SOxz16f__LdqQ8niQf?usp=sharing

<br>

```python
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_camera_spherical_lut(
    K, D, origin_width, origin_height, target_width, target_height, hfov_deg, vfov_deg, DEFAULT_OUT_VALUE=-1):
    
    '''
    - K : (3, 3) intrinsic matrix
    - D : (5, ) distortion coefficient
    - origin_width, origin_height: input image size
    - target_width, target_height: output image size
    - hfov_deg: 0 ~ 360
    - vfov_deg: 0 ~ 180
    '''

    # 구면 투영 시 생성할 azimuth/elevation 각도 범위
    # hfov: azimuth
    # vfov: elevation
    hfov=np.deg2rad(hfov_deg)
    vfov=np.deg2rad(vfov_deg)
    
    # 구면 투영 시, normalized → image 로 적용하기 위한 intrinsic 행렬
    new_K = np.array([
        [target_width/hfov,       0,                     target_width/2],
        [0,                       target_height/vfov,    target_height/2],
        [0,                       0,                     1]
    ], dtype=np.float32)
    
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
    # azimuthal angle
    phi = p_norm[:, :, 0]
    # elevation angle
    theta = p_norm[:, :, 1]   
    
    RDF_cartesian = np.zeros(p_norm.shape).astype(np.float32)
    
    # x, y, z : cartesian coordinate in camera coordinate system (RDF, Right-Down-Front)
    # hemisphere
    x =np.cos(theta)*np.sin(phi) # -1 ~ 1
    y =np.sin(theta) # -1 ~ 1
    z =np.cos(theta)*np.cos(phi) # 0 ~ 1
    
    RDF_cartesian[:,:,0,:]=x
    RDF_cartesian[:,:,1,:]=y
    RDF_cartesian[:,:,2,:]=z    
    
    # x_un, y_un, z_un: (target_height, target_width)
    x_un = RDF_cartesian[:, :, 0, 0]
    y_un = RDF_cartesian[:, :, 1, 0]
    z_un = RDF_cartesian[:, :, 2, 0]
    
    theta = np.arccos(z_un / np.sqrt(x_un**2 + y_un**2 + z_un**2))
    
    mask = theta > np.pi/2

    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    r_dn = D[0]*theta + D[1]*theta**3 + D[2]*theta**5 + D[3]*theta**7 + D[4]*theta**9
    
    r_un = np.sqrt(x_un**2 + y_un**2)
    
    x_dn = r_dn * x_un / (r_un + 1e-6) # horizontal
    y_dn = r_dn * y_un / (r_un + 1e-6) # vertical    
    
    map_x_origin2new = K[0][0]*x_dn + K[0][1]*y_dn + K[0][2]
    map_y_origin2new = K[1][1]*y_dn + K[1][2]
    
    map_x_origin2new[mask] = DEFAULT_OUT_VALUE
    map_y_origin2new[mask] = DEFAULT_OUT_VALUE
    
    map_x_origin2new = map_x_origin2new.astype(np.float32)
    map_y_origin2new = map_y_origin2new.astype(np.float32)
    
    return map_x_origin2new, map_y_origin2new, new_K

calib = json.load(open("camera_calibration.json", "r"))
image = cv2.cvtColor(cv2.imread("front_fisheye_camera.png", -1), cv2.COLOR_BGR2RGB)

origin_height, origin_width, _ = image.shape
target_height, target_width  = origin_height//2, origin_width//2
hfov_deg = 180
vfov_deg = 150

K = np.array(calib['front_fisheye_camera']['Intrinsic']['K']).reshape(3, 3)
D = np.array(calib['front_fisheye_camera']['Intrinsic']['D'])

map_x, map_y, new_K = get_camera_spherical_lut(K, D, origin_width, origin_height, target_width, target_height, hfov_deg=hfov_deg, vfov_deg=vfov_deg)
new_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
plt.imshow(new_image)
```

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 코드를 통하여 왼쪽의 원본 이미지를 오른쪽의 `구면 투영 이미지`와 같이 변경할 수 있습니다. 구면 투영 이미지는 원본 이미지의 절반 사이즈로 생성하였습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/9.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 설명한 바와 같이 $$ \color{red}{X} $$ 가 증가하는 방향으로 $$ \phi $$ (`azimuth`)가 증가합니다. 이미지의 중점에서 $$ \phi $$ 는 0이고 우측 방향으로 최대 $$ \text{hfov} / 2 $$ 만큼 커지고 좌측 방향으로 최소 $$ -\text{hfov} / 2 $$ 만큼 작아집니다.
- 마찬가지로 $$ \color{green}{Y} $$ 가 증가하는 방향으로 $$ \theta $$ (`elevation`)이 증가합니다. 이미지의 중점에서 $$ \theta $$ 는 0이고 아래 방향으로 최대 $$ \text{vfov} / 2 $$ 만큼 커지고 윗 방향으로 최소 $$ -\text{vfov} / 2 $$ 만큼 작아집니다.
- `new_K`를 생성하였을 때, 정의한 $$ c_{x}, c_{y} $$ 로 인하여 $$ \phi, \theta $$ 가 좌/우, 상/하 대칭이 되도록 이미지를 생성하였습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 지금까지 확인한 내용을 정리하면 위 그림과 같습니다. 직교 좌표계에 정의된 원본 이미지를 구면 좌표계로 변환하기 위해서는 $$ \text{target height}, \text{target width}, \text{hfov}, \text{vfov} $$ 가 정의 되어야 합니다. 그리고 각 이미지 픽셀 $$ \phi_{i}, \theta{j} $$ 의 의미는 구면 좌표계에서 정의된 `azimuth`, `elevation`를 뜻합니다.

<br>

## **회전을 고려한 카메라 기준 구면 투영법**

<br>

- 앞에서 살펴본 내용에서는 이미지의 중점을 구면 좌표 축과 동일하게 두어 이미지의 중점에 $$ \phi = 0, \theta = 0 $$ 인 상태로 `구면 투영 이미지`를 생성하였습니다.
- 만약 `Roll`, `Pitch`, `Yaw` 축의 각 방향에 `Rotation`을 적용하여 `구면 투영 이미지`를 생성한다면 어떻게 생성할 수 있을까요? 이와 같이 이미지를 생성한다면 카메라의 장착이 회전되었을 때를 고려하여 이미지를 생성할 수 있습니다. 이 방법에 대하여 살펴보도록 하겠습니다.

<br>

- 먼저 카메라 좌표계 기준에서 `roll`, `pitch`, `yaw`를 변환하는 방법은 아래와 같습니다.
- 카메라 좌표계 또는 `RDF` 좌표계에서는 $$ X $$ 축 회전이 `pitch` 회전에 대응되고, $$ Y $$ 축 회전이 `yaw` 회전, $$ Z $$ 축 회전이 `roll` 회전에 대응됩니다. 각 좌표축에서 양의 방향으로 회전 시 **반시계 방향**으로 회전합니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 주황색 회전 방향과 같이 `RDF` 좌표계에서는 양의 방향으로 회전 시 **반시계 방향**으로 회전합니다.
- 예를 들어 $$ X $$ 축 즉, `pitch` 방향으로 +30도 회전하게 되면 카메라는 아래쪽을 바라보도록 회전합니다. $$ Y $$ 축 즉, `yaw` 방향으로 +30도 회전하게 되면 카메라는 왼쪽을 바라보도록 회전합니다. 마지막으로 $$ Z $$ 축 즉, `roll` 방향으로 +30도 회전하게 되면 카메라는 광축을 기준으로 +30도 반시계 방향으로 회전합니다. 이 내용을 그림으로 살펴보면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 `pitch` 기준의 회전을 살펴보면 위 그림과 같습니다. $$ X $$ 축을 기준으로 반시계 방향으로 회전한 것을 볼 수 있습니다. (위 그림에서는 그림 상 축 외부에서 회전을 표시할 수 밖에 없어서 시계 방향으로 회전한 것처럼 보이지만 원점에서 바라보면 반시계 방향으로 회전한 것을 알 수 있습니다.)

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음은 `yaw` 기준의 회전을 살펴보겠습니다. $$ Y $$ 축을 기준으로 반시계 방향으로 회전한 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음은 `roll` 기준의 회전을 살펴보겠습니다. $$ Z $$ 축을 기준으로 반시계 방향으로 회전한 것을 볼 수 있습니다.

<br>

- 이와 같이 `roll`, `pitch`, `yaw` 각 방향으로 회전을 하였을 때, 카메라는 각 축을 기준으로 반시계 방향 회전을 하게 됩니다.
- 카메라의 회전을 반영하는 방법은 2가지 방법이 있습니다. 첫번째로 카메라 축을 회전하는 방법이고 있고 두번째로 앞의 코드에서 `RDF_cartesian`인 포인트 샘플을 회전하는 방법이 있습니다. 이번 글에서는 두번째 방법인 **포인트 샘플을 회전하는 방법**을 기준으로 설명을 진행하려고 합니다.
- 포인트 샘플을 회전하는 방법을 이용하는 이유는 ① 궁극적으로 회전을 해야 하는 것은 구면 좌표계에서 부터 정의되어 직교 좌표계로 변환된 `RDF_cartesian`이기 때문입니다. 그리고 ② 회전해야 할 포인트들을 직접 회전시키는 것이 더 이해하기도 쉽고 설명하기도 쉽기 때문입니다.
- 카메라 축 회전과 포인트 자체를 회전하는 것을 각각 `Passive Transformtation`, `Active Transformation`이라고 합니다. 즉, 본 글에서는 `Active Transformation`을 사용하여 내용을 전개할 예정입니다. 이와 관련된 내용은 아래 링크를 참조해 보시기 바랍니다.
    - 링크: https://gaussian37.github.io/vision-concept-calibration/ (글 내부에서 Active/Passive Transformation을 확인)

<br>
<center><img src="../assets/img/vision/concept/spherical_projection/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 다룬 이 그림을 다시 한번 살펴보도록 하겠습니다.

<br>

```python
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
```

<br>

<br>

## **회전을 고려한 World 기준 구면 투영법**

<br>


<br>

## **회전을 고려한 World 기준 구면 투영법의 World-to-Image, Image-to-World**

<br>

<br>

## **회전을 고려한 World 기준 구면 파노라마 투영법**

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

## **카메라 기준 구면 투영법**

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

## **회전을 고려한 World 기준 구면 투영법**

<br>

<br>

## **회전을 고려한 World 기준 구면 투영법의 World-to-Image, Image-to-World**

<br>

<br>

## **회전을 고려한 World 기준 구면 투영 이미지의 Topview 이미지 생성**

<br>

<br>

## **회전을 고려한 World 기준 구면 파노라마 투영법**

<br>

<br>


<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>