---
layout: post
title: Monocular 3D Object Detection 에서의 Orientation Estimation (방향 추정)
date: 2021-08-10 00:00:00
img: vision/detection/orientation_estimation_monocular_3d_od/0.png
categories: [vision-detection] 
tags: [object detection, 3d object detection, monocular, orientation, estimation, orientation estimation] # add tag
---

<br>

[Detection 관련 글 목록](https://gaussian37.github.io/vision-detection-table/)

<br>

- 참조 : https://towardsdatascience.com/orientation-estimation-in-monocular-3d-object-detection-f850ace91411

<br>

- 이번 글에서는 단안 카메라에서의 3D Object Detection에서의 방향 추정을 하는 방법에 대하여 알아보도록 하겠습니다.
- Monocular 3D Object Detection은 2D RGB 이미지에서 객체 주변에 3D 방향의 Bounding Box를 그리는 Task를 의미합니다.
- 단일 2D 이미지 입력으로 3D를 추론하는 작업은 어려우며 차량 방향 추정은 이 중요한 작업을 위한 중요한 단계 중 하나압니다.

<br>
<center><img src="../assets/img/vision/detection/orientation_estimation_monocular_3d_od/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- Monocular 3D object detection에서 orientation 관련 개념으로 언급되는 개념이 `allocentric orientation`과 `egocentric orientation`입니다. 이번 글에서는 자율주행차 관점에서의 2가지 orientation에 대하여 다루어보도록 하겠습니다.

<br>

## **Egocentric과 Allocentric**

<br>

- `egocentric orientation`에서 `egocentric`의 사전적 의미는 `자기 중심적인`입니다. 자율주행차량 관점에서의 의미는 **자차의 카메라를 기준으로 한 방향을 의미**합니다. 반면 `allocentric orientation`의 `allocentric`의 사전적 의미는 `타인 중심의`이며 자율주행차량 관점에서는 자차 이외의 차량을 기준으로 한 방향을 의미합니다.
- `egocentric orientation`은 차량들의 `global orientation`이라고도 하며 자차의 카마레 좌표계에 
- `allocentric orientation`은 `local orientation` 또는 `observation angle`이라고도 하며 `egocentric`과는 다르게 참조하는 frame이 관심 대상에 따라 변합니다. 각각의 차량에 따라 개별적인 좌표계를 

<br>
<center><img src="../assets/img/vision/detection/orientation_estimation_monocular_3d_od/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 (a)를 살펴보면 차들이 왼쪽에서 오른쪽으로 이동하고 있는 상태 입니다. 차들의 입장에서는 같은 방향을 바라보고 있지만 차와 카메라의 방향은 왼쪽에서 오른쪽으로 이동하면서 계속 바뀌는 것을 확인할 수 있습니다.
- 반면 그림 (b)에서는 차들의 방향은 서로 다릅니다. 하지만 차들과 카메라의 방향은 모두 같은 것을 알 수 있습니다.
- 위 그림을 통해 확인할 수 있는 내용은 단안 카메라의 이미지는 `local orientation`을 따르며 추정해야 하는 것 또한 `local orientation`임을 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/detection/orientation_estimation_monocular_3d_od/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림에서 왼쪽 crop 된 차들의 이미지를 보면 crop된 이미지 상에서 차의 방향은 계속 변합니다. 하지만 전체 이미지를 보면 차의 방향은 변화하지 않고 같은 방향의 직선 구간을 주행하고 있습니다.
- 따라서 왼쪽의 crop된 패치만을 이용해서는 차의 `global orientation`을 추정하기는 어렵습니다. 따라서 **이미지 전체에서 차의 의미를 파악하여 global orientation을 추정하는 것이 중요**합니다. 반면에, `local orientation`는 이미지 패치 하나만으로도 구할 수 있습니다.

<br>

- KITTI 데이터셋에서는 roll과 pitch를 0으로 둡니다. 이와 같은 방법을 사용하면 orientation은 단순히 `yaw` 값으로 축소시킬 수 있습니다. 따라서  위 그림의 2가지 방향을 `global yaw`와 `local yaw` 2가지로 나타낼 수 있습니다.

<br>

## **local → global yaw**

<br>



<br>

[Detection 관련 글 목록](https://gaussian37.github.io/vision-detection-table/)

<br>