---
layout: post
title: Ashok Elluswamy (Tesla) CVPR 2022 Workshop on Autonomous Vehicles (Occupancy Network) 정리
date: 2022-09-11 00:00:00
img: autodrive/concept/tesla_cvpr_2022/0.png
categories: [autodrive-concept] 
tags: [tesla, 테슬라, cvpr, cvpr 2022 workshop, occupancy network] # add tag
---

<br>

- 이번 글에서는 CVPR 2022의 Workshop에서 Tesla가 발표한 `Occupancy Network` 내용에 대하여 정리해 보도록 하겠습니다. 아래는 발표 내용입니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/jPCV4GKX9Dw" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 발표 전에 `Occupancy Network`가 적용된 `Full Self-Driving Beta 10.69.1`의 성능을 한번 보면 2021년 버전 보다 얼만큼 개선된 지 알 수 있습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/N4X4GMFmTb0" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

## **목차**

<br>

- ### Autopilot과 Full Self-Driving Beta Software
- ### Classical Drivable Space 인식의 한계
- ### Occupancy Network의 소개
- ### Occupancy Network Architecture

<br>

## **Autopilot과 Full Self-Driving Beta Software**

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 슬라이드에서는 현재 개발된 `Autopilot`과 `FSD` (Full Self-Driving) Beta software에 대한 설명이 되어 있습니다.
- **현재 모든 차량**에는 기본적인 `Autopolut`은 탑재되어 있고 이 기능은 자차가 차선을 벗어나지 않도록 유지하도록 하는 기능이고 주변 차량을 따라가는 역할을 합니다. 또한 안전 기능으로써 다양한 충돌을 피하기 위한 긴급 정지 및 회피 (emergency & steering) 기능이 적용되어 있습니다.
- 그 다음 단계로 약 100만대의 차량에 [Navigation On Autopilot](https://www.tesla.com/ownersmanual/modely/en_kr/GUID-0535381F-643F-4C60-85AB-1783E723B9B6.html)이 적용되어 있습니다. 이 기능은 차선 변경과 고속도로에서 IC/JC를 자동으로 빠져나가는 기능을 지원합니다.
- 마지막으로 약 10만대의 차량에서 `FSD`를 사용중이며 이 기능은 주차장에서 부터 도심과 고속도로 전체에서 주행 보조를 지원합니다. 이 기능부터는 인식 범위가 확장되어 신호등과 정지 신호를 감지하여 멈출 수 있으며 교차로 및 보호/비호호 좌/우 회전에서 다른 차량에게 길을 양보하여 적당한 상황에서 자동 주행을 할 수 있으며 이 때 주차된 차들이나 장애물들을 피해갈 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 슬라이드에서는 테슬라의 `FSD` 화면과 `FSD`에 사용된 센서의 간략한 사양을 확인할 수 있습니다.
- 8개의 카메라와 1.2 백만개의 픽셀 사이즈의 영상을 받는 카메라를 사용하고 8개의 카메라를 이용하여 360도 전체를 볼 수 있으며 (볼 수 있는 거리는 미확인) 초당 36 Frame을 입력으로 받을 수 있습니다. (실제 기능의 처리 시간은 아니며 카메라가 처리할 수 있는 FPS로 생각하면 됩니다.)
- 카메라 입력을 처리하는 하드웨어는 144 TOPS의 처리 속도를 가집니다.
- 이전에 테슬라에서 공개한 바와 같이 레이더, 라이다는 인식 기능을 위해 사용되지는 않았고 초음파 센서는 사용되었으며 HD map도 배제한 것으로 설명합니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 자세한 인식 영상은 글 상단의 영상을 확인하시면 되며 몇가지 내용만 확인해 보겠습니다.
- 위 그림에서 보면 다양한 차들을 인식하며 인식 결과도 깜빡이지 않고 일관성 있게 출력하고 있습니다. 단, 위 그림과 같이 많은 차들이 일렬로 나열되어 인식 난이도가 올라가면 차량의 앞/뒤 또는 차량의 종류 구분에는 오인식이 발생하는 것으로 확인됩니다.
- 그럼에도 불구하고 차량 인식 성능이 과거에 비해 향상된 것으로 확인됩니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 사거리에서의 영상 인식 성능도 향상된 것을 확인할 수 있으며 건너편의 사람도 인식이 되는 것을 확인할 수 있습니다. 

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 우회전 상황이며 이 때, 파란색으로 나타나는 차량이 화면에 표시되며 정확한 의미는 확인이 어렵지만, 충돌 가능한 차량으로 이 차가 지나가기를 기다리는 것으로 추정합니다.

<br>

## **Classical Drivable Space 인식의 한계**

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 기존에 3D 공간의 주행 가능 영역 (drivable space)를 확인하는 방법은 **2D 이미지 상에서 픽셀 별 (uv 좌표계 기준)로 주행 가능 영역인 지 Semantic Segmentation 모델을 이용하여 확인**하고 **Depth Estimation을 통해 3D 공간으로 확장하는 방법**을 사용하였습니다.
- 테슬라에서 최근에 공개한 컴퓨터 비전 기반의 인식 모델의 컨셉은 다른 방향으로 바뀌었는데 어떤 문제가 있어서 컨셉의 변경이 있었는 지 슬라이드에서 제공하는 `기존 문제점`에 대하여 먼저 살펴보겠습니다.
- **Doesn't get overhanging obstacles & provide 3D structure** : 2D 이미지 → 3D 공간으로 변경 시 물체의 3D 형상을 예측하기 어렵습니다. 위 슬라이드와 같이 포크레인 머리 부분의 돌출부나 건물 벽과 같은 영역의 돌출부의 3D 정보를 추정하는 데 한계가 있습니다.
- **Extremely sensitive to depth at the horizon** : 원거리에 있는 수평선 라인의 경우 주행 가능 영역인 지 또는 주행 불가능 영역인 지 확인 시 Segmentation의 결과를 이용하여 판단하고 주행 가능 영역의 거리는 Depth Estimation을 통해서 거리를 예측합니다. 하지만 원거리 영역에서의 Depth Estimation은 몇 픽셀에 따라서 큰 차이가 날 수 있고 Segmentation의 결과가 몇 픽셀 부정확하게 예측하면 오차가 크게 발생하는 문제가 생깁니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 2D 이미지 → 3D 공간으로 변환 (`unproejct to 3d points`)하는 것에 한계점은 `Depth Estimation`의 출력에 한계가 있기 때문입니다. 2D 이미지 → 3D 공간으로 변환은 아래 링크를 참조하시기 바랍니다.
    - 링크 : [포인트 클라우드와 뎁스 맵의 변환 관계 정리](https://gaussian37.github.io/vision-depth-pcd_depthmap/)
- 위 슬라이드에서 지적하는 Depth Estimation의 단점은 크게 5가지가 있고 각 내용은 **Depth Estimation의 해상도가 높지 않다는 것과 2D 이미지에서 Depth를 검출하는 것의 한계에 관련된 내용**들입니다.
- **Depth can be inconsistent locally** : local 영역에 대하여 Depth 정보가 일관적이지 않는 경우의 문제 입니다. 이 경우 평평한 벽과 같은 물체에 대해서도 깊이가 일관적이지 않고 들쑥날쑥하게 됩니다.
- **Too Sparse closer to the horizon** : local 영역에 대하여 Depth가 일관적이지 않아 너무 듬성 듬성 Depth 정보가 존재하게 되면 horizon으로 잘못 인식 하는 경우가 발생하게 됩니다.
- **Cannot predict through occlusion** : 2D 이미지를 통해 3D 공간을 복원하기 때문에 다른 물체에 의해 가려진다면 가려진 가려진 부분은 3D 공간에 복원할 수 없습니다. 이는 사람 또한 상상으로 복원하는 것이지 가려진 물체의 깊이 정보는 복원할 수 없으나 테슬라에서는 이 부분을 문제로 간주하고 개선하였습니다.
- **Doesn't distinguish between moving & static obstacles** : 정적인 물체와 동적인 물체를 구분할 수 없습니다. 

<br>

## **Occupancy Network의 소개**

<br>

- Classical Drivable Space 인식 방법에는 앞에서 소개한 문제가 있고 이 문제를 개선하기 위하여 `Occupancy Network`를 사용합니다.
- Occupancy Network의 출력은 아래와 같습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 아래 링크의 영상 클립을 살펴보시면 데모 영상을 볼 수 있습니다.
    - 링크 : https://youtube.com/clip/UgkxAWHFUi1Y-Jznqwh9zOLNJfnqjZ2Tc4oU

<br> 
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2022/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 슬라이드에서는 테슬라에서 사용하는 `Occupancy Network`의 좋은 장점들을 소개합니다. 
- 8개의 카메라에서 입력되는 이미지를 동시에 처리하여 3D 공간 하나를 출력으로 만들고 그 공간에서 어떤 물체가 차지하고 있는 지, 그 물체가 무엇인 지 까지 확인합니다.
- 이와 같이 3D 공간 상에서 voxel 별 어떤 물체가 점유하고 있는 지 확인할 수 있으므로 `volumetric occupancy`라고 칭합니다.
- Multi Camera를 Video로 처리하기 때문에 일부 보이지 않는 영역을 연속된 영상의 정보를 이용하여 처리할 수 있고 이 내용을 `Multi-camera & video context, Dynamic occupancy`으로 설명합니다.
- 앞에서 다룬 2D 이미지 → 3D 공간으로 변환하는 경우 occlusion이 발생하여 2D 이미지 상에서 보이지 않는 물체는 형상을 그려내지 못하지만 위 슬라이드와 같이 `Multi-camera & video` 환경에서 2D 이미지를 거치지 않고 바로 3D로 변환하는 경우 occlusion이 발생한 물체에 대해서도 일부 출력이 가능해 짐을 보여줍니다. (`Persistent through occlusion`)
- 앞에서 살펴본 2D 이미지의 Depth Prediction은 근거리에서는 해상도가 높지만 원거리에서는 오차 범위가 커져서 해상도가 낮은 문제가 발생합니다. 이 문제로 인하여 Segmentation의 불안정한 출력이 거리 오차를 크게 만드는 문제가 있음을 이전 슬라이드에서 언급하였습니다. `Occupancy Network`에서는 일정한 간격으로 Voxel을 형성하고 각 영역에 물체가 있는 지 여부를 확인하기 때문에 해상도가 급격히 안좋아지는 문제를 개선할 수 있음을 언급합니다. (`Resolution were it matters`)
- 마지막으로 이와 같은 과정을 10 ms 이내에 처리할 수 있도록 메모리와 계산 측면에서 최적화 하여 `Occupancy Network`의 주기가 10 ms 가 될 수 있도록 구현하였다고 설명합니다. 이전 슬라이드에서 카메라 입력이 35 ms 주기이고 Occupancy Network의 연산이 10 ms 이므로 `Occupancy Network` 기준 전처리 및 후처리를 할 시간이 충분함을 시사합니다.

<br>

## **Occupancy Network Architecture**

<br>

<br>

