---
layout: post
title: Ashok Elluswamy (Tesla) CVPR 2022 Workshop on Autonomous Vehicles (Occupancy Network) 정리
date: 2022-09-11 00:00:00
img: autodrive/concept/tesla_cvpr_2022/0.png
categories: [autodrive-concept] 
tags: [tesla, 테슬라, cvpr, cvpr 2022 workshop, occupancy network] # add tag
---

<br>

- 이번 글에서는 CVPR 2022의 Workshop에서 Tesla가 발표한 `Occupancy Network` 내용에 대하여 정리해 보도록 하겠습니다.

<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/jPCV4GKX9Dw" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

## **목차**

<br>

- ### Autopilot과 Full Self-Driving Beta Software
- ### 


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