---
layout: post
title: 라이다 포인트 클라우드 처리 및 응용
date: 2022-02-10 00:00:00
img: autodrive/concept/lidar_pointcloud_process/0.png
categories: [autodrive-concept] 
tags: [라이다, 포인트 클라우드, lidar, point cloud] # add tag
---

<br>

- 이 글은 라이다 포인트 클라우드 처리 및 응용 (건국대학교 조기춘 교수님) 강의를 듣고 정리한 내용입니다.

<br>

## **목차**

<br>

- ### 라이다의 기본 원리
- ### 3 Big factors of LiDAR sensors
    - ### 3D Beam-scanning technology
    - ### Distance measurement
    - ### Laser wavelength, Power
- ### 포인트 클라우드 처리
- ### 포인트 클라우드 처리 응용

<br>

## **라이다의 기본 원리**

<br>

- 라이다를 통해 얻을 수 있는 데이터는 포인트 클라우드 이고 포인트 클라우드를 통하여 대상을 인식하는 것이 라이다를 이용한 인식 기법의 핵심입니다. 즉, 포인트 클라우드를 잘 처리하는 것이 라이다를 잘 사용하는 것의 핵심이 됩니다.
- 따라서 포인트 클라우드를 잘 처리하기 위해 라이다의 원리를 먼저 이해해 보고자 합니다. 예를 들어 영상 인식 기법을 잘 이해하기 위하여 Resolution, FoV, Dynamic Range, Color Space, Focal Length, Distortion 등의 카메라 특성을 잘 이해하는 것이 필요하다는 것과 같은 관점입니다. 마찬가지로 라이다의 포인트 클라우드 처리를 위하여 이 글에서 라이다 작동 원리를 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/lidar_pointcloud_process/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- ① `PC` : 프로세싱을 하는 컴퓨터 모듈을 의미합니다.
- ② `timing module` : 라이다 전체 동작의 타이밍을 처리하는 모듈을 의미합니다. 위 그림과 같은 Scanning type Lidar는 시간차를 이용하기 때문에 타이밍 처리가 중요합니다. timing module은 laser를 쏘우라고 laser에 신호를 보냅니다. 이 때, 시간은 $$ t_{1} $$ 이라고 해보겠습니다.
- ③ `laser` : laser는 이 때, 신호를 받고 laser를 쏘게 됩니다. 일반적인 laser beam 처럼 계속 쏘는 것은 아니며 일시적으로 Pusle를 쏘게 됩니다.
- ④, ⑤ `mirror` : mirror를 통하여 원하는 곳으로 laser를 전달하게 됩니다.
- ⑥ `laser pulse` : $$ t_{1} $$ 시간에 쏜 laser의 pulse가 출력됩니다.
- ⑦ `refelected pulse` : laser pulse가 물체를 맞고 반사되어 들어옵니다. 맞고 돌아오는 pulse는 일반적으로 laser pulse 보다 energy가 작습니다.
- ⑧ `mirror` : 반사되어 돌아온 pulse는 거울에 반사되어 detector로 전달합니다.
- ⑨ `detector` : detector에서는 반사되어 돌아온 pulse를 인식합니다.
- ⑩ `timing module` : 반사되어 돌아온 pulse의 수신 시간을 $$ t_{2} $$ 라고 하면 위 그림의 $$ t_{1}, t_{2} $$ 시간 차이를 이용하여 거리를 구할 수 있습니다. 위 식에서 사용된 $$ C $$ 는 빛의 속도를 의미합니다.

<br>
<center><img src="../assets/img/autodrive/concept/lidar_pointcloud_process/2.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 즉, lidar는 laser가 반사되어 돌아오는 것을 통하여 물체의 거리를 측정하는 방식이며 반사되어 돌아오는 시간을 이용하여 물체의 거리 계산합니다. 위 그림의 초록색 점이 point cloud라고 하며 lidar를 통해 인식하는 데이터가 됩니다. 마치 카메라를 통해 인식하면 픽셀마다 컬러 값이 존재하게 되는데 이와 대응된다고 볼 수 있습니다.
- 위 그림과 같이 `mirror`를 회전시켜서 laser를 원하는 영역까지 송신하고 반사되는 위치를 인식하므로 물체의 `최근접점`을 인식하게 됩니다. 반사된 거리를 인식하기 때문에 뒤에 가려진 물체는 인식하기 어렵습니다.
- lidar 구조 이미지에서 ③ laser (발광부)와 ⑧ detector (수광부)는 한 쌍으로 이루어져 있습니다. 만약 많은 point cloud를 얻기 위하여 여러개의 발광부와 수광부를 사용할 수도 있지만 이런 경우에는 가격이 비싸지게 되므로 거울을 통해 반사하는 형식을 사용하고 있습니다.

<br>

- 지금까지 라이다의 기본 원리에 대하여 살펴보았습니다. 최근에는 라이다의 3가지 요소를 기준으로 라이다가 발전하고 있습니다. 라이다의 3가지 요소는 ① 3D Beam-scanning technology, ② Distance measurement, ③ Laser wavelength, Power 입니다. 지금부터 이 3가지 요소에 대하여 자세히 살펴보도록 하겠습니다. 각 항목에서 다룰 내용은 다음과 같습니다.
- `① 3D Beam-scanning technology` : 어떻게 Beam을 사용할 지에 대한 기술로 앞의 예제에서 거울을 이용하여 laser beam의 방향을 바꾸는 것에 해당합니다.
- `② Distance measurement` : 거리를 어떻게 측정할 지에 해당합니다. 앞의 예제에서는 ToF(Time of Flight)라는 방법 즉, laser가 송출되었다가 수신되는 시간 차이를 이용한 거리 추정 방법에 해당합니다.
- `③ Laser wavelength, Power` : 라이다의 wavelength, Power를 어떻게 설정할 지에 해당합니다.
- 그러면 위 3가지 요소를 차례대로 살펴보도록 하겠습니다.

<br>

## **3D Beam-scanning technology**

<br>

- 앞에서 가장 기본적인 lidar의 구조에 대하여 살펴보았고 성능 개선, 가격 경쟁력을 위하여 다양한 시도들이 이루어 지고 있습니다.
- 먼저 Beam sanning 기술은 앞에서 설명한 거울을 이용하여 beam의 방향을 바꾸는 것에 해당하며 이와 같이 beam의 방향을 바꾸는 이유는 원하는 방향으로 laser를 설치하여 사용 하면 laser의 갯수가 늘어나서 가격이 비싸지는 문제점이 있기 때문입니다.

<br>
<center><img src="../assets/img/autodrive/concept/lidar_pointcloud_process/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Beam scanning 관점에서 라이다는 크게 Non-scanning과 scanning 방식의 라이다로 나눌 수 있습니다. 
- 먼저 Non-scanning 라이다는 laser를 scanning 하지 않고 고정된 방식으로 쏘게 되나 laser를 퍼트려 줌으로써 다양한 영역을 감지할 수 있는 방식입니다.
- scanning 방식에는 크게 Non-mechanical scanner와 Mechanical scanner 2가지 방식이 있습니다. Non-mechanical scanner 전자식을 의미하며 Mechanical scanner는 말 그대로 기계식 방식을 의미합니다. 라이다가 동작할 때, 내부적으로 회전이 발생하는 것이 보인다면 Mechanical scanner입니다.
- `Motorized Optomechanical Scanners`라고 불리는 방식은 Beam Scanning을 위하여 모터가 돌아가는 방식이며 자동차에서 사용 중인 벨로다인 라이다를 보면 내부적으로 모터가 돌아가는 것을 확인할 수 있는데 이 라이다가 대표적인 예시입니다.
- `MEMS Scanners`의 MEMS는 Micro Electro Mechanical Systems의 약자로 굉장히 작은 전자 기계 방식을 이용합니다. 앞의 모터 방식은 모터가 사람 눈으로 볼 수 있을 정도로 크지만 `MEMS Scanner`는 그렇지 않습니다.