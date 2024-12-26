---
layout: post
title: 원통 좌표계와 구면 좌표계
date: 2022-03-01 00:00:00
img: math/calculus/cylindrical_spherical_coordinate_system/0.png
categories: [math-calculus] 
tags: [원통 좌표계, 구면 좌표계, cylindrical, spherical] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 이번 글에서는 원통 좌표계와 구면 좌표계에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [직교 좌표계, 원통 좌표계, 구면 좌표계](#직교-좌표계-원통-좌표계-구면-좌표계-1)
- ### [Foward-Left-Up 좌표축에서의 좌표계 변환](#foward-left-up-좌표축에서의-좌표계-변환-1)
- ### [Right-Down-Foward 좌표축에서의 좌표계 변환](#right-down-foward-좌표축에서의-좌표계-변환-1)

<br>

## **직교 좌표계, 원통 좌표계, 구면 좌표계**

<br>

- 3차원 좌표계에서 일반적으로 많이 사용하는 `직교 좌표계`는 임의의 점 $$ P = (x, y, z) $$ 로 나타냅니다.
- 반면 이 글에서 다룰 `원통 좌표계(Cylindrical Coordinate)` 또는 `구면 좌표계(Spherical Coordinate)`는 직교좌표계의 $$ x, y, z $$ 좌표를 다른 방식으로 표현하며 이 글에서 이 내용에 대하여 살펴보고자 합니다.
- `원통 좌표계`나 `구면 좌표계`를 사용하는 이유는 다양한 상황에서 이와 같은 좌표계를 사용하였을 때, 계산이 편리해지는 상황이 발생하기 때문입니다. 이와 관련된 내용은 대학 미적분학에서도 충분히 다루기도 합니다.
- 먼저 개념을 이해하기 위해 `원통 좌표계`와 `구면 좌표계`의 표현 방식에 대하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 가운데 좌표계가 `원통 좌표계`입니다. 위 좌표축과 같이 검지가 $$ X $$ 축, 중지가 $$ Y $$ 축, 엄지가 $$ Z $$ 축으로 표현할 수 있는 **오른손 좌표계**에서 $$ X $$ 축에서 $$ Y $$ 축 방향으로의 회전 각도를 $$ \phi $$ 로 나타내며 이 각도를 `azimuth angle`이라고 부릅니다. 점 $$ P $$ 를 $$ XY $$ 평면에 투영하였을 때, 원점과 점 $$ P $$ 의 `거리(distance)`를 $$ r $$ 로 나타냅니다. $$ z $$ 는 `직교 좌표계`와 동일하게 사용됩니다.
- `원통 좌표계`는 기존에 사용하는 `직교 좌표계`의 $$ x, y $$ 좌표 표현을 사용하는 대신에 $$ \phi, r $$ 을 이용하여 나타냅니다. 회전 각도와 원점으로 부터의 거리인 $$ \phi, r $$ 을 이용하여 점의 위치를 나타내기 때문에 원과 같은 곡면을 표현하기 유리해지며 높이는 그대로 $$ z $$ 를 사용하기 때문에 원기둥의 표면을 좌표로 나타내기 쉬운 장점이 있습니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림을 참조하면 `원통 좌표계`의 의미를 이해하시는 데 도움이 되실 것입니다.
- 앞에서 `직교 좌표계`와 비교하기 위하여 $$ X, Y, Z $$ 축을 그대로 사용하였지만 위 그림의 $$ A $$ 에 해당하는 $$ \phi $$ 의 시작점을 나타내는 축은 `Polar Axis`라 하고 높이를 나타내는 $$ L $$ 은 `Longitudinal Axis`라고 합니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 가장 오른쪽 좌표계는 구면 좌표계를 나타냅니다. 원통 좌표계와 비교하면 $$ r $$ 의 의미가 달라진 점과 $$ z $$ 대신에 $$ \theta $$ 를 사용하여 점 $$ P $$ 를 표현하였다는 점입니다.
- 원통 좌표계에서는 $$ r = \sqrt{x^{2} + y^{2}} $$ 인 반면 구면 좌표계에서는 $$ r = \sqrt{x^{2} + y^{2} + z^{2} } $$ 으로 정의됩니다.
- 구면 좌표계에서의 $$ \theta $$ 는 `elevation` 또는 `inclination` 이라고 하며 높이 축으로 부터 아래로 내려오는 각도를 의미합니다.
- 이와 같이 $$ \phi, \theta, r $$ 을 이용하여 점 $$ P $$ 를 나타내면 $$ x, y, z $$ 를 사용하지 않고도 점의 위치를 표현할 수 있으며 구 형태 표면에서의 좌표를 나타낼 때 용이합니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 추가적으로 참고 자료에 따라 표기 방법이 다른 경우가 발생하여 그 부분을 정하고 넘어가고자 합니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 내용은 위키피디아에서 발췌한 내용입니다. 왼쪽의 경우가 물리학이나 공학에서 많이 사용하는 표기법입니다. 즉, `azimuth angle`에 $$ \phi $$ 를 사용하고 `elevation angle`에 $$ \theta $$ 를 사용합니다. 본 글에서도 이 표기법을 따를 예정입니다.
- 반면에 오른쪽 표기 방법은 수학에서 많이 사용하는 방법이라고 합니다. 참조하시면 되겠습니다.

<br>

## **Foward-Left-Up 좌표축에서의 좌표계 변환**

<br>

- 지금 부터는 `FLU(Foward-Left-Up)` 좌표축에서 직교 좌표계, 원통 좌표계, 구면 좌표계의 변환 방법과 코드를 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/calculus/cylindrical_spherical_coordinate_system/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `FLU`는 위 축의 방향과 같이 $$ X, Y, Z $$ 축이 정의된 형태를 의미합니다. 앞에서 계속 살펴본 것과 동일한 축 방향입니다. 글 아랫 부분에서는 `RDF(Right-Down-Foward)` 좌표축에서의 변환 방법을 다루어 볼 예정이므로 `FLU`와 비교해서 살펴보면 도움이 될 것 같습니다.

<br>


## **Right-Down-Foward 좌표축에서의 좌표계 변환**

<br>


<br>


<br>



[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>