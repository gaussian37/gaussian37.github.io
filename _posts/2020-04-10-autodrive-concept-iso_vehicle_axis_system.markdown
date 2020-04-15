---
layout: post
title: ISO Vehicle Axis System
date: 2020-04-10 00:00:00
img: autodrive/concept/vehicle_axis/0.png
categories: [autodrive-concept] 
tags: [autonomous drive, 자율 주행, vehicle axis system] # add tag
---

<br>

- [Autonomous Driving 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>

- `ISO 8855-2011`에서 제시하는 ISO 차량 축 시스템에 대하여 간략하게 소개하겠습니다.
- 여기서 소개하는 축 시스템은 `SAE`에서 제안하는 것과는 다릅니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 나의 차를 기준으로 정면이 $$ X $$ 축 양의 방향, 왼쪽이 $$ Y $$축 양의 방향, 위 쪽이 $$ Z $$ 축 양의 방향입니다. 

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 중고등학교 시절 때 배운, 벡터의 외적에 대하여 생각해 보면 오른손으로 검지, 중지, 엄지가 가르키는 방향을 각각 $$ X, Y, Z $$의 양의 방향이라고 생각하셔도 됩니다.
- 그리고 $$ X $$ 축의 방향을 `Roll` 또는 `Longitudinal` 이라하고 $$ Y $$ 축의 방향을 `Pitch` 또는 `Lateral` 이라 하며 마지막으로 $$ Z $$ 축의 방향을 `Yaw` 또는 `Vertial` 이라고 합니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 정리하면 위 그림과 같습니다.

<br>

- 이 때, 각 축을 기준으로 회전이 발생할 수 있습니다.
- 먼저 X축을 기준으로 회전한 `Rolling`이 발생하는 경우는 다음과 같습니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/2.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음으로 Y축을 기준으로 회전한 `Pitching`이 발생하는 경우는 다음과 같습니다.
- 도로에 요철이 있어서 위아래로 흔들리는 경우입니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/4_1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음으로 Z축을 기준으로 회전하는 `Yawing`이 발생하는 경우입니다.
- 좌회전, 우회전 할 때 회전하는 경우입니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/5_1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 여기까지가 `ISO 8855-2011`에서 제시하는 차량 축에 관한 내용입니다.

<br>

- 관점에 따라 다를 수 있지만 3개의 축을 2개씩 짝을 지어 평면 위에 둔다면 일반적으로 사용하는 좌표 평면과 조금 다릅니다.
- 지금 부터 보는 관점은 자동차의 입장 또는 운전자가 탑승하였을 때의 위치 기준으로 설명하는 것입니다.
- 먼저 `YZ 평면` 부터 살펴보겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음은 `ZX 평면`을 살펴보겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그리고 `XY 평면`을 살펴보겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/vehicle_axis/8.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 세 평면 모두 세로 축은 중앙에서 위쪽으로 갈수록 값이 증가하지만 가로 축은 중앙에서 왼쪽으로 갈수록 값이 증가합니다.
- 가로축의 경우 일반적인 좌표평면과 축의 방향이 반대가 되므로 계산할 때 주의가 필요하오니 반드시 가로 축의 방향을 고려해서 부호 및 계산을 해야합니다.

<br>

- [Autonomous Driving 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>