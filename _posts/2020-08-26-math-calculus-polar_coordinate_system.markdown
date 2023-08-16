---
layout: post
title: 극 좌표계 (Polar Coordinate System)
date: 2020-08-26 00:00:00
img: math/calculus/polar_coordinate_system/0.png
categories: [math-calculus] 
tags: [극좌표계, Polar Coordinate] # add tag
---

<br>

[calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 이번 글은 간단하게 `극 좌표계 (Polar Coordinate)`에 관련된 내용을 다루어 보려고 합니다.

<br>

- 일반적으로 좌표계는 테카르트 좌표계 (직교 좌표계)를 사용합니다. 가로축이 $$ x $$ 축이 되고 세로축이 $$ y $$ 축이 되는 형태입니다. 
- 하지만 물체의 이동 및 회전이 동시에 고려되어야 한다면 데카르트 좌표계에서는 회전에 따른 $$ x, y $$ 의 변화를 매번 계산해야 하는 불편함이 발생합니다.
- 따라서 회전 동작을 편리하게 사용하기 위하여 고안된 좌표계가 본 글에서 소개하는 `극 좌표계 (Polar Coordinate System)` 입니다.

<br>
<center><img src="../assets/img/math/calculus/polar_coordinate/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 극 좌표계는 위 그림과 같이 원점으로부터의 거리 $$ r $$ 과 각 $$ \theta $$ 의 두 요소로 구성되며 극좌표계의 좌표는 $$ (r, \theta) $$ 로 표시됩니다.
- 극 좌표계는 동심원의 형태로 평면의 모든 점을 표현할 수 있으며 주로 시간에 따른 회전의 움직임을 구현하기에 용이합니다.

<br>

- 데카르트 좌표계에서 표현된 벡터 $$ (x, y) $$ 는 벡터의 크기와 `arctan` 함수를 사용하여 다음과 같이 $$ (r, \theta) $$ 로 변환할 수 있습니다.

<br>

- $$ r = \sqrt(x^{2} + y^{2}) \tag{1} $$

- $$ \theta = \text{atan2}{(y, x)} \tag{2} $$

<br>

- 참고로 $$ \text{atan2}{(y, x)} $$ 는 1사분면에서의 $$ \text{atan}{(y/x)} $$ 와 같습니다. 하지만 다른 사분면에서는 부호에 따라 값이 달라질 수 있습니다.
    - 참고 : [두 점 사이의 절대각도를 재는 atan2](https://spiralmoon.tistory.com/entry/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9D%B4%EB%A1%A0-%EB%91%90-%EC%A0%90-%EC%82%AC%EC%9D%B4%EC%9D%98-%EC%A0%88%EB%8C%80%EA%B0%81%EB%8F%84%EB%A5%BC-%EC%9E%AC%EB%8A%94-atan2)
- `atan`은 두 점 사이의 **탄젠트 값**을 받아 $$ -\pi/2 $$ ~ $$ \pi/2 $$ 범위의 라디안 값 (-90도 ~ 90도)을 반환하는 반면 `atan2`는 **두 점 사이의 상대좌표 $$ (x, y) $$** 를 받아 $$ -\pi $$ ~ $$ \pi $$ 범위의 라디안 값 (-180도 ~ 180도)의 라디안 값을 반환합니다.

<br>

- 출처 : [두 점 사이의 절대각도를 재는 atan2](https://spiralmoon.tistory.com/entry/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9D%B4%EB%A1%A0-%EB%91%90-%EC%A0%90-%EC%82%AC%EC%9D%B4%EC%9D%98-%EC%A0%88%EB%8C%80%EA%B0%81%EB%8F%84%EB%A5%BC-%EC%9E%AC%EB%8A%94-atan2)

<br>
<center><img src="../assets/img/math/calculus/polar_coordinate/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- + 또는 -의 부호가 표시되는 데카르트 좌표계에서는 `atan2`를 사용하면 좌표값을 인자로 사용하면 되기 때문에 보통 `atan2`를 사용합니다.

<br>

- 반대로 극 좌표계의 좌표 $$ (r, \theta) $$ 를 데카르트 좌표계 $$ (x, y) $$ 로 변환하는 식은 삼각함수를 사용해 구할 수 있습니다.

<br>

- $$ x = r \cdot \cos{(\theta)} \tag{3} $$

- $$ y = r \cdot \sin{(\theta)} \tag{4} $$

<br>

[calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

