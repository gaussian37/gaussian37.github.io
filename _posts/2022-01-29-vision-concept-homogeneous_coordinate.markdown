---
layout: post
title: Homogeneous coordinate (동차 좌표계)
date: 2022-04-01 00:00:00
img: vision/concept/homogeneous_coordinate/0.png
categories: [vision-concept] 
tags: [homogeneous coordinate, 동차 좌표계] # add tag
---

<br>

- 참조 : https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/homo-coor.html
- 참조 : https://blog.daum.net/shksjy/229
- 참조 : https://darkpgmr.tistory.com/78
- 참조 : [Homogeneous Coordinates and Transformations ofthe Plane](https://drive.google.com/file/d/1MgpwWjSDtFtHXOCrrJQLeTXcjh9X4bhM/view?usp=sharing)
- 참조 : [The Homogeneous Perspective Transform](https://drive.google.com/file/d/12zjuYN98rGY42gxFtykUNl4FsxghpZ5_/view?usp=sharing)

<br>

- Homogeneous coordinate는 Computer Vision에서 연산을 할 때 많이 사용하는 개념입니다. 그러면 Homogeneous coordinate가 무엇인 지, 왜 사용하는 지 등에 대한 개념을 알아보도록 하겠습니다.

<br>

## **Homogeneous coordinate의 의미와 사용 이유**

<br>

- 먼저 Homogeneous coordinate를 다루기 이전에 Affine Transformation에 대한 개념이 필요합니다. [Affine Transformation의 상세 내용](https://gaussian37.github.io/vision-concept-image_transformation/)에 대한 내용을 확인하시길 바랍니다.
- `Affine Transformation`은 간단하게 `선형 변환 + 이동`으로 정의할 수 있습니다. 보통 어떤 벡터를 Transformation을 할 때, Transformation Matrix를 곱해주어서 변환하게 됩니다. 그런데 벡터라는 것은 크기와 방향을 가지게 되는데 벡터 자체만으로는 **위치가 없기 때문에** 이동 변환을 할 수 없습니다. 즉, 벡터를 Transformation Matrix를 곱하여 이동 변환을 바로 적용할 수 없다는 뜻입니다.

<br>
<center><img src="../assets/img/vision/concept/homogeneous_coordinate/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그러면 위 그림과 같이 벡터 $$ v $$ 에 포인트 개념을 추가하여 위치를 표현해 보도록 하겠습니다. 포인트 $$ P, Q $$ 가 있고 벡터 $$ v $$ 가 있을 때, 아래 식과 같이 정의 됩니다.

<br>

- $$ v = P - Q \tag{1} $$

- $$ P = Q + v \tag{2} $$

<br>

- 식(1)을 통하여 포인트 - 포인트는 벡터가 됨을 알 수 있고 식(2)를 통하여 포인트 더하기 벡터는 포인트가 됨을 알 수 있습니다.
- 단순히 벡터만 존재하는 `벡터 공간 (Vector Space)`에서 포인트를 도입하여 위치 정보가 추가되면 이 때부터 `어파인 공간 (Affine Space)`이라고 부르게 됩니다. 벡터 공간에서 발생하는 transformation은 linear transformation이고 어파인 공간에서 발생하는 transformation은 affine transformation인데 가장 큰 차이점은 앞에서 언급한 바와 같이 이동 성분입니다.

<br>

- $$ y = Ax \tag{3} $$

- $$ y = Ax + b \tag{4} $$

<br>

- 식 (3)과 같이 변환할 때 linear transformation이고 식 (4)와 같이 변환할 때 affine transformation입니다. 즉 $$ b $$ 라는 이동 성분으로 인하여 affine transformation이 되는 것입니다.

<br>

- 그런데 여기서 affine transfomation 조차 $$ y = Ax $$ 와 같이 표현할 수 있을까요? 즉, 단순히 벡터에 transformation matrix를 곱하여 affine transformation을 하는 방법이 있을까요? 이 방법을 적용하기 위해서 `homogeneous coordinate`를 도입해야 합니다. 다시 말하면 **포인트의 transformation과 벡터의 transformation을 한번에 표현하는 transformation matrix를 구해야 합니다.**

<br>

- $$ v = \begin{bmatrix} v_{x} \\ v_{y} \\ v_{z} \\ 0 \end{bmatrix} \tag{5} $$

- $$ p = \begin{bmatrix} p_{x} \\ p_{y} \\ p_{z} \\ 1 \end{bmatrix} \tag{6} $$

<br>

- 사용법을 먼저 다루고 상세 내용은 이후에 다루도록 하겠습니다. `homogeneous coordinate`의 사용 방법은 $$ N $$ 차원의 벡터와 포인트를 표현할 때, $$ N + 1 $$ 차원의 벡터와 포인트를 사용하는 것입니다. 예를 들어 벡터의 경우 식 (5)와 같이 차원을 하나 추가하고 추가된 차원에는 0을 사용합니다. 반면 포인트의 경우 식 (6)과 같이 차원을 하나 추가하고 추가된 차원에는 1을 사용합니다. 
- 이와 같은 형태로 transformation matrix를 사용하게 되면 벡터와 포인트에 대하여 linear transformation과 translation을 하나의 transformation matrix로 표현할 수 있습니다.

<br>

- 먼저 translation matrix를 `homogeneous coordinate` 방식으로 나타내고 각각 `homogeneous coordinate`에서 벡터와 포인트에 각각 곱해보겠습니다.
- translation matrix를 이용하여 벡터 자체는 이동 할 수 없고 포인트는 이동할 수 있는 성질을 이용하여 `homogeneous coordinate`에서 translation matrix를 벡터와 포인트에 각각 곱하면 어떻게 되는지 살펴보겠습니다.

<br>

- $$ \begin{bmatrix} 1 & 0 & 0 & d_{x} \\ 0 & 1 & 0 & d_{y} \\ 0 & 0 & 1 & d_{z} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} v_{x} \\ v_{y} \\ v_{z} \\ 0 \end{bmatrix} = \begin{bmatrix} v_{x} \\ v_{y} \\ v_{z} \\ 0 \end{bmatrix} \tag{7} $$

- $$ \begin{bmatrix} 1 & 0 & 0 & d_{x} \\ 0 & 1 & 0 & d_{y} \\ 0 & 0 & 1 & d_{z} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} p_{x} \\ p_{y} \\ p_{z} \\ 1 \end{bmatrix} = \begin{bmatrix} p_{x} + d_{x} \\ p_{y} + d_{y} \\ p_{z} + d_{z} \\ 1 \end{bmatrix} \tag{8} $$

<br>

- 식 (7)에서 벡터는 이동 행렬을 곱하여도 변화가 없는 반면에 식 (8)에서 포인트는 이동 행렬을 곱하면 이동이 발생하는 것을 확인할 수 있습니다.

<br>

- $$ \begin{bmatrix} \cos{(\theta)} & 0 & \sin{(\theta)} & d_{x} \\ 0 & 1 & 0 & d_{y} \\ -\sin{(\theta)} & 0 & \cos{(\theta)} & d_{z} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} v_{x} \\ v_{y} \\ v_{z} \\ 0 \end{bmatrix} = \begin{bmatrix} v_{x}\cos{(\theta)} + v_{z}\sin{(\theta)} \\ v_{y} \\ -v_{x}\sin{(\theta)} + v_{z}\cos{(\theta)} \\ 0 \end{bmatrix} \tag{9} $$

- $$ \begin{bmatrix} \cos{(\theta)} & 0 & \sin{(\theta)} & d_{x} \\ 0 & 1 & 0 & d_{y} \\ -\sin{(\theta)} & 0 & \cos{(\theta)} & d_{z} \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} p_{x} \\ p_{y} \\ p_{z} \\ 1 \end{bmatrix} = \begin{bmatrix} p_{x}\cos{(\theta)} + p_{z}\sin{(\theta)} + d_{x} \\ p_{y} + d{y} \\ -p_{x}\sin{(\theta)} + p_{z}\cos{(\theta)} + d_{z} \\ 1 \end{bmatrix} \tag{10} $$

<br>

- 식 (9)와 식 (10) 에 사용된 transformation matrix는 모두 $$ y $$ 축에 대한 회전과 이동행렬을 적용한 것입니다.
- 식 (9)의 결과를 보면 벡터는 회전만 할 뿐 이동은 발생하지 않음을 알 수 있습니다.
- 반면 식 (10)의 결과를 보면 포인트는 원점에 대해 회전하고 이동함을 알 수 있습니다.

<br>

- 이와 같이 식 (7) ~ (10) 까지의 결과를 통해 `homogeneous coordinate` 상에서 transformation matrix를 만들면 벡터와 포인트 각각에 대하여 행렬곱 연산만으로 transformation을 할 수 있음을 알 수 있었습니다.

<br>

- 지금까지 `homogeneous coordinate`의 의미와 사용 방법에 다루어 보았습니다. 그러면 이 좌표계를 왜 사용하는 것일까요? 가장 직접적인 이유는 `projective transformation`을 편리하게 다루기 위함입니다. 

<br>
<center><img src="../assets/img/vision/concept/homogeneous_coordinate/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 3차원 좌표계에서 2차원 평면으로 투영을 한다고 가정해 보겠습니다. 이와 같은 투영을 `projective transformation` 통해 발생하게 되는데 원점으로부터 얼만큼 떨어져 있는 평면에 투영하느냐에 따라서 투영한 결과가 달라지게 됩니다.
- 물론 3차원 좌표계에서는 의미하는 점이 변하지 않지만 투영하고자 하는 평면에 따라서 평면 상에서의 값은 달라지게 됩니다.

<br>
<center><img src="../assets/img/vision/concept/homogeneous_coordinate/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 3차원 상에서 빨간색 점이 존재하고 이 점을 2차원 평면에 투영한다고 하였을 때, 어떤 평면에 투영하는 지에 따라서 같은 점이 다른 좌표에 표현되게 됩니다. 
- 그런데 같은 평면이 달라진다고 해서 값이 의미 없이 달라지는 것은 아니고 스케일에 비례하여 달라지는 것을 알 수 있습니다.  (간단한 도형의 닮음을 이용하여 비례하는 것을 살펴보면 됩니다.)
- 이 때, 스케일을 의미하는 것이 앞에서 살펴본 **추가된 차원의 값**입니다. 추가된 차원을 $$ w $$ 라고 하겠습니다. ($$ x, y, z $$ 축과 별도로 $$ w $$ 축을 추가하였음)
- 앞에서 포인트의 경우 $$ w = 1 $$ 을 적용하였습니다. 이것의 의미는 normalized 된 평면에 3차원 좌표의 어떤 점을 2차원 normalized 평면에 투영시킨 것을 의미합니다. 만약 $$ w = 2 $$가 된다면 normalized 평면보다 원점에 2배 가깝게 투영된 것입니다. 반대로 $$ w = 0.5 $$가 되면 normalized 평면보다 원점에서 2배 멀게 투영된 것입니다.
- 만약 $$ w = 0 $$이 된다면 어떻게 될까요? 이 경우 투영되는 평면이 무한대로 멀어지게 되어 단순히 평면에 투영되는 점이 아닌 무한대로 뻗어나아가는 벡터로 이해할 수 있습니다. 따라서 따로 무한대의 표현을 하지않고 $$ w = 0 $$을 사용하면 평면에 투영되는 특정 점이 아닌 벡터로 나타낼 수 있습니다.
