---
layout: post
title: 축-각 회전 (Axis-Angle rotation) 또는 로드리게스 회전 (Rodrigues rotation)
date: 2022-05-10 00:00:00
img: vision/concept/axis_angle_rotation/0.png
categories: [vision-concept] 
tags: [로드리게스 회전 공식, 축 각 회전, axis-angle rotation] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 참조 : 이득우의 게임 수학
- 참조 : https://mycom333.blogspot.com/2014/01/axis-angle-rotation.html

<br>

- 사전 지식 : https://gaussian37.github.io/math-la-rotation_matrix/
- 사전 지식 : https://gaussian37.github.io/math-la-projection/
- 사전 지식 : https://gaussian37.github.io/math-la-cross_product/

<br>

- 이번 글에서는 3차원 회전의 대표적인 방법 중 하나인 `Axis-Angle Rotation`에 대하여 다루어 보도록 하겠습니다. 이 방법은 방법론을 제시한 로드리게스의 이름을 따서 로드리게스 회전이라고도 불립니다. 본 글에서는 `Axis-Angle Rotation`으로 사용하겠습니다.

<br>

## **목차**

<br>

- ### [Axis-Angle Rotation의 필요성](#axis-angle-rotation의-필요성-1)
- ### [Axis-Angle Rotation 수식 설명](#axis-angle-rotation-수식-설명-1)
- ### [Axis-Angle Rotation의 행렬식 표현](#axis-angle-rotation의-행렬식-표현-1)
- ### [Axis-Angle Rotation의 Python code](#axis-angle-rotation의-python-code-1)
- ### [Axis-Angle Rotation의 단점](#axis-angle-rotation의-단점-1)

<br>

## **Axis-Angle Rotation의 필요성**

<br>

- `Axis-Angle Rotation`은 `Euler Angle`의 단점을 개선할 수 있는 3차원 회전 방법으로 사용 됩니다. `Euler Angle`을 이용한 3차원 회전은 직관적이며 단순하다는 장점이 있지만 크게 2가지 문제점이 있습니다. 바로 `Gimbal Lock (짐벌 락)` 현상과 `Rotaional Interpolation (회전 보간)`의 한계점 입니다.
    - 이와 관련된 상세 내용은 아래 링크에서도 확인할 수 있습니다.
    - 링크 : [https://gaussian37.github.io/math-la-rotation_matrix/](https://gaussian37.github.io/math-la-rotation_matrix/)

<br>

- ① `Gimbal Lock` 현상은 X, Y, Z 축을 90도 회전 시키다 보면 어떤 2개의 축의 회전이 동일해져 버리는 현상을 말합니다. 세 개의 축으로 자유롭게 회전할 수 있었는데 두 개의 축만 회전할 수 있도록 어떤 축의 회전이 `Lock`이 걸려 버리기 때문에 `Gimbal Lock`이라고 표현합니다. 아래 영상을 참조하시면 바로 이해가 되실 겁니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/zc8b2Jo7mno" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 90도 회전 시, 축이 사라지는 현상이 발생하는 것을 의미합니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 각 축으로 어떻게 90도 회전 하는 것에 따라서 위 그림 처럼 다양한 형태로 `Gimbal Lock`이 발생합니다.
- 물론 이와 같이 항상 축이 사라지지는 않으며 어떤 각도로 먼저 회전할 지에 따라서 문제가 되지 않을 수 있습니다. 따라서 `Euler Angle`에서의 `Gimbal Lock` 문제를 회피하기 위한 방법들이 존재 (위 링크 참조) 하지만, 근본적인 해결을 위해서는 다른 방법의 3차원 회전이 필요합니다.

<br>

- ② `Rotaional Interpolation`은 3차원 회전 시 시작 회전과 끝 회전을 지정하면 두 회전 사이를 부드럽게 전환할 수 있어야 함을 의미합니다. 파워포인트부터 다양한 3D 툴에서 회전량을 지정하면 애니매이션 효과를 줄 수 있는데, 이 때 사용되는 중간 중간의 움직임을 의미합니다.
- 이러한 동작을 구현하기 위해서는 경과된 시간에 따라 회전이 변화되도록 중간 회전값을 계산할 수 있어야 하는데 결과적으로 `Euler Angle`에서는 **두 축 이상을 사용하는 `Euler Angle`의 `linear interpolation`을 사용할 수 없습니다.** 1개의 축만 사용하여 회전할 때에는 문제가 없지만 2개의 축 이상을 사용할 때에는 이 부분이 문제가 발생합니다. 이 문제 또한 자세한 문제는 위 링크를 참조하시면 됩니다.

<br>

- 이와 같은 2가지 문제점을 해결하기 위하여 본 글에서는 다루는 `Axis-Angle Rotation`을 사용하거나 `Quaternion`을 사용하여 3차원 회전을 이용합니다. 이 2가지 해결책에도 각각 장단점이 있습니다.
- 본 글에서는 `Axis-Angle Rotation`을 알아볼 것입니다. `Axis-Angle Rotation`은 임의의 축에 직교하는 평면에 대한 임의의 회전을 할 수 있는 컨셉입니다. 따라서 `Euler Angle` 보다 더 유연하게 3차원 회전을 할 수 있습니다.

<br>


## **Axis-Angle Rotation 수식 설명**

<br>

- 앞에서 언급한 바와 같이 `Axis-Angle Rotation`은 임의의 축에 대한 평면의 회전 방식을 이용합니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 3차원 공간에서 임의의 주황색 축을 이용하고 주황색 축에 직교 하는 검은색 점선 평면에서 회전을 하는 방식으로 3차원 회전을 합니다. 벡터의 내적과 외적 연산을 통하여 3차원 회전을 계산할 수 있으며 이 방법에 대하여 알아보도록 하겠습니다.
- 회전축이 되는 주황색 선을 $$ \vec{n} $$, 회전시킬 점을 $$ P $$, 회전할 각을 $$ \theta $$, 최종 회전한 점을 $$ P' $$, 월드 공간의 원점을 $$ O $$, 회전 평면의 중심을 $$ O' $$ 로 정하겠습니다.
- 즉, 회전축은 월드 공간의 원점을 기점으로 뻗은 벡터이며 그 벡터 상의 특정 위치에 직교하는 평면이 생기고 그 평면을 기준으로 $$ \theta $$ 만큼 회전하기 때문에 임의의 방향 및 크기대로 3차원 회전을 할 수 있는 원리 입니다.

<br>

- 지금부터 살펴볼 전개 방식은 위 그림에서 $$ \vec{OO'} + \vec{O'P'} = \vec{u'} $$ 가 되는 구조로 먼저 $$ \vec{OO'} = \vec{v} $$ 를 살펴보고 그 다음 $$ \vec{O'P'} $$ 를 살펴보도록 하겠습니다.
- 이 과정을 통하여 최종적으로 구하고 싶은 $$ \vec{u'} $$ 를 구할 수 있습니다.

<br>

#### **① OO' 벡터 구하기**

<br>

- 위 그림에서 점 $$ P $$ 의 좌표를 동차좌표계로 $$ P = (x, y, z, 1) $$ 로 나타내 보겠습니다.

<br>

- 그러면 다음과 같이 간단하게 $$ \vec{u} $$ 를 구할 수 있습니다.

<br>

- $$ \vec{u} = P - O = (x, y, z, 0) $$

<br>

- 최종적으로 구하고자 하는 벡터는 위 그림에서 $$ \vec{u'} $$ 이고 이 값을 구하기 위해서는 $$ \vec{u} $$ , $$ \vec{n} $$ , $$ \theta $$ 가 필요합니다.
- 임의의 축 $$ \vec{n} $$ 에 대하여 $$ \vec{u} $$ 를 각 $$ \theta $$ 만큼 회전시켜 $$ \vec{u'} $$ 를 계산하는 `Axis-Angle Rotation`의 식은 다음과 같습니다. 아래 식의 $$ \hat{n} $$ 은 $$ \vec{n} $$ 의 크기가 1인 벡터 입니다.

<br>

- $$ \vec{u'} = \cos{(\theta)} \cdot \vec{u} + (1 - \cos{(\theta)}) \cdot (\vec{u} \cdot \hat{n}) \cdot \hat{n} + \sin{(\theta)} \cdot (\hat{n} \times \vec{u}) \tag{1} $$

<br>

- 지금부터는 이 식의 유도 과정을 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 위 그림과 같이 $$ O \to P $$ 로 향하는 $$ \vec{u} $$ 를 $$ O \to O' $$ 로 향하는 $$ \vec{v} $$ 로 만들기 위하여 다음과 같이 수식을 이용하여 만듭니다.

<br>

- $$ \vec{OO'} = \vec{v} = (\vec{u} \cdot \hat{n}) \cdot \hat{n} \tag{2} $$

<br>

- 식 (2)와 같이 전개되는 이유는 아래 링크를 참조하시기 바랍니다.
- [vector projection](https://gaussian37.github.io/math-la-projection/#scalar-projection--vector-projection-1)

<br>

- 따라서 $$ O' \to P $$ 로 향하는 벡터는 $$ \vec{u}, \vec{v} $$ 를 이용하여 $$ \vec{u} - \vec{v} $$ 와 같이 구할 수 있습니다.

<br>

#### **② O'P' 벡터 구하기**

<br>

- 앞에서 다룬 회전 평면을 위에서 아래로 내려다 보면서 회전 동작을 살펴보겠습니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 회전 평면에서 $$ \vec{O'P} $$ 를 $$ \theta $$ 만큼 회전 시킨 $$ \vec{O'P'} $$ 를 구하는 것이 첫번째 목표입니다.
- 먼저 위 그림과 같이 $$ \vec{O'P'} $$ 의 가로 성분은 $$ \cos{(\theta)} \cdot (\vec{u} - \vec{v}) $$ 와 같이 구할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림에서 $$ \vec{n} $$ 과 $$ \vec{O'P'} $$ 의 `cross product`를 통해 $$ \vec{O'Q} $$ 를 구할 수 있습니다. 직교하는 성분이기 때문입니다.

<br>

- $$ \vec{O'Q} = \hat{n} \times (\vec{u} - \vec{v}) \tag{3} $$

<br>

- 지금 까지 확인한 내용을 이용하여 $$ \vec{O'Q} $$ 의 값을 확인해 보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 수식을 전개하여 $$ \vec{O'P'} $$ 를 구하면 다음과 같습니다.

<br>

- $$ \begin{align}\vec{O'P'} &= \cos{(\theta)} \cdot \vec{O'P} + \sin{(\theta)} \cdot \vec{O'Q} \\ &= \cos{(\theta)} \cdot (\vec{u} - \vec{v}) + \sin{(\theta)} \cdot (\hat{n} \times (\vec{u} - \vec{v})) \\ $= \cos{(\theta)} \cdot (\vec{u} - \vec{v}) + \sin{(\theta)} \cdot (\hat{n} \times \vec{u} - \hat{n} \times \vec{v}) \\ $= \cos{(\theta)} \cdot (\vec{u} - \vec{v}) + \sin{(\theta)} \cdot (\hat{n} \times \vec{u}) \end{align} \tag{4} $$

<br>

- 위 식에서 마지막의 $$ \hat{n} \times \vec{u} $$ 가 소거된 이유는 두 벡터가 평행하기 때문에 cross product가 0이 되어서 소거하였습니다.
- 따라서 식을 정리하면 다음과 같습니다.

<br>

- $$ \vec{O'P'} = \cos{(\theta)} \cdot (\vec{u} - \vec{v}) + \sin{(\theta)} \cdot (\hat{n} \times \vec{u}) \tag{5} $$

<br>

#### **③ OP' 벡터 구하기**

<br>
<center><img src="../assets/img/vision/concept/axis_angle_rotation/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 식 (2)의 벡터 $$ \vec{OO'} $$ 와 식 (5)의 벡터 $$ \vec{O'P'} $$ 를 더하면 최종적으로 구하고자 하는 벡터 $$ \vec{OP'} $$ 를 구할 수 있습니다.

<br>

- $$ \vec{OP'} = \vec{v} + \cos{(\theta)} \cdot (\vec{u} - \vec{v}) + \sin{(\theta)} \cdot (\hat{n} \times \vec{u}) \tag{6} $$

<br>

- 마지막으로 $$ \vec{v} = (\vec{u} \cdot \hat{n}) \cdot \hat{n} $$ 를 대입하면 최종적으로 다음과 같습니다.

<br>

- $$ \vec{OP'} = \vec{u'} = \cos{(\theta)} \cdot \vec{u} + (1 - \cos{(\theta)})(\vec{u} \cdot \hat{n}) \cdot \hat{n} + \sin{(\theta)} \cdot (\hat{n} \times \vec{u}) \tag{7} $$

<br>

- 지금까지가 `Axis-Angle Rotation`을 수식적으로 살펴보았습니다. 이와 같은 회전 방식을 고안해 낸 로드리게스의 이름을 따서 로드리게스 회전 이라고도 부릅니다.

<br>

## **Axis-Angle Rotation의 행렬식 표현**

<br>

- 지금까지 살펴본 공식을 통하여 3차원 공간에서 어떻게 임의의 축을 회전하여 3차원 회전하는 지 살펴보았습니다.
- 이번에는 `Axis-Angle Rotation`을 행렬로 나타내는 방법에 대하여 다루어 보도록 하겠습니다.


<br>

## **Axis-Angle Rotation의 Python code**

<br>

<br>


## **Axis-Angle Rotation의 단점**

<br>

<br>





<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>