---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 1. 2D and 1D projective geometry
date: 2022-04-20 00:00:01
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [(멀티플 뷰 지오메트리). Multiple View Geometry] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/LAHQ_qIzNGU?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/gQ7IUS8NKCI?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision
- 참조 : https://www.cuemath.com/learn/mathematics/conics-in-real-life/

<br>

- 이번 글에서는 **2D and 1D projective geometry** 내용의 강의를 듣고 정리해 보도록 하겠습니다.

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/T-p6d7av32Y" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

- 지금 부터는 **2D and 1D projective geometry** 강의의 후반부로 conics, dual conics와 관련된 내용과 transform 관련 내용에 대하여 다루어 보도록 하겠습니다.

<br>
<br>
<br>

- `projective plane`이란 3차원 공간에서 원점을 지나는 모든 직선들의 모임으로 해석할 수 있습니다. 이 관점에서 `projective plane`의 `point`는 원점을 지나는 각각의 직선 (`line`)이고 `line`은 원점을 지나는 3차원 공간 속의 2차원 평면 (`plane`)으로 정의할 수 있습니다.
- `projective plane`은 일반적인 plane과 유사하지만, `point at infinity`라는 `무한대의 점`이 존재하여 모든 두 직선이 항상 교차가 되는 특성이 있습니다. 모든 `point at infinity` 들이 지나는 직선을 `line at infinity`라고 합니다.

<br>
<br>
<br>


<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/gQ7IUS8NKCI" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/46.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- point $$ x_{i} $$ 가 line $$ l $$ 위에 있을 때, $$ l^{T} x_{i} = 0 $$ 으로 표현할 수 있습니다. 만약 transformed point (`projective transformation`) 인 $$ x_{i}' = H x_{i} $$ 가 $$ l' $$ 위에 있다면 $$ {l'}^{T} x_{i}' = 0 $$ 이 되고 $$ l $$ 과 $$ l' $$ 두 line의 관계로 나타내면 $$ l' = H^{-T} l $$ 으로 표현할 수 있습니다. 수식 전개 과정은 아래와 같습니다.

<br>

- $$ x_{i}' = H x_{i} $$

- $$ {l'}^{T} x_{i}' = 0 $$

- $$ \therefore \quad {l'}^{T} H x_{i} = 0 $$

- $$ {l'}^{T} H x_{i} = l^{t} x_{i} $$

- $$ {l'}^{T} H = l^{t} $$

- $$ H^{T} l' = l $$

- $$ \therefore \quad l' = H^{-T} l $$

<br>


<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/50.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `Similarity Transformation`은 `isotropic scaling` 으로 구성되며 위 슬라이드의 식 처럼 표현됩니다.
- `isotropic`이란 한글로 등방형이며 방향에 상관없이 일정하다는 뜻입니다. 즉, `Similarity Transformation`은 모든 방향에 동일한 효과를 적용합니다.
- 위 식에서 $$ s $$ 는 `isotropic scaling`이라고 하며 Similarity Transformation 적용 시 변환의 크기를 조절합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/51.png" alt="Drawing" style="width: 8000px;"/></center>
<br>

- `Similarity Transformation`은 `shape`을 보존하기 때문에 `equi-form transformation`이라고도 합니다.
- 식에서 나타난 바와 같이 3개의 `isometry` DoF ( $$ \theta, t_{x}, t_{y} $$ )와 `isotropic scaling` $$ s $$ 를 가지므로 `Similarity Transformation`은 총 4 DoF를 가지며 **DoF가 4개이므로** 변환된 2개의 점을 알 때, 이 값들을 추정할 수 있습니다.
- `shape`이 보존되기 때문에 `Angle`, `ratio of two lengths`, `ratio of areas`는 보존이되는 성질을 가집니다. **평행선은 평행선으로 유지되는 것 또한 중요한 성질**입니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/52.png" alt="Drawing" style="width: 8000px;"/></center>
<br>

- 그 다음으로 `Affinity`에 대하여 다루어 보도록 하겠습니다.
- 위 슬라이드에서 설명하는 `Affine Transformation` 행렬은 `non-singular linear transformation`으로 간단히 말하면 역행렬이 존재하는 행렬이며 $$ A $$ 부분의 2 X 2 행렬 또한 `non-singular matrix`로 역행렬이 존재합니다.
- 앞에서 살펴본 `Similarity Transformation`과 다르게 6 DoF로 DoF가 2개가 더 추가가 되었습니다. Similarity Transformation에서는 $$ \theta $$ 가 정해지면 $$ H_{S} $$ 가 정해졌으나 `Affine Transformation`에서는 4개의 각 성분이 모두 DoF를 가지기 때문에 DoF가 총 6개가 됩니다.
- **DoF가 6개이므로** 기존의 3개의 점이 `Affine Transformation` 적용 시 어떻게 변환되는 지 관계를 알면 `Affine Transformation`을 구할 수 있습니다. (Similarity Transformation에서는 점 2개가 필요하였습니다.)
- `Affine Transformation`을 적용하더라도 **평행선은 그대로 유지**되며 **평행 선분의 길이 비율과 면적 비율은 유지**됩니다. 반면 Similarity Transformation에서는 보존되었던 임의의 선의 길이 비율과 선 사이의 각도는 보존되지 않습니다. 그 이유에 대하여 `Affine Transformation Matrix`를 분해하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/53.png" alt="Drawing" style="width: 8000px;"/></center>
<br>

- 이번 슬라이드와 다음 슬라이드에서는 `Affine Transformation`을 `SVD` (Singular Value Decomposition)으로 분해하였을 때, 각 성분이 가지는 의미를 나타냅니다. 위 슬라이드에서는 먼저 어떻게 분해될 수 있는 지 설명합니다.

<br>

- `Affine Transformation`의 의미를 기하학적으로 이해하기 위한 방법으로 `rotation`과 `non-isotropic scaling` 두 가지 선형 변환의 합성으로 생각하는 방법이 있습니다. 먼저 `Affine Transformation`은 아래와 같이 분해 됩니다.

<br>

- $$ A = R(\theta) R(-\phi) D R(\phi) $$

- $$ R(\theta), R(\phi) \text{ : rotation by} \theta, \phi \text{ respectively} $$

- $$ D = \begin{bmatrix} \lambda_{1} & 0 \\ 0 & \lambda_{2} \end{bmatrix} \text{ : diagonal matrix} $$

<br>

- 위 식과 같이 분해되는 이유는 아래와 같습니다. `affine transformation matrix`를 Singular Value Decomposition을 하고 SVD 결과의 $$ U, V $$ 가 `orthogonal matrix`이므로 $$ U^{-1} = U^{T} $$ , $$ V^임을{-1} = V^{T} $$ 임을 이용하여 전개하였습니다.

<br>

- $$ A = U D V^{T} = U (V^{-1}V) D V^{T} = U (V^{T}V) D V^{T} = (U V^{T})V D V^{T} $$

- $$ = R(\theta)(R(-\phi) D R(\phi)) $$

<br>

- [회전 행렬 관련 글](https://gaussian37.github.io/math-la-rotation_matrix)에서 다룬 바와 같이 `orthogonal matrix`는 rotation 임을 만족하기 때문에 $$ U = R(\theta) $$, $$ V = R(-\phi) $$, $$ V^{T} = R(\phi) $$ 로 표현할 수 있어서 위 식과 같이 전개됩니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/54.png" alt="Drawing" style="width: 8000px;"/></center>
<br>

- 따라서 식을 위 슬라이드와 같이 시각적으로 나타낼 수 있습니다. 위 슬라이드는 `affine transformation`인 $$ A $$ 를 바로 적용한 결과를 $$ R(\phi) $$ , $$ D $$ , $$ R(\theta)R(\-phi) $$ 순서로 나누어서 보여줍니다.
- 즉 `affine transformation` $$ A $$ 에서의 연산 순서는
    - ① $$ \phi $$ 만큼 회전합니다.
    - ② $$ x, y $$ 방향으로 각각 $$ \lambda_{1}, \lambda{2} $$ 만큼 `scaling`을 조정합니다. ($$ \lambda_{1}x_{1} + \lambda{2}x_{2} $$ )
    - ③ $$ -\phi $$ 만큼 역회전 합니다. 즉, $$ \phi $$ 만큼 회전한 영역에서 주성분 방향으로 `scaling`을 조정하고 다시 역회전하여 회전을 없앱니다.
    - ④ 마지막으로 $$ \theta $$ 만큼 회전합니다.
- 이와 같은 순서로 연산을 살펴보았을 때, `similarity transformation`에 비하여 추가된 개념은 `non-isotropic scaling`입니다. 즉, `scaling` 조정 방향을 지정하는 각도 $$ \phi $$ 와 scaling 조정 비율인 $$ \lambda{1}, \lambda{2} $$ 가 이에 해당합니다.
- 따라서 `affine transformation`에서는 **특정 각도에 대하여 직교하는 방향으로 scaling을 조정하는 것이 중요합니다.**

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/53_1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 그림으로 나타내면 왼쪽 그림은 최종 $$ R(\theta) $$ 에 의한 회전을 나타내고 오른쪽 그림은 $$ R(-\phi)DR(\phi) $$ 에 의한 변형을 나타냅니다. `scaling` 방향으로 `orthogonal` 함을 유심히 살펴보면 이해하는 데 도움이 됩니다.



<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>