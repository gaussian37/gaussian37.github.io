---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 7. The fundamental and essential matrices
date: 2022-04-20 00:00:07
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [멀티플 뷰 지오메트리, Multiple View Geometry, The fundamental and essential matrices] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/eJnG1vwGJkE?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/mNThwULGR-g?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/7QYq7qNkmtg?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/WQvJICS3Ecc?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

- Lecture 7에서는 `fundamental matrix`와 `essential matrix` 내용을 다룹니다. 이 개념들을 통하여 카메라 간의 관계를 정의하기 위한 기본적인 개념을 익힐 수 있습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/eJnG1vwGJkE" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 먼저 본 강의 내용에 앞서 다음 내용을 간략히 숙지하면 이해하는 데 도움이 됩니다.

<br>

- `epipolar geometry`란 2개이상의 카메라에서 카메라 간의 관계를 추정하는 것으로 생각할 수 있습니다.
- 만약 카메라가 2개라면 스테레오 비전 또는 2-view라고 하며 스테레오 비전에서의 두 카메라의 관계를 `epipolar geometry`로 표현할 수 있습니다. 아래 그림과 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림에서 동일한 3차원의 점인 $$ P $$ 를 서로 다른 두 카메라에서 획득하였을 때, 매칭 쌍 $$ (p, p') $$ 사이의 기하학적 관계를 다룹니다. 먼저 위 그림에 각 기호에 대한 설명을 하면 다음과 같습니다.
- 　$$ P(P') $$ : 이미지 상에 맺힐 3차원 공간 상의 점을 의미합니다.
- 　$$ C, C' $$ : 각 영상의 카메라 센터점을 의미합니다.
- 　$$ \text{base line} $$ : 카메라의 센터점을 이은 선을 `base line`이라고 합니다. $$ C, C' $$ 의 거리가 가까운 경우를 `narrow base line`이라고 하고 거리가 먼 경우를 `wide base line`이라고 합니다.
- 　$$ p, p' $$ : 각 영상에서 $$ P(P') $$ 가 투영된 점을 의미합니다.
- 　$$ e, e' $$ : 반대 영상의 카메라 센터점에서 해당 영상의 이미지 상에 맺힌 점을 의미합니다. 이 점을 `epipole` 이라고 합니다.
- 　$$ l, l' $$ : 각 영상에서 `epipole` ( $$ e, e' $$ )과 이미지 상의 점 ( $$ p, p' $$ )를 이은 선을 의미합니다. 이 선을 `epolar line` 이라고 합니다.

<br>

- 위 그림에서 $$ C $$ 와 $$ C' $$ 간의 3차원 위치 관계인 $$ [R \vert T] $$ 와 $$ p $$ 를 알더라도 3차원 공간 상의 점 $$ P $$ 에 대한 실제 깊이 (`depth`) 를 알지 못하면 유일한 $$ p' $$ 를 결정하지 못합니다.
- 반면에 $$ P $$ 는 $$ C $$ 와 $$ p $$ 를 잇는 `ray` 상에 존재하므로 이 선이 반대 영상에 투영된 `epipolar line` $$ l' $$ 은 유일하게 존재합니다.
- 이 때, $$ A $$ 이미지에서 $$ B $$ 이미지로 대응되는 `epipolar line` ( $$ l' $$ ) 의 관계를 나타내는 행렬이 $$ F, E $$ 이며 각각 `Fundamental Matrix`, `Essential Matrix`라고 합니다.

<br>

- `Essential Matrix` $$ E $$ 는 `normalized image plane` 에서의 매칭쌍들 사이의 기하학적 관계를 설명하는 행렬을 의미하고 `Fundamental Matrix` $$ F $$ 는 카메라 파라미터 까지 포함한 두 이미지의 실제 픽셀 좌표 사이의 기하학적 관계를 표현하는 행렬을 의미합니다. 따라서 $$ E $$ 는 $$ F $$ 의 특수한 형태라고 생각할 수 있습니다.

<br>

- 두 이미지 평면 간의 기하학적 관계가 $$ E, F $$ 가 주어지고 두 이미지 평면상의 매칭쌍 $$ p, p' $$ 가 주어질 때, 3D 공간 상의 좌표 $$ P $$ 를 결정할 수 있습니다. 따라서 스테레오 비전에서의 깊이 ('depth')를 구할 수 있습니다.

<br>

- 상세 내용은 본 강의 내용을 통하여 자세하게 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/5.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/6.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/7.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/12.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/14.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/15.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/16.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/17.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/18.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/19.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/20.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/21.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/22.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 하나의 3D 물체를 두 이미지에 각각 투영하여 얻은 점을 $$ x, x' $$ 라고 하였을 때, 지금까지 살펴본 `Fundamental Matrix` $$ F $$ 와 $$ x, x' $$ 를 이용하여 다음과 같은 식을 나타낼 수 있습니다.

<br>

- $$ x'^{T} F x = 0 $$

<br>

- 이와 같이 `Fundamental Matrix`를 통하여 위 식과 같은 조건을 만족하면 $$ x, x' $$ 를 `corresponding points` 관계라고 말할 수 있습니다. 위 식이 만족하는 이유는 슬라이드의 Proof 부분을 보면 쉽게 이해할 수 있습니다. 위 관계를 성립하기 위하여 $$ l' = Fx $$ `epipolar line`을 이용합니다. 간략히 살펴보면 다음과 같습니다.

<br>

- ① **Definition of $$ l' $$**: 두번째 이미지에서의 `epipolar line` $$ l' $$ 는 첫번째 이미지의 포인트 $$ x $$ 와 $$ F $$ 행렬을 통해 만들어집니다. $$ l' = Fx $$
- ② **Point on the Line** : $$ x' $$ 는 $$ x $$ 의 `corresponding point` 이기 때문에 반드시 `epipolar line` $$ l' $$ 상에 있어야 합니다. `homogeneous coordinate`에서 $$ x' $$ 가 $$ l' $$ 상에 있으면 $$ x'^{T}l' = 0 $$ 으로 표현할 수 있습니다.
- ③ **Substitue $$ l'$$** : $$ x'^{T}l' = x'^{T}(Fx) = 0 $$

<br>

- 식 $$ x'^{T} F x = 0 $$ 은 `epipolar constraint` 라고도 불립니다. `epipolar constraint`는 첫번째 이미지에서의 임의의 점 $$ x $$ 와 $$ F $$ 가 두번째 이미지에서 만들어내는 `epipolar line` 상에 두번째 이미지에서 $$ x $$ 와 `corresponding point` 관계인 $$ x' $$ 가 존재한다는 조건입니다.
- 본 글의 뒷부분에서 살펴볼 내용으로 만약 카메라가 캘리브레이션이 완료되었다면 $$ F $$ 는 $$ E $$ 즉, `essential matrix`로 변환하여 사용할 수 있습니다. 이와 같은 경우 $$ x'^{T} E x = 0 $$ 으로 표현할 수 있습니다.
- `epipoelar constraint`는 `multiple view geometry`의 가장 기본적이면서도 중요한 개념입니다. `corresponding points`를 이용하여 `Stereo Matching`, `3D reconstruction`과 같은 분야를 접근할 수 있기 때문입니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/23.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- `epipolar constraint` $$ x'^{T} F x = 0 $$ 의 관계를 구하기 위해서는 `Fundamental Matrix` $$ F $$ 를 알아야 합니다. $$ F $$ 를 알기 위해서는 카메라 파라미터 정보와 상관 없이 `corresponding points`인 $$ x, x' $$ 의 관계들만을 필요로 합니다. 
- 그러면 `Fundamental matrix` $$ F $$ 를 구하기 위하여 필요한 `corresponding points`의 갯수에 대하여 간략히 알아보겠습니다. 상세 내용은 본 글의 뒷부분에서 다룰 예정입니다.

<br>

- 참조 : https://stackoverflow.com/questions/49763903/why-does-fundamental-matrix-have-7-degrees-of-freedom

<br>

- `Fundamental Matrix` $$ F $$ 는 $$ 3 \times 3 $$ 크기를 가지는 행렬이므로 9 개의 값을 가집니다. 따라서 9개의 자유도를 가지기 때문에 9 개의 `corresponding points`가 필요한 것으로 보이지만 `Fundamental Matrix`는 7개의 자유도인 `7-DOF`를 가지므로 7개의 `corresponding points`를 필요로 합니다.
- 7개의 자유도를 가지는 이유는 `Fundamental Matrix`의 `scale ambiguity`와 `rank-2 constraint`와 연관되어 있습니다.
- `scale ambiguity`는 스케일 값에 무관함을 나타내며 $$ F $$ 와 $$ \alpha F (\alpha \ne 0 ) $$ 가 같은 의미를 가지는 행렬이므로 이러한 `ambiguity`가 발생합니다. 이 컨셉은 `homogeneous coordinate`에서 충분히 많이 다룬 개념입니다. 수식으로 좀 더 자세히 살펴보면 다음과 같습니다.

<br>

- $$ x = \begin{matrix} u & v & 1 \end{bmatrix}^{T} $$

- $$ x' = \begin{matrix} u' & v' & 1 \end{bmatrix}^{T} $$

- $$ F = \begin{bmatrix} f_{1} & f_{2} & f_{3} \\ f_{4} & f_{5} & f_{6} \\ f_{7} & f_{8} & f_{9} \end{bmatrix} $$

- $$ x'^{T} F x  = 0 $$

<br>

- 위 식을 풀어서 정리해 보면 다음과 같습니다.

<br>

- $$ uu'f_{1} + vu'f_{2} + u'f_{3} + uv'f_{4} + vv'f_{5} + v'f_{6} + uf_{7} + vf_{8} + f_{9} = 0 $$

<br>

- 위 식을 $$ AF = 0 $$ 형태로 나타내면 $$ A $$ 와 $$ F $$ 는 다음과 같습니다.

<br>

- $$ AF = 0 $$

- $$ A = \begin{bmatrix} uu' & vu' & u' &  uv' & vv' & v' & u & v & 1 \end{bmatrix} $$

- $$ F = \begin{bmatrix} f_{1} & f_{2} & f_{3} & f_{4} & f_{5} & f_{6} & f_{7} & f_{8} & f_{9} \end{bmatrix} $$

<br>

- 행렬 $$ A $$ 를 살펴보면 8개의 미지수가 있고 위 예시에서는 1이라는 상수가 한개 있습니다. 이 값이 1이기 때문에 이미 미지수를 풀 수 있는 하나의 정보는 주어졌고 이 정보가 `scale`에 관한 정보입니다.
- 만약 1이 아니라 $$ \alpha $$ 라는 값이 적용 된다면 $$ A $$ 의 모든 원소에 $$ \alpha $$ 를 곱하여 $$ AF = 0 $$ 을 만족시킬 수 있습니다. 따라서 이 값을 통해 `scale ambiguity`의 성질을 만족시킬 수 있습니다.

<br>

- 두번째로 `rank-2 constraint`에 관한 것입니다. $$ l' = Fx $$ 를 통하여 임의의 점 $$ x $$ 에 $$ F $$ 를 곱하면 선 형태인 `epipolar line`을 얻을 수 있음을 확인하였습니다. 즉, $$ F $$ 는 모든 column 또는 row 성분이 independent 한 `full rank` 행렬이 아니기 때문에 점이 선으로 변환될 수 있습니다. (만약 `full rank` 행렬이면 $$ l' = Fx $$ 를 얻을 수 없고 $$ Fx $$ 는 포인트가 되어야 합니다.)
- `full rank`가 아닌 행렬은 `determinant`가 0이 되기 때문에 다음과 같이 식을 전개해 볼 수 있습니다.

<br>

- $$ F = \begin{bmatrix} f_{1} & f_{2} & f_{3} & f_{4} & f_{5} & f_{6} & f_{7} & f_{8} & f_{9} \end{bmatrix} $$

- $$ \text{det}(F) = (f_{1}*f_{5}*f_{8})+(f_{2}*f_{6}*f_{7})+(f_{3}*f_{4}*f_{8})-(f_{3}*f_{5}*f_{7})-(f_{2}*f_{4}*f_{9})-(f_{1}*f_{6}*f_{8}) \quad \text{by determinant formula} = 0 $$

<br>

- 식 $$ \text{det}(F) = 0 $$ 에서 단 하나의 미지수를 제외하고 다른 미지수의 해를 구하면 나머지 하나의 미지수는 자동으로 결정되기 때문에 자유도가 하나 줄어들게 됩니다.

<br>

- 추가적으로 `rank 2`가 가지는 의미와 `rank 1`을 가지면 `epipolar constrain`을 만족할 수 없음을 보이면 다음과 같습니다.

<br>

- 지금까지 살펴본 `Fundamental Matrix`의 성질에 대하여 몇가지 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/24.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 먼저 `Transpose` 성질에 대하여 살펴보도록 하겠습니다. $$ F $$ 는 $$ (P, P') $$ 와 $$ (P', P) $$ 양쪽에 모두 적용될 수 있습니다. $$ (P, P') $$ 에 대한 관계를 $$ F $$ 로 나타날 때, $$ (P', P) $$ 는 $$ F^{T} $$ 로 나타낼 수 있습니다. 이러한 대칭 관계가 성립하는 이유는 `epipolar geometry`에서 두 이미지 간의 역 관계가 성립하기 때문입니다. 역관계를 살펴보면 다음과 같습니다.

<br>

- $$ x'^{T} F x = 0 $$

- $$ (x'^{T} F x)^{T} = 0 $$

- $$ (x'^{T} F x)^{T} = x^{T} F^{T} x' = 0 $$

<br>

- 다음으로 `Epipolar lines` 성질에 대하여 살펴보도록 하겠습니다. 관련 내용은 앞에서 다룬 내용입니다. $$ l' = Fx $$ 로 정의되는 `epipolar line`은 두번째 이미지에 형성되며 첫번째 이미지의 포인트 $$ x $$ 에 대응됩니다.
- 반대 방향으로 $$ l = F^{T} x' $$ 로 식을 적용할 수 있습니다. $$ l $$ 은 첫번째 이미지에 형성되는 `epipolar line`이고 두번째 이미지의 포인트 $$ x' $$ 에 대응됩니다.
- 2가지 방향 모두 아래 식과 같이 `epipolar constrain`을 통해 정의될 수 있습니다.

<br>

- $$ x'^{T} F x = (x'^{T}) (F x) = (x'^{T}) l' = 0 $$

- $$ x^{T} F^{T} x' = (x^{T}) (F^{T} x') = (x^{T}) l = 0 $$

<br>

- 다음으로 `Epipole` 성질에 대하여 살펴보도록 하겠습니다.
- 두번째 이미지의 임의의 `epipolar line` $$ l' = Fx $$ 은 `epipole` $$ e' $$ 를 포함하고 있습니다. 왜냐하면 두번째 이미지의 `epipole`은 첫번째 이미지를 촬영한 카메라의 센터점이기 때문에 $$ l' $$ 는 항상 $$ e' $$ 에 수렴하게 되어있습니다. 이 원리는 반대 방향도 동일하게 적용됩니다.
- 다음으로 `Epipole`과 `Fundamental Matrix`의 관계를 살펴보면 다음과 같습니다. 다음 수식을 살펴보겠습니다.

<br>

- $$ e'^{T} (Fx) = (e'^{T}F)(x) = 0 $$

<br>

- `epipole` $$ e' $$ 는 `epipolar line` $$ l' $$ 상에 있기 때문에 위 식과 같이 적을 수 있습니다. 그리고 $$ x $$ 는 영벡터가 아닌 유효한 이미지 포인트이기 때문에 $$ (e'^{T}F) = 0 $$ 를 만족해야 위 식을 만족시킬 수 있습니다.
- 따라서 $$ e' $$ 는 $$ F $$ 에 대하여 `left null-vector`가 됩니다. 

<br>

- 반대 방향으로 적용하면 다음과 같습니다.

<br>

- $$ e^{T} F^{T} x' = x'^{T} F e =  x'^{T} (F e) = 0 $$

<br>

- 같은 논리로 $$ e $$ 는 $$ F $$ 에 대하여 `right null-vector`가 됩니다. 

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/25.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 앞에서 설명한 바와 같이 $$ F $$ 는 `7-DOF`를 가집니다. 9개의 elements 중 `scale ambiguity`와 `Rank-2 constraint`로 2개의 DOF가 제외되어 최종적으로 `7-DOF`를 가지게 됩니다.

<br>

- `Fundamental Matrix`는 임의의 포인트 $$ x $$ 를 `epipolar line` $$ l' $$ 로 변환하는 행렬입니다. 즉, 점을 선으로 매핑하는 관계를 가지게 됩니다.
- 첫번째 카메라에서 형성되는 $$ l $$ 과 두번째 카메라에서 형성되는 $$ l' $$ 은 `corresponding epipolar lines` 라고 부르며 $$ l $$ 선상에 있는 임의의 점 $$ x $$ 를 $$ F $$ 통해 매핑하였을 때, $$ l' $$ 선상에 위치하게 됩니다. 물론 그 반대 관계도 성립합니다.
- 이와 같은 점 → 선으로의 매핑 관계를 가지므로 역관계는 가질 수 없습니다. (1:1 대응이 되지 않기 때문입니다.) 이와 같은 조건이 $$ F $$ 가 `full rank`가 될 수 없는 이유이기도 합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/26.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 지금까지 살펴본 `Fundamental Matrix` 내용을 살펴보면 위 표와 같습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/mNThwULGR-g" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/7QYq7qNkmtg" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/WQvJICS3Ecc" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
