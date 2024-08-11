---
layout: post
title: Triangulation 
date: 2021-02-15 00:00:00
img: vision/concept/triangulation/0.png
categories: [vision-concept] 
tags: [triangluation, stereo, multiple view] # add tag
---

<br>

- 참조 : https://www.cs.cmu.edu/~16385/s17/Slides/

<br>

- 사전 지식 : [Direct Linear Transformation](https://gaussian37.github.io/vision-concept-direct_linear_transformation/)
- 사전 지식 : [특이값 분해 (Singular Value Decomposition)](https://gaussian37.github.io/math-la-svd/)
- 사전 지식 : [카메라 모델 및 카메라 캘리브레이션의 이해와 Python 실습](https://gaussian37.github.io/vision-concept-calibration/)

<br>

## **목차**

<br>

- ### [Triangulation 개념](#triangulation-개념-1)
- ### [Python 실습](#python-실습-1)

<br>

## **Triangulation 개념**

<br>
<center><img src="../assets/img/vision/concept/triangulation/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 표의 3가지 항목인 `Pose Estimation`, `Triangulation`, `Reconstruction` 중 이번 글에서 다룰 내용은 `Triangulation`이며 이 태스크는 도표의 분류와 같이 `Pose Estimation` 또는 `Reconstruction`과 차이가 있음을 알 수 있습니다. 그 내용에 대하여 좀 더 자세하게 살펴보겠습니다.

<br>
<center><img src="../assets/img/vision/concept/triangulation/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `Triangulation`은 2개의 이미지에서 같은 지점을 가리키는 `2D point` 셋과 `카메라 파라미터`가 있을 때, `2D point`의 실제 `3D point`를 추정하는 작업을 의미합니다. 
- `2D Point`는 보통 Feature Extraction & Matching 작업 등과 같은 방식을 통하여 얻기 때문에 일부 오차가 있음을 가정합니다. 또한 `Triangulation`에서는 이미지의 왜곡 보정이 반영되어 있음을 가정합니다.
- 이번 글에서는 다음과 같이 용어 및 기호를 통일하여 사용하도록 하겠습니다.

<br>

- $$ \text{Given a set of (noisy) matched points: } x, x' \tag{1} $$

- $$ \text{Camera projection matrices: } P, P' \tag{2} $$

- $$ \text{Estimated 3D points: } \mathbf{X} = \{X, Y, Z \} \tag{3} $

<br>

- 식 (1)에서 $$ x, x' $$ 각각은 이미지 상에서 서로 같은 지점의 이미지 좌표를 의미합니다.
- 식 (2)에서 $$ P $$ 는 `Camera Extrinsic Matrix`와 `Camera Intrinsic Matrix`를 모두 반영한 `Camera Projection Matrix`를 의미합니다. 상세 내용은 글 상단의 `카메라 캘리브레이션` 링크를 참조해 보시기 바랍니다.
- 식 (3)의 $$ \mathbf{X} $$ 는 식 (1)의 이미지 2D points인 $$ x, x' $$ 가 이미지에 투영되기 이전의 3D points를 의미하며 **최종 추정하고자 하는 값**이 됩니다.

<br>

- $$ x = P\mathbf{X} \tag{4} $$

<br>

- 식 (4)에서 `3D points`인 $$ \mathbf{X} $$ 를 `Camera Projection Matrix`를 통해 이미지에 투영하면 $$ x $$ 가 됩니다. 
- 이 때, $$ \mathbf{X} $$ 를 알고 싶으나 $$ \mathbf{X} $$ 의 후보가 되는 값은 무수히 많이 있습니다. 관련 내용은 아래 링크를 참조하시면 됩니다.
    - 동차좌표계 : https://gaussian37.github.io/vision-concept-homogeneous_coordinate/

<br>
<center><img src="../assets/img/vision/concept/triangulation/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 빨간색 선인 `Ray`에 대응되는 모든 점들이 $$ \mathbf{X} $$ 가 될 수 있으므로 유일한 해를 결정할 수 없습니다. `scale` $$ \alpha $$ 값에 따라 빨간색 선의 어떤 점으로도 대응될 수 있기 때문에 1개의 카메라 만을 이용하면 무수히 많은 해를 가지게 됩니다.

<br>
<center><img src="../assets/img/vision/concept/triangulation/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 반면 이상적인 상황에서 2개의 카메라에서 생성한 `Ray`의 교점을 이용하여 $$ X $$ 의 좌표를 구하는 방법을 사용해 볼 수 있습니다. 이 경우에는 다음과 같은 식을 이용합니다.

<br>

- $$ \begin{cases} x = P\mathbf{X} \\ x' = P'\mathbf{X} \end{cases} \tag{5} $$

<br>

- 앞에서 가정한 바와 같이 $$ x, x' $$ 의 `matched points`는 노이즈를 포함할 수 밖에 없음을 가정하였습니다. 따라서 식(5)에서 완벽히 일치하는 해 $$ X $$ 를 찾기는 어렵습니다. 대신에 가장 적합한 해를 찾는 최적화 방법을 이용해야 합니다.
- 한 개의 점 $$ \mathbf{X} = \{X, Y, Z\} $$ 에 대하여 찾고자 하는 값은 3개이므로 최소 3개식을 이용해야 $$ X, Y, Z $$ 를 추정할 수 있습니다.

<br>

- 식을 구하기 위하여 **2개의 `Ray` $$ x $$ 와 $$ \alpha P\mathbf{X} $$ 가 서로 평행하다는 조건을 이용**하도록 하겠습니다. 2개의 `Ray`가 평행하다면 `cross product`가 0이 되기 때문에 이 값을 이용하면 식을 구할 수 있습니다.

<br>

- 먼저 아래와 같이 $$ x = \alpha P \mathbf{X} $$ 를 전개해 보도록 하겠습니다.

<br>

- $$ x = P\mathbf{X} \quad \text{(homogeneous coordinate)} $$

- $$ \Rightarrow x = \alpha P\mathbf{X} \quad \text{(inhomogeneous coordinate)} \tag{6} $$

- $$ \Rightarrow \begin{align} \begin{bmatrix} x \\ y \\ z \end{bmatrix} &= \alpha \begin{bmatrix} p_{11} & p_{12} & p_{13} & p_{14} \\ p_{21} & p_{22} & p_{23} & p_{24} \\ p_{31} & p_{32} & p_{33} & p_{34} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} \\ &= \alpha \begin{bmatrix} P_{1}^{T} \\ P_{2}^{T} \\ P_{3}^{T}\end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} \end{align} \tag{7} $$

- $$ \begin{cases} P_{1}^{T} = \begin{bmatrix} p_{11} & p_{12} & p_{13} & p_{14} \\ \end{bmatrix} \\ P_{2}^{T} = \begin{bmatrix} p_{21} & p_{22} & p_{23} & p_{24} \\ \end{bmatrix} \\ P_{3}^{T} = \begin{bmatrix} p_{31} & p_{32} & p_{33} & p_{34} \\ \end{bmatrix}\end{cases} \tag{8}  $$

<br>

- 2개의 `Ray` $$ x $$ 와 $$ \alpha P\mathbf{X} $$ 가 평행하기 때문에 다음 식을 만족합니다. `scale factor` $$ \alpha $$ 는 아래 식에 영향이 없으므로 제거 하였습니다.

<br>

- $$ x \times P\mathbf{X} = 0 \tag{9} $$

<br>
<center><img src="../assets/img/vision/concept/triangulation/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- $$ a^{T} = \begin{bmatrix} a_{1} & a_{2} & a_{3} \end{bmatrix} \tag{10} $$

- $$ b^{T} = \begin{bmatrix} b_{1} & b_{2} & b_{3} \end{bmatrix} \tag{11} $$

- $$ a \times b = \begin{bmatrix} a_{2}b_{3} - a_{3}b_{2} \\ a_{3}b_{1} - a_{1}b_{3} \\ a_{1}b_{2} - a_{2}b_{1} \end{bmatrix} \tag{12} $$

- $$ \text{Cross product of two vectors in the same direction is zero: } a \times a = 0 \tag{13} $$

<br>

- 식 (7)을 이용하여 전개하면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \alpha \begin{bmatrix} P_{1}^{T} \\ P_{2}^{T} \\ P_{3}^{T}\end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \alpha \begin{bmatrix} P_{1}^{T}\mathbf{X} \\ P_{2}^{T}\mathbf{X} \\ P_{3}^{T}\mathbf{X} \end{bmatrix} \tag{13} $$

- $$ \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \alpha \begin{bmatrix} P_{1}^{T}\mathbf{X} \\ P_{2}^{T}\mathbf{X} \\ P_{3}^{T}\mathbf{X} \end{bmatrix} \tag{14} $$

<br>

- 식 (14)에서 좌변의 $$ z $$ 는 `scale`에 해당하므로 `cross proudct` 식을 적용할 때, `normalized scale`인 $$ z = 1 $$ 로 두고 $$ \alpha $$ 는 소거해도 `cross product` 식 전개에는 영향이 없습니다. 따라서 다음과 같이 식을 전개할 수 있습니다.

<br>

- $$ \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} P_{1}^{T}\mathbf{X} \\ P_{2}^{T}\mathbf{X} \\ P_{3}^{T}\mathbf{X} \end{bmatrix} = \begin{bmatrix} y P_{3}^{T}\mathbf{X} - P_{2}^{T}\mathbf{X} \\ P_{1}^{T}\mathbf{X} - x P_{3}^{T}\mathbf{X} \\ xP_{2}^{T}\mathbf{X} - yP_{1}^{T}\mathbf{X}\end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} \tag{15} $$

<br>

- 식 (15)에서 3행은 1행과 2행의 선형 결합으로 이루어져 있습니다. (1행에 $$ x $$ 를 곱하고 2행에 $$ y $$ 를 곱하여 더하면 3행이 도출됩니다.) 따라서 1행과 2행의 식만 이용할 수 있습니다. 즉, 식 (15)를 통해 2개의 `2D point`와 `3D point` 간의 대응 관계를 구할 수 있습니다.

<br>

- $$ \begin{bmatrix} y P_{3}^{T}\mathbf{X} - P_{2}^{T}\mathbf{X} \\ P_{1}^{T}\mathbf{X} - x P_{3}^{T}\mathbf{X} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \tag{16} $$

- $$ \begin{bmatrix} y P_{3}^{T} - P_{2}^{T} \\ P_{1}^{T} - x P_{3}^{T} \end{bmatrix} \mathbf{X} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \tag{17} $$

<br>

- 같은 방식으로 $$ x' \times P'\mathbf{X} = 0 $$ 을 이용한 후 식(17)의 좌변 행렬에 누적시키면 다음과 같이 식을 구성할 수 있습니다.

<br>

- $$ \begin{bmatrix} y P_{3}^{T} - P_{2}^{T} \\ P_{1}^{T} - x P_{3}^{T} \\ y' P_{3}^{'T} - P_{2}^{'T} \\ P_{1}^{'T} - x' P_{3}^{'T}\end{bmatrix} \mathbf{X} = \begin{bmatrix} 0 \\ 0 \\0 \\ 0 \end{bmatrix} \tag{18} $$

<br>

- 식 (18)의 좌변을 간단히 다음과 같이 나타낼 수 있습니다.

<br>

- $$ A\mathbf{X} = 0 \tag{19} $$

<br>

- 식 (19)에서 $$ \mathbf{X} $$ 이 `trivial solution`이 아닌 해를 구하려면 [특이값 분해 (Singular Value Decomposition)](https://gaussian37.github.io/math-la-svd/)를 이용하여 구할 수 있습니다. 상세 내용은 아래 링크에서 확인할 수 있습니다.
    - 링크 : [SVD 활용](https://gaussian37.github.io/math-la-svd/#svd%EC%9D%98-%ED%99%9C%EC%9A%A9-1)

<br>

## **Python 실습**

<br>

