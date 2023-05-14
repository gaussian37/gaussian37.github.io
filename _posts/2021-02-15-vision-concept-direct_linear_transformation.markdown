---
layout: post
title: Direct Linear Transformation 
date: 2021-02-15 00:00:00
img: vision/concept/direct_linear_transformation/0.png
categories: [vision-concept] 
tags: [direct linear transformation, DLT] # add tag
---

<br>

- 참조 : http://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf
- 참조 : https://gaussian37.github.io/vision-concept-geometric_transformation/

<br>

- 사전 지식 : [특이값 분해 (Singular Value Decomposition)](https://gaussian37.github.io/math-la-svd/)

<br>

- 이번 글에서는 `Homography` 적용 시 4개의 점을 이용하여 3 X 3 Homography 행렬을 만드는 방법에 대하여 다루어 보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이미지 변환을 할 때, 위 그림과 같이 왼쪽의 이미지를 오른쪽 이미지와 같이 기하학적 변환을 적용하곤 합니다.
- 이 때, 동일 평면 (coplanar) 상의 점들을 3차원 변환을 하기 위하여 `Homography(또는 Perspective Transformation, Projective Transformation)` 방법을 사용합니다.
- 이번 글에서는 `Homography`에 대한 자세한 개념 보다는 두 이미지에서 대응되는 4개의 점을 이용하여 3 X 3 Homography를 구하는 방법에 대하여 다루어 보겠습니다.
- `Homography`에 대한 개념은 아래 링크에서 확인하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/vision-concept-geometric_transformation/](https://gaussian37.github.io/vision-concept-geometric_transformation/)
    - 링크 : [https://gaussian37.github.io/vision-concept-camera_and_geometry/](https://gaussian37.github.io/vision-concept-camera_and_geometry/)

<br>

- $$ \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \alpha H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$

<br>

- $$ H = \begin{bmatrix} h_{1} & h_{2} & h_{3} \\ h_{4} & h_{5} & h_{6} \\ h_{7} & h_{8} & h_{9} \end{bmatrix} $$

<br>

- 먼저 행렬 `H`를 Homography 행렬이라고 합니다. 위 식의 좌변과 우변의 $$ x, y $$ 쌍을 대응해 주기 때문입니다.
- `H`에서 $$ h _{9} $$는 스케일과 관련된 값으로 1 또는 사용할 스케일 값을 사용합니다. 즉, $$ h_{i} $$해를 구할 때, 크게 고려하지 않아도 됩니다.
- 따라서 8개의 파라미터 $$ h_{1}, h_{2}, \cdots h_{8} $$을 구하기 위하여 8개의 식이 필요합니다. 즉, $$ (x, y) $$로 이루어진 4개 점을 통하여 8개의 식을 얻고 8개의 식을 이용하여 파라미터 8개를 연립방정식을 통하여 구할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 Homography의 파라미터를 구하기 위하여 위 식을 homogeneous linear equation 형태로 변형한 뒤 해를 구하는 방법을 사용하도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 우변의 행렬을 전개하여 위 식과 같이 3개의 식으로 풀어 보겠습니다. 그 다음, 좌변이 1인 세번째 식을 첫번째, 두번째 식에 나누어 보겠습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 세번째 식으로 나눈 식에서 분모가 없도록 정리를 하면 위 식과 같이 2개의 식으로 정리할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 때, 우변이 0이 되도록 좌변과 우변을 정리하면 위 식과 같이 정리할 수 있습니다. 이제 우변을 0으로 만들었으므로 homogeneous 형태의 식을 만들 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 식에서 $$ A_{i} $$는 (2 X 9) 크기의 행렬이고 $$ h $$는 (9 X 1) 크기의 행렬이므로 연산 결과 (2 X 1) 크기의 영행렬을 얻을 수 있습니다.
- 이를 확장하여 4개의 좌표 쌍을 사용한다면 $$ A_{1}, A_{2}, A_{3}, A_{4} $$를 사용하여 (8 X 9) 크기의 행렬 $$ A $$를 만들 수 있고 $$ h $$는 (9 X 1) 크기의 행렬이므로 우변은 (8 X 1) 크기의 행렬을 만들 수 있습니다.

<br>

- $$ Ah = 0 $$

<br>

- $$ \begin{bmatrix} -x_{1} & -y_{1} & -1 & 0 & 0 & 0 & x_{1}x_{1}' & y_{1}x_{1}' & x_{1}' \\ 0 & 0 & 0 & -x_{1} & -y_{1} & -1 & x_{1}y_{1}' & y_{1}y_{1}' & y_{1}' \\ -x_{2} & -y_{2} & -1 & 0 & 0 & 0 & x_{2}x_{2}' & y_{2}x_{2}' & x_{2}' \\ 0 & 0 & 0 & -x_{2} & -y_{2} & -1 & x_{2}y_{2}' & y_{2}y_{2}' & y_{2}' \\ -x_{3} & -y_{3} & -1 & 0 & 0 & 0 & x_{3}x_{3}' & y_{3}x_{3}' & x_{3}' \\ 0 & 0 & 0 & -x_{3} & -y_{3} & -1 & x_{3}y_{3}' & y_{3}y_{3}' & y_{3}' \\ -x_{4} & -y_{4} & -1 & 0 & 0 & 0 & x_{4}x_{4}' & y_{4}x_{4}' & x_{4}' \\ 0 & 0 & 0 & -x_{4} & -y_{4} & -1 & x_{4}y_{4}' & y_{4}y_{4}' & y_{4}' \end{bmatrix}  \begin{bmatrix} h_{1} \\ h_{2} \\ h_{3} \\ h_{4} \\ h_{5} \\ h_{6} \\ h_{7} \\ h_{8} \\ h_{9}\end{bmatrix} =  \begin{bmatrix}0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} $$

<br>

- 위 식에서 $$ (x_{i}, y_{i}) $$와 $$ (x_{i}', y_{i}') $$는 변환 전, 변환 후에 대응되는 좌표값으로 실제 값이 입력됩니다.
- 즉, 구해야 하는 미지수는 $$ h_{i} $$ 값이 됩니다. 
- 따라서 이 문제는 `Homogeneous Linear Least Squares` 문제가 되며 `SVD(Singular Value Decomposition)`을 이용하여 풀 수 있습니다.

<br>

- $$ A = U \Sigma V^{T} $$

<br>

- 행렬 A를 `SVD`를 이용하여 분해하였을 때, `Singular Value`가 최소가 되는 `Singular Vector`를 $$ V $$ 행렬에서 선택하면 $$ Ah = 0 $$ 에 가장 근사하는 $$ h $$ 를 구할 수 있습니다.
- `Singular Vector`는 행렬 $$ A $$ 가 선형 변환할 때, 어떤 `basis` 방향으로 얼만큼 늘리거나 줄여서 변환할 지 방향과 크기를 나타냅니다. 따라서 가장 작은 `Singular Value`는 행렬 $$ A $$ 에 의해 변환되는 가장 작은 변화량을 나타냅니다. 이 가장 작은 `Singular Value`에 해당하는 `Singular Vector`는 행렬 $$ A $$ 가 벡터를 0 또는 거의 0 에 가까운 길이로 압축하는 방향을 알려줍니다.
따라서 $$ Ah = 0 $$ 의 해를 구할 때, 가장 작은 `Singular Value`에 해당하는 $$ V $$ 의 벡터가 좋은 후보군이 됩니다. 최대한 압축되어 A의 영공간에 가까워지는 방향이기 때문입니다.
- 이러한 이유로 `Singular Value`가 가장 작은 `Singular Vector`를 구하면 $$ Ah = 0 $$ 을 만족하는 가장 근사한 값을 구할 수 있습니다.
- 따라서 최종적으로 $$ h $$ 벡터를 구하면 기존에 원하는 모양인 행렬 $$ H $$ 로 모양을 맞춰주면 `Homography`를 구할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `Direct Linear Transformation`의 순서를 다시 정리하면 위 절차와 같습니다.