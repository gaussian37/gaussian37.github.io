---
layout: post
title: matrices transform into the new basis vector set  
date: 2018-09-27 15:00:00
img: math/la/overall.jpg
categories: [math-la] 
tags: [선형대수학, linear algebra, basis, 기저, 기저 변환] # add tag
---

- 이번 글에서는 어떤 좌표계에서 bais였던 벡터들을 다른 좌표계로 변환시키는 방법에 대하여 배워보려고 합니다.
- 자 그러면 2차원의 2개의 basis 벡터를 어떻게 새로운 좌표계로 변환시키는 지 알아보겠습니다. 

<br>

<center><img src="../assets/img/math/la/matrices_transform_into_the_new_basis_vector_set/1.png" alt="Drawing" style="width: 800px;"/></center>

<br>

- 만약 2개의 frame이 있다고 가정해 보겠습니다. 현재 그래프에서 사용하고 있는 frame1은 basis 벡터가 $$ \hat{e_{1}}, \hat{e_{2}} $$로 이루어진 공간입니다.
    - 검은색 화살표로 이루어진 축입니다.
- frame1에서 노란색으로 칠해진 벡터는 frame2의 basis 벡터입니다. 
- frame2의 basis 벡터는 frame2에서는 유닛 벡터 형태의 basis 벡터이지만 frame1에서 표현될 때는 변형 되어서 $$ [1, 1]^{T}, [3, 1]^{T} $$로 표현됩니다.
- 그러면 frame1과 frame2 각각의 basis 벡터에 대하여 알고 있고 frame2의 basis 벡터가 frame1에 어떻게 표현되는 지 알고 있으므로 frame2의 모든 벡터를 frame1에 표현할 수 있습니다.

<br>

- 위 그래프의 갈색 벡터를 보면 frame2에서는 $$ \frac{1}{2}[3, 1]^{T} $$ 였습니다
- 이 벡터가 frame1에서는 어떻게 나타날 수 있을까요? 이 때, 필요한 개념이 transformation 입니다.
- 연산하는 방법은 다음과 같습니다.

$$
    \begin{bmatrix}
    3 & 1 \\
    1 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    \frac{3}{2} \\
    \frac{1}{2} \\
    \end{bmatrix}
=
    \begin{bmatrix} 
    5 \\
    2 \\
    \end{bmatrix}
    
$$
   
- 이 때, 첫번째 행렬인 $$ \begin{bmatrix} 3 & 1 \\ 1 & 1 \\ \end{bmatrix} $$ 은 frame1 상에서의 fram2의 basis 벡터에 해당합니다.
    - 이 행렬을 **transformation matirx**라고 합니다.
- 두번째 항인 벡터 $$ \begin{bmatrix} \frac{3}{2} \\ \frac{1}{2} \\ \end{bmatrix} $$는 frame2 상에서의 벡터입니다.
- 첫번째 항인 행렬과 두번째 항인 벡터의 곱은 변형된 frame1상에서의 frame2 벡터가 됩니다.
- 따라서 연산 과정을 정리하면 frame1 vector = transformation matrix * frame2 vector 

<br>

- 그러면 반대로 frame1상에서의 벡터를 frame2에서 표현할 수 있도록 변형하려면 어떻게 하면 될까요?
- 방법은 간단합니다. 앞에서 구한 transformation matrix의 역행렬을 이용하여 구할 수 있습니다.
- 즉, fram2 vector = inverse of transformation matrix * frame1 vector 가 됩니다.
- 따라서 역행렬인 $$ \begin{bmatrix} 1 & -1 \\ -1 & 3 \\ \end{bmatrix} $$ 가 frame1 → frame2로의 transformation matrix가 됩니다.

<br>

- 이번에는 다른 예제를 한번 살펴보도록 하겠습니다.    



  