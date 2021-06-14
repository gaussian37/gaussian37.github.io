---
layout: post
title: 2 x 2 행렬의 고유값과 고유벡터 공식
date: 2019-12-10 00:00:00
img: math/la/2_by_2_eigen/0.png
categories: [math-la] 
tags: [linear algebra, 고유값, 고유벡터, 고유값 공식] # add tag
---

<br>

- 출처 : http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html

<br>

- 이번 글에서는 2 x 2 행렬에서만 사용할 수 있는 간단한 공식 방식의 고유값, 고유행렬을 구하는 방법에 대하여 알아보도록 하겠습니다.

<br>

- 위 글에 따르면 2 x 2, 3 x 3, 4 x 4 행렬의 경우에는 closed form 형태의 고유값, 고유벡터의 값을 구하는 방법이 있다고 합니다.
- 확인해 보니 2 x 2의 경우에는 깔끔하지만 3 x 3 이상의 경우에는 closed form의 형태도 상당히 복잡해서 일반적인 방식으로 구하는 것이 나을 것 같습니다. 울프람 알파를 통해 확인한 3 x 3 행렬에서의 고유값, 고유벡터의 값을 구하는 방법은 아래 그림을 참조하시기 바랍니다. 상당히 복잡해서 스킵하겠습니다.

<br>
<center><img src="../assets/img/math/la/2_by_2_eigen/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 그럼 본론으로 2 x 2 행렬에서의 고유값, 고유벡터를 구하는 방식을 살펴보겠습니다.

<br>

- $$ A = \begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} $$

<br>

- $$ T = a + d, \ \ D = ad - bc $$

<br>

- $$ L_{1} = \frac{T}{2} + (\frac{T^{2}}{4} - D)^{0.5} $$

<br>

- $$ L_{2} = \frac{T}{2} - (\frac{T^{2}}{4} - D)^{0.5} $$

<br>

- 여기서 고유값은 $$ L1, L2 $$ 입니다.
- 만약 행렬 $$ A $$의 $$ c $$가 0이 아니면 고유 벡터는 다음과 같습니다.

<br>

- $$ \begin{bmatrix} L_{1} - d \\ c \end{bmatrix}, \ \ \begin{bmatrix} L_{2} - d \\ c \end{bmatrix}$$

<br>

- 만약 행렬 $$ A $$의 $$ b $$가 0이 아니면 고유 벡터는 다음과 같습니다.

<br>

- $$ \begin{bmatrix} b \\ L_{1} - a \end{bmatrix}, \ \ \begin{bmatrix} b \\ L_{2} - a \end{bmatrix} $$

<br>

- 만약 행렬 $$ A $$의 $$ b , c $$ 둘 다 0이라면 고유 벡터는 다음과 같습니다.

<br>

- $$ \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \ \ \begin{bmatrix} 0 \\ 1 \end{bmatrix} $$

<br>

- 이 공식이 유도된 것은 2 x 2 행렬의 고유값 및 고유벡터는 $$ Av = \lambda v$$를 푸는 과정 중 $$ \lambda^{2} -(a+d)\lambda + (ad - bc) = 0 $$ 식을 푸는 과정 속에서 전개됩니다.