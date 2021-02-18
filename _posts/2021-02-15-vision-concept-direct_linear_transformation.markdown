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

- 이번 글에서는 `Homography` 적용 시 4개의 점을 이용하여 3 X 3 Homography 행렬을 만드는 방법에 대하여 다루어 보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/direct_linear_transformation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이미지 변환을 할 때, 위 그림과 같이 왼쪽의 이미지를 오른쪽 이미지와 같이 기하학적 변환을 적용하곤 합니다.
- 이 때, 동일 평면 (coplanar) 상의 점들을 3차원 변환을 하기 위하여 `Homography(또는 Perspective Transformation, Projective Transformation)` 방법을 사용합니다.

<br>

