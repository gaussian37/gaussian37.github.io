---
layout: post
title: 이미지 Geometric Transformation 알아보기
date: 2019-12-26 00:00:00
img: vision/concept/geometric_transformation/0.png
categories: [vision-concept] 
tags: [vision, 2d transformation, ] # add tag
---

<br>

- 참조 : Computer Vision Algorithms and Applications (Richard Szeliski)
- 참조 : Introduction to Computer Vision
- 참조 : OpenCV를 활용한 컴퓨터 비전

<br>

- 이번 글에서는 2D 이미지를 다양한 방법으로 Transformation하는 방법들에 대하여 총 정리해 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### Geometric Transformation (기하학적 변환)의 정의
- ### Translation transformation (이동 변환)

<br>

## **Geometric Transformation (기하학적 변환)의 정의**

<br>

- 흔히 알려진 기본적인 이미지 처리 방법인 필터 적용, 밝기 변화, 블러 적용등은 픽셀 단위 별로 변환을 주는 작업입니다. 이는 픽셀 값의 변화가 있을 뿐 픽셀의 위치 이동은 없습니다. 
- 영상의 `기하학적 변환 (Geometric Transformation)`이란 영상을 구성하는 픽셀이 배치된 구조를 변경함으로써 **전체 영상의 모양을 바꾸는 작업**을 뜻합니다. 즉, 어떤 픽셀의 좌표가 다른 좌표로 이동되는 경우를 말합니다.

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 대표적인 Geometric Transformation의 예시를 나타냅니다. 이 중에서 `affine` 변환과 `projective` 변환에 대해서는 이 글의 내용을 읽어 보시길 권장드립니다. 그 만큼 중요합니다.

<br>

## **Translation transformation (이동 변환)**

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 영상의 이동 변환에 대하여 알아보도록 하겠습니다. 영상의 픽셀 값이 한 쪽 방향으로 이동한 경우를 나타내며 shift 라고도 합니다.
- 이동 변환에서는 영상이 $$ x $$방향 또는 $$ y $$방향으로 이동이 발생합니다. 따라서 `이동 변위`를 지정해 주어야 합니다.
- 위 그림에서 $$ x $$ 방향으로 $$ a $$ 만큼, $$ y $$ 방향으로 $$ b $$ 만큼 이동한 것을 확인할 수 있습니다.

<br>

- $$ \begin{cases} x' = x + a \\[2ex] y' = y + b \end{cases} $$

<br>

- 영상 처리에서 이러한 수식을 편리하게 사용하기 위해서는 수식을 행렬로 나타내어야 합니다. 따라서 다음과 같이 나타낼 수 있습니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} a \\ b \end{bmatrix} $$

<br>

- 간단하게 수식을 위 행렬처럼 변환할 수 있습니다. 영상 처리에서는 위 수식을 다음과 같이 **곱셈 하나의 형태**로 표시하기도 합니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \end{bmatrix} =\begin{bmatrix} 1 & 0 & a \\ 0 & 1 & b\end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$

<br>

- 여기서 $$ \begin{bmatrix} 1 & 0 & a \\ 0 & 1 & b\end{bmatrix} $$을 **이동 변환**을 나타내기 위한 `affine` 변환 행렬 이라고 합니다.

