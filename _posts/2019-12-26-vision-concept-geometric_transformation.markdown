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

- ### [Geometric Transformation (기하학적 변환)의 정의](#geometric-transformation-기하학적-변환의-정의-1)
- ### [Translation Transformation (이동 변환)](#translation-transformation-이동-변환-1)
- ### [Shear Transformation (전단 변환)](#shear-transformation-전단-변환-1)
- ### [Scale Transformation (크기 변환)](#scale-transformation-크기-변환-1)
- ### [Reflection Transformation (대칭 변환)](#reflection-transformation-대칭-변환-1)
- ### [Rotation Transformation (회전 변환)](#rotation-transformation-회전-변환-1)

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

<br>

- OpenCV에서 Affine 변환 행렬을 이용하여 영상의 이동 변환을 하는 방법은 아래 링크를 참조하시기 바랍니다.
    - 링크 : [warpAffine을 이용한 기하학적 변환](https://gaussian37.github.io/vision-opencv_python_snippets/#warpaffine%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EA%B8%B0%ED%95%98%ED%95%99%EC%A0%81-%EB%B3%80%ED%99%98-1)

<br>

## **Shear Transformation (전단 변환)**

<br>

- 영상의 전단 변환은 $$ x, y $$ 축 방향으로 영상이 밀리는 것 처럼 보이는 변환을 뜻합니다. (아래 그림 참조) 따라서 축에 따라서 픽셀의 이동 비율이 달라집니다.
- 따라서 전단 변환의 결과로 한쪽으로 기울어진 영상을 만들어 낼 수 있습니다. 따라서 각 방향으로 기울어진 변환을 적용하기 위하여 $$ x $$축과 $$ y $$축 각각에 대하여 변환을 적용하면 됩니다. 아래와 같습니다.

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 전단 변환 또한 affine 행렬을 통하여 변환 상태를 나타낼 수 있습니다. affine 행렬로 나타내는 방식은 이동 변환과 똑같은 방식입니다.

<br>

- $$ \begin{cases} x' = x + my \\ y' = y\end{cases} $$

<br>

- 위 수식을 보면 $$ x $$ 좌표에 $$ y $$ 좌표 값에 스케일링 값 $$ m $$을 곱한 값을 더하여 $$ x $$값을 변환하였습니다.
- 이 방식을 통하여 $$ x $$ 값은 변환하고 $$ y $$ 값은 고정 시킬 수 있습니다.

<br>

- $$ \begin{cases} x' = x \\ y' = mx + y \end{cases} $$

<br>

- 위 식은 앞의 식과 기준 축을 바꾼 경우이고 원리는 같습니다.

<br>

## **Scale Transformation (크기 변환)**

<br>

- 영상을 확대하거나 축소할 수 있는 크기 변환에 대하여 알아보도록 하겠습니다.
- 크기 변환을 하기 위해서는 $$ x $$ 방향 또는 $$ y $$ 방향으로 얼마나 크게 또는 작게 만들기 위한 `scale factor`가 필요로 합니다.

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 확대 및 축소를 위한 비율은 입력 영상과 출력 영상의 비율을 이용하여 `scale factor`를 구할 수 있습니다.
- 이 factor를 위 수식과 같이 $$ S_{x}, S_{y} $$로 구할 수 있고 이 값을 이용하여 affine 행렬을 만들 수 있습니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} S_{x} & 0 & 0 \\ 0 & S_{y} & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1\end{bmatrix} $$

<br>

- 따라서 위 식과 같은 affine 행렬을 이용하면 이미지의 크기 변환을 적용할 수 있습니다.

<br>

- 영상의 크기 변환은 기하학적 변환 중에서도 굉장히 많이 사용되는 변환입니다. 따라서 affine 변환이 아닌 변도의 `resize` 함수를 이용하여 사용하곤 합니다.

<br>

- 영상의 크기 변환은 Affine 변환을 이용하여 할 수도 있지만 좀 더 편하게 사용하기 위하여 `resize` 함수라는 별도 함수를 이용하여 크기 변환을 할 수 있습니다. 다음 링크에서 기능을 확인하시기 바랍니다.
- 링크 : [https://gaussian37.github.io/vision-opencv_python_snippets/#resize를-이용한-크기-변환-1](https://gaussian37.github.io/vision-opencv_python_snippets/#resize를-이용한-크기-변환-1)

<br>

## **Reflection Transformation (대칭 변환)**

<br>

- 이미지의 대칭 변환은 **크기 변환 + 이동 변환**을 조합한 결과를 통해 만들어 낼 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들어 좌우 대칭된 변환은 크기를 -1배하여 변환할 수 있고 (위 그림의 왼쪽) 그 결과를 이동 변환을 통하여 (위 그림의 오른쪽) 기존 이미지를 대칭한 것 처럼 변환할 수 있습니다.
- 이 방법은 좌우 대칭, 상하 대칭, 좌우 상하 대칭에 모두 적용할 수 있습니다.
- 실제 구현할 때에는 간단하게 opencv의 flip 함수를 사용하면 됩니다. 아래 링크를 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/vision-opencv_python_snippets/#flip을-이용한-대칭-변환-1](https://gaussian37.github.io/vision-opencv_python_snippets/#flip을-이용한-대칭-변환-1)

<br>

## **Rotation Transformation (회전 변환)**

<br>

- 이번에는 영상을 회전하는 회전 변환에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/geometric_transformation/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 영상을 회전 하기 위한 변환은 위 식과 같이 $$ \sin{\theta}, \cos{\theta} $$의 식을 이용하여 표현할 수 있습니다.
- 위 식을 통하여 변환되는 $$ x, y $$ 좌표를 통하여 원하는 각도 만큼 이미지를 회전할 수 있습니다.
- 앞의 다른 변환과 마찬가지로 affine 행렬을 이용하여 변환하려면 2 X 3 크기의 행렬을 사용할 수 있습니다.

<br>

- $$ \begin{bmatrix} \cos{\theta} & \sin{\theta} & 0 \\ -\sin{\theta} & \cos{\theta} & 0 \end{bmatrix} $$

<br>