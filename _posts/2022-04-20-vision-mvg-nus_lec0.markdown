---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 0. Multiple View Geometry 용어 둘러보기
date: 2022-04-20 00:00:00
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [(멀티플 뷰 지오메트리). Multiple View Geometry] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : Multiple View Geometry in Computer Vision

<br>

- 이번 글에서는 Multiple View Geometry 책의 서론에 해당하는 부분을 요약하였습니다.
- 강의에 따로 포함되어 있지는 않기 때문에 책 내용을 요약하였고 전체 내용을 한번 소개하는 부분임에도 불구하고 중요한 개념이 많이 수록되어 있습니다.
- 추가적으로 Multiple View Geometry 책 전체에 걸쳐서 필요한 용어들을 추가적으로 정리합니다.

<br>

## **목차**

<br>

- ### [Introduction and Basic Concepts](#introduction-and-basic-concepts-1)
- ### [Affine and Euclidean Geometry](#affine-and-euclidean-geometry-1)
- ### [Camera projections](#)
- ### [Reconstruction from more than one view](#)
- ### [Three-view geometry](#)
- ### [Four view geometry and n-view reconstruction](#)
- ### [Transfer](#)
- ### [Euclidean reconstruction](#)
- ### [Auto-calibration](#)
- ### [3D graphical models](#)
- ### [video augmentation](#)

<br>

## **Introduction and Basic Concepts**

<br>

- `euclidean geometry`는 물체의 각도와 모양을 서술하는 기하입니다. 일반적인 공간에서 흔히 사용하는 기하학입니다. `euclidean geometry`에서 두 선은 교차한다는 성질을 이용하고 싶지만 평행하는 선에 대한 예외를 두어야 하는데 이 부분을 예외로 두지 않고 평행선은 무한에서 만난다고 가정하여 사고를 확장하고자 합니다. 이 때 두 평행선이 만나는 점을 `point at infinity`라고하며 `euclidiean geometry` → `projective geometry`로 개념이 확장 됩니다.

<br>

- 무한이라는 공간을 표현하기 위하여 `projective geometry`에서는 좌표 형식을 `homogeneous coordinate`로 변환을 하고자 합니다.
- 2D euclidean space에서는 (x, y)로 나타내었던 좌표를 (x, y, 1)로 나타내고자 합니다. 마지막 좌표인 1은 스케일 값 형태로 나타낸다면 (kx, ky, k)같은 형태로 나타낼 수 있으며 따라서 (x, y, 1)과 (kx, ky, k)는 `equivalent class`라고 표현합니다. 두 값 모두 (x/1, y/1) = (kx/k, ky/k) = (x, y)로 나타낼 수 있기 때문입니다.
- 이 때, (x, y, 0)은 `equivalent class`로 나타내어 지지 않습니다. x/0, y/0이 정의되어 지지 않기 때문입니다. 따라서 마지막 차원의 값이 0인 `homogeneous coordinate` 상의 좌표를 `point at infinity`라고 정의합니다.

<br>

- 이와 같은 방식으로 euclidean space $$ \mathbb{R}^n $$ 의 점을 $$ \mathbb{P}^{n} $$ 공간으로 확장할 수 있으며 $$ \mathbb{P}^{2} $$ 에서 무한대에 있는 점은 `line at infinity`라고 불리는 선을 만들 수 있고 $$ \mathbb{P}^{3} $$ 에서는 `plane at infinity`를 만들 수 있습니다.

<br>

- 어떤 점을 좌표로 표현하기 위해서는 특정 한 점을 원점으로 선택해야 합니다. 즉, 다른 점을 원점으로 이용한다면 기존의 좌표값은 달라지게 됩니다.
- 어떤 원점을 다른 위치로 이동하고 회전하는 방식을 통하여 euclidean space의 좌표를 변경할 수 있습니다. 이 때, euclidean space에서 발생하는 이동 및 회전에 관한 변환을 `euclidean transformation`이라고 합니다.
- `euclidean transformation`은 $$ \mathbb{R}^{n} $$ 에 선형 변환을 적용한 후 공간의 원점을 이동하는 것을 의미하는 반면 공간을 **다른 방향의 다른 비율**로 선형적으로 움직이거나 회전 및 늘리는 방법을 `affine transformation` 이라고 합니다.
- `affine(euclidean 포함) transformation`을 적용하면 `point at infinity`는 여전히 무한대에 존재합니다. 하지만 `projective transformation`을 적용하면 `point at infinity`는 더 이상 무한대에 존재하지 않으며 다른 점과 차이가 없어집니다. 즉, $$ \mathbb{R}^{n} $$ 공간 상에서의 `point at infinity`는 $$ \mathbb{P}^{n} $$ 에서 임의의 다른 점으로 변환되며 아래 식과 같이 나타내 집니다.

<br>

- $$ X' = H_{(n+1) \times (n+1)} X $$

<br>

- 컴퓨터 비전에서는 `projective space`는 **3D 세계를 표현하는 방법으로 사용**됩니다. 즉, 2D 상의 이미지를 2D projective space에 놓인 것으로 생각하고 이것을 3D로 생각하며 `point at infinity`, `line at infinity` 등을 `projective space`에서 다른 점 또는 선과 동일하게 취급하는 방식을 사용합니다.

<br>

## **Affine and Euclidean Geometry**

<br>

- 무한대 공간에 선 또는 면을 추가하여 `euclidean space`에서 `projective space`로 확장한 것을 살펴보았습니다.
- `euclidean space` $$ \mathbb{R}^{n} $$ 에서 사용하는 기하학을 `euclidean geometry`라고 하고 `projective space` $$ \mathbb{P}^{n} $$ 에서 사용하는 기하학을 `projective geometry`라고 정의하면 `affine geometry`는 어떤 의미를 가질까요?

<br>

- 먼저 `projective geometry`와 `affine geometry`를 구분해 보겠습니다.
- `affine geometry`의 정의는 `projective geometry` + `line at infinity`로 정의할 수 있고 한 공간에서의 `line at infinity`를 다른 공간에서의 `line at infinity`로 `projective transformation` 하는 것으로 이 방식을 `affine transformation`이라고 말할 수 있습니다.
- 기존의 `projective geometry`와 `affine geometry`의 차이점은 `line at infinity`가 무한에 있는 것인 지의 차이점입니다. `projective geometry`의 `line at infinity`는 유한한 공간에 존재하여 실제 평면 상에 평행선이 교차하는 반면에 `affine geometry`에서는 `line at infinity`가 무한한 공간에 존재하여 실제 평면 상에 평행선이 존재하지 않습니다.

<br>

- 따라서 지금 까지 내용을 정리하면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec0/1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>

## **Camera projections**

<br>

<br>

## **Reconstruction from more than one view**

<br>

<br>

## **Three-view geometry**

<br>

<br>

## **Four view geometry and n-view reconstruction**

<br>

<br>

## **Transfer**

<br>

<br>

## **Euclidean reconstruction**

<br>

<br>

## **Auto-calibration**

<br>

<br>

## **3D graphical models**

<br>

<br>

## **video augmentation**

<br>

<br>


<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>