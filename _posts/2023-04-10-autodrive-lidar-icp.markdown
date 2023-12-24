---
layout: post
title: ICP (Iterative Closest Point) 와 Point Cloud Registration
date: 2023-04-10 00:00:00
img: autodrive/lidar/icp/0.png
categories: [autodrive-lidar] 
tags: [icp, iterative closest point, point cloud registration, svd, known data association, ] # add tag
---

<br>

- 이번 글은 이론적으로 `ICP (Iterative Closest Point)`에 대한 내용을 다루는 `Cyrill Stachniss`의 강의를 정리하는 내용입니다.
- 강의는 총 1시간 분량의 강의 3개로 구성되어 총 3시간 분량의 강의입니다.
- 아래 참조 내용은 실제 코드 구현 시 도움 받은 내용입니다.

<br>

- 참조 : https://mr-waguwagu.tistory.com/36
- 참조 : https://github.com/minsu1206/3D/tree/main

<br>

## **목차**

<br>

- ### [3D-3D ICP의 기본 내용](#3d-3d-icp의-기본-내용-1)
- ### [Part 1: Known Data Association & SVD](#part-1-known-data-association--svd-1)
- ### [Part 2: Unknown Data Association](#part-2-unknown-data-association-1)
- ### [Part 3: Non-linear Least Squares](#part-3-non-linear-least-squares-1)

<br>

## **3D-3D ICP의 기본 내용**

<br>

- 먼저 이번 글에서 다룰 강의 내용에 앞서서 간략하게 다룰 내용은 아래 강의 내용 중 `Part 1: Known Data Association & SVD`에 해당하는 내용입니다. 뒤의 강의 내용의 이해를 돕기 위하여 아래와 같이 먼저 정리하였습니다.

<br>

- 아래와 같이 2개의 점군 $$ P, P' $$ 가 있고 각 $$ i $$ 번째 점인 $$ p_{i}, p'_{i} $$ 가 서로 대응되는 것을 알고 있는 상황이라고 가정합니다.

<br>

- $$ P = {p_{1}, p_{2}, ... , p_{n}} $$

- $$ P' = {p'_{1}, p'_{2}, ..., p'_{n}} $$

<br>

- 이 때, $$ p_{i}, p'_{i} $$ 의 관계를 알기 위해서는 `Rotation`을 위한 행렬 $$ R $$ 과 `Translation`을 위한 행렬 $$ t $$ 가 필요합니다. 

<br>

- $$ \forall_{i} \ \ p_{i} = Rp'_{i} + t $$

<br>

- 이상적인 환경에서는 모든 $$ i $$ 에 대하여 $$ p_{i} = Rp'_{i} + t $$ 를 만족해야 하지만 현실적으로 오차가 포함되기 때문에 **전체의 오차가 최소화 되는 방향으로 근사화** 시키는 최적해를 구하는 방법을 이용하여 `ICP`를 적용합니다.
- `RGB-D 카메라`를 이용하거나 이미지에서 `Feature Extraction 및 Matching`을 하여 점들끼리 쌍을 매칭한 점 군 $$ P, P' $$ 를 구한 경우에 지금부터 설명할 방법을 사용할 수 있습니다.

<br>

- 먼저 아래와 같이 $$ i $$ 점의 오차를 정의 합니다.

<br>

- $$ e_{i} = p_{i} - (Rp'_{i} + t) $$

<br>

- 풀어야 할 문제는 모든 에러 $$ e_{i} $$ 를 최소화 시키는 목적 함수를 만들고 목적 함수를 최소화 시키는 문제를 푸는 것입니다. 따라서 다음과 같이 `오차 제곱 합`의 목적 함수를 만듭니다.

<br>

- $$ \min_{R,t} \frac{1}{2} \sum_{i=1}^{n} \Vert (p_{i} - (Rp'_{i} + t)) \Vert^{2} $$

<br>

- 위 식에서 $$ p_{i}, p'_{i} $$ 는 벡터이기 때문에 `norm`을 적용하여 크기값인 스칼라 값으로 바꾸어서 목적 함수의 결과로 둡니다.
- 위 식을 좀 더 단순화하여 전개하기 위해 `두 점군의 중심 위치 (centroid)`를 정의해 보도록 하겠습니다.

<br>

- $$ p = \frac{1}{n} \sum_{i=1}^{n}(p_{i}) $$

- $$ p' = \frac{1}{n} \sum_{i=1}^{n}(p'_{i}) $$

<br>





<br>

- 지금까지 살펴본 방법은 매칭이 주어질 때, $$ R, t $$ 를 추정하는 문제에 해당합니다.
- 매칭을 알고있는 경우에는 최소 제곱 문제를 해결하기 위한 `analytic solution`이 존재하기 때문에 `numerical solution`을 이용한 최적화가 반드시 필요하진 않습니다.
- 하지만 Feature 매칭에 오류가 있거나 추출한 Feature의 $$ X, Y, Z $$ 값이 부정확한 경우 별도의 노이즈를 제거해야 좋은 $$ R, t $$ 값을 구할 수 있으므로 이 부분은 고려해야 합니다.

<br>

- 따라서 간단하면서도 효율적인 `RANSAC`을 적용하여 




<br>

- 지금까지 살펴본 내용은 두 점군의 쌍을 매칭할 수 있을 때, 

<br>

## **Part 1: Known Data Association & SVD**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/dhzLQfDBx2Q" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

## **Part 2: Unknown Data Association**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/ktRqKxddjJk" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

## **Part 3: Non-linear Least Squares**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/CJE59i8oxIE" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

